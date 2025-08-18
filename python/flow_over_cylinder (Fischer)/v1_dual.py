# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import njit, prange
import os
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc
import h5py
import logging
from tqdm import tqdm

# Set matplotlib backend and style
plt.switch_backend('Agg')
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/v1_dual_re_600.log')
    ]
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

@dataclass
class DualCylinderTurbulentConfig:
    L: float = 1.0
    R_cylinder: float = 0.4  # Slightly smaller radius for better resolution
    V_inf: float = 1.0
    # Two cylinders positioned side by side
    cylinder1_center: tuple[float, float] = (4.0, 1.5)  # Upper cylinder
    cylinder2_center: tuple[float, float] = (4.0, 2.5)  # Lower cylinder
    cylinder_separation: float = 1.0  # Distance between cylinder centers
    x_min: float = 0.0
    x_max: float = 20.0
    y_min: float = 0.0
    y_max: float = 4.0
    nx: int = 600
    ny: int = 180
    T_total: float = 30.0
    dt_base: float = 0.00005
    cfl_target: float = 0.1
    adaptive_dt: bool = True
    dt_min: float = 1e-6
    dt_max: float = 0.0001
    Re: float = 600.0
    use_les: bool = False
    smagorinsky_constant: float = 0.0
    use_supg: bool = True
    artificial_viscosity: float = 0.001
    pressure_iterations: int = 1500
    pressure_tolerance: float = 1e-8
    max_velocity: float = 5.0
    initial_steps: int = 1000
    parallel_threads: int = 4
    use_fast_pressure: bool = True
    memory_efficient: bool = True
    vectorized_ops: bool = True
    save_interval: int = 200
    output_dir: str = "v1_dual_re_600"
    hdf5_file: str = f"{output_dir}.h5"
    dpi: int = 200

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dy = (self.y_max - self.y_min) / (self.ny - 1)
        self.nu = np.float32(1.0 / self.Re)
        self.dt = np.float32(self.dt_base)
        self.artificial_viscosity = np.float32(self.artificial_viscosity)
        self.parallel_threads = min(self.parallel_threads, os.cpu_count())
        memory_mb = (self.nx * self.ny * 4 * 6) / (1024 * 1024)
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Calculate actual separation distance
        self.cylinder_separation = np.sqrt(
            (self.cylinder2_center[0] - self.cylinder1_center[0])**2 + 
            (self.cylinder2_center[1] - self.cylinder1_center[1])**2
        )
        
        logger.info("--- Dual Cylinder Turbulent Flow Configuration ---")
        logger.info(f"Reynolds Number: {self.Re}")
        logger.info(f"Grid: {self.nx}x{self.ny}")
        logger.info(f"Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}")
        logger.info(f"Cylinder 1 center: ({self.cylinder1_center[0]:.2f}, {self.cylinder1_center[1]:.2f})")
        logger.info(f"Cylinder 2 center: ({self.cylinder2_center[0]:.2f}, {self.cylinder2_center[1]:.2f})")
        logger.info(f"Cylinder separation: {self.cylinder_separation:.2f}L")
        logger.info(f"Cylinder radius: {self.R_cylinder:.2f}L")
        logger.info(f"Separation/Diameter ratio: {self.cylinder_separation/(2*self.R_cylinder):.2f}")
        logger.info(f"Parallel threads: {self.parallel_threads}")
        logger.info(f"Estimated Memory: {memory_mb:.1f} MB")
        logger.info(f"Available Memory: {available_mb:.1f} MB")
        logger.info(f"Target performance: >15 steps/sec")
        logger.info("--------------------------------------------")

@njit(parallel=True, fastmath=True, cache=True)
def compute_smagorinsky_viscosity_fast(u, v, dx, dy, cs):
    ny, nx = u.shape
    nu_t = np.zeros_like(u)
    delta = (dx * dy) ** 0.5
    cs_delta_sq = (cs * delta) ** 2
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            dudx = (u[i, j+1] - u[i, j]) / dx
            dudy = (u[i+1, j] - u[i, j]) / dy
            dvdx = (v[i, j+1] - v[i, j]) / dx
            dvdy = (v[i+1, j] - v[i, j]) / dy
            S_mag = (2 * (dudx*dudx + dvdy*dvdy) + (dudy + dvdx)**2) ** 0.5
            nu_t[i, j] = cs_delta_sq * S_mag
    return nu_t

@njit(parallel=True, fastmath=True, cache=True)
def compute_convection_fast(u, v, phi, dx, dy):
    ny, nx = phi.shape
    conv = np.zeros_like(phi)
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            u_vel = u[i, j]
            v_vel = v[i, j]
            dphidx = (phi[i, j] - phi[i, j-1]) * dx_inv if u_vel > 0 else (phi[i, j+1] - phi[i, j]) * dx_inv
            dphidy = (phi[i, j] - phi[i-1, j]) * dy_inv if v_vel > 0 else (phi[i+1, j] - phi[i, j]) * dy_inv
            conv[i, j] = u_vel * dphidx + v_vel * dphidy
    return conv

@njit(parallel=True, fastmath=True, cache=True)
def compute_convection_supg_fast(u, v, phi, dx, dy, tau_supg):
    ny, nx = phi.shape
    conv = np.zeros_like(phi)
    dx_inv = 0.5 / dx
    dy_inv = 0.5 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            u_vel = u[i, j]
            v_vel = v[i, j]
            dphidx = (phi[i, j+1] - phi[i, j-1]) * (0.5 * dx_inv)
            dphidy = (phi[i+1, j] - phi[i-1, j]) * (0.5 * dy_inv)
            conv_standard = u_vel * dphidx + v_vel * dphidy
            if tau_supg[i, j] > 0:
                d2phidx2 = (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) * (dx_inv * dx_inv)
                d2phidy2 = (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) * (dy_inv * dy_inv)
                supg_term = tau_supg[i, j] * (u_vel * d2phidx2 + v_vel * d2phidy2)
                conv[i, j] = conv_standard - supg_term
            else:
                conv[i, j] = conv_standard
    return conv

@njit(parallel=True, fastmath=True, cache=True)
def compute_supg_stabilization_fast(u, v, dx, dy, dt, nu_eff):
    ny, nx = u.shape
    tau_supg = np.zeros_like(u)
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            vel_mag = (u[i, j]**2 + v[i, j]**2) ** 0.5
            h_elem = min(dx, dy)
            if vel_mag > 1e-10:
                Pe = vel_mag * h_elem / (nu_eff[i, j] + 1e-10)
                tau_supg[i, j] = h_elem / (2 * vel_mag) * min(1.0, Pe / 2.0)
            else:
                tau_supg[i, j] = dt / 2
    return tau_supg

@njit(parallel=True, fastmath=True, cache=True)
def compute_laplacian_fast(phi, dx, dy, nu_eff):
    ny, nx = phi.shape
    lap = np.zeros_like(phi)
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            lap[i, j] = nu_eff[i, j] * (
                (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) * dx2_inv +
                (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) * dy2_inv
            )
    return lap

@njit(parallel=True, fastmath=True, cache=True)
def compute_divergence_fast(u, v, dx, dy):
    ny, nx = u.shape
    div = np.zeros_like(u)
    dx_inv = 0.5 / dx
    dy_inv = 0.5 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            div[i, j] = (u[i, j+1] - u[i, j-1]) * dx_inv + (v[i+1, j] - v[i-1, j]) * dy_inv
    return div

@njit(parallel=True, fastmath=True, cache=True)
def compute_gradient_fast(phi, dx, dy):
    ny, nx = phi.shape
    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    dx_inv = 0.5 / dx
    dy_inv = 0.5 / dy
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            grad_x[i, j] = (phi[i, j+1] - phi[i, j-1]) * dx_inv
            grad_y[i, j] = (phi[i+1, j] - phi[i-1, j]) * dy_inv
    return grad_x, grad_y

@njit(parallel=True, fastmath=True, cache=True)
def solve_pressure_gauss_seidel_fast(phi, div_u_star, dx, dy, dt, mask, iterations, tolerance):
    ny, nx = phi.shape
    dx2 = dx * dx
    dy2 = dy * dy
    dx2_inv = 1.0 / dx2
    dy2_inv = 1.0 / dy2
    denom_inv = 1.0 / (2.0 * (dx2_inv + dy2_inv))
    dt_inv = 1.0 / dt
    for iteration in range(iterations):
        max_change = 0.0
        for color in range(2):
            for i in prange(1, ny-1):
                for j in range(1 + (i + color) % 2, nx-1, 2):
                    if not mask[i, j]:
                        rhs = -div_u_star[i, j] * dt_inv
                        phi_new = (dx2_inv * (phi[i, j+1] + phi[i, j-1]) +
                                   dy2_inv * (phi[i+1, j] + phi[i-1, j]) - rhs) * denom_inv
                        change = abs(phi_new - phi[i, j])
                        if change > max_change:
                            max_change = change
                        phi[i, j] = phi_new
        if max_change < tolerance:
            break
    return phi

@njit(parallel=True, fastmath=True, cache=True)
def apply_dual_ibm_fast(u, v, ibm_mask1, ibm_mask2, force_strength):
    ny, nx = u.shape
    for i in prange(ny):
        for j in prange(nx):
            # Apply force from both cylinders (taking maximum)
            mask_val = max(ibm_mask1[i, j], ibm_mask2[i, j])
            if mask_val > 0:
                u[i, j] *= (1.0 - mask_val * force_strength)
                v[i, j] *= (1.0 - mask_val * force_strength)
    return u, v

@njit(parallel=True, fastmath=True, cache=True)
def clean_divergence_fast(u, v, dx, dy, iterations=2):
    ny, nx = u.shape
    phi = np.zeros_like(u)
    dx2 = dx * dx
    dy2 = dy * dy
    dx2_inv = 1.0 / dx2
    dy2_inv = 1.0 / dy2
    denom_inv = 1.0 / (2.0 * (dx2_inv + dy2_inv))
    for _ in range(iterations):
        div = compute_divergence_fast(u, v, dx, dy)
        for i in prange(1, ny-1):
            for j in prange(1, nx-1):
                phi[i, j] = (dx2_inv * (phi[i, j+1] + phi[i, j-1]) +
                             dy2_inv * (phi[i+1, j] + phi[i-1, j]) - div[i, j]) * denom_inv
        grad_x, grad_y = compute_gradient_fast(phi, dx, dy)
        u[1:-1, 1:-1] -= grad_x[1:-1, 1:-1]
        v[1:-1, 1:-1] -= grad_y[1:-1, 1:-1]
    return u, v

class DualCylinderTurbulentSolver:
    def __init__(self, config: DualCylinderTurbulentConfig):
        self.config = config
        self.setup_grid()
        self.setup_dual_boundary_masks()
        self.initialize_fields()
        self.step = 0
        self.energy_history = []
        self.times = []

    def setup_grid(self):
        cfg = self.config
        self.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        self.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

    def setup_dual_boundary_masks(self):
        cfg = self.config
        x1_c, y1_c = cfg.cylinder1_center
        x2_c, y2_c = cfg.cylinder2_center
        
        # Distance to each cylinder
        self.dist1 = np.sqrt((self.X - x1_c)**2 + (self.Y - y1_c)**2)
        self.dist2 = np.sqrt((self.X - x2_c)**2 + (self.Y - y2_c)**2)
        
        # Cylinder masks
        self.cylinder1_mask = self.dist1 <= cfg.R_cylinder
        self.cylinder2_mask = self.dist2 <= cfg.R_cylinder
        self.cylinder_mask = self.cylinder1_mask | self.cylinder2_mask
        
        # IBM masks with smooth transitions
        sigma = 2 * cfg.dx
        self.ibm_mask1 = np.exp(-((self.dist1 - cfg.R_cylinder) / sigma)**2)
        self.ibm_mask1 = np.where(self.dist1 < cfg.R_cylinder, 1.0,
                                 np.where(self.dist1 < cfg.R_cylinder + 5*cfg.dx, self.ibm_mask1, 0.0))
        
        self.ibm_mask2 = np.exp(-((self.dist2 - cfg.R_cylinder) / sigma)**2)
        self.ibm_mask2 = np.where(self.dist2 < cfg.R_cylinder, 1.0,
                                 np.where(self.dist2 < cfg.R_cylinder + 5*cfg.dx, self.ibm_mask2, 0.0))

    def initialize_fields(self):
        cfg = self.config
        dtype = np.float32 if cfg.memory_efficient else np.float64
        self.u = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.v = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.p = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.nu_t = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.tau_supg = np.zeros((cfg.ny, cfg.nx), dtype=dtype, order='C')
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.div_u_star = np.zeros_like(self.u)
        self.phi = np.zeros_like(self.u)
        self.initialize_dual_potential_flow()

    def initialize_dual_potential_flow(self):
        cfg = self.config
        x1_c, y1_c = cfg.cylinder1_center
        x2_c, y2_c = cfg.cylinder2_center
        
        for i in range(cfg.ny):
            for j in range(cfg.nx):
                # Check if point is far enough from both cylinders
                r1 = self.dist1[i, j]
                r2 = self.dist2[i, j]
                min_dist = min(r1, r2)
                
                mask_val = max(self.ibm_mask1[i, j], self.ibm_mask2[i, j])
                
                if min_dist > cfg.R_cylinder + 4*cfg.dx:
                    # Superposition of potential flows around both cylinders
                    # Flow around cylinder 1
                    theta1 = np.arctan2(self.Y[i, j] - y1_c, self.X[i, j] - x1_c)
                    factor1 = (cfg.R_cylinder / r1)**2
                    u1 = cfg.V_inf * (1 - factor1 * np.cos(2 * theta1))
                    v1 = -cfg.V_inf * factor1 * np.sin(2 * theta1)
                    
                    # Flow around cylinder 2
                    theta2 = np.arctan2(self.Y[i, j] - y2_c, self.X[i, j] - x2_c)
                    factor2 = (cfg.R_cylinder / r2)**2
                    u2 = cfg.V_inf * (1 - factor2 * np.cos(2 * theta2))
                    v2 = -cfg.V_inf * factor2 * np.sin(2 * theta2)
                    
                    # Superposition with weighting based on distance
                    w1 = 1.0 / (r1 + 1e-6)
                    w2 = 1.0 / (r2 + 1e-6)
                    w_total = w1 + w2
                    
                    self.u[i, j] = ((w1 * u1 + w2 * u2) / w_total + cfg.V_inf) * 0.5 * (1 - mask_val)
                    self.v[i, j] = (w1 * v1 + w2 * v2) / w_total * (1 - mask_val)
                else:
                    # Blend to uniform flow in near-cylinder regions
                    blend = min(1.0, ((min_dist - cfg.R_cylinder) / (4 * cfg.dx))**2)
                    self.u[i, j] = cfg.V_inf * blend * (1 - mask_val)
                    self.v[i, j] = 0.0

    def adaptive_time_step(self):
        if not self.config.adaptive_dt:
            return self.config.dt_base
        cfg = self.config
        if self.step < 1000:
            return np.float32(0.00002)
        vel_max = max(np.max(np.abs(self.u)), np.max(np.abs(self.v)), 1e-10)
        dt_cfl = cfg.cfl_target * min(cfg.dx, cfg.dy) / vel_max
        nu_total = cfg.nu + np.mean(self.nu_t) + cfg.artificial_viscosity
        dt_visc = 0.4 * min(cfg.dx, cfg.dy)**2 / nu_total
        return np.float32(np.clip(min(dt_cfl, dt_visc), cfg.dt_min, cfg.dt_max))

    def solve_pressure_fast(self, div_u_star):
        cfg = self.config
        if cfg.use_fast_pressure:
            self.phi.fill(0.0)
            self.phi = solve_pressure_gauss_seidel_fast(
                self.phi, div_u_star, cfg.dx, cfg.dy, cfg.dt,
                self.cylinder_mask, cfg.pressure_iterations, cfg.pressure_tolerance
            )
        else:
            self.phi.fill(0.0)
            for _ in range(cfg.pressure_iterations):
                phi_new = self.phi.copy()
                phi_new[1:-1, 1:-1] = 0.25 * (
                    self.phi[1:-1, 2:] + self.phi[1:-1, :-2] +
                    self.phi[2:, 1:-1] + self.phi[:-2, 1:-1] -
                    cfg.dx**2 * div_u_star[1:-1, 1:-1] / cfg.dt
                )
                phi_new[self.cylinder_mask] = 0
                self.phi = phi_new
        return self.phi

    def apply_boundary_conditions(self, u, v):
        cfg = self.config
        # Add slight perturbation for instability initiation
        pert_scale = min(1.0, self.step / 1000.0) * 0.01
        perturbation = pert_scale * np.sin(2 * np.pi * self.y / cfg.y_max + 0.02 * self.step)
        
        # Inlet boundary
        u[:, 0] = cfg.V_inf * (1 + perturbation)
        v[:, 0] = 0
        
        # Outlet boundary (convective)
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        
        # Wall boundaries
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

    def compute_energy(self):
        return 0.5 * (self.u**2 + self.v**2)

    def compute_vorticity(self):
        cfg = self.config
        vorticity = np.zeros_like(self.u)
        vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * cfg.dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * cfg.dy)
        )
        vorticity[self.cylinder_mask] = np.nan
        return vorticity

    def time_step(self):
        cfg = self.config
        dt = self.adaptive_time_step()
        u_old = self.u.copy()
        v_old = self.v.copy()
        
        # Compute effective viscosity
        nu_eff = cfg.nu + self.config.artificial_viscosity
        if cfg.use_les:
            self.nu_t = compute_smagorinsky_viscosity_fast(
                u_old, v_old, cfg.dx, cfg.dy, cfg.smagorinsky_constant
            )
        else:
            self.nu_t.fill(0.0)

        nu_eff = cfg.nu + self.nu_t + cfg.artificial_viscosity
        
        # Compute convection and diffusion
        if cfg.use_supg:
            self.tau_supg = compute_supg_stabilization_fast(
                u_old, v_old, cfg.dx, cfg.dy, dt, nu_eff
            )
            conv_u = compute_convection_supg_fast(u_old, v_old, u_old, cfg.dx, cfg.dy, self.tau_supg)
            conv_v = compute_convection_supg_fast(u_old, v_old, v_old, cfg.dx, cfg.dy, self.tau_supg)
        else:
            conv_u = compute_convection_fast(u_old, v_old, u_old, cfg.dx, cfg.dy)
            conv_v = compute_convection_fast(u_old, v_old, v_old, cfg.dx, cfg.dy)

        lap_u = compute_laplacian_fast(u_old, cfg.dx, cfg.dy, nu_eff)
        lap_v = compute_laplacian_fast(v_old, cfg.dx, cfg.dy, nu_eff)

        # Predictor step
        self.u_star[:] = u_old + dt * (-conv_u + lap_u)
        self.v_star[:] = v_old + dt * (-conv_v + lap_v)

        # Apply boundary conditions and IBM
        self.apply_boundary_conditions(self.u_star, self.v_star)
        force_strength = min(1.0, self.step / cfg.initial_steps)
        self.u_star, self.v_star = apply_dual_ibm_fast(
            self.u_star, self.v_star, self.ibm_mask1, self.ibm_mask2, force_strength
        )

        # Pressure correction
        self.div_u_star = compute_divergence_fast(self.u_star, self.v_star, cfg.dx, cfg.dy)
        self.phi = self.solve_pressure_fast(self.div_u_star)

        dpdx, dpdy = compute_gradient_fast(self.phi, cfg.dx, cfg.dy)
        self.u[:] = self.u_star - dt * dpdx
        self.v[:] = self.v_star - dt * dpdy

        # Clean up divergence
        self.u, self.v = clean_divergence_fast(self.u, self.v, cfg.dx, cfg.dy, iterations=2)

        # Final boundary conditions and IBM
        self.apply_boundary_conditions(self.u, self.v)
        self.u, self.v = apply_dual_ibm_fast(
            self.u, self.v, self.ibm_mask1, self.ibm_mask2, force_strength
        )

        # Monitoring
        vorticity = self.compute_vorticity()
        energy = self.compute_energy()
        energy_mean = np.nanmean(energy)
        self.energy_history.append((self.step, energy_mean))
        self.times.append(self.step * dt)

        # Velocity limiting
        np.clip(self.u, -cfg.max_velocity, cfg.max_velocity, out=self.u)
        np.clip(self.v, -cfg.max_velocity, cfg.max_velocity, out=self.v)

        self.step += 1
        return dt

class DualCylinderVisualizer:
    def __init__(self, config: DualCylinderTurbulentConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        self.velocity_path = self.output_path / "velocity_frames"
        self.vorticity_path = self.output_path / "vorticity_frames"
        self.velocity_path.mkdir(exist_ok=True)
        self.vorticity_path.mkdir(exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def save_data_to_hdf5(self, solver: DualCylinderTurbulentSolver, step: int, current_time: float):
        cfg = self.config
        try:
            with h5py.File(self.output_path / cfg.hdf5_file, 'a') as f:
                group_name = f"step_{step:06d}"
                if group_name not in f:
                    group = f.create_group(group_name)
                    group.attrs['time'] = current_time
                    group.create_dataset('u', data=solver.u, compression='gzip', compression_opts=4)
                    group.create_dataset('v', data=solver.v, compression='gzip', compression_opts=4)
                    group.create_dataset('vorticity', data=solver.compute_vorticity(),
                                        compression='gzip', compression_opts=4)
                    group.create_dataset('X', data=solver.X, compression='gzip', compression_opts=4)
                    group.create_dataset('Y', data=solver.Y, compression='gzip', compression_opts=4)
                logger.info(f"Saved data for step {step} to HDF5")
        except Exception as e:
            logger.error(f"Error saving HDF5 data for step {step}: {e}")

    def generate_frames_from_hdf5(self):
        cfg = self.config
        output_path = self.output_path
        hdf5_path = output_path / cfg.hdf5_file
        if not hdf5_path.exists():
            logger.warning(f"HDF5 file {hdf5_path} does not exist. Skipping frame generation.")
            return
        with h5py.File(hdf5_path, 'r') as f:
            steps = sorted([k for k in f.keys() if k.startswith('step_')],
                          key=lambda x: int(x.split('_')[1]))
            for step_key in tqdm(steps, desc="Generating Frames", unit="frame"):
                step = int(step_key.split('_')[1])
                group = f[step_key]
                u = group['u'][:]
                v = group['v'][:]
                vorticity = group['vorticity'][:]
                X = group['X'][:]
                Y = group['Y'][:]
                current_time = group.attrs['time']
                
                try:
                    # Velocity field visualization
                    fig = plt.figure(figsize=(14, 7))
                    ax = fig.add_subplot(111)
                    vel_mag = np.sqrt(u**2 + v**2)
                    levels = np.linspace(0, np.nanmax(vel_mag)*0.9, 31)
                    cf = ax.contourf(X, Y, vel_mag, levels=levels, cmap='viridis')
                    plt.colorbar(cf, ax=ax, label='Velocity Magnitude |V|', shrink=0.8)
                    
                    # Streamlines
                    skip = max(15, min(X.shape) // 15)
                    seed_points = np.array([
                        [cfg.x_min + 1, y] for y in np.linspace(cfg.y_min + 0.3, cfg.y_max - 0.3, 7)
                    ])
                    ax.streamplot(X, Y, u, v,
                                  color='white', linewidth=0.6, density=0.8,
                                  start_points=seed_points, maxlength=50)
                    
                    # Vector field
                    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                              u[::skip, ::skip], v[::skip, ::skip],
                              color='lightgray', scale=40, alpha=0.3)
                    
                    # Draw both cylinders
                    cyl1 = patches.Circle(cfg.cylinder1_center, cfg.R_cylinder,
                                         facecolor='darkred', edgecolor='gold', linewidth=1.5)
                    cyl2 = patches.Circle(cfg.cylinder2_center, cfg.R_cylinder,
                                         facecolor='darkblue', edgecolor='gold', linewidth=1.5)
                    ax.add_patch(cyl1)
                    ax.add_patch(cyl2)
                    
                    ax.set_xlim(cfg.x_min, cfg.x_max)
                    ax.set_ylim(cfg.y_min, cfg.y_max)
                    ax.set_aspect('equal')
                    ax.set_xlabel('x/L')
                    ax.set_ylabel('y/L')
                    ax.set_title(f'Dual Cylinder Velocity Field, Re={cfg.Re:.0f}, t={current_time:.2f}')
                    ax.grid(True, alpha=0.2)
                    
                    # Add statistics
                    max_vel = np.nanmax(vel_mag)
                    mean_vel = np.nanmean(vel_mag)
                    fig.text(0.02, 0.02, f'Max |V|: {max_vel:.3f} | Mean |V|: {mean_vel:.3f} | S/D: {cfg.cylinder_separation/(2*cfg.R_cylinder):.2f}',
                             fontsize=8, color='white')
                    
                    plt.tight_layout()
                    filename = self.velocity_path / f"velocity_frame_{step:06d}.png"
                    plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Error plotting velocity frame {step}: {e}")
                    plt.close('all')
                
                try:
                    # Vorticity field visualization
                    fig = plt.figure(figsize=(14, 7))
                    ax = fig.add_subplot(111)
                    vort_max = min(np.nanmax(np.abs(vorticity)), 20.0)
                    levels = np.linspace(-vort_max, vort_max, 51)
                    cf = ax.contourf(X, Y, vorticity, levels=levels, cmap='RdBu_r', extend='both')
                    plt.colorbar(cf, ax=ax, label='Vorticity ω', shrink=0.8)
                    
                    # Draw both cylinders
                    cyl1 = patches.Circle(cfg.cylinder1_center, cfg.R_cylinder,
                                         facecolor='darkred', edgecolor='gold', linewidth=1.5)
                    cyl2 = patches.Circle(cfg.cylinder2_center, cfg.R_cylinder,
                                         facecolor='darkblue', edgecolor='gold', linewidth=1.5)
                    ax.add_patch(cyl1)
                    ax.add_patch(cyl2)
                    
                    ax.set_xlim(cfg.x_min, cfg.x_max)
                    ax.set_ylim(cfg.y_min, cfg.y_max)
                    ax.set_aspect('equal')
                    ax.set_xlabel('x/L')
                    ax.set_ylabel('y/L')
                    ax.set_title(f'Dual Cylinder Vorticity Field, Re={cfg.Re:.0f}, t={current_time:.2f}')
                    ax.grid(True, alpha=0.2)
                    
                    fig.text(0.02, 0.02, f'Vorticity Range: ±{vort_max:.2f} | S/D: {cfg.cylinder_separation/(2*cfg.R_cylinder):.2f}', 
                             fontsize=8, color='white')
                    
                    plt.tight_layout()
                    filename = self.vorticity_path / f"vorticity_frame_{step:06d}.png"
                    plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Error plotting vorticity frame {step}: {e}")
                    plt.close('all')
                
                if cfg.memory_efficient:
                    gc.collect()

    def plot_energy_history(self, solver: DualCylinderTurbulentSolver):
        cfg = self.config
        try:
            if not solver.energy_history:
                logger.warning("No energy history data to plot.")
                return
            steps, energies = zip(*solver.energy_history)
            fig = plt.figure(figsize=(12, 6))
            
            ax1 = fig.add_subplot(121)
            ax1.semilogx(steps, energies, label='Mean Kinetic Energy (0.5 * |V|^2)', color='cyan')
            ax1.set_xlabel('Steps (log scale)')
            ax1.set_ylabel('Mean Kinetic Energy')
            ax1.set_title(f'Dual Cylinder Energy History, Re={cfg.Re:.0f}')
            ax1.grid(True, which='both', alpha=0.3)
            ax1.legend()

            ax2 = fig.add_subplot(122)
            interval = 200
            energy_intervals = [np.mean([e for _, e in solver.energy_history[i:i+interval]])
                              for i in range(0, len(solver.energy_history), interval)]
            interval_steps = [s for s, _ in solver.energy_history[::interval]]
            ax2.bar(interval_steps, energy_intervals, color='orange', alpha=0.7, width=interval*0.8)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Mean Kinetic Energy (Averaged over 200 Steps)')
            ax2.set_title('Energy Intervals')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = self.output_path / "energy_history.png"
            plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight')
            plt.close(fig)
            if cfg.memory_efficient:
                gc.collect()
            logger.info("Saved energy history plot")
        except Exception as e:
            logger.error(f"Error plotting energy history: {e}")
            plt.close('all')

    def cleanup(self):
        self.executor.shutdown(wait=False)
        logger.info("Visualizer cleanup completed")

def monitor_dual_simulation_health(solver, step):
    cfg = solver.config
    if np.any(~np.isfinite(solver.u)) or np.any(~np.isfinite(solver.v)):
        logger.error(f"Non-finite values at step {step}")
        return False
    vel_max = max(np.max(np.abs(solver.u)), np.max(np.abs(solver.v)))
    if vel_max > cfg.max_velocity:
        logger.warning(f"High velocity {vel_max:.3f} at step {step}")
        return False
    div_max = np.max(np.abs(compute_divergence_fast(solver.u, solver.v, cfg.dx, cfg.dy)))
    div_threshold = 20.0 if step <= 1000 else 2.0
    if div_max > div_threshold:
        logger.warning(f"High divergence {div_max:.3f} at step {step}")
        return False
    return True

def main():
    # Configuration for dual cylinder simulation
    config = DualCylinderTurbulentConfig(
        Re=600.0,
        R_cylinder=0.4,  # Slightly smaller radius for better resolution
        cylinder1_center=(4.0, 1.5),  # Upper cylinder
        cylinder2_center=(4.0, 2.5),  # Lower cylinder
        nx=600,
        ny=180,
        T_total=30.0,
        dt_base=0.00005,
        dt_max=0.0001,
        cfl_target=0.1,
        use_les=False,
        smagorinsky_constant=0.0,
        use_supg=True,
        artificial_viscosity=0.001,
        pressure_iterations=1500,
        pressure_tolerance=1e-8,
        save_interval=200,
        use_fast_pressure=True,
        adaptive_dt=True,
        memory_efficient=True
    )

    logger.info("🚀 Initializing Dual Cylinder Ultra-High-Performance Turbulent CFD Solver...")
    logger.info("Target: >15 steps/second with dual cylinder interaction modeling")
    logger.info(f"Configuration: S/D = {config.cylinder_separation/(2*config.R_cylinder):.2f}")

    solver = DualCylinderTurbulentSolver(config)
    visualizer = DualCylinderVisualizer(config)

    start_time = time.time()
    step = 0
    current_time = 0.0

    try:
        # Save initial state
        logger.info(f"Saving initial data to HDF5 at t={current_time:.2f}...")
        visualizer.save_data_to_hdf5(solver, step, current_time)

        with tqdm(total=config.T_total, desc="Dual Cylinder Simulation Progress", unit="time",
                  bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while current_time < config.T_total:
                dt = solver.time_step()
                current_time += dt

                # Health monitoring
                if step % 20 == 0:
                    if not monitor_dual_simulation_health(solver, step):
                        logger.error("Dual cylinder simulation became unstable. Stopping.")
                        break

                # Data saving
                if step % config.save_interval == 0:
                    logger.info(f"Saving dual cylinder data to HDF5 at t={current_time:.2f}...")
                    visualizer.save_data_to_hdf5(solver, step, current_time)
                    if config.memory_efficient:
                        memory_usage = psutil.Process().memory_info().rss / (1024*1024)
                        logger.info(f"Memory usage: {memory_usage:.1f} MB")

                # Progress logging
                if step % 500 == 0:
                    vorticity = solver.compute_vorticity()
                    vort_max = np.nanmax(np.abs(vorticity))
                    energy_mean = np.nanmean(solver.compute_energy())
                    logger.info(f"Step {step}: t={current_time:.2f}, max vorticity={vort_max:.3f}, energy={energy_mean:.3f}")

                step += 1
                pbar.update(float(dt))

        logger.info("Generating dual cylinder visualization frames from HDF5 data...")
        visualizer.generate_frames_from_hdf5()

    except KeyboardInterrupt:
        logger.warning("Dual cylinder simulation interrupted by user.")
    except Exception as e:
        logger.error(f"Dual cylinder simulation error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        visualizer.plot_energy_history(solver)
        end_time = time.time()
        total_time = end_time - start_time
        final_speed = step / total_time if total_time > 0 else 0

        logger.warning("\n🏆 Dual Cylinder Simulation Performance Report:")
        logger.warning(f"Total steps: {step}")
        logger.warning(f"Final time: {current_time:.2f}")
        logger.warning(f"Wall time: {total_time/60:.1f} minutes")
        logger.warning(f"Average speed: {final_speed:.1f} steps/second")
        logger.warning(f"Cylinder separation ratio (S/D): {config.cylinder_separation/(2*config.R_cylinder):.2f}")
        logger.warning(f"Results saved in: {visualizer.output_path}")
        logger.warning(f"Data saved in: {visualizer.output_path / config.hdf5_file}")
        logger.warning("Expected phenomena: Complex wake interactions, asymmetric vortex shedding")

        visualizer.cleanup()
        del solver, visualizer
        gc.collect()

if __name__ == "__main__":
    main()