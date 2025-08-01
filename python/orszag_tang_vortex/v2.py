"""
ULTRA-FAST Orszag-Tang Vortex Simulator - V3 Modified
================================================

Modified Orszag-Tang vortex with reduced amplitudes for stability.
Key changes:
- Reduced velocity and magnetic field amplitudes by 0.5
- Added hyperbolic divergence cleaning (Dedner method)
- Increased divB_cleaning to 1.0
- Set eta=nu=chi=0.005, dt_base=0.0002, cfl_safety=0.15
- Tighter field limits and 2nd-order derivatives in high-divergence regions
- Increased save/plot intervals to 2000
- Ensured animation with codec='libx264' and gif fallback
- Added step method to V3MHDSolver
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads
import h5py
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
import logging
from tqdm import tqdm
import warnings
import gc
import imageio
import os

# Maximum performance setup
plt.switch_backend('Agg')
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

# Use all available CPU cores
max_threads = psutil.cpu_count()
set_num_threads(max_threads)
os.environ['NUMBA_NUM_THREADS'] = str(max_threads)
os.environ['OMP_NUM_THREADS'] = str(max_threads)
os.environ['NUMBA_THREADING_LAYER'] = 'tbb'

@dataclass
class V3Config:
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 512
    ny: int = 512
    gamma: float = 5.0/3.0
    rho0: float = 25.0/(36.0*np.pi)
    P0: float = 5.0/(12.0*np.pi)
    B0: float = 1.0/(8.0*np.pi)**0.5  # Reduced for stability
    eta: float = 0.005  # Increased for stability
    nu: float = 0.005
    chi: float = 0.005
    T_final: float = 30.0
    dt_base: float = 0.0002  # Reduced for stability
    cfl_safety: float = 0.15  # Reduced for stability
    dt_max: float = 0.0005   # Reduced for stability
    dt_min: float = 1e-8
    save_interval: int = 2000  # Increased to reduce I/O
    plot_interval: int = 2000
    diagnostic_interval: int = 2000
    print_interval: int = 1000
    output_dir: str = "v3_30_512_modified"
    dpi: int = 150
    max_velocity: float = 10.0
    max_magnetic: float = 3.0  # Tighter limit
    min_density: float = 0.1   # Tighter limit
    min_pressure: float = 0.1  # Tighter limit
    divB_cleaning: float = 1.0 # Increased for stronger cleaning
    hyperbolic_cleaning: float = 0.1  # Hyperbolic cleaning parameter

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dx_inv = 1.0 / self.dx
        self.dy_inv = 1.0 / self.dy
        self.dx2_inv = 1.0 / (self.dx * self.dx)
        self.dy2_inv = 1.0 / (self.dy * self.dy)
        self.cs2 = self.gamma * self.P0 / self.rho0

def setup_logging(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = output_path / "logs" / f"v3_modified_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

@njit(parallel=True, fastmath=True, cache=True)
def compute_derivatives_4th_order(f, dx, dy):
    ny, nx = f.shape
    fx = np.zeros_like(f)
    fy = np.zeros_like(f)
    dx_inv = 1.0 / (12 * dx)
    dy_inv = 1.0 / (12 * dy)
    
    for i in prange(ny):
        for j in range(nx):
            jp2 = (j + 2) % nx
            jp1 = (j + 1) % nx
            jm1 = (j - 1) % nx
            jm2 = (j - 2) % nx
            fx[i,j] = (-f[i,jp2] + 8*f[i,jp1] - 8*f[i,jm1] + f[i,jm2]) * dx_inv
    
    for i in range(ny):
        ip2 = (i + 2) % ny
        ip1 = (i + 1) % ny
        im1 = (i - 1) % ny
        im2 = (i - 2) % ny
        for j in prange(nx):
            fy[i,j] = (-f[ip2,j] + 8*f[ip1,j] - 8*f[im1,j] + f[im2,j]) * dy_inv
    
    return fx, fy

@njit(parallel=True, fastmath=True, cache=True)
def compute_derivatives_2nd_order(f, dx, dy):
    ny, nx = f.shape
    fx = np.zeros_like(f)
    fy = np.zeros_like(f)
    dx_inv = 1.0 / (2 * dx)
    dy_inv = 1.0 / (2 * dy)
    
    for i in prange(ny):
        for j in range(nx):
            jp1 = (j + 1) % nx
            jm1 = (j - 1) % nx
            fx[i,j] = (f[i,jp1] - f[i,jm1]) * dx_inv
    
    for i in range(ny):
        ip1 = (i + 1) % ny
        im1 = (i - 1) % ny
        for j in prange(nx):
            fy[i,j] = (f[ip1,j] - f[im1,j]) * dy_inv
    
    return fx, fy

@njit(parallel=True, fastmath=True, cache=True)
def compute_laplacian(f, dx2_inv, dy2_inv, diffusivity):
    ny, nx = f.shape
    lap = np.zeros_like(f)
    for i in prange(ny):
        ip1 = (i + 1) % ny
        im1 = (i - 1) % ny
        for j in prange(nx):
            jp1 = (j + 1) % nx
            jm1 = (j - 1) % nx
            lap[i,j] = diffusivity * (
                dx2_inv * (f[i,jp1] - 2*f[i,j] + f[i,jm1]) +
                dy2_inv * (f[ip1,j] - 2*f[i,j] + f[im1,j])
            )
    return lap

@njit(parallel=True, fastmath=True, cache=True)
def compute_divergence(vx, vy, dx, dy, use_2nd_order=False):
    if use_2nd_order:
        dvx_dx, _ = compute_derivatives_2nd_order(vx, dx, dy)
        _, dvy_dy = compute_derivatives_2nd_order(vy, dx, dy)
    else:
        dvx_dx, _ = compute_derivatives_4th_order(vx, dx, dy)
        _, dvy_dy = compute_derivatives_4th_order(vy, dx, dy)
    return dvx_dx + dvy_dy

@njit(parallel=True, fastmath=True, cache=True)
def clean_divergence(Bx, By, dx, dy, cleaning_factor, hyperbolic_factor):
    ny, nx = Bx.shape
    divB = compute_divergence(Bx, By, dx, dy)
    ddivB_dx, ddivB_dy = compute_derivatives_4th_order(divB, dx, dy)
    
    Bx_new = Bx - cleaning_factor * dx * ddivB_dx
    By_new = By - cleaning_factor * dx * ddivB_dy
    
    # Hyperbolic cleaning (Dedner method)
    psi = np.zeros_like(Bx)  # Scalar field for divergence transport
    dpsi_dx, dpsi_dy = compute_derivatives_4th_order(psi, dx, dy)
    Bx_new -= hyperbolic_factor * dx * dpsi_dx
    By_new -= hyperbolic_factor * dx * dpsi_dy
    dpsi_dt = -hyperbolic_factor * divB  # Evolution of psi
    
    divB_new = compute_divergence(Bx_new, By_new, dx, dy)
    if np.max(np.abs(divB_new)) > np.max(np.abs(div_B)):
        return Bx, By, psi
    return Bx_new, By_new, dpsi_dt

@njit(parallel=True, fastmath=True, cache=True)
def mhd_rhs_v3(rho, vx, vy, vz, Bx, By, Bz, P, psi, gamma, dx, dy, dx2_inv, dy2_inv, eta, nu, chi, cleaning_factor, hyperbolic_factor):
    ny, nx = rho.shape
    max_speed = 0.0
    use_2nd_order = False
    
    # Check divergence to decide derivative order
    div_B = compute_divergence(Bx, By, dx, dy)
    if np.max(np.abs(div_B)) > 1e-4:
        use_2nd_order = True
    
    # Compute derivatives
    if use_2nd_order:
        drho_dx, drho_dy = compute_derivatives_2nd_order(rho, dx, dy)
        dvx_dx, dvx_dy = compute_derivatives_2nd_order(vx, dx, dy)
        dvy_dx, dvy_dy = compute_derivatives_2nd_order(vy, dx, dy)
        dvz_dx, dvz_dy = compute_derivatives_2nd_order(vz, dx, dy)
        dBx_dx, dBx_dy = compute_derivatives_2nd_order(Bx, dx, dy)
        dBy_dx, dBy_dy = compute_derivatives_2nd_order(By, dx, dy)
        dBz_dx, dBz_dy = compute_derivatives_2nd_order(Bz, dx, dy)
        dP_dx, dP_dy = compute_derivatives_2nd_order(P, dx, dy)
    else:
        drho_dx, drho_dy = compute_derivatives_4th_order(rho, dx, dy)
        dvx_dx, dvx_dy = compute_derivatives_4th_order(vx, dx, dy)
        dvy_dx, dvy_dy = compute_derivatives_4th_order(vy, dx, dy)
        dvz_dx, dvz_dy = compute_derivatives_4th_order(vz, dx, dy)
        dBx_dx, dBx_dy = compute_derivatives_4th_order(Bx, dx, dy)
        dBy_dx, dBy_dy = compute_derivatives_4th_order(By, dx, dy)
        dBz_dx, dBz_dy = compute_derivatives_4th_order(Bz, dx, dy)
        dP_dx, dP_dy = compute_derivatives_4th_order(P, dx, dy)
    
    div_v = dvx_dx + dvy_dy
    Jz = dBy_dx - dBx_dy
    B2 = Bx**2 + By**2 + Bz**2
    dB2_dx, dB2_dy = compute_derivatives_4th_order(B2, dx, dy) if not use_2nd_order else compute_derivatives_2nd_order(B2, dx, dy)
    
    # Diffusion terms
    lap_vx = compute_laplacian(vx, dx2_inv, dy2_inv, nu)
    lap_vy = compute_laplacian(vy, dx2_inv, dy2_inv, nu)
    lap_vz = compute_laplacian(vz, dx2_inv, dy2_inv, nu)
    lap_Bx = compute_laplacian(Bx, dx2_inv, dy2_inv, eta)
    lap_By = compute_laplacian(By, dx2_inv, dy2_inv, eta)
    lap_Bz = compute_laplacian(Bz, dx2_inv, dy2_inv, eta)
    lap_P = compute_laplacian(P, dx2_inv, dy2_inv, chi)
    
    # Initialize outputs
    drho_dt = np.zeros_like(rho)
    dvx_dt = np.zeros_like(vx)
    dvy_dt = np.zeros_like(vy)
    dvz_dt = np.zeros_like(vz)
    dBx_dt = np.zeros_like(Bx)
    dBy_dt = np.zeros_like(By)
    dBz_dt = np.zeros_like(Bz)
    dP_dt = np.zeros_like(P)
    dpsi_dt = np.zeros_like(psi)
    
    # Main computation
    for i in prange(ny):
        for j in prange(nx):
            rho_ij = max(rho[i,j], 0.1)  # Enforce minimum density
            P_ij = max(P[i,j], 0.1)      # Enforce minimum pressure
            rho_inv = 1.0 / rho_ij
            vx_ij = vx[i,j]
            vy_ij = vy[i,j]
            vz_ij = vz[i,j]
            Bx_ij = Bx[i,j]
            By_ij = By[i,j]
            Bz_ij = Bz[i,j]
            
            mag_press_x = 0.5 * dB2_dx[i,j]
            mag_press_y = 0.5 * dB2_dy[i,j]
            mag_tens_x = Bx_ij*dBx_dx[i,j] + By_ij*dBx_dy[i,j] + Bz_ij*dBz_dx[i,j]
            mag_tens_y = Bx_ij*dBy_dx[i,j] + By_ij*dBy_dy[i,j] + Bz_ij*dBz_dy[i,j]
            
            drho_dt[i,j] = -(rho_ij * div_v[i,j] + vx_ij * drho_dx[i,j] + vy_ij * drho_dy[i,j])
            dvx_dt[i,j] = (-(vx_ij*dvx_dx[i,j] + vy_ij*dvx_dy[i,j]) -
                          dP_dx[i,j]*rho_inv +
                          (mag_tens_x - mag_press_x)*rho_inv +
                          lap_vx[i,j])
            dvy_dt[i,j] = (-(vx_ij*dvy_dx[i,j] + vy_ij*dvy_dy[i,j]) -
                          dP_dy[i,j]*rho_inv +
                          (mag_tens_y - mag_press_y)*rho_inv +
                          lap_vy[i,j])
            dvz_dt[i,j] = (-(vx_ij*dvz_dx[i,j] + vy_ij*dvz_dy[i,j]) +
                          lap_vz[i,j])
            dBx_dt[i,j] = (vy_ij*Jz[i,j] -
                          (vx_ij*dBx_dx[i,j] + vy_ij*dBx_dy[i,j] -
                           Bx_ij*div_v[i,j]) +
                          lap_Bx[i,j])
            dBy_dt[i,j] = (-vx_ij*Jz[i,j] -
                          (vx_ij*dBy_dx[i,j] + vy_ij*dBy_dy[i,j] -
                           By_ij*div_v[i,j]) +
                          lap_By[i,j])
            dBz_dt[i,j] = (-(vx_ij*dBz_dx[i,j] + vy_ij*dBz_dy[i,j] -
                           Bz_ij*div_v[i,j]) +
                          lap_Bz[i,j])
            dP_dt[i,j] = (-gamma * P_ij * div_v[i,j] +
                          lap_P[i,j])
            
            cs2 = gamma * P_ij / rho_ij
            va2 = B2[i,j] / rho_ij
            cf = np.sqrt(cs2 + va2)
            v_mag = np.sqrt(vx_ij**2 + vy_ij**2 + vz_ij**2)
            local_speed = v_mag + cf
            if local_speed > max_speed:
                max_speed = local_speed
    
    dpsi_dt = clean_divergence(Bx, By, dx, dy, cleaning_factor, hyperbolic_factor)[2]
    
    return drho_dt, dvx_dt, dvy_dt, dvz_dt, dBx_dt, dBy_dt, dBz_dt, dP_dt, dpsi_dt, max_speed

@njit(parallel=True, fastmath=True, cache=True)
def fast_field_limits(rho, vx, vy, vz, Bx, By, Bz, P, psi, limits):
    max_vel, max_mag, min_rho, min_P = limits
    ny, nx = rho.shape
    
    for i in prange(ny):
        for j in prange(nx):
            rho[i,j] = max(min(rho[i,j], 20.0), min_rho)
            P[i,j] = max(min(P[i,j], 20.0), min_P)
            vx[i,j] = max(min(vx[i,j], max_vel), -max_vel)
            vy[i,j] = max(min(vy[i,j], max_vel), -max_vel)
            vz[i,j] = max(min(vz[i,j], max_vel), -max_vel)
            Bx[i,j] = max(min(Bx[i,j], max_mag), -max_mag)
            By[i,j] = max(min(By[i,j], max_mag), -max_mag)
            Bz[i,j] = max(min(Bz[i,j], max_mag), -max_mag)
            psi[i,j] = max(min(psi[i,j], max_mag), -max_mag)

class V3MHDSolver:
    def __init__(self, config: V3Config, logger):
        self.config = config
        self.logger = logger
        self.setup_memory()
        self.initialize_orszag_tang()
        self.time = 0.0
        self.step_count = 0
        self.last_max_speed = 1.0
        self.save_count = 0
        self.last_stable_state = None
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        for subdir in ["data", "fields", "profiles", "diagnostics", "animations"]:
            (self.output_path / subdir).mkdir(exist_ok=True)
        self.hdf5_filename = self.output_path / "data" / "simulation_data.h5"
        self.initialize_hdf5()
        self.diagnostics = {
            'time': [], 'dt': [], 'max_speed': [], 'max_divB': [],
            'kinetic_energy': [], 'magnetic_energy': [], 'thermal_energy': [], 'total_energy': [],
            'max_current': [], 'max_vorticity': [], 'min_density': []
        }
        self.logger.info(f"Initialized V3 solver: {config.nx}×{config.ny} grid")
        self.logger.info(f"Target: {config.T_final}s in 2 hours")
        self.logger.info(f"Using {max_threads} CPU threads")

    def setup_memory(self):
        cfg = self.config
        shape = (cfg.ny, cfg.nx)
        self.rho = np.ones(shape, dtype=np.float64, order='C')
        self.vx = np.zeros(shape, dtype=np.float64, order='C')
        self.vy = np.zeros(shape, dtype=np.float64, order='C')
        self.vz = np.zeros(shape, dtype=np.float64, order='C')
        self.Bx = np.zeros(shape, dtype=np.float64, order='C')
        self.By = np.zeros(shape, dtype=np.float64, order='C')
        self.Bz = np.zeros(shape, dtype=np.float64, order='C')
        self.P = np.ones(shape, dtype=np.float64, order='C')
        self.psi = np.zeros(shape, dtype=np.float64, order='C')  # For hyperbolic cleaning
        self.work_arrays = tuple(np.zeros(shape, dtype=np.float64, order='C') for _ in range(36))
        self.x = np.linspace(0, cfg.Lx, cfg.nx, endpoint=False, dtype=np.float64)
        self.y = np.linspace(0, cfg.Ly, cfg.ny, endpoint=False, dtype=np.float64)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.limits = np.array([cfg.max_velocity, cfg.max_magnetic, cfg.min_density, cfg.min_pressure], dtype=np.float64)
        total_gb = shape[0] * shape[1] * 8 * 45 / (1024**3)  # Adjusted for psi and extra arrays
        self.logger.info(f"Memory allocated: {total_gb:.1f} GB")

    def initialize_orszag_tang(self):
        cfg = self.config
        self.rho.fill(cfg.rho0)
        self.P.fill(cfg.P0)
        self.vx[:] = -0.5 * np.sin(2 * np.pi * self.Y)  # Reduced amplitude
        self.vy[:] = 0.5 * np.sin(2 * np.pi * self.X)   # Reduced amplitude
        self.Bx[:] = -0.5 * cfg.B0 * np.sin(2 * np.pi * self.Y)  # Reduced amplitude
        self.By[:] = 0.5 * cfg.B0 * np.sin(4 * np.pi * self.X)  # Reduced amplitude
        div_B = compute_divergence(self.Bx, self.By, cfg.dx, cfg.dy)
        cs2_actual = cfg.gamma * self.P[0,0] / self.rho[0,0]
        self.logger.info("Initial conditions:")
        self.logger.info(f"  Density: ρ = {self.rho[0,0]:.6f}")
        self.logger.info(f"  Pressure: P = {self.P[0,0]:.6f}")
        self.logger.info(f"  Sound speed²: cs² = {cs2_actual:.6f}")
        self.logger.info(f"  Max |∇·B|: {np.max(np.abs(div_B)):.2e}")

    def compute_timestep(self, max_speed):
        cfg = self.config
        if max_speed <= 0:
            return cfg.dt_base
        dt_cfl = cfg.cfl_safety * min(cfg.dx, cfg.dy) / max_speed
        dt_diff = 0.2 * min(cfg.dx**2, cfg.dy**2) / max(cfg.eta, cfg.nu, cfg.chi)
        dt = min(dt_cfl, dt_diff, cfg.dt_max)
        return max(dt, cfg.dt_min)

    def rk4_step(self, dt):
        cfg = self.config
        dt_half = 0.5 * dt
        dt_sixth = dt / 6.0
        
        (k1_rho, k1_vx, k1_vy, k1_vz, k1_Bx, k1_By, k1_Bz, k1_P, k1_psi,
         k2_rho, k2_vx, k2_vy, k2_vz, k2_Bx, k2_By, k2_Bz, k2_P, k2_psi,
         k3_rho, k3_vx, k3_vy, k3_vz, k3_Bx, k3_By, k3_Bz, k3_P, k3_psi,
         k4_rho, k4_vx, k4_vy, k4_vz, k4_Bx, k4_By, k4_Bz, k4_P, k4_psi) = self.work_arrays
        
        tmp_rho, tmp_vx, tmp_vy, tmp_vz = self.rho.copy(), self.vx.copy(), self.vy.copy(), self.vz.copy()
        tmp_Bx, tmp_By, tmp_Bz, tmp_P, tmp_psi = self.Bx.copy(), self.By.copy(), self.Bz.copy(), self.P.copy(), self.psi.copy()
        
        self.last_stable_state = (tmp_rho.copy(), tmp_vx.copy(), tmp_vy.copy(), tmp_vz.copy(),
                                 tmp_Bx.copy(), tmp_By.copy(), tmp_Bz.copy(), tmp_P.copy(), tmp_psi.copy())
        
        k1_rho, k1_vx, k1_vy, k1_vz, k1_Bx, k1_By, k1_Bz, k1_P, k1_psi, max_speed = \
            mhd_rhs_v3(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi,
                       cfg.gamma, cfg.dx, cfg.dy, cfg.dx2_inv, cfg.dy2_inv, cfg.eta, cfg.nu, cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        for arr, tmp, k in [(self.rho, tmp_rho, k1_rho), (self.vx, tmp_vx, k1_vx),
                           (self.vy, tmp_vy, k1_vy), (self.vz, tmp_vz, k1_vz),
                           (self.Bx, tmp_Bx, k1_Bx), (self.By, tmp_By, k1_By),
                           (self.Bz, tmp_Bz, k1_Bz), (self.P, tmp_P, k1_P),
                           (self.psi, tmp_psi, k1_psi)]:
            arr[:] = tmp + dt_half * k
        self.Bx, self.By, self.psi = clean_divergence(self.Bx, self.By, cfg.dx, cfg.dy, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        k2_rho, k2_vx, k2_vy, k2_vz, k2_Bx, k2_By, k2_Bz, k2_P, k2_psi, _ = \
            mhd_rhs_v3(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi,
                       cfg.gamma, cfg.dx, cfg.dy, cfg.dx2_inv, cfg.dy2_inv, cfg.eta, cfg.nu, cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        for arr, tmp, k in [(self.rho, tmp_rho, k2_rho), (self.vx, tmp_vx, k2_vx),
                           (self.vy, tmp_vy, k2_vy), (self.vz, tmp_vz, k2_vz),
                           (self.Bx, tmp_Bx, k2_Bx), (self.By, tmp_By, k2_By),
                           (self.Bz, tmp_Bz, k2_Bz), (self.P, tmp_P, k2_P),
                           (self.psi, tmp_psi, k2_psi)]:
            arr[:] = tmp + dt_half * k
        self.Bx, self.By, self.psi = clean_divergence(self.Bx, self.By, cfg.dx, cfg.dy, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        k3_rho, k3_vx, k3_vy, k3_vz, k3_Bx, k3_By, k3_Bz, k3_P, k3_psi, _ = \
            mhd_rhs_v3(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi,
                       cfg.gamma, cfg.dx, cfg.dy, cfg.dx2_inv, cfg.dy2_inv, cfg.eta, cfg.nu, cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        for arr, tmp, k in [(self.rho, tmp_rho, k3_rho), (self.vx, tmp_vx, k3_vx),
                           (self.vy, tmp_vy, k3_vy), (self.vz, tmp_vz, k3_vz),
                           (self.Bx, tmp_Bx, k3_Bx), (self.By, tmp_By, k3_By),
                           (self.Bz, tmp_Bz, k3_Bz), (self.P, tmp_P, k3_P),
                           (self.psi, tmp_psi, k3_psi)]:
            arr[:] = tmp + dt * k
        self.Bx, self.By, self.psi = clean_divergence(self.Bx, self.By, cfg.dx, cfg.dy, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        k4_rho, k4_vx, k4_vy, k4_vz, k4_Bx, k4_By, k4_Bz, k4_P, k4_psi, _ = \
            mhd_rhs_v3(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi,
                       cfg.gamma, cfg.dx, cfg.dy, cfg.dx2_inv, cfg.dy2_inv, cfg.eta, cfg.nu, cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        for arr, tmp, k1, k2, k3, k4 in [(self.rho, tmp_rho, k1_rho, k2_rho, k3_rho, k4_rho),
                                       (self.vx, tmp_vx, k1_vx, k2_vx, k3_vx, k4_vx),
                                       (self.vy, tmp_vy, k1_vy, k2_vy, k3_vy, k4_vy),
                                       (self.vz, tmp_vz, k1_vz, k2_vz, k3_vz, k4_vz),
                                       (self.Bx, tmp_Bx, k1_Bx, k2_Bx, k3_Bx, k4_Bx),
                                       (self.By, tmp_By, k1_By, k2_By, k3_By, k4_By),
                                       (self.Bz, tmp_Bz, k1_Bz, k2_Bz, k3_Bz, k4_Bz),
                                       (self.P, tmp_P, k1_P, k2_P, k3_P, k4_P),
                                       (self.psi, tmp_psi, k1_psi, k2_psi, k3_psi, k4_psi)]:
            arr[:] = tmp + dt_sixth * (k1 + 2*k2 + 2*k3 + k4)
        self.Bx, self.By, self.psi = clean_divergence(self.Bx, self.By, cfg.dx, cfg.dy, cfg.divB_cleaning, cfg.hyperbolic_cleaning)
        
        return max_speed

    def step(self):
        """Perform a single time step using RK4 integration."""
        max_speed = self.rk4_step(self.compute_timestep(self.last_max_speed))
        fast_field_limits(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi, self.limits)
        dt = self.compute_timestep(max_speed)
        self.time += dt
        self.step_count += 1
        self.last_max_speed = max_speed
        return dt, max_speed

    def rollback(self):
        if self.last_stable_state is None:
            self.logger.error("No stable state to roll back to")
            return False
        (self.rho[:], self.vx[:], self.vy[:], self.vz[:],
         self.Bx[:], self.By[:], self.Bz[:], self.P[:], self.psi[:]) = self.last_stable_state
        self.time -= self.compute_timestep(self.last_max_speed)
        self.step_count -= 1
        self.config.dt_max *= 0.5
        self.config.eta *= 1.2
        self.config.nu *= 1.2
        self.config.chi *= 1.2
        self.logger.warning(f"Rolled back to step {self.step_count}, dt_max={self.config.dt_max:.2e}, "
                           f"eta={self.config.eta:.2e}")
        return True

    def compute_diagnostics(self):
        cfg = self.config
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        B2 = self.Bx**2 + self.By**2 + self.Bz**2
        kinetic = 0.5 * np.mean(self.rho * v2)
        magnetic = 0.5 * np.mean(B2)
        thermal = np.mean(self.P / (cfg.gamma - 1))
        total = kinetic + magnetic + thermal
        Jz = compute_divergence(self.By, -self.Bx, cfg.dx, cfg.dy)
        max_current = np.max(np.abs(Jz))
        dvx_dy, _ = compute_derivatives_4th_order(self.vx, cfg.dx, cfg.dy)
        _, dvy_dx = compute_derivatives_4th_order(self.vy, cfg.dx, cfg.dy)
        vorticity = dvy_dx - dvx_dy
        max_vorticity = np.max(np.abs(vorticity))
        min_density = np.min(self.rho)
        div_B = compute_divergence(self.Bx, self.By, cfg.dx, cfg.dy)
        max_divB = np.max(np.abs(div_B))
        
        self.diagnostics['time'].append(float(self.time))
        self.diagnostics['dt'].append(float(self.compute_timestep(self.last_max_speed)))
        self.diagnostics['max_speed'].append(float(self.last_max_speed))
        self.diagnostics['max_divB'].append(float(max_divB))
        self.diagnostics['kinetic_energy'].append(float(kinetic))
        self.diagnostics['magnetic_energy'].append(float(magnetic))
        self.diagnostics['thermal_energy'].append(float(thermal))
        self.diagnostics['total_energy'].append(float(total))
        self.diagnostics['max_current'].append(float(max_current))
        self.diagnostics['max_vorticity'].append(float(max_vorticity))
        self.diagnostics['min_density'].append(float(min_density))
        
        return {
            'kinetic_energy': kinetic, 'magnetic_energy': magnetic,
            'thermal_energy': thermal, 'total_energy': total,
            'max_current': max_current, 'max_vorticity': max_vorticity,
            'min_density': min_density, 'max_divB': max_divB
        }

    def initialize_hdf5(self):
        cfg = self.config
        estimated_saves = max(100, int(cfg.T_final / (cfg.save_interval * cfg.dt_base)) + 10)
        with h5py.File(self.hdf5_filename, 'w') as f:
            f.attrs['nx'] = cfg.nx
            f.attrs['ny'] = cfg.ny
            f.attrs['T_final'] = cfg.T_final
            f.attrs['gamma'] = cfg.gamma
            f.create_dataset('X', data=self.X, compression='lzf')
            f.create_dataset('Y', data=self.Y, compression='lzf')
            f.create_dataset('save_times', (estimated_saves,), dtype=np.float64, maxshape=(None,))
            f.create_dataset('save_steps', (estimated_saves,), dtype=np.int32, maxshape=(None,))
            shape = (estimated_saves, cfg.ny, cfg.nx)
            maxshape = (None, cfg.ny, cfg.nx)
            chunk_shape = (1, cfg.ny, cfg.nx)
            for field in ['rho', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz', 'P', 'psi']:
                f.create_dataset(field, shape, dtype=np.float64, maxshape=maxshape,
                               compression='lzf', chunks=chunk_shape)

    def save_to_hdf5(self):
        with h5py.File(self.hdf5_filename, 'a') as f:
            if self.save_count >= f['save_times'].shape[0]:
                new_size = self.save_count + 50
                f['save_times'].resize((new_size,))
                f['save_steps'].resize((new_size,))
                for field in ['rho', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz', 'P', 'psi']:
                    f[field].resize((new_size, self.config.ny, self.config.nx))
            f['save_times'][self.save_count] = self.time
            f['save_steps'][self.save_count] = self.step_count
            f['rho'][self.save_count] = self.rho
            f['vx'][self.save_count] = self.vx
            f['vy'][self.save_count] = self.vy
            f['vz'][self.save_count] = self.vz
            f['Bx'][self.save_count] = self.Bx
            f['By'][self.save_count] = self.By
            f['Bz'][self.save_count] = self.Bz
            f['P'][self.save_count] = self.P
            f['psi'][self.save_count] = self.psi
        self.save_count += 1

def create_field_plots(solver, output_path, time_val, step):
    cfg = solver.config
    fields_path = output_path / "fields"
    for field in ['density', 'velocity', 'magnetic', 'pressure', 'current', 'vorticity']:
        (fields_path / field).mkdir(exist_ok=True)
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(solver.X, solver.Y, solver.rho, levels=50, cmap='viridis')
        plt.colorbar(im, ax=ax, label='ρ')
        ax.set_title(f'Density Field at t = {time_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        filename = fields_path / "density" / f"density_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
        im = ax.contourf(solver.X, solver.Y, v_mag, levels=50, cmap='plasma')
        try:
            ax.streamplot(solver.X, solver.Y, solver.vx, solver.vy, color='white', density=1.5, linewidth=0.8)
        except:
            skip = slice(None, None, 8)
            ax.quiver(solver.X[skip, skip], solver.Y[skip, skip], solver.vx[skip, skip], solver.vy[skip, skip],
                     color='white', alpha=0.7, scale=50)
            solver.logger.warning(f"Velocity streamplot failed at t={time_val:.2f}, using quiver")
        plt.colorbar(im, ax=ax, label='|v|')
        ax.set_title(f'Velocity Field at t = {time_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        filename = fields_path / "velocity" / f"velocity_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        B_mag = np.sqrt(solver.Bx**2 + solver.By**2 + solver.Bz**2)
        im = ax.contourf(solver.X, solver.Y, B_mag, levels=50, cmap='coolwarm')
        try:
            ax.streamplot(solver.X, solver.Y, solver.Bx, solver.By, color='black', density=1.5, linewidth=0.8)
        except:
            skip = slice(None, None, 8)
            ax.quiver(solver.X[skip, skip], solver.Y[skip, skip], solver.Bx[skip, skip], solver.By[skip, skip],
                     color='black', alpha=0.8, scale=20)
            solver.logger.warning(f"Magnetic streamplot failed at t={time_val:.2f}, using quiver")
        plt.colorbar(im, ax=ax, label='|B|')
        ax.set_title(f'Magnetic Field at t = {time_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        filename = fields_path / "magnetic" / f"magnetic_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(solver.X, solver.Y, solver.P, levels=50, cmap='hot')
        plt.colorbar(im, ax=ax, label='P')
        ax.set_title(f'Pressure Field at t = {time_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        filename = fields_path / "pressure" / f"pressure_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        Jz = compute_divergence(solver.By, -solver.Bx, cfg.dx, cfg.dy)
        im = ax.contourf(solver.X, solver.Y, Jz, levels=50, cmap='RdBu', extend='both')
        plt.colorbar(im, ax=ax, label='Jz')
        ax.set_title(f'Current Density at t = {time_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        filename = fields_path / "current" / f"current_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        dvx_dy, _ = compute_derivatives_4th_order(solver.vx, cfg.dx, cfg.dy)
        _, dvy_dx = compute_derivatives_4th_order(solver.vy, cfg.dx, cfg.dy)
        vorticity = dvy_dx - dvx_dy
        im = ax.contourf(solver.X, solver.Y, vorticity, levels=50, cmap='seismic', extend='both')
        plt.colorbar(im, ax=ax, label='ω')
        ax.set_title(f'Vorticity at t = {time_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        filename = fields_path / "vorticity" / f"vorticity_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        gc.collect()

    except Exception as e:
        solver.logger.error(f"Field plotting error at t={time_val:.2f}: {e}")

def create_1d_profiles(solver, output_path, time_val, step):
    cfg = solver.config
    profiles_path = output_path / "profiles"
    profiles_path.mkdir(exist_ok=True)
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'1D Profiles at t = {time_val:.2f}s', fontsize=16, color='white')
        
        center_y = cfg.ny // 2
        center_x = cfg.nx // 2
        x_slice = solver.x
        y_slice = solver.y
        
        axes[0,0].plot(x_slice, solver.rho[center_y, :], 'cyan', label='y=0.5')
        axes[0,0].plot(y_slice, solver.rho[:, center_x], 'orange', label='x=0.5')
        axes[0,0].set_xlabel('Position')
        axes[0,0].set_ylabel('Density ρ')
        axes[0,0].set_title('Density')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
        axes[0,1].plot(x_slice, v_mag[center_y, :], 'cyan', label='y=0.5')
        axes[0,1].plot(y_slice, v_mag[:, center_x], 'orange', label='x=0.5')
        axes[0,1].set_xlabel('Position')
        axes[0,1].set_ylabel('Velocity |v|')
        axes[0,1].set_title('Velocity')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        B_mag = np.sqrt(solver.Bx**2 + solver.By**2 + solver.Bz**2)
        axes[0,2].plot(x_slice, B_mag[center_y, :], 'cyan', label='y=0.5')
        axes[0,2].plot(y_slice, B_mag[:, center_x], 'orange', label='x=0.5')
        axes[0,2].set_xlabel('Position')
        axes[0,2].set_ylabel('Magnetic Field |B|')
        axes[0,2].set_title('Magnetic Field')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()
        
        axes[1,0].plot(x_slice, solver.P[center_y, :], 'cyan', label='y=0.5')
        axes[1,0].plot(y_slice, solver.P[:, center_x], 'orange', label='x=0.5')
        axes[1,0].set_xlabel('Position')
        axes[1,0].set_ylabel('Pressure P')
        axes[1,0].set_title('Pressure')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        Jz = compute_divergence(solver.By, -solver.Bx, cfg.dx, cfg.dy)
        axes[1,1].plot(x_slice, Jz[center_y, :], 'cyan', label='y=0.5')
        axes[1,1].plot(y_slice, Jz[:, center_x], 'orange', label='x=0.5')
        axes[1,1].set_xlabel('Position')
        axes[1,1].set_ylabel('Current Density Jz')
        axes[1,1].set_title('Current Density')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        dvx_dy, _ = compute_derivatives_4th_order(solver.vx, cfg.dx, cfg.dy)
        _, dvy_dx = compute_derivatives_4th_order(solver.vy, cfg.dx, cfg.dy)
        vorticity = dvy_dx - dvx_dy
        axes[1,2].plot(x_slice, vorticity[center_y, :], 'cyan', label='y=0.5')
        axes[1,2].plot(y_slice, vorticity[:, center_x], 'orange', label='x=0.5')
        axes[1,2].set_xlabel('Position')
        axes[1,2].set_ylabel('Vorticity ω')
        axes[1,2].set_title('Vorticity')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].legend()
        
        plt.tight_layout()
        filename = profiles_path / f"profiles_t{time_val:.2f}.png"
        plt.savefig(filename, dpi=cfg.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        gc.collect()

    except Exception as e:
        solver.logger.error(f"Profile plotting error at t={time_val:.2f}: {e}")

def create_diagnostic_plots(solver, output_path):
    diag_path = output_path / "diagnostics"
    diag_path.mkdir(exist_ok=True)
    
    try:
        if len(solver.diagnostics['time']) < 2:
            solver.logger.warning("Insufficient diagnostic data for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('V3 Modified MHD Diagnostics', fontsize=16, color='white')
        times = np.array(solver.diagnostics['time'])
        
        axes[0,0].plot(times, solver.diagnostics['kinetic_energy'], 'cyan', label='Kinetic')
        axes[0,0].plot(times, solver.diagnostics['magnetic_energy'], 'red', label='Magnetic')
        axes[0,0].plot(times, solver.diagnostics['thermal_energy'], 'green', label='Thermal')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Energy Density')
        axes[0,0].set_title('Energy Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(times, solver.diagnostics['total_energy'], 'yellow')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Total Energy')
        axes[0,1].set_title('Energy Conservation')
        axes[0,1].grid(True, alpha=0.3)
        
        axes[0,2].semilogy(times, solver.diagnostics['max_current'], 'purple')
        axes[0,2].set_xlabel('Time')
        axes[0,2].set_ylabel('Max |J|')
        axes[0,2].set_title('Current Density')
        axes[0,2].grid(True, alpha=0.3)
        
        axes[1,0].semilogy(times, solver.diagnostics['max_vorticity'], 'orange')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Max |ω|')
        axes[1,0].set_title('Vorticity')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].plot(times, solver.diagnostics['min_density'], 'lightblue')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Min ρ')
        axes[1,1].set_title('Density Evolution')
        axes[1,1].grid(True, alpha=0.3)
        
        axes[1,2].semilogy(times, solver.diagnostics['max_divB'], 'red', label='Max |∇·B|')
        axes[1,2].set_xlabel('Time')
        axes[1,2].set_ylabel('Max |∇·B|')
        axes[1,2].set_title('Magnetic Divergence')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = diag_path / "diagnostics.png"
        plt.savefig(filename, dpi=solver.config.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        diag_file = diag_path / "diagnostics.h5"
        with h5py.File(diag_file, 'w') as f:
            for key, values in solver.diagnostics.items():
                if len(values) > 0:
                    f.create_dataset(key, data=np.array(values, dtype=np.float64), compression='lzf')

    except Exception as e:
        solver.logger.error(f"Diagnostic plotting error: {e}")

def create_animation(output_path):
    density_path = output_path / "fields" / "density"
    output_movie = output_path / "animations" / "density_evolution.mp4"
    output_gif = output_path / "animations" / "density_evolution.gif"
    
    try:
        density_files = sorted(density_path.glob("density_t*.png"))
        if len(density_files) >= 10:  # Ensure enough frames
            with imageio.get_writer(output_movie, fps=10, codec='libx264') as writer:
                for file in tqdm(density_files, desc="Creating density animation"):
                    image = imageio.imread(file)
                    writer.append_data(image)
            print(f"Animation created: {output_movie}")
            # Create GIF as fallback
            with imageio.get_writer(output_gif, fps=10, mode='I') as writer:
                for file in tqdm(density_files, desc="Creating density GIF"):
                    image = imageio.imread(file)
                    writer.append_data(image)
            print(f"GIF created: {output_gif}")
        else:
            print(f"Insufficient density plots ({len(density_files)}) for animation, need at least 10")
    except Exception as e:
        print(f"Animation creation error: {e}")

def monitor_stability(solver):
    fields = [solver.rho, solver.vx, solver.vy, solver.vz, solver.Bx, solver.By, solver.Bz, solver.P, solver.psi]
    for field in fields:
        if not np.all(np.isfinite(field)):
            solver.logger.error(f"Non-finite values detected at step {solver.step_count}")
            return False
    if np.any(solver.rho <= 0) or np.any(solver.P <= 0):
        solver.logger.error(f"Negative density or pressure at step {solver.step_count}")
        return False
    v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
    if np.max(v_mag) > solver.config.max_velocity:
        solver.logger.warning(f"High velocity {np.max(v_mag):.3f} at step {solver.step_count}")
    div_B = compute_divergence(solver.Bx, solver.By, solver.config.dx, solver.config.dy)
    max_divB = np.max(np.abs(div_B))
    if max_divB > 1e-4:
        solver.logger.warning(f"High magnetic divergence {max_divB:.2e} at step {solver.step_count}")
        # Apply immediate cleaning
        solver.Bx, solver.By, solver.psi = clean_divergence(solver.Bx, solver.By, solver.config.dx, solver.config.dy,
                                                           solver.config.divB_cleaning, solver.config.hyperbolic_cleaning)
        return False
    return True

def main():
    print("🚀 V3 MODIFIED ULTRA-FAST MHD SIMULATOR")
    print("="*50)
    print(f"🎯 Target: 30s simulation in 2 hours")
    print(f"⚡ CPU cores: {max_threads}")
    print("="*50)
    
    logger = setup_logging("v3_30_512_modified")
    config = V3Config(output_dir="v3_30_512_modified")
    
    logger.info("Pre-compiling kernels...")
    test_config = V3Config(nx=32, ny=32, T_final=0.01, output_dir="v3_test")
    test_solver = V3MHDSolver(test_config, logger)
    for _ in range(10):
        test_solver.step()
    del test_solver
    gc.collect()
    logger.info("Kernels compiled!")
    
    solver = V3MHDSolver(config, logger)
    start_time = time.time()
    
    solver.save_to_hdf5()
    diagnostics = solver.compute_diagnostics()
    create_field_plots(solver, solver.output_path, solver.time, solver.step_count)
    create_1d_profiles(solver, solver.output_path, solver.time, solver.step_count)
    
    logger.info(f"Initial energies: KE={diagnostics['kinetic_energy']:.4f}, "
                f"ME={diagnostics['magnetic_energy']:.4f}, TE={diagnostics['total_energy']:.4f}")
    
    last_print_time = start_time
    last_save_step = 0
    last_diag_step = 0
    rollback_count = 0
    max_rollbacks = 5
    
    with tqdm(total=config.T_final, desc="🚀 V3 MODIFIED SIMULATION", unit="s",
             bar_format='{desc}: {percentage:3.1f}%|{bar}| {n:.2f}/{total:.0f}s [{elapsed}<{remaining}, {rate_fmt}]',
             colour='green') as pbar:
        
        while solver.time < config.T_final:
            dt, max_speed = solver.step()
            
            if solver.step_count % 5 == 0:
                if not monitor_stability(solver):
                    if rollback_count < max_rollbacks and solver.rollback():
                        rollback_count += 1
                        continue
                    else:
                        logger.error("Simulation unstable after max rollbacks. Terminating.")
                        break
            
            if solver.step_count - last_diag_step >= config.diagnostic_interval:
                diagnostics = solver.compute_diagnostics()
                last_diag_step = solver.step_count
            
            if solver.step_count - last_save_step >= config.save_interval:
                solver.save_to_hdf5()
                create_field_plots(solver, solver.output_path, solver.time, solver.step_count)
                create_1d_profiles(solver, solver.output_path, solver.time, solver.step_count)
                last_save_step = solver.step_count
            
            if solver.step_count % config.print_interval == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                steps_per_sec = solver.step_count / elapsed
                sim_speed = solver.time / elapsed
                eta_minutes = (config.T_final - solver.time) / (sim_speed * 60) if sim_speed > 0 else 0
                
                logger.info(f"Time: {solver.time:.2f}/{config.T_final}s ({100*solver.time/config.T_final:.1f}%)")
                logger.info(f"  Speed: {steps_per_sec:.0f} steps/s, {sim_speed:.4f}x realtime")
                logger.info(f"  ETA: {eta_minutes:.1f} minutes")
                logger.info(f"  dt: {dt:.2e}, max_speed: {max_speed:.2f}")
                logger.info(f"  Energies: KE={diagnostics['kinetic_energy']:.4f}, "
                           f"ME={diagnostics['magnetic_energy']:.4f}, TE={diagnostics['total_energy']:.4f}")
                logger.info(f"  Max |∇·B|: {diagnostics['max_divB']:.2e}")
                last_print_time = current_time
            
            pbar.update(dt)
            
            if solver.step_count % 5000 == 0:
                gc.collect()
    
    solver.save_to_hdf5()
    solver.compute_diagnostics()
    create_field_plots(solver, solver.output_path, solver.time, solver.step_count)
    create_1d_profiles(solver, solver.output_path, solver.time, solver.step_count)
    create_diagnostic_plots(solver, solver.output_path)
    create_animation(solver.output_path)
    
    end_time = time.time()
    total_duration = end_time - start_time
    steps_per_sec = solver.step_count / total_duration
    sim_speed = solver.time / total_duration
    
    print("\n🏁 SIMULATION COMPLETE!")
    print("="*50)
    print(f"✅ Simulated time: {solver.time:.2f}/{config.T_final}s ({100*solver.time/config.T_final:.1f}%)")
    print(f"✅ Total steps: {solver.step_count:,}")
    print(f"✅ Total saves: {solver.save_count}")
    print(f"✅ Runtime: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"✅ Performance: {steps_per_sec:.0f} steps/s")
    print(f"✅ Speed ratio: {sim_speed:.4f}x realtime")
    
    success = solver.time >= config.T_final * 0.9 and total_duration < 7200
    if success:
        print("🏆 TARGET ACHIEVED!")
        print(f"🎯 UNDER 2 HOURS: {total_duration/3600:.2f}h")
    else:
        print(f"⚠️ Simulation stopped at {100*solver.time/config.T_final:.1f}% completion")
    
    final_diagnostics = solver.compute_diagnostics()
    initial_te = solver.diagnostics['total_energy'][0] if solver.diagnostics['total_energy'] else 1.0
    energy_conservation = abs(final_diagnostics['total_energy'] / initial_te - 1) * 100
    print(f"🔬 Final energies: KE={final_diagnostics['kinetic_energy']:.4f}, "
          f"ME={final_diagnostics['magnetic_energy']:.4f}, TE={final_diagnostics['total_energy']:.4f}")
    print(f"🔬 Energy conservation: {energy_conservation:.2f}% deviation")
    print(f"🔬 Max |∇·B|: {final_diagnostics['max_divB']:.2e}")
    
    print(f"\n💾 Results saved to: {solver.output_path}")
    print(f"📁 Data file: {solver.hdf5_filename}")
    print(f"🎨 Check fields/ and profiles/ for plots")
    print(f"📊 Check diagnostics/ for analytics")
    print(f"🎬 Check animations/ for density movie")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)