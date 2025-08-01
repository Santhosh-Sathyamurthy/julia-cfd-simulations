import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import h5py
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
import logging
from tqdm import tqdm
import gc

# Environment setup
plt.switch_backend('Agg')
plt.style.use('dark_background')

# Logging setup
output_dir = "v3_30_256"
Path(output_dir).mkdir(exist_ok=True)
Path(output_dir + "/logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f'{output_dir}/logs/v3_30_256.log'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class V3Config:
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 256
    ny: int = 256
    gamma: float = 5.0/3.0
    rho0: float = 25.0/(36.0*np.pi)
    P0: float = 5.0/(12.0*np.pi)
    B0: float = 1.0/(4.0*np.pi)**0.5
    eta: float = 0.1  # Increased for stability
    nu: float = 0.1   # Increased for stability
    chi: float = 0.1  # Increased for stability
    T_final: float = 30.0
    dt_base: float = 0.0002
    cfl_safety: float = 0.3   # Relaxed for speed
    dt_max: float = 5e-5
    dt_min: float = 1e-8
    save_interval: int = 10000  # Increased to reduce I/O
    plot_interval: int = 10000  # Increased to reduce I/O
    output_dir: str = output_dir
    dpi: int = 150
    divB_cleaning: float = 0.02   # Reduced
    hyperbolic_cleaning: float = 0.01  # Increased
    parabolic_cleaning: float = 0.05  # New parabolic damping
    artificial_viscosity: float = 0.02  # Increased
    div_v_max: float = 10.0  # Tightened
    divB_max: float = 1e-2   # Threshold for dt reduction

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.cs2 = self.gamma * self.P0 / self.rho0
        beta = 2 * self.P0 / (self.B0**2)
        logger.info(f"Config: Grid={self.nx}x{self.ny}, T_final={self.T_final}, cs2={self.cs2:.2f}, beta={beta:.2f}")

@njit(parallel=True, fastmath=True)
def minmod(a, b):
    """Minmod flux limiter."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

@njit(parallel=True, fastmath=True)
def compute_derivatives_2nd_order(f, dx, dy):
    ny, nx = f.shape
    fx = np.zeros_like(f)
    fy = np.zeros_like(f)
    dx_inv = 1.0 / (2 * dx)
    dy_inv = 1.0 / (2 * dy)
    for i in prange(ny):
        for j in range(nx):
            fx[i,j] = minmod(f[i,(j+1)%nx] - f[i,j], f[i,j] - f[i,(j-1)%nx]) * dx_inv
    for i in range(ny):
        for j in prange(nx):
            fy[i,j] = minmod(f[(i+1)%ny,j] - f[i,j], f[i,j] - f[(i-1)%ny,j]) * dy_inv
    return fx, fy

@njit(parallel=True, fastmath=True)
def compute_laplacian(f, dx, dy, diffusivity):
    ny, nx = f.shape
    lap = np.zeros_like(f)
    dx2_inv = diffusivity / (dx * dx)
    dy2_inv = diffusivity / (dy * dy)
    for i in prange(ny):
        for j in prange(nx):
            lap[i,j] = dx2_inv * (f[i,(j+1)%nx] - 2*f[i,j] + f[i,(j-1)%nx]) + \
                       dy2_inv * (f[(i+1)%ny,j] - 2*f[i,j] + f[(i-1)%ny,j])
    return lap

@njit(parallel=True, fastmath=True)
def compute_divergence(vx, vy, dx, dy):
    dvx_dx, _ = compute_derivatives_2nd_order(vx, dx, dy)
    _, dvy_dy = compute_derivatives_2nd_order(vy, dx, dy)
    return dvx_dx + dvy_dy

@njit(parallel=True, fastmath=True)
def clean_divergence(Bx, By, psi, dx, dy, cleaning_factor, hyperbolic_factor, parabolic_factor):
    divB = compute_divergence(Bx, By, dx, dy)
    ddivB_dx, ddivB_dy = compute_derivatives_2nd_order(divB, dx, dy)
    Bx_new = Bx - cleaning_factor * dx * ddivB_dx
    By_new = By - cleaning_factor * dx * ddivB_dy
    dpsi_dx, dpsi_dy = compute_derivatives_2nd_order(psi, dx, dy)
    Bx_new -= hyperbolic_factor * dx * dpsi_dx
    By_new -= hyperbolic_factor * dx * dpsi_dy
    dpsi_dt = -hyperbolic_factor * divB + parabolic_factor * compute_laplacian(psi, dx, dy, parabolic_factor)
    Bx_new = np.clip(Bx_new, -1e2, 1e2)
    By_new = np.clip(By_new, -1e2, 1e2)
    return Bx_new, By_new, dpsi_dt

@njit(parallel=True, fastmath=True)
def mhd_rhs(rho, vx, vy, vz, Bx, By, Bz, P, psi, gamma, dx, dy, eta, nu, chi, divB_cleaning, hyperbolic_cleaning, parabolic_cleaning, artificial_viscosity, div_v_max):
    rho = np.maximum(rho, 0.01 * 25.0/(36.0*np.pi)).astype(np.float32)
    P = np.clip(P, 0.01 * 5.0/(12.0*np.pi), 5e1 * 5.0/(12.0*np.pi)).astype(np.float32)
    Bx = np.clip(Bx, -1e2, 1e2).astype(np.float32)
    By = np.clip(By, -1e2, 1e2).astype(np.float32)
    Bz = np.clip(Bz, -1e2, 1e2).astype(np.float32)

    drho_dx, drho_dy = compute_derivatives_2nd_order(rho, dx, dy)
    dvx_dx, dvx_dy = compute_derivatives_2nd_order(vx, dx, dy)
    dvy_dx, dvy_dy = compute_derivatives_2nd_order(vy, dx, dy)
    dvz_dx, dvz_dy = compute_derivatives_2nd_order(vz, dx, dy)
    dBx_dx, dBx_dy = compute_derivatives_2nd_order(Bx, dx, dy)
    dBy_dx, dBy_dy = compute_derivatives_2nd_order(By, dx, dy)
    dBz_dx, dBz_dy = compute_derivatives_2nd_order(Bz, dx, dy)
    dP_dx, dP_dy = compute_derivatives_2nd_order(P, dx, dy)
    
    div_v = np.clip(compute_divergence(vx, vy, dx, dy), -div_v_max, div_v_max)
    Jz = dBy_dx - dBx_dy
    B2 = Bx**2 + By**2 + Bz**2
    dB2_dx, dB2_dy = compute_derivatives_2nd_order(B2, dx, dy)
    
    lap_vx = compute_laplacian(vx, dx, dy, nu)
    lap_vy = compute_laplacian(vy, dx, dy, nu)
    lap_vz = compute_laplacian(vz, dx, dy, nu)
    lap_Bx = compute_laplacian(Bx, dx, dy, eta)
    lap_By = compute_laplacian(By, dx, dy, eta)
    lap_Bz = compute_laplacian(Bz, dx, dy, eta)
    lap_P = compute_laplacian(P, dx, dy, chi)
    lap_rho = compute_laplacian(rho, dx, dy, artificial_viscosity)
    
    drho_dt = -(rho * div_v + vx * drho_dx + vy * drho_dy) + lap_rho
    dvx_dt = -(vx * dvx_dx + vy * dvx_dy) - dP_dx / rho + \
             (Bx * dBx_dx + By * dBx_dy + Bz * dBz_dx - 0.5 * dB2_dx) / rho + lap_vx
    dvy_dt = -(vx * dvy_dx + vy * dvy_dy) - dP_dy / rho + \
             (Bx * dBy_dx + By * dBy_dy + Bz * dBz_dy - 0.5 * dB2_dy) / rho + lap_vy
    dvz_dt = -(vx * dvz_dx + vy * dvz_dy) + lap_vz
    dBx_dt = vy * Jz - (vx * dBx_dx + vy * dBx_dy - Bx * dvx_dx - By * dvx_dy) + lap_Bx
    dBy_dt = -vx * Jz - (vx * dBy_dx + vy * dBy_dy - Bx * dvy_dx - By * dvy_dy) + lap_By
    dBz_dt = -(vx * dBz_dx + vy * dBz_dy - Bx * dvz_dx - By * dvz_dy) + lap_Bz
    dP_dt = -gamma * P * div_v + lap_P + artificial_viscosity * compute_laplacian(P, dx, dy, artificial_viscosity)
    
    cs = np.sqrt(np.maximum(gamma * P / rho, 1e-10))
    va = np.sqrt(np.maximum(B2 / rho, 1e-10))
    max_speed = np.minimum(np.max(np.sqrt(vx**2 + vy**2 + vz**2) + np.sqrt(cs**2 + va**2)), 2e3)
    
    divB = compute_divergence(Bx, By, dx, dy)
    dpsi_dt = -hyperbolic_cleaning * divB + parabolic_cleaning * compute_laplacian(psi, dx, dy, parabolic_cleaning)
    
    return drho_dt, dvx_dt, dvy_dt, dvz_dt, dBx_dt, dBy_dt, dBz_dt, dP_dt, dpsi_dt, max_speed, div_v

class V3MHDSolver:
    def __init__(self, config):
        self.config = config
        self.setup_grid()
        self.initialize_fields()
        self.time = 0.0
        self.step = 0
        self.save_count = 0
        self.last_max_speed = 1.0
        self.output_path = Path(config.output_dir)
        for subdir in ["data", "fields", "profiles", "diagnostics"]:
            (self.output_path / subdir).mkdir(exist_ok=True)
        self.hdf5_file = self.output_path / "data" / "mhd_data.h5"
        self.diagnostics = {'time': [], 'kinetic_energy': [], 'magnetic_energy': [], 'total_energy': [], 'max_divB': []}
        
    def setup_grid(self):
        cfg = self.config
        self.x = np.linspace(0, cfg.Lx, cfg.nx, endpoint=False, dtype=np.float32)
        self.y = np.linspace(0, cfg.Ly, cfg.ny, endpoint=False, dtype=np.float32)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        
    def initialize_fields(self):
        cfg = self.config
        self.rho = np.full((cfg.ny, cfg.nx), cfg.rho0, dtype=np.float32)
        self.P = np.full((cfg.ny, cfg.nx), cfg.P0, dtype=np.float32)
        self.vx = -np.sin(2 * np.pi * self.Y, dtype=np.float32)
        self.vy = np.sin(2 * np.pi * self.X, dtype=np.float32)
        self.vz = np.zeros((cfg.ny, cfg.nx), dtype=np.float32)
        self.Bx = -cfg.B0 * np.sin(2 * np.pi * self.Y, dtype=np.float32)
        self.By = cfg.B0 * np.sin(4 * np.pi * self.X, dtype=np.float32)
        self.Bz = np.zeros((cfg.ny, cfg.nx), dtype=np.float32)
        self.psi = np.zeros((cfg.ny, cfg.nx), dtype=np.float32)
        div_B = compute_divergence(self.Bx, self.By, cfg.dx, cfg.dy)
        logger.info(f"Initial max |∇·B|: {np.max(np.abs(div_B)):.2e}")
        
    def compute_timestep(self, max_speed, max_cs, max_divB):
        cfg = self.config
        dt_cfl = cfg.cfl_safety * min(cfg.dx, cfg.dy) / max(np.maximum(max_speed, 1.0), 1e-10)
        dt_diff = 0.1 * min(cfg.dx**2, cfg.dy**2) / max(cfg.eta, cfg.nu, cfg.chi, cfg.artificial_viscosity, cfg.parabolic_cleaning)
        dt = np.clip(min(dt_cfl, dt_diff), cfg.dt_min, cfg.dt_max)
        if max_cs > 1e2 or max_divB > cfg.divB_max:
            dt *= 0.1
        return dt
    
    def _clean_divergence(self):
        Bx_new, By_new, dpsi_dt = clean_divergence(self.Bx, self.By, self.psi,
                                                  self.config.dx, self.config.dy,
                                                  self.config.divB_cleaning,
                                                  self.config.hyperbolic_cleaning,
                                                  self.config.parabolic_cleaning)
        self.Bx, self.By, self.psi = Bx_new, By_new, self.psi + dpsi_dt
        div_B = compute_divergence(self.Bx, self.By, self.config.dx, self.config.dy)
        logger.debug(f"Max |∇·B| after cleaning: {np.max(np.abs(div_B)):.2e}")
        
    def rk4_step(self, dt):
        cfg = self.config
        state = [self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi]
        k1 = mhd_rhs(*state, cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu, cfg.chi,
                     cfg.divB_cleaning, cfg.hyperbolic_cleaning, cfg.parabolic_cleaning, cfg.artificial_viscosity, cfg.div_v_max)
        logger.debug(f"RK4 k1: max_speed={k1[-2]:.2e}, min(rho)={np.min(state[0]):.2e}, max(div_v)={np.max(np.abs(k1[-1])):.2e}")
        self._clean_divergence()
        
        tmp = [s + 0.5 * dt * k for s, k in zip(state, k1)]
        tmp[0] = np.maximum(tmp[0], 0.01 * cfg.rho0)
        tmp[7] = np.clip(tmp[7], 0.01 * cfg.P0, 5e1 * cfg.P0)
        self._assign_state(tmp)
        k2 = mhd_rhs(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz,
                     self.P, self.psi, cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu,
                     cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning, cfg.parabolic_cleaning, cfg.artificial_viscosity, cfg.div_v_max)
        logger.debug(f"RK4 k2: max_speed={k2[-2]:.2e}, min(rho)={np.min(self.rho):.2e}, max(div_v)={np.max(np.abs(k2[-1])):.2e}")
        self._clean_divergence()
        
        tmp = [s + 0.5 * dt * k for s, k in zip(state, k2)]
        tmp[0] = np.maximum(tmp[0], 0.01 * cfg.rho0)
        tmp[7] = np.clip(tmp[7], 0.01 * cfg.P0, 5e1 * cfg.P0)
        self._assign_state(tmp)
        k3 = mhd_rhs(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz,
                     self.P, self.psi, cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu,
                     cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning, cfg.parabolic_cleaning, cfg.artificial_viscosity, cfg.div_v_max)
        logger.debug(f"RK4 k3: max_speed={k3[-2]:.2e}, min(rho)={np.min(self.rho):.2e}, max(div_v)={np.max(np.abs(k3[-1])):.2e}")
        self._clean_divergence()
        
        tmp = [s + dt * k for s, k in zip(state, k3)]
        tmp[0] = np.maximum(tmp[0], 0.01 * cfg.rho0)
        tmp[7] = np.clip(tmp[7], 0.01 * cfg.P0, 5e1 * cfg.P0)
        self._assign_state(tmp)
        k4 = mhd_rhs(self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz,
                     self.P, self.psi, cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu,
                     cfg.chi, cfg.divB_cleaning, cfg.hyperbolic_cleaning, cfg.parabolic_cleaning, cfg.artificial_viscosity, cfg.div_v_max)
        logger.debug(f"RK4 k4: max_speed={k4[-2]:.2e}, min(rho)={np.min(self.rho):.2e}, max(div_v)={np.max(np.abs(k4[-1])):.2e}")
        self._clean_divergence()
        
        fields = ['rho', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz', 'P', 'psi']
        for i, field in enumerate(fields):
            increment = dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            updated_value = state[i] + increment
            if np.any(np.isnan(updated_value)) or np.any(np.isinf(updated_value)):
                logger.error(f"NaN or Inf detected in {field} at step {self.step}, time {self.time}")
                raise ValueError(f"Simulation diverged: {field} contains NaN or Inf")
            setattr(self, field, updated_value)
        
        self._clean_divergence()
        
        self.rho = np.maximum(self.rho, 0.01 * cfg.rho0)
        self.P = np.clip(self.P, 0.01 * cfg.P0, 5e1 * cfg.P0)
        self.Bx = np.clip(self.Bx, -1e2, 1e2)
        self.By = np.clip(self.By, -1e2, 1e2)
        self.Bz = np.clip(self.Bz, -1e2, 1e2)
        
        max_cs = np.max(np.sqrt(np.maximum(cfg.gamma * self.P / self.rho, 1e-10)))
        max_divB = np.max(np.abs(compute_divergence(self.Bx, self.By, cfg.dx, cfg.dy)))
        logger.info(f"Step {self.step}: min(rho)={np.min(self.rho):.2e}, max(P)={np.max(self.P):.2e}, min(P)={np.min(self.P):.2e}, max(cs)={max_cs:.2e}, max_speed={k4[-2]:.2e}, dt={dt:.2e}, max_divB={max_divB:.2e}")
        return k4[-2], max_cs, max_divB

    def _assign_state(self, tmp):
        self.rho, self.vx, self.vy, self.vz, self.Bx, self.By, self.Bz, self.P, self.psi = tmp

    def save_to_hdf5(self):
        with h5py.File(self.hdf5_file, 'w') as f:
            group_name = f"step_{self.step:06d}"
            if group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            group.attrs['time'] = self.time
            for field in ['rho', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz', 'P']:
                group.create_dataset(field, data=getattr(self, field), compression='gzip', dtype=np.float32)
            if 'grid' not in f:
                f.create_dataset('X', data=self.X, compression='gzip')
                f.create_dataset('Y', data=self.Y, compression='gzip')
        self.save_count += 1
    
    def compute_diagnostics(self):
        cfg = self.config
        kinetic = 0.5 * np.mean(self.rho * (self.vx**2 + self.vy**2 + self.vz**2))
        magnetic = 0.5 * np.mean(self.Bx**2 + self.By**2 + self.Bz**2)
        total = kinetic + magnetic + np.mean(self.P / (cfg.gamma - 1))
        div_B = compute_divergence(self.Bx, self.By, cfg.dx, cfg.dy)
        max_divB = np.max(np.abs(div_B))
        self.diagnostics['time'].append(self.time)
        self.diagnostics['kinetic_energy'].append(kinetic)
        self.diagnostics['magnetic_energy'].append(magnetic)
        self.diagnostics['total_energy'].append(total)
        self.diagnostics['max_divB'].append(max_divB)
        logger.info(f"Diagnostics at t={self.time:.2e}: kinetic={kinetic:.2e}, magnetic={magnetic:.2e}, total={total:.2e}, max_divB={max_divB:.2e}")
        return {'kinetic': kinetic, 'magnetic': magnetic, 'total': total, 'max_divB': max_divB}

def generate_plots_from_hdf5(config):
    hdf5_file = Path(config.output_dir) / "data" / "mhd_data.h5"
    fields_path = Path(config.output_dir) / "fields"
    profiles_path = Path(config.output_dir) / "profiles"
    diag_path = Path(config.output_dir) / "diagnostics"
    for path in [fields_path, profiles_path, diag_path]:
        path.mkdir(exist_ok=True)
    
    with h5py.File(hdf5_file, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        steps = sorted([k for k in f.keys() if k.startswith('step_')], key=lambda x: int(x.split('_')[1]))
        for step_key in tqdm(steps, desc="Generating Plots"):
            step = int(step_key.split('_')[1])
            group = f[step_key]
            time_val = group.attrs['time']
            rho = group['rho'][:]
            vx = group['vx'][:]
            vy = group['vy'][:]
            vz = group['vz'][:]
            Bx = group['Bx'][:]
            By = group['By'][:]
            Bz = group['Bz'][:]
            P = group['P'][:]
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            Jz = compute_derivatives_2nd_order(By, config.dx, config.dy)[0] - compute_derivatives_2nd_order(Bx, config.dx, config.dy)[1]
            
            for name, data, cmap, label in [
                ('density', rho, 'viridis', 'ρ'), ('velocity', v_mag, 'plasma', '|v|'),
                ('magnetic', B_mag, 'coolwarm', '|B|'), ('pressure', P, 'hot', 'P'),
                ('current', Jz, 'RdBu', 'Jz')
            ]:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.contourf(X, Y, data, levels=50, cmap=cmap, extend='both' if name == 'current' else None)
                if name in ['velocity', 'magnetic']:
                    ax.streamplot(X, Y, vx if name == 'velocity' else Bx, vy if name == 'velocity' else By,
                                  color='white' if name == 'velocity' else 'black', density=1.5)
                plt.colorbar(im, ax=ax, label=label)
                ax.set_title(f'{name.capitalize()} at t = {time_val:.2f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal')
                plt.savefig(fields_path / f"{name}_{step:06d}.png", dpi=config.dpi, bbox_inches='tight', facecolor='black')
                plt.close(fig)
            
            center_x, center_y = config.nx // 2, config.ny // 2
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Profiles at t = {time_val:.2f}')
            for ax, data, label in [
                (axes[0,0], rho, 'ρ'), (axes[0,1], vx, 'vx'), (axes[0,2], vy, 'vy'),
                (axes[1,0], Bx, 'Bx'), (axes[1,1], By, 'By')
            ]:
                ax.plot(X[center_y, :], data[center_y, :], 'cyan', label='y=0.5')
                ax.plot(Y[:, center_x], data[:, center_x], 'orange', label='x=0.5')
                ax.set_xlabel('Position')
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                ax.legend()
            axes[1,2].axis('off')
            plt.savefig(profiles_path / f"profiles_{step:06d}.png", dpi=config.dpi, bbox_inches='tight', facecolor='black')
            plt.close(fig)
    
    solver = V3MHDSolver(config)
    solver.diagnostics = h5py.File(hdf5_file, 'r')['step_000000'].parent.attrs.get('diagnostics', solver.diagnostics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('MHD Diagnostics')
    axes[0,0].plot(solver.diagnostics['time'], solver.diagnostics['kinetic_energy'], label='Kinetic')
    axes[0,0].plot(solver.diagnostics['time'], solver.diagnostics['magnetic_energy'], label='Magnetic')
    axes[0,0].plot(solver.diagnostics['time'], solver.diagnostics['total_energy'], label='Total')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Energy')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,1].semilogy(solver.diagnostics['time'], solver.diagnostics['max_divB'], label='Max |∇·B|')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Max |∇·B|')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[1,0].axis('off')
    axes[1,1].axis('off')
    plt.savefig(diag_path / "diagnostics.png", dpi=config.dpi, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    gc.collect()

def main():
    config = V3Config()
    solver = V3MHDSolver(config)
    logger.info(f"Starting simulation: {config.nx}x{config.ny}, T_final={config.T_final}")
    start_time = time.time()
    
    solver.save_to_hdf5()
    solver.compute_diagnostics()
    
    with tqdm(total=config.T_final, desc="Simulation") as pbar:
        while solver.time < config.T_final:
            max_divB = np.max(np.abs(compute_divergence(solver.Bx, solver.By, config.dx, config.dy)))
            dt = solver.compute_timestep(solver.last_max_speed, np.max(np.sqrt(np.maximum(config.gamma * solver.P / solver.rho, 1e-10))), max_divB)
            max_speed, max_cs, max_divB = solver.rk4_step(dt)
            solver.time += dt
            solver.step += 1
            solver.last_max_speed = max_speed
            if solver.step % config.save_interval == 0:
                solver.save_to_hdf5()
                solver.compute_diagnostics()
            pbar.update(dt)
    
    solver.save_to_hdf5()
    solver.compute_diagnostics()
    end_time = time.time()
    logger.info(f"Simulation completed in {(end_time - start_time)/3600:.2f} hours")
    logger.info(f"Generating plots from HDF5...")
    generate_plots_from_hdf5(config)
    logger.info(f"Results saved in {config.output_dir}")

if __name__ == "__main__":
    main()