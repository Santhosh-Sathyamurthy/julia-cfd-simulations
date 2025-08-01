"""
Optimized Orszag-Tang Vortex: High-Performance MHD Turbulence Simulation
=========================================================================

A professional-grade 2D compressible magnetohydrodynamics simulation incorporating
advanced numerical methods and optimization techniques for MHD turbulence research.

Features:
- High-performance Numba-accelerated computations
- Adaptive time stepping with CFL condition
- Advanced visualization with real-time monitoring
- HDF5 data storage for large-scale analysis
- Memory-efficient operations for extended simulations
- Comprehensive diagnostics and energy conservation tracking

Physics Demonstrated:
- Magnetic reconnection phenomena
- Current sheet formation and evolution  
- MHD turbulent cascade development
- Energy transfer between kinetic and magnetic modes
- Compressible plasma dynamics

Author: MHD Research Team
Target: Advanced MHD Turbulence Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from numba import njit, prange
import h5py
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
import logging
from tqdm import tqdm
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor

# Configure environment
plt.switch_backend('Agg')
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/v1_512.log')]
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)

@dataclass
class OptimizedMHDConfig:
    """Configuration class for the Orszag-Tang MHD simulation"""
    
    # Domain parameters
    Lx: float = 1.0          # Domain size x
    Ly: float = 1.0          # Domain size y  
    nx: int = 512            # Grid points x
    ny: int = 512            # Grid points y
    
    # Physical parameters  
    gamma: float = 5.0/3.0   # Adiabatic index
    rho0: float = 25.0/(36.0*np.pi)  # Initial density
    P0: float = 5.0/(12.0*np.pi)     # Initial pressure
    B0: float = 1.0/(4.0*np.pi)**0.5 # Magnetic field strength
    
    # Numerical parameters
    eta: float = 0.0001      # Magnetic diffusivity
    nu: float = 0.0001       # Kinematic viscosity  
    chi: float = 0.0001      # Thermal diffusivity
    
    # Time integration
    T_final: float = 5.0     # Final simulation time
    dt_base: float = 0.0001  # Base time step
    cfl_safety: float = 0.3  # CFL safety factor
    adaptive_dt: bool = True # Adaptive time stepping
    dt_min: float = 1e-6     # Minimum time step
    dt_max: float = 0.001    # Maximum time step
    
    # Output and performance
    save_interval: int = 100  # Save data every N steps
    plot_interval: int = 50   # Plot every N steps
    output_dir: str = "v1_512"
    hdf5_file: str = "mhd_data.h5"
    parallel_threads: int = 4
    memory_efficient: bool = True
    dpi: int = 150
    
    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.cs2 = self.gamma * self.P0 / self.rho0  # Should equal 1.0
        self.parallel_threads = min(self.parallel_threads, psutil.cpu_count())
        
        # Memory estimation
        memory_mb = (self.nx * self.ny * 8 * 8) / (1024 * 1024)  # 8 fields, 8 bytes each
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        logger.info("=== Optimized Orszag-Tang MHD Configuration ===")
        logger.info(f"Domain: {self.Lx} x {self.Ly}, Grid: {self.nx} x {self.ny}")
        logger.info(f"Resolution: dx={self.dx:.6f}, dy={self.dy:.6f}")
        logger.info(f"Physical: ρ₀={self.rho0:.6f}, P₀={self.P0:.6f}, B₀={self.B0:.6f}")
        logger.info(f"Sound speed²: {self.cs2:.6f} (should be 1.0)")
        logger.info(f"Estimated memory: {memory_mb:.1f} MB / {available_mb:.1f} MB available")
        logger.info(f"Parallel threads: {self.parallel_threads}")
        logger.info("=" * 48)

# High-performance numerical kernels
@njit(parallel=True, fastmath=True, cache=True)
def compute_derivatives_4th_order(f, dx, dy):
    """High-accuracy 4th order spatial derivatives with periodic BC"""
    ny, nx = f.shape
    fx = np.zeros_like(f)
    fy = np.zeros_like(f)
    
    # 4th order central differences for interior points
    for i in prange(ny):
        for j in range(2, nx-2):
            fx[i,j] = (-f[i,(j+2)%nx] + 8*f[i,(j+1)%nx] - 8*f[i,(j-1)%nx] + f[i,(j-2)%nx]) / (12*dx)
        # Boundary points with periodic BC
        fx[i,0] = (-f[i,2] + 8*f[i,1] - 8*f[i,nx-1] + f[i,nx-2]) / (12*dx)
        fx[i,1] = (-f[i,3] + 8*f[i,2] - 8*f[i,0] + f[i,nx-1]) / (12*dx)
        fx[i,nx-2] = (-f[i,0] + 8*f[i,nx-1] - 8*f[i,nx-3] + f[i,nx-4]) / (12*dx)
        fx[i,nx-1] = (-f[i,1] + 8*f[i,0] - 8*f[i,nx-2] + f[i,nx-3]) / (12*dx)
    
    for i in range(2, ny-2):
        for j in prange(nx):
            fy[i,j] = (-f[(i+2)%ny,j] + 8*f[(i+1)%ny,j] - 8*f[(i-1)%ny,j] + f[(i-2)%ny,j]) / (12*dy)
    # Boundary points with periodic BC
    for j in prange(nx):
        fy[0,j] = (-f[2,j] + 8*f[1,j] - 8*f[ny-1,j] + f[ny-2,j]) / (12*dy)
        fy[1,j] = (-f[3,j] + 8*f[2,j] - 8*f[0,j] + f[ny-1,j]) / (12*dy)
        fy[ny-2,j] = (-f[0,j] + 8*f[ny-1,j] - 8*f[ny-3,j] + f[ny-4,j]) / (12*dy)
        fy[ny-1,j] = (-f[1,j] + 8*f[0,j] - 8*f[ny-2,j] + f[ny-3,j]) / (12*dy)
    
    return fx, fy

@njit(parallel=True, fastmath=True, cache=True)
def compute_curl_z(Bx, By, dx, dy):
    """Compute z-component of curl (current density)"""
    dBx_dy, _ = compute_derivatives_4th_order(Bx, dx, dy)
    _, dBy_dx = compute_derivatives_4th_order(By, dx, dy)
    return dBy_dx - dBx_dy

@njit(parallel=True, fastmath=True, cache=True)
def compute_divergence(vx, vy, dx, dy):
    """Compute divergence of vector field"""
    dvx_dx, _ = compute_derivatives_4th_order(vx, dx, dy)
    _, dvy_dy = compute_derivatives_4th_order(vy, dx, dy)
    return dvx_dx + dvy_dy

@njit(parallel=True, fastmath=True, cache=True)
def compute_laplacian(f, dx, dy, diffusivity):
    """Compute Laplacian with diffusivity coefficient"""
    ny, nx = f.shape
    lap = np.zeros_like(f)
    dx2_inv = diffusivity / (dx * dx)
    dy2_inv = diffusivity / (dy * dy)
    
    for i in prange(ny):
        for j in prange(nx):
            lap[i,j] = dx2_inv * (f[i,(j+1)%nx] - 2*f[i,j] + f[i,(j-1)%nx]) + \
                       dy2_inv * (f[(i+1)%ny,j] - 2*f[i,j] + f[(i-1)%ny,j])
    return lap

@njit(parallel=True, fastmath=True, cache=True)
def compute_mhd_rhs(rho, vx, vy, vz, Bx, By, Bz, P, gamma, dx, dy, eta, nu, chi):
    """Compute right-hand side of MHD equations with high performance"""
    
    # Compute all necessary derivatives
    drho_dx, drho_dy = compute_derivatives_4th_order(rho, dx, dy)
    dvx_dx, dvx_dy = compute_derivatives_4th_order(vx, dx, dy)
    dvy_dx, dvy_dy = compute_derivatives_4th_order(vy, dx, dy)
    dvz_dx, dvz_dy = compute_derivatives_4th_order(vz, dx, dy)
    dBx_dx, dBx_dy = compute_derivatives_4th_order(Bx, dx, dy)
    dBy_dx, dBy_dy = compute_derivatives_4th_order(By, dx, dy)
    dBz_dx, dBz_dy = compute_derivatives_4th_order(Bz, dx, dy)
    dP_dx, dP_dy = compute_derivatives_4th_order(P, dx, dy)
    
    # Velocity divergence
    div_v = dvx_dx + dvy_dy
    
    # Current density
    Jz = dBy_dx - dBx_dy
    
    # Magnetic field components and pressure
    B2 = Bx**2 + By**2 + Bz**2
    dB2_dx, dB2_dy = compute_derivatives_4th_order(B2, dx, dy)
    
    # Continuity equation
    drho_dt = -(rho * div_v + vx * drho_dx + vy * drho_dy)
    
    # Momentum equations with Lorentz force
    magnetic_pressure_x = 0.5 * dB2_dx
    magnetic_pressure_y = 0.5 * dB2_dy
    
    magnetic_tension_x = Bx * dBx_dx + By * dBx_dy + Bz * dBz_dx
    magnetic_tension_y = Bx * dBy_dx + By * dBy_dy + Bz * dBz_dy
    
    # Viscous terms
    lap_vx = compute_laplacian(vx, dx, dy, nu)
    lap_vy = compute_laplacian(vy, dx, dy, nu)
    lap_vz = compute_laplacian(vz, dx, dy, nu)
    
    dvx_dt = -(vx * dvx_dx + vy * dvx_dy) - dP_dx / rho + \
             (magnetic_tension_x - magnetic_pressure_x) / rho + lap_vx
    
    dvy_dt = -(vx * dvy_dx + vy * dvy_dy) - dP_dy / rho + \
             (magnetic_tension_y - magnetic_pressure_y) / rho + lap_vy
    
    dvz_dt = -(vx * dvz_dx + vy * dvz_dy) + lap_vz
    
    # Induction equation
    lap_Bx = compute_laplacian(Bx, dx, dy, eta)
    lap_By = compute_laplacian(By, dx, dy, eta)
    lap_Bz = compute_laplacian(Bz, dx, dy, eta)
    
    dBx_dt = vy * Jz - (vx * dBx_dx + vy * dBx_dy - Bx * dvx_dx - By * dvx_dy) + lap_Bx
    dBy_dt = -vx * Jz - (vx * dBy_dx + vy * dBy_dy - Bx * dvy_dx - By * dvy_dy) + lap_By
    dBz_dt = -(vx * dBz_dx + vy * dBz_dy - Bx * dvz_dx - By * dvz_dy) + lap_Bz
    
    # Energy equation (simplified ideal gas)
    dP_dt = -gamma * P * div_v + compute_laplacian(P, dx, dy, chi)
    
    return drho_dt, dvx_dt, dvy_dt, dvz_dt, dBx_dt, dBy_dt, dBz_dt, dP_dt

class OptimizedOrszagTangMHD:
    """High-performance Orszag-Tang vortex MHD simulation"""
    
    def __init__(self, config: OptimizedMHDConfig):
        self.config = config
        self.setup_grid()
        self.initialize_fields()
        self.setup_diagnostics()
        self.step = 0
        self.current_time = 0.0
        
    def setup_grid(self):
        """Initialize computational grid"""
        cfg = self.config
        self.x = np.linspace(0, cfg.Lx, cfg.nx, endpoint=False)
        self.y = np.linspace(0, cfg.Ly, cfg.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        
    def initialize_fields(self):
        """Initialize Orszag-Tang vortex with exact initial conditions"""
        cfg = self.config
        dtype = np.float32 if cfg.memory_efficient else np.float64
        
        # Initialize all fields
        self.rho = np.full((cfg.ny, cfg.nx), cfg.rho0, dtype=dtype)
        self.P = np.full((cfg.ny, cfg.nx), cfg.P0, dtype=dtype)
        
        # Velocity field: Vx = -sin(2πy), Vy = sin(2πx)
        self.vx = -np.sin(2 * np.pi * self.Y).astype(dtype)
        self.vy = np.sin(2 * np.pi * self.X).astype(dtype)
        self.vz = np.zeros_like(self.vx)
        
        # Magnetic field from vector potential
        # Az = B0 * (cos(4πx)/(4π) + cos(2πy)/(2π))
        # Bx = -∂Az/∂y = B0 * sin(2πy)
        # By = ∂Az/∂x = -B0 * sin(4πx)
        self.Bx = -cfg.B0 * np.sin(2 * np.pi * self.Y).astype(dtype)
        self.By = cfg.B0 * np.sin(4 * np.pi * self.X).astype(dtype)
        self.Bz = np.zeros_like(self.Bx)
        
        # Verify initial conditions
        div_B = compute_divergence(self.Bx, self.By, cfg.dx, cfg.dy)
        cs2_actual = cfg.gamma * self.P[0,0] / self.rho[0,0]
        
        logger.info("Initial conditions verified:")
        logger.info(f"  Density: ρ = {self.rho[0,0]:.6f}")
        logger.info(f"  Pressure: P = {self.P[0,0]:.6f}")
        logger.info(f"  Sound speed²: cs² = {cs2_actual:.6f} (target: 1.0)")
        logger.info(f"  Max |∇·B|: {np.max(np.abs(div_B)):.2e}")
        logger.info(f"  Velocity range: |v| ∈ [{np.min(np.sqrt(self.vx**2 + self.vy**2)):.3f}, {np.max(np.sqrt(self.vx**2 + self.vy**2)):.3f}]")
        logger.info(f"  Magnetic field range: |B| ∈ [{np.min(np.sqrt(self.Bx**2 + self.By**2)):.3f}, {np.max(np.sqrt(self.Bx**2 + self.By**2)):.3f}]")
        
    def compute_timestep(self):
        """Adaptive time step based on CFL condition"""
        if not self.config.adaptive_dt:
            return self.config.dt_base
            
        cfg = self.config
        
        # Sound speed
        cs = np.sqrt(cfg.gamma * self.P / self.rho)
        
        # Alfven speed
        B2 = self.Bx**2 + self.By**2 + self.Bz**2
        va = np.sqrt(B2 / self.rho)
        
        # Fast magnetosonic speed
        cf = np.sqrt(cs**2 + va**2)
        
        # Velocity magnitude
        v_mag = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        
        # Maximum characteristic speed
        max_speed = np.max(v_mag + cf)
        
        # CFL time step
        dt_cfl = cfg.cfl_safety * min(cfg.dx, cfg.dy) / max_speed
        
        # Diffusion time step
        dt_diff = 0.2 * min(cfg.dx**2, cfg.dy**2) / max(cfg.eta, cfg.nu, cfg.chi)
        
        dt = min(dt_cfl, dt_diff)
        return np.clip(dt, cfg.dt_min, cfg.dt_max)
    
    def rk4_step(self, dt):
        """4th order Runge-Kutta time integration"""
        cfg = self.config
        
        # Store initial state
        rho0 = self.rho.copy()
        vx0, vy0, vz0 = self.vx.copy(), self.vy.copy(), self.vz.copy()
        Bx0, By0, Bz0 = self.Bx.copy(), self.By.copy(), self.Bz.copy()
        P0 = self.P.copy()
        
        # k1
        k1_rho, k1_vx, k1_vy, k1_vz, k1_Bx, k1_By, k1_Bz, k1_P = \
            compute_mhd_rhs(self.rho, self.vx, self.vy, self.vz, 
                           self.Bx, self.By, self.Bz, self.P,
                           cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu, cfg.chi)
        
        # k2
        self.rho = rho0 + 0.5 * dt * k1_rho
        self.vx, self.vy, self.vz = vx0 + 0.5 * dt * k1_vx, vy0 + 0.5 * dt * k1_vy, vz0 + 0.5 * dt * k1_vz
        self.Bx, self.By, self.Bz = Bx0 + 0.5 * dt * k1_Bx, By0 + 0.5 * dt * k1_By, Bz0 + 0.5 * dt * k1_Bz
        self.P = P0 + 0.5 * dt * k1_P
        
        k2_rho, k2_vx, k2_vy, k2_vz, k2_Bx, k2_By, k2_Bz, k2_P = \
            compute_mhd_rhs(self.rho, self.vx, self.vy, self.vz,
                           self.Bx, self.By, self.Bz, self.P,
                           cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu, cfg.chi)
        
        # k3
        self.rho = rho0 + 0.5 * dt * k2_rho
        self.vx, self.vy, self.vz = vx0 + 0.5 * dt * k2_vx, vy0 + 0.5 * dt * k2_vy, vz0 + 0.5 * dt * k2_vz
        self.Bx, self.By, self.Bz = Bx0 + 0.5 * dt * k2_Bx, By0 + 0.5 * dt * k2_By, Bz0 + 0.5 * dt * k2_Bz
        self.P = P0 + 0.5 * dt * k2_P
        
        k3_rho, k3_vx, k3_vy, k3_vz, k3_Bx, k3_By, k3_Bz, k3_P = \
            compute_mhd_rhs(self.rho, self.vx, self.vy, self.vz,
                           self.Bx, self.By, self.Bz, self.P,
                           cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu, cfg.chi)
        
        # k4
        self.rho = rho0 + dt * k3_rho
        self.vx, self.vy, self.vz = vx0 + dt * k3_vx, vy0 + dt * k3_vy, vz0 + dt * k3_vz
        self.Bx, self.By, self.Bz = Bx0 + dt * k3_Bx, By0 + dt * k3_By, Bz0 + dt * k3_Bz
        self.P = P0 + dt * k3_P
        
        k4_rho, k4_vx, k4_vy, k4_vz, k4_Bx, k4_By, k4_Bz, k4_P = \
            compute_mhd_rhs(self.rho, self.vx, self.vy, self.vz,
                           self.Bx, self.By, self.Bz, self.P,
                           cfg.gamma, cfg.dx, cfg.dy, cfg.eta, cfg.nu, cfg.chi)
        
        # Final update
        self.rho = rho0 + dt/6 * (k1_rho + 2*k2_rho + 2*k3_rho + k4_rho)
        self.vx = vx0 + dt/6 * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        self.vy = vy0 + dt/6 * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)
        self.vz = vz0 + dt/6 * (k1_vz + 2*k2_vz + 2*k3_vz + k4_vz)
        self.Bx = Bx0 + dt/6 * (k1_Bx + 2*k2_Bx + 2*k3_Bx + k4_Bx)
        self.By = By0 + dt/6 * (k1_By + 2*k2_By + 2*k3_By + k4_By)
        self.Bz = Bz0 + dt/6 * (k1_Bz + 2*k2_Bz + 2*k3_Bz + k4_Bz)
        self.P = P0 + dt/6 * (k1_P + 2*k2_P + 2*k3_P + k4_P)
        
        # Ensure positivity
        self.rho = np.maximum(self.rho, 0.01 * self.config.rho0)
        self.P = np.maximum(self.P, 0.01 * self.config.P0)
    
    def setup_diagnostics(self):
        """Initialize diagnostic tracking"""
        self.time_history = []
        self.kinetic_energy_history = []
        self.magnetic_energy_history = []
        self.thermal_energy_history = []
        self.total_energy_history = []
        self.max_current_history = []
        self.max_vorticity_history = []
        self.min_density_history = []
        
    def compute_diagnostics(self):
        """Compute comprehensive diagnostic quantities"""
        cfg = self.config
        
        # Kinetic energy density
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        kinetic_energy = 0.5 * np.mean(self.rho * v2)
        
        # Magnetic energy density
        B2 = self.Bx**2 + self.By**2 + self.Bz**2
        magnetic_energy = 0.5 * np.mean(B2)
        
        # Thermal energy density
        thermal_energy = np.mean(self.P / (cfg.gamma - 1))
        
        # Total energy
        total_energy = kinetic_energy + magnetic_energy + thermal_energy
        
        # Current density
        Jz = compute_curl_z(self.Bx, self.By, cfg.dx, cfg.dy)
        max_current = np.max(np.abs(Jz))
        
        # Vorticity
        dvx_dy, _ = compute_derivatives_4th_order(self.vx, cfg.dx, cfg.dy)
        _, dvy_dx = compute_derivatives_4th_order(self.vy, cfg.dx, cfg.dy)
        vorticity = dvy_dx - dvx_dy
        max_vorticity = np.max(np.abs(vorticity))
        
        # Density extrema
        min_density = np.min(self.rho)
        
        # Store diagnostics
        self.time_history.append(self.current_time)
        self.kinetic_energy_history.append(kinetic_energy)
        self.magnetic_energy_history.append(magnetic_energy)
        self.thermal_energy_history.append(thermal_energy)
        self.total_energy_history.append(total_energy)
        self.max_current_history.append(max_current)
        self.max_vorticity_history.append(max_vorticity)
        self.min_density_history.append(min_density)
        
        return {
            'kinetic_energy': kinetic_energy,
            'magnetic_energy': magnetic_energy,
            'thermal_energy': thermal_energy,
            'total_energy': total_energy,
            'max_current': max_current,
            'max_vorticity': max_vorticity,
            'min_density': min_density
        }

class OptimizedMHDVisualizer:
    """High-performance visualization system for MHD data"""
    
    def __init__(self, config: OptimizedMHDConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.frames_path = self.output_path / "frames"
        self.data_path = self.output_path / "data"
        self.frames_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def save_data_hdf5(self, solver: OptimizedOrszagTangMHD):
        """Save simulation data to HDF5"""
        hdf5_path = self.data_path / self.config.hdf5_file
        
        try:
            with h5py.File(hdf5_path, 'a') as f:
                group_name = f"step_{solver.step:08d}"
                if group_name in f:
                    del f[group_name]
                    
                group = f.create_group(group_name)
                group.attrs['time'] = solver.current_time
                group.attrs['step'] = solver.step
                
                # Save all field data
                group.create_dataset('rho', data=solver.rho, compression='gzip')
                group.create_dataset('vx', data=solver.vx, compression='gzip')
                group.create_dataset('vy', data=solver.vy, compression='gzip')
                group.create_dataset('vz', data=solver.vz, compression='gzip')
                group.create_dataset('Bx', data=solver.Bx, compression='gzip')
                group.create_dataset('By', data=solver.By, compression='gzip')
                group.create_dataset('Bz', data=solver.Bz, compression='gzip')
                group.create_dataset('P', data=solver.P, compression='gzip')
                
                # Save derived quantities
                v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
                B_mag = np.sqrt(solver.Bx**2 + solver.By**2 + solver.Bz**2)
                
                group.create_dataset('velocity_magnitude', data=v_mag, compression='gzip')
                group.create_dataset('magnetic_magnitude', data=B_mag, compression='gzip')
                
                # Current density and vorticity
                Jz = compute_curl_z(solver.Bx, solver.By, solver.config.dx, solver.config.dy)
                dvx_dy, _ = compute_derivatives_4th_order(solver.vx, solver.config.dx, solver.config.dy)
                _, dvy_dx = compute_derivatives_4th_order(solver.vy, solver.config.dx, solver.config.dy)
                vorticity = dvy_dx - dvx_dy
                
                group.create_dataset('current_density', data=Jz, compression='gzip')
                group.create_dataset('vorticity', data=vorticity, compression='gzip')
                
                # Grid data (save once)
                if 'grid' not in f:
                    grid_group = f.create_group('grid')
                    grid_group.create_dataset('X', data=solver.X, compression='gzip')
                    grid_group.create_dataset('Y', data=solver.Y, compression='gzip')
                    grid_group.create_dataset('x', data=solver.x, compression='gzip')
                    grid_group.create_dataset('y', data=solver.y, compression='gzip')
                
        except Exception as e:
            logger.error(f"Error saving HDF5 data at step {solver.step}: {e}")
    
    def create_comprehensive_plot(self, solver: OptimizedOrszagTangMHD):
        """Create comprehensive visualization of MHD fields"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Orszag-Tang Vortex: t = {solver.current_time:.3f}', fontsize=16, color='white')
            
            # Compute derived quantities
            v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
            B_mag = np.sqrt(solver.Bx**2 + solver.By**2 + solver.Bz**2)
            Jz = compute_curl_z(solver.Bx, solver.By, solver.config.dx, solver.config.dy)
            
            dvx_dy, _ = compute_derivatives_4th_order(solver.vx, solver.config.dx, solver.config.dy)
            _, dvy_dx = compute_derivatives_4th_order(solver.vy, solver.config.dx, solver.config.dy)
            vorticity = dvy_dx - dvx_dy
            
            # Row 1: Primary fields
            # Density
            im1 = axes[0,0].contourf(solver.X, solver.Y, solver.rho, levels=50, cmap='viridis')
            axes[0,0].set_title('Density ρ')
            axes[0,0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
            
            # Velocity field with streamlines
            im2 = axes[0,1].contourf(solver.X, solver.Y, v_mag, levels=50, cmap='plasma')
            axes[0,1].streamplot(solver.X, solver.Y, solver.vx, solver.vy, 
                                color='white', linewidth=0.8, density=1.5)
            axes[0,1].set_title('Velocity |V|')
            axes[0,1].set_aspect('equal')
            plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
            
            # Magnetic field with field lines
            im3 = axes[0,2].contourf(solver.X, solver.Y, B_mag, levels=50, cmap='coolwarm')
            axes[0,2].streamplot(solver.X, solver.Y, solver.Bx, solver.By,
                                color='black', linewidth=0.8, density=1.5)
            axes[0,2].set_title('Magnetic Field |B|')
            axes[0,2].set_aspect('equal')
            plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
            
            # Pressure
            im4 = axes[0,3].contourf(solver.X, solver.Y, solver.P, levels=50, cmap='hot')
            axes[0,3].set_title('Pressure P')
            axes[0,3].set_aspect('equal')
            plt.colorbar(im4, ax=axes[0,3], shrink=0.8)
            
            # Row 2: Derived quantities
            # Current density
            Jz_max = np.max(np.abs(Jz))
            im5 = axes[1,0].contourf(solver.X, solver.Y, Jz, 
                                    levels=np.linspace(-Jz_max, Jz_max, 51), 
                                    cmap='RdBu_r', extend='both')
            axes[1,0].set_title('Current Density Jz')
            axes[1,0].set_aspect('equal')
            plt.colorbar(im5, ax=axes[1,0], shrink=0.8)
            
            # Vorticity
            vort_max = np.max(np.abs(vorticity))
            im6 = axes[1,1].contourf(solver.X, solver.Y, vorticity,
                                    levels=np.linspace(-vort_max, vort_max, 51),
                                    cmap='seismic', extend='both')
            axes[1,1].set_title('Vorticity ωz')
            axes[1,1].set_aspect('equal')
            plt.colorbar(im6, ax=axes[1,1], shrink=0.8)
            
            # Energy density
            kinetic_density = 0.5 * solver.rho * v_mag**2
            magnetic_density = 0.5 * B_mag**2
            im7 = axes[1,2].contourf(solver.X, solver.Y, kinetic_density + magnetic_density,
                                    levels=50, cmap='inferno')
            axes[1,2].set_title('Total Energy Density')
            axes[1,2].set_aspect('equal')
            plt.colorbar(im7, ax=axes[1,2], shrink=0.8)
            
            # Beta parameter (plasma beta)
            beta = 2 * solver.P / B_mag**2
            beta = np.clip(beta, 0.01, 100)  # Clip for visualization
            im8 = axes[1,3].contourf(solver.X, solver.Y, np.log10(beta), 
                                    levels=50, cmap='RdYlBu_r')
            axes[1,3].set_title('log₁₀(β) Plasma Beta')
            axes[1,3].set_aspect('equal')
            plt.colorbar(im8, ax=axes[1,3], shrink=0.8)
            
            # Add grid and labels
            for ax in axes.flat:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save frame
            filename = self.frames_path / f"mhd_frame_{solver.step:08d}.png"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.close(fig)
            
            if self.config.memory_efficient:
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error creating plot at step {solver.step}: {e}")
            plt.close('all')
    
    def plot_1d_profiles(self, solver: OptimizedOrszagTangMHD):
        """Create 1D profiles showing spatial variation"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'1D Profiles: t = {solver.current_time:.3f}', fontsize=16, color='white')
            
            # Take slices through domain center
            center_y = solver.config.ny // 2
            center_x = solver.config.nx // 2
            
            x_slice = solver.x
            y_slice = solver.y
            
            # Density vs position
            axes[0,0].plot(x_slice, solver.rho[center_y, :], 'cyan', linewidth=2, label='y = 0.5')
            axes[0,0].plot(y_slice, solver.rho[:, center_x], 'orange', linewidth=2, label='x = 0.5')
            axes[0,0].set_xlabel('Position')
            axes[0,0].set_ylabel('Density ρ')
            axes[0,0].set_title('Density vs Position')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()
            
            # Velocity magnitude vs position
            v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
            axes[0,1].plot(x_slice, v_mag[center_y, :], 'cyan', linewidth=2, label='y = 0.5')
            axes[0,1].plot(y_slice, v_mag[:, center_x], 'orange', linewidth=2, label='x = 0.5')
            axes[0,1].set_xlabel('Position')
            axes[0,1].set_ylabel('Velocity |V|')
            axes[0,1].set_title('Velocity vs Position')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
            
            # Magnetic field magnitude vs position
            B_mag = np.sqrt(solver.Bx**2 + solver.By**2 + solver.Bz**2)
            axes[0,2].plot(x_slice, B_mag[center_y, :], 'cyan', linewidth=2, label='y = 0.5')
            axes[0,2].plot(y_slice, B_mag[:, center_x], 'orange', linewidth=2, label='x = 0.5')
            axes[0,2].set_xlabel('Position')
            axes[0,2].set_ylabel('Magnetic Field |B|')
            axes[0,2].set_title('Magnetic Field vs Position')
            axes[0,2].grid(True, alpha=0.3)
            axes[0,2].legend()
            
            # Pressure vs position
            axes[1,0].plot(x_slice, solver.P[center_y, :], 'cyan', linewidth=2, label='y = 0.5')
            axes[1,0].plot(y_slice, solver.P[:, center_x], 'orange', linewidth=2, label='x = 0.5')
            axes[1,0].set_xlabel('Position')
            axes[1,0].set_ylabel('Pressure P')
            axes[1,0].set_title('Pressure vs Position')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
            
            # Current density vs position
            Jz = compute_curl_z(solver.Bx, solver.By, solver.config.dx, solver.config.dy)
            axes[1,1].plot(x_slice, Jz[center_y, :], 'cyan', linewidth=2, label='y = 0.5')
            axes[1,1].plot(y_slice, Jz[:, center_x], 'orange', linewidth=2, label='x = 0.5')
            axes[1,1].set_xlabel('Position')
            axes[1,1].set_ylabel('Current Density Jz')
            axes[1,1].set_title('Current Density vs Position')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
            
            # Energy density vs position
            kinetic_density = 0.5 * solver.rho * v_mag**2
            magnetic_density = 0.5 * B_mag**2
            total_energy_density = kinetic_density + magnetic_density
            
            axes[1,2].plot(x_slice, total_energy_density[center_y, :], 'cyan', linewidth=2, label='y = 0.5')
            axes[1,2].plot(y_slice, total_energy_density[:, center_x], 'orange', linewidth=2, label='x = 0.5')
            axes[1,2].set_xlabel('Position')
            axes[1,2].set_ylabel('Total Energy Density')
            axes[1,2].set_title('Energy Density vs Position')
            axes[1,2].grid(True, alpha=0.3)
            axes[1,2].legend()
            
            plt.tight_layout()
            
            # Save profile plot
            filename = self.frames_path / f"profiles_{solver.step:08d}.png"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating 1D profiles at step {solver.step}: {e}")
            plt.close('all')
    
    def plot_diagnostics(self, solver: OptimizedOrszagTangMHD):
        """Plot comprehensive diagnostic time series"""
        try:
            if len(solver.time_history) < 2:
                return
                
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('MHD Simulation Diagnostics', fontsize=16, color='white')
            
            times = np.array(solver.time_history)
            
            # Energy evolution
            axes[0,0].plot(times, solver.kinetic_energy_history, 'cyan', linewidth=2, label='Kinetic')
            axes[0,0].plot(times, solver.magnetic_energy_history, 'red', linewidth=2, label='Magnetic')
            axes[0,0].plot(times, solver.thermal_energy_history, 'green', linewidth=2, label='Thermal')
            axes[0,0].set_xlabel('Time')
            axes[0,0].set_ylabel('Energy Density')
            axes[0,0].set_title('Energy Evolution')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Total energy conservation
            axes[0,1].plot(times, solver.total_energy_history, 'yellow', linewidth=2)
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_ylabel('Total Energy')
            axes[0,1].set_title('Energy Conservation')
            axes[0,1].grid(True, alpha=0.3)
            
            # Current density growth
            axes[0,2].semilogy(times, solver.max_current_history, 'purple', linewidth=2)
            axes[0,2].set_xlabel('Time')
            axes[0,2].set_ylabel('Max |J|')
            axes[0,2].set_title('Current Density Evolution')
            axes[0,2].grid(True, alpha=0.3)
            
            # Vorticity evolution
            axes[1,0].semilogy(times, solver.max_vorticity_history, 'orange', linewidth=2)
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_ylabel('Max |ω|')
            axes[1,0].set_title('Vorticity Evolution')
            axes[1,0].grid(True, alpha=0.3)
            
            # Density evolution
            axes[1,1].plot(times, solver.min_density_history, 'lightblue', linewidth=2)
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('Min ρ')
            axes[1,1].set_title('Density Evolution')
            axes[1,1].grid(True, alpha=0.3)
            
            # Energy ratios
            total_energy = np.array(solver.total_energy_history)
            kinetic_ratio = np.array(solver.kinetic_energy_history) / total_energy
            magnetic_ratio = np.array(solver.magnetic_energy_history) / total_energy
            
            axes[1,2].plot(times, kinetic_ratio, 'cyan', linewidth=2, label='Kinetic/Total')
            axes[1,2].plot(times, magnetic_ratio, 'red', linewidth=2, label='Magnetic/Total')
            axes[1,2].set_xlabel('Time')
            axes[1,2].set_ylabel('Energy Fraction')
            axes[1,2].set_title('Energy Partitioning')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save diagnostics plot
            filename = self.output_path / "mhd_diagnostics.png"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting diagnostics: {e}")
            plt.close('all')
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=False)

def monitor_mhd_stability(solver: OptimizedOrszagTangMHD) -> bool:
    """Monitor simulation for numerical stability"""
    
    # Check for NaN or infinity
    fields = [solver.rho, solver.vx, solver.vy, solver.vz, 
              solver.Bx, solver.By, solver.Bz, solver.P]
    
    for field in fields:
        if not np.all(np.isfinite(field)):
            logger.error(f"Non-finite values detected at step {solver.step}")
            return False
    
    # Check for negative density or pressure
    if np.any(solver.rho <= 0) or np.any(solver.P <= 0):
        logger.error(f"Negative density or pressure at step {solver.step}")
        return False
    
    # Check velocity magnitude
    v_mag = np.sqrt(solver.vx**2 + solver.vy**2 + solver.vz**2)
    if np.max(v_mag) > 10.0:  # Reasonable upper bound
        logger.warning(f"High velocity {np.max(v_mag):.3f} at step {solver.step}")
    
    # Check magnetic divergence
    div_B = compute_divergence(solver.Bx, solver.By, solver.config.dx, solver.config.dy)
    max_div_B = np.max(np.abs(div_B))
    if max_div_B > 1e-3:  # Tolerance for div B = 0
        logger.warning(f"High magnetic divergence {max_div_B:.2e} at step {solver.step}")
    
    return True

def main():
    """Main execution function for optimized MHD simulation"""
    
    # Configuration for high-performance simulation
    config = OptimizedMHDConfig(
        nx=512,
        ny=512,
        T_final=5.0,
        dt_base=0.0001,
        cfl_safety=0.3,
        eta=0.0001,
        nu=0.0001,
        chi=0.0001,
        adaptive_dt=True,
        save_interval=100,
        plot_interval=50,
        memory_efficient=True,
        parallel_threads=psutil.cpu_count(),
        dpi=150
    )
    
    logger.warning("🚀 LAUNCHING OPTIMIZED ORSZAG-TANG MHD SIMULATION")
    logger.warning("=" * 60)
    logger.warning("Target: Professional MHD turbulence analysis")
    logger.warning("Features: High-performance computing, comprehensive diagnostics")
    logger.warning("=" * 60)
    
    # Initialize solver and visualizer
    solver = OptimizedOrszagTangMHD(config)
    visualizer = OptimizedMHDVisualizer(config)
    
    start_time = time.time()
    
    try:
        # Initial diagnostics and save
        diagnostics = solver.compute_diagnostics()
        visualizer.save_data_hdf5(solver)
        visualizer.create_comprehensive_plot(solver)
        visualizer.plot_1d_profiles(solver)
        
        logger.info("Initial conditions established. Starting time evolution...")
        
        # Main simulation loop
        with tqdm(total=config.T_final, desc="MHD Simulation", unit="time",
                  bar_format="{l_bar}{bar}| {n:.3f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            while solver.current_time < config.T_final:
                # Compute adaptive time step
                dt = solver.compute_timestep()
                
                # Time integration
                solver.rk4_step(dt)
                solver.current_time += dt
                solver.step += 1
                
                # Monitor stability
                if solver.step % 10 == 0:
                    if not monitor_mhd_stability(solver):
                        logger.error("Simulation became unstable. Terminating.")
                        break
                
                # Compute diagnostics
                diagnostics = solver.compute_diagnostics()
                
                # Periodic output
                if solver.step % config.save_interval == 0:
                    logger.info(f"Step {solver.step:6d}, t={solver.current_time:.4f}, dt={dt:.2e}")
                    logger.info(f"  Energies - K:{diagnostics['kinetic_energy']:.4f}, "
                              f"M:{diagnostics['magnetic_energy']:.4f}, "
                              f"T:{diagnostics['thermal_energy']:.4f}")
                    logger.info(f"  Max |J|:{diagnostics['max_current']:.3f}, "
                              f"Max |ω|:{diagnostics['max_vorticity']:.3f}")
                    
                    # Save data and create visualizations
                    visualizer.save_data_hdf5(solver)
                
                # Create plots
                if solver.step % config.plot_interval == 0:
                    visualizer.create_comprehensive_plot(solver)
                    visualizer.plot_1d_profiles(solver)
                
                # Update progress bar
                pbar.update(float(dt))
                
                # Memory management
                if config.memory_efficient and solver.step % 200 == 0:
                    gc.collect()
        
        # Final analysis
        logger.warning("\n🎯 SIMULATION COMPLETED SUCCESSFULLY")
        
        # Create final diagnostic plots
        visualizer.plot_diagnostics(solver)
        
        # Performance summary
        end_time = time.time()
        total_time = end_time - start_time
        steps_per_second = solver.step / total_time if total_time > 0 else 0
        
        logger.warning("\n📊 PERFORMANCE SUMMARY:")
        logger.warning(f"Total simulation time: {solver.current_time:.3f}")
        logger.warning(f"Total steps: {solver.step}")
        logger.warning(f"Wall clock time: {total_time/60:.2f} minutes")
        logger.warning(f"Performance: {steps_per_second:.2f} steps/second")
        logger.warning(f"Memory usage: {psutil.Process().memory_info().rss/(1024**3):.2f} GB")
        
        # Physics summary
        final_diagnostics = solver.compute_diagnostics()
        logger.warning("\n🔬 PHYSICS SUMMARY:")
        logger.warning(f"Final kinetic energy: {final_diagnostics['kinetic_energy']:.4f}")
        logger.warning(f"Final magnetic energy: {final_diagnostics['magnetic_energy']:.4f}")
        logger.warning(f"Maximum current density: {final_diagnostics['max_current']:.3f}")
        logger.warning(f"Energy conservation: {abs(solver.total_energy_history[-1]/solver.total_energy_history[0] - 1)*100:.2f}% deviation")
        
        logger.warning(f"\n💾 Results saved in: {visualizer.output_path}")
        
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        visualizer.cleanup()
        del solver, visualizer
        gc.collect()
        
        logger.warning("🏁 Simulation cleanup completed")

if __name__ == "__main__":
    main()