#!/usr/bin/env python3
"""
Generate clean 3D FDTD data WITHOUT source excitation period.

Strategy:
1. Run FDTD with source for first 83 steps (warmup period)
2. Save fields at step 83 as initial conditions
3. Continue simulation from step 84 to nmax WITHOUT source
4. Save only the clean data (steps 84-500) to .mat file
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

# Define floattype
floattype = np.float64

from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0

c0 = 2.99792458e8

# Grid parameters
nx = 20
ny = 20
nz = 20
dx = 2e-3
dy = 2e-3
dz = 2e-3

# Source location
Is = 10 - 1  # Index 9
Js = 10 - 1
Ks = 10 - 1

# Time stepping
nmax = 500
dt = dx / (2.0 * c0)

# Source parameters
rtau = 50.0e-12
tau = rtau / dt
ndelay = 3 * tau
srcconst = -dt * (3.0e11)
SOURCE_END_STEP = 83  # From analysis

print("="*70)
print("3D FDTD DATA GENERATION (WITHOUT SOURCE PERIOD)")
print("="*70)
print(f"Grid: {nx} x {ny} x {nz}")
print(f"Cell size: dx={dx*1e3:.1f}mm, dy={dy*1e3:.1f}mm, dz={dz*1e3:.1f}mm")
print(f"Time step dt: {dt:.6e} s")
print(f"Total steps: {nmax}")
print(f"Source excitation: steps 1-{SOURCE_END_STEP}")
print(f"Clean data saved: steps {SOURCE_END_STEP+1}-{nmax}")
print("="*70)

# Initialize fields (Yee grid)
Ex = np.zeros((nx, ny + 1, nz + 1), dtype=floattype)
Ey = np.zeros((nx + 1, ny, nz + 1), dtype=floattype)
Ez = np.zeros((nx + 1, ny + 1, nz), dtype=floattype)
Hx = np.zeros((nx + 1, ny, nz), dtype=floattype)
Hy = np.zeros((nx, ny + 1, nz), dtype=floattype)
Hz = np.zeros((nx, ny, nz + 1), dtype=floattype)

# Update coefficients for free space
er = 1.0
se = 0.0
mr = 1.0
sm = 0.0

# E-field coefficients
EA = (e0 * er / dt) + 0.5 * se
EB = (e0 * er / dt) - 0.5 * se
CA = EB / EA
CBx = (1 / dx) / EA
CBy = (1 / dy) / EA
CBz = (1 / dz) / EA

# H-field coefficients
HA = (m0 * mr / dt) + 0.5 * sm
HB = (m0 * mr / dt) - 0.5 * sm
DA = HB / HA
DBx = (1 / dx) / HA
DBy = (1 / dy) / HA
DBz = (1 / dz) / HA

# Source signal
source = np.zeros(nmax + 1, dtype=floattype)
for n in range(1, nmax + 1):
    source[n] = srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / (tau**2)))

# Storage for clean data (from SOURCE_END_STEP+1 to nmax)
n_clean = nmax - SOURCE_END_STEP
Ex_save = np.zeros((nx, ny + 1, nz + 1, n_clean), dtype=floattype)
Ey_save = np.zeros((nx + 1, ny, nz + 1, n_clean), dtype=floattype)
Ez_save = np.zeros((nx + 1, ny + 1, nz, n_clean), dtype=floattype)
Hx_save = np.zeros((nx + 1, ny, nz, n_clean), dtype=floattype)
Hy_save = np.zeros((nx, ny + 1, nz, n_clean), dtype=floattype)
Hz_save = np.zeros((nx, ny, nz + 1, n_clean), dtype=floattype)

print("\nStarting FDTD simulation...")
print(f"Phase 1: Warmup with source (steps 1-{SOURCE_END_STEP})")


for n in range(1, nmax + 1):
    
    # ========== Update E fields ==========
    
    # Ex (interior only, PEC boundaries set to 0)
    for i in range(0, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                Ex[i, j, k] = CA * Ex[i, j, k] + CBy * (Hz[i, j, k] - Hz[i, j - 1, k]) - CBz * (Hy[i, j, k] - Hy[i, j, k - 1])
    
    # Ey (interior only)
    for i in range(1, nx):
        for j in range(0, ny):
            for k in range(1, nz):
                Ey[i, j, k] = CA * Ey[i, j, k] + CBz * (Hx[i, j, k] - Hx[i, j, k - 1]) - CBx * (Hz[i, j, k] - Hz[i - 1, j, k])
    
    # Ez (interior only)
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(0, nz):
                Ez[i, j, k] = CA * Ez[i, j, k] + CBx * (Hy[i, j, k] - Hy[i - 1, j, k]) - CBy * (Hx[i, j, k] - Hx[i, j - 1, k])
    
    # Add source ONLY during warmup period
    if n <= SOURCE_END_STEP:
        Ez[Is, Js, Ks] = Ez[Is, Js, Ks] + source[n]
    
    # ========== Update H fields ==========
    
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                Hx[i, j, k] = DA * Hx[i, j, k] - DBy * (Ez[i, j + 1, k] - Ez[i, j, k]) + DBz * (Ey[i, j, k + 1] - Ey[i, j, k])
                Hy[i, j, k] = DA * Hy[i, j, k] - DBz * (Ex[i, j, k + 1] - Ex[i, j, k]) + DBx * (Ez[i + 1, j, k] - Ez[i, j, k])
                Hz[i, j, k] = DA * Hz[i, j, k] - DBx * (Ey[i + 1, j, k] - Ey[i, j, k]) + DBy * (Ex[i, j + 1, k] - Ex[i, j, k])
    
    # ========== Save data ==========
    
    if n == SOURCE_END_STEP:
        print(f"\nPhase 1 complete. Fields at step {SOURCE_END_STEP} saved as initial condition.")
        print(f"Phase 2: Clean simulation without source (steps {SOURCE_END_STEP+1}-{nmax})")
        print(f"  Ez at source point [{Is},{Js},{Ks}]: {Ez[Is, Js, Ks]:.6e}")
    
    # Save only clean data (after source period)
    if n > SOURCE_END_STEP:
        save_idx = n - SOURCE_END_STEP - 1  # 0-indexed
        Ex_save[:, :, :, save_idx] = Ex
        Ey_save[:, :, :, save_idx] = Ey
        Ez_save[:, :, :, save_idx] = Ez
        Hx_save[:, :, :, save_idx] = Hx
        Hy_save[:, :, :, save_idx] = Hy
        Hz_save[:, :, :, save_idx] = Hz
    
    # Progress reporting
    if n % 50 == 0:
        print(f"  Step {n}/{nmax} - Ez[{Is},{Js},{Ks}] = {Ez[Is, Js, Ks]:.6e}")

print("\nSimulation complete!")

# Save data to .mat file

output_dir = '/Users/zyanxin/Documents/code/GEM_N/data'
output_file = os.path.join(output_dir, 'FDTD_3D_cavity_clean_data.mat')

# Create coordinate arrays
x = np.linspace(0, nx * dx, nx + 1)
y = np.linspace(0, ny * dy, ny + 1)
z = np.linspace(0, nz * dz, nz + 1)
t = np.arange(SOURCE_END_STEP + 1, nmax + 1) * dt

# Create meshgrid for visualization
xx_3d, yy_3d, zz_3d = np.meshgrid(x[:-1], y[:-1], z[:-1], indexing='ij')

print(f"\nSaving data to: {output_file}")
print(f"  Ex shape: {Ex_save.shape}")
print(f"  Ey shape: {Ey_save.shape}")
print(f"  Ez shape: {Ez_save.shape}")
print(f"  Hx shape: {Hx_save.shape}")
print(f"  Hy shape: {Hy_save.shape}")
print(f"  Hz shape: {Hz_save.shape}")
print(f"  Time array: {len(t)} steps from t={t[0]:.6e} to t={t[-1]:.6e} s")

scipy.io.savemat(output_file, {
    'Ex': Ex_save,
    'Ey': Ey_save,
    'Ez': Ez_save,
    'Hx': Hx_save,
    'Hy': Hy_save,
    'Hz': Hz_save,
    'x': x,
    'y': y,
    'z': z,
    't': t,
    'xx_3d': xx_3d,
    'yy_3d': yy_3d,
    'zz_3d': zz_3d,
    'dx': dx,
    'dy': dy,
    'dz': dz,
    'dt': dt,
    'nx': nx,
    'ny': ny,
    'nz': nz,
    'source_end_step': SOURCE_END_STEP,
    'description': 'Clean 3D FDTD data without source excitation period. Fields are from step 84 to 500.'
})

print(f"\n{'='*70}")
print("SUCCESS! Clean 3D FDTD data saved.")
print(f"{'='*70}")
print(f"Output file: {output_file}")
print(f"Data contains {n_clean} time steps (steps {SOURCE_END_STEP+1}-{nmax})")
print(f"Grid: {nx} x {ny} x {nz}")
print(f"Boundary conditions: PEC (E=0 at boundaries)")
print(f"\nThis data can be used for:")
print(f"  - Training GNN/ML models")
print(f"  - Testing FDTD implementations")
print(f"  - Initial condition for further simulations")
print("="*70)

# Plot a sample field at middle time step
mid_idx = n_clean // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Ez at z=nz//2
im0 = axes[0].imshow(Ez_save[:, :, nz//2, mid_idx].T, origin='lower', cmap='RdBu_r', aspect='equal')
axes[0].set_title(f'Ez at z={nz//2}, step {SOURCE_END_STEP + mid_idx}')
axes[0].set_xlabel('x index')
axes[0].set_ylabel('y index')
plt.colorbar(im0, ax=axes[0])

# Hx at z=nz//2  
im1 = axes[1].imshow(Hx_save[:, :, nz//2, mid_idx].T, origin='lower', cmap='RdBu_r', aspect='equal')
axes[1].set_title(f'Hx at z={nz//2}, step {SOURCE_END_STEP + mid_idx}')
axes[1].set_xlabel('x index')
axes[1].set_ylabel('y index')
plt.colorbar(im1, ax=axes[1])

# Hy at z=nz//2
im2 = axes[2].imshow(Hy_save[:, :, nz//2, mid_idx].T, origin='lower', cmap='RdBu_r', aspect='equal')
axes[2].set_title(f'Hy at z={nz//2}, step {SOURCE_END_STEP + mid_idx}')
axes[2].set_xlabel('x index')
axes[2].set_ylabel('y index')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'FDTD_3D_sample_fields.png'), dpi=150, bbox_inches='tight')
print(f"\nSample field plot saved to: {os.path.join(output_dir, 'FDTD_3D_sample_fields.png')}")
plt.show()
