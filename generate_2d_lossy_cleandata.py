#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate CLEAN 2D FDTD data for LOSSY Medium (TM Mode).

Scenario:
- Grid: 100x100 (1mm spacing)
- Medium: Conductivity sigma = 4.0 S/m (Lossy)
- Process: 
  1. Warmup (Steps 1-200): Inject source to excite the field.
  2. Clean Phase (Steps 201-1000): Turn off source, record field evolution.
  3. Save: Export fields to .mat file.

Output: './data/FDTD_2D_lossy_clean_data.mat'
"""

import numpy as np
import scipy.io
import os
import time
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0
from scipy.constants import speed_of_light as c0

# ============================================================================
# 1. Configuration & Constants
# ============================================================================
floattype = np.float64

# Simulation Parameters
nmax = 1000          # Total time steps
dx = 1.0E-3          # Grid spacing x (m)
dy = 1.0E-3          # Grid spacing y (m)

nx, ny = 100, 100    # Grid size
Is, Js = nx // 2, ny // 2  # Source at center

# CFL stability condition
dt = 0.99 / (c0 * np.sqrt(1.0/dx**2 + 1.0/dy**2))

# Material Parameters (Lossy)
sigma_medium = 4.0   # Conductivity (S/m)
eps_r = 1.0          # Relative Permittivity

# Source Parameters
rtau = 50.0e-12        
tau = rtau / dt        
ndelay = 3 * tau       
srcconst = -dt * 3.0e11

# Data Generation Settings
SOURCE_END_STEP = 200  # Stop injecting source after this step
n_clean = nmax - SOURCE_END_STEP
output_dir = './data'

print("=" * 60)
print("2D FDTD Data Generator: LOSSY MEDIUM (TM Mode)")
print("=" * 60)
print(f"Grid: {nx}x{ny} | dx=dy={dx*1000}mm")
print(f"Sigma: {sigma_medium} S/m")
print(f"Warmup steps: 1 - {SOURCE_END_STEP}")
print(f"Clean Data steps: {SOURCE_END_STEP+1} - {nmax} (Total: {n_clean} frames)")
print("=" * 60)

# ============================================================================
# 2. Arrays Initialization
# ============================================================================
# Field Arrays (TM: Ez, Hx, Hy)
Ez = np.zeros((nx, ny), dtype=floattype)
Hx = np.zeros((nx, ny - 1), dtype=floattype)
Hy = np.zeros((nx - 1, ny), dtype=floattype)

# Storage for Clean Data (Time is the last dimension)
Ez_save = np.zeros((nx, ny, n_clean), dtype=floattype)
Hx_save = np.zeros((nx, ny - 1, n_clean), dtype=floattype)
Hy_save = np.zeros((nx - 1, ny, n_clean), dtype=floattype)

# Coefficient Arrays for Lossy Medium
CA_Ez = np.zeros((nx, ny), dtype=floattype)
CB_Ez = np.zeros((nx, ny), dtype=floattype)

# Magnetic update coefficients (Constant for non-magnetic)
DA = 1.0
DB = dt / m0

# Spatial Derivative Denominators (Optimization)
inv_dx = 1.0 / dx
inv_dy = 1.0 / dy

# ============================================================================
# 3. Setup Physics (Coefficients)
# ============================================================================
# Initialize uniform lossy medium
# Equation: epsilon * dE/dt + sigma * E = Curl(H)
# Discrete: E(n+1) = CA * E(n) + CB * Curl(H)
for i in range(nx):
    for j in range(ny):
        # Local material properties
        sig = sigma_medium
        eps = eps_r * e0
        
        # Semi-implicit FDTD coefficients for lossy medium
        denom = 1.0 + (sig * dt) / (2.0 * eps)
        CA_Ez[i, j] = (1.0 - (sig * dt) / (2.0 * eps)) / denom
        CB_Ez[i, j] = (dt / eps) / denom

# Pre-calculate Source Pulse
source_pulse = np.zeros(nmax + 1, dtype=floattype)
for n in range(1, nmax + 1):
    source_pulse[n] = (n - ndelay) * np.exp(-((n - ndelay)**2 / tau**2))

# ============================================================================
# 4. Main Simulation Loop
# ============================================================================
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Starting simulation loop...")
start_time = time.time()

for n in range(1, nmax + 1):
    
    # --- Update Hx ---
    # Hx(i, j) depends on Ez(i, j+1) - Ez(i, j)
    Hx[:, :] = DA * Hx[:, :] - DB * (Ez[:, 1:] - Ez[:, :-1]) * inv_dy
    
    # --- Update Hy ---
    # Hy(i, j) depends on Ez(i+1, j) - Ez(i, j)
    Hy[:, :] = DA * Hy[:, :] + DB * (Ez[1:, :] - Ez[:-1, :]) * inv_dx
    
    # --- Update Ez ---
    # Ez(i, j) depends on (Hy(i, j) - Hy(i-1, j)) - (Hx(i, j) - Hx(i, j-1))
    # We ignore the PEC boundaries (i=0, i=nx-1, j=0, j=ny-1) implicitly by loop range
    # Or using vectorized slicing for inner points:
    Ez[1:-1, 1:-1] = (CA_Ez[1:-1, 1:-1] * Ez[1:-1, 1:-1] + 
                      CB_Ez[1:-1, 1:-1] * (
                          (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) * inv_dx - 
                          (Hx[1:-1, 1:] - Hx[1:-1, :-1]) * inv_dy
                      ))
    
    # --- Apply Source (WARMUP ONLY) ---
    if n <= SOURCE_END_STEP:
        # Soft source injection
        Ez[Is, Js] += CB_Ez[Is, Js] * srcconst * source_pulse[n]
    
    # --- Save Clean Data ---
    if n > SOURCE_END_STEP:
        idx = n - SOURCE_END_STEP - 1
        Ez_save[:, :, idx] = Ez
        Hx_save[:, :, idx] = Hx
        Hy_save[:, :, idx] = Hy

    # Progress Log
    if n % 100 == 0:
        print(f"Step {n}/{nmax} | Max Ez: {np.max(np.abs(Ez)):.4e}")

# ============================================================================
# 5. Export Data
# ============================================================================
output_file = os.path.join(output_dir, 'FDTD_2D_lossy_clean_data.mat')

# Coordinates for reference
x_axis = np.arange(nx) * dx
y_axis = np.arange(ny) * dy
t_axis = np.arange(SOURCE_END_STEP + 1, nmax + 1) * dt

print(f"\nSaving data to {output_file}...")
scipy.io.savemat(output_file, {
    'Ez': Ez_save, 
    'Hx': Hx_save, 
    'Hy': Hy_save,
    'x': x_axis, 
    'y': y_axis, 
    't': t_axis,
    'sigma': sigma_medium,
    'dx': dx, 'dy': dy, 'dt': dt,
    'info': '2D TM Lossy Clean Data (Source Off Phase)'
})
print("Done!")
print(f"Total time: {time.time() - start_time:.2f}s")

# ============================================================================
# 6. Verification Plot
# ============================================================================
plt.figure(figsize=(10, 4))

# Plot 1: Snapshot of the field
plt.subplot(1, 2, 1)
clean_step_idx = 10 # Look at the 10th frame of clean data
plt.imshow(Ez_save[:, :, clean_step_idx].T, origin='lower', cmap='RdBu_r')
plt.title(f'Ez Field (Clean Step {clean_step_idx})')
plt.colorbar(label='V/m')
plt.xlabel('nx')
plt.ylabel('ny')

# Plot 2: Time decay at the source point
plt.subplot(1, 2, 2)
center_decay = Ez_save[Is, Js, :]
plt.plot(t_axis * 1e12, center_decay)
plt.title('Ez Decay at Center (Source OFF)')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plot_file = os.path.join(output_dir, 'check_2d_decay.png')
plt.savefig(plot_file)
print(f"Verification plot saved to {plot_file}")
plt.show()