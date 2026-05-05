#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate CLEAN 3D FDTD data for LOSSY Medium.

Scenario:
- Grid: 20x20x20 (2mm spacing)
- Medium: Conductivity sigma = 4.0 S/m (Lossy)
- Process: 
  1. Warmup (Steps 1-83): Inject source.
  2. Clean Phase (Steps 84-500): Turn off source, record field evolution.
  3. Save: Export fields to .mat file for Deep Learning training.

Output: 'FDTD_3D_lossy_clean_data.mat'
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
# Define high precision for data generation
floattype = np.float64
complextype = np.complex128

# Simulation Parameters
nmax = 500           
dx = 2e-3            
dy = 2e-3            
dz = 2e-3            

nx, ny, nz = 20, 20, 20
Is, Js, Ks = 9, 9, 9  # Python 0-indexed (10-1)

# Time step (CFL)
dt = 0.99 / (c0 * np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)))

# Material Parameters (Lossy)
sigma_medium = 4.0   # S/m
eps_r = 1.0          
mr = 1.0             
sm = 0.0             

# Source Parameters
rtau = 50.0e-12        
tau = rtau / dt        
ndelay = 3 * tau       
srcconst = -dt * 3.0e11

# Data Generation Settings
SOURCE_END_STEP = 83  # Cutoff step for source
n_clean = nmax - SOURCE_END_STEP
output_dir = './data'

print("=" * 60)
print("3D FDTD Data Generator: LOSSY MEDIUM")
print("=" * 60)
print(f"Sigma: {sigma_medium} S/m")
print(f"Warmup steps: 1 - {SOURCE_END_STEP}")
print(f"Clean Data steps: {SOURCE_END_STEP+1} - {nmax} (Total: {n_clean} frames)")
print("=" * 60)

# ============================================================================
# 2. Arrays Initialization
# ============================================================================
Ex = np.zeros((nx, ny + 1, nz + 1), dtype=floattype)
Ey = np.zeros((nx + 1, ny, nz + 1), dtype=floattype)
Ez = np.zeros((nx + 1, ny + 1, nz), dtype=floattype)
Hx = np.zeros((nx + 1, ny, nz), dtype=floattype)
Hy = np.zeros((nx, ny + 1, nz), dtype=floattype)
Hz = np.zeros((nx, ny, nz + 1), dtype=floattype)

# Storage for Clean Data
Ex_save = np.zeros((nx, ny + 1, nz + 1, n_clean), dtype=floattype)
Ey_save = np.zeros((nx + 1, ny, nz + 1, n_clean), dtype=floattype)
Ez_save = np.zeros((nx + 1, ny + 1, nz, n_clean), dtype=floattype)
Hx_save = np.zeros((nx + 1, ny, nz, n_clean), dtype=floattype)
Hy_save = np.zeros((nx, ny + 1, nz, n_clean), dtype=floattype)
Hz_save = np.zeros((nx, ny, nz + 1, n_clean), dtype=floattype)

# Material ID Arrays
# 1 = Lossy Medium, 2 = PEC (if needed)
ID = np.zeros((6, nx + 1, ny + 1, nz + 1), dtype=np.int32)
updatecoeffsH = np.zeros((6, 6), dtype=floattype)
updatecoeffsE = np.zeros((6, 6), dtype=floattype)

# ============================================================================
# 3. Physics Kernel (Coefficients)
# ============================================================================
def update_H_coeff(IDnum, dx, dy, dz, dt, sm):
    if IDnum == 1:
        HA = (m0 * mr / dt) + 0.5 * sm
        HB = (m0 * mr / dt) - 0.5 * sm
        DA = HB / HA
        DBx = (1 / dx) * 1 / HA
        DBy = (1 / dy) * 1 / HA
        DBz = (1 / dz) * 1 / HA
        updatecoeffsH[IDnum, 0] = DA
        updatecoeffsH[IDnum, 1] = DBx
        updatecoeffsH[IDnum, 2] = DBy
        updatecoeffsH[IDnum, 3] = DBz

def update_E_coeff(IDnum, dx, dy, dz, dt, se):
    if IDnum == 1:
        EA = (e0 * eps_r / dt) + 0.5 * se
        EB = (e0 * eps_r / dt) - 0.5 * se
        CA = EB / EA
        CBx = (1 / dx) * 1 / EA
        CBy = (1 / dy) * 1 / EA
        CBz = (1 / dz) * 1 / EA
        srce = 1 / EA
        updatecoeffsE[IDnum, 0] = CA
        updatecoeffsE[IDnum, 1] = CBx
        updatecoeffsE[IDnum, 2] = CBy
        updatecoeffsE[IDnum, 3] = CBz
        updatecoeffsE[IDnum, 4] = srce
    elif IDnum == 2: # PEC
        updatecoeffsE[IDnum, :] = 0.0

# Initialize Materials (Set All to ID=1 Lossy)
ID[:] = 1 

# Pre-calculate coefficients
update_E_coeff(1, dx, dy, dz, dt, sigma_medium)
update_H_coeff(1, dx, dy, dz, dt, sm)

# Source pulse
source = np.zeros(nmax + 1, dtype=floattype)
for n in range(1, nmax + 1):
    source[n] = srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / tau**2))

# ============================================================================
# 4. Main Simulation Loop
# ============================================================================
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Starting simulation loop...")
start_time = time.time()

for n in range(1, nmax + 1):
    
    # --- Update Electric Fields ---
    # Optimized loops using numpy slicing where possible could be faster, 
    # but we stick to the provided explicit loops for correctness assurance.
    
    # Ex
    for i in range(0, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                mat = ID[0, i, j, k]
                Ex[i, j, k] = (updatecoeffsE[mat, 0] * Ex[i, j, k] + 
                              updatecoeffsE[mat, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - 
                              updatecoeffsE[mat, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))
    
    # Ey
    for i in range(1, nx):
        for j in range(0, ny):
            for k in range(1, nz):
                mat = ID[1, i, j, k]
                Ey[i, j, k] = (updatecoeffsE[mat, 0] * Ey[i, j, k] + 
                              updatecoeffsE[mat, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - 
                              updatecoeffsE[mat, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))
    
    # Ez
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(0, nz):
                mat = ID[2, i, j, k]
                Ez[i, j, k] = (updatecoeffsE[mat, 0] * Ez[i, j, k] + 
                              updatecoeffsE[mat, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - 
                              updatecoeffsE[mat, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))

    # --- Apply Source (WARMUP ONLY) ---
    if n <= SOURCE_END_STEP:
        Ez[Is, Js, Ks] += source[n]
    
    # --- Update Magnetic Fields ---
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                mHx, mHy, mHz = ID[3, i, j, k], ID[4, i, j, k], ID[5, i, j, k]
                
                Hx[i, j, k] = (updatecoeffsH[mHx, 0] * Hx[i, j, k] - 
                              updatecoeffsH[mHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + 
                              updatecoeffsH[mHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k]))
                
                Hy[i, j, k] = (updatecoeffsH[mHy, 0] * Hy[i, j, k] - 
                              updatecoeffsH[mHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + 
                              updatecoeffsH[mHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k]))
                
                Hz[i, j, k] = (updatecoeffsH[mHz, 0] * Hz[i, j, k] - 
                              updatecoeffsH[mHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + 
                              updatecoeffsH[mHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k]))

    # --- Save Clean Data ---
    if n > SOURCE_END_STEP:
        idx = n - SOURCE_END_STEP - 1
        Ex_save[:, :, :, idx] = Ex
        Ey_save[:, :, :, idx] = Ey
        Ez_save[:, :, :, idx] = Ez
        Hx_save[:, :, :, idx] = Hx
        Hy_save[:, :, :, idx] = Hy
        Hz_save[:, :, :, idx] = Hz

    if n % 50 == 0:
        print(f"Step {n}/{nmax} complete. Ez_center={Ez[Is,Js,Ks]:.2e}")

# ============================================================================
# 5. Export Data
# ============================================================================
output_file = os.path.join(output_dir, 'FDTD_3D_lossy_clean_data.mat')

# Grid coordinates for reference
x = np.linspace(0, nx * dx, nx + 1)
y = np.linspace(0, ny * dy, ny + 1)
z = np.linspace(0, nz * dz, nz + 1)
t = np.arange(SOURCE_END_STEP + 1, nmax + 1) * dt

print(f"\nSaving data to {output_file}...")
scipy.io.savemat(output_file, {
    'Ex': Ex_save, 'Ey': Ey_save, 'Ez': Ez_save,
    'Hx': Hx_save, 'Hy': Hy_save, 'Hz': Hz_save,
    'x': x, 'y': y, 'z': z, 't': t,
    'sigma': sigma_medium,
    'dx': dx, 'dy': dy, 'dz': dz, 'dt': dt,
    'info': 'Clean data (source off) for Lossy Medium'
})
print("Done!")
print(f"Total time: {time.time() - start_time:.2f}s")

# Simple check plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(Ez_save[:,:,nz//2, 10].T, cmap='RdBu_r', origin='lower')
plt.title(f'Ez (z-slice) at Clean Step 10')
plt.subplot(1,2,2)
# Plot decay at center point to verify lossy nature
center_decay = Ez_save[Is, Js, Ks, :]
plt.plot(center_decay)
plt.title('Ez at center (Time Decay)')
plt.xlabel('Clean Steps')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'check_lossy_decay.png'))
print("Verification plot saved.")