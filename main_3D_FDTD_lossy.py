# -*- coding: utf-8 -*-
"""
3D FDTD Simulation with Lossy Medium (Conductivity)

This script simulates electromagnetic wave propagation in a 3D domain
using the Finite-Difference Time-Domain (FDTD) method with a lossy medium
characterized by electric conductivity sigma.

Based on main_good_3D_cavity_PEC_boudary_to_test.py

@author: yangzf (modified)
"""

from constants_CPU import floattype, complextype
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0
from scipy.constants import speed_of_light as c0

import time

# ============================================================================
# Simulation Parameters
# ============================================================================
nmax = 500           # Maximum number of time steps
dx = 2e-3            # Grid spacing in x-direction (m)
dy = 2e-3            # Grid spacing in y-direction (m)
dz = 2e-3            # Grid spacing in z-direction (m)

# Grid dimensions
nx = 20
ny = 20
nz = 20

# Source point at the center
Is = 10 - 1
Js = 10 - 1
Ks = 10 - 1

# CFL stability condition for time step
dt = 0.99 / (c0 * np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)))
# Alternative: dt = dx / (2.0 * c0)

# ============================================================================
# Material Parameters - Lossy Medium
# ============================================================================
# Conductivity (S/m):
#   - Perfect insulator: sigma = 0
#   - Lossy dielectric:  sigma = 0.001 - 0.1 S/m
#   - Semiconductor:     sigma = 0.1 - 100 S/m  
#   - Sea water:         sigma ~ 4 S/m
#   - Good conductor:    sigma > 1e6 S/m (metals)
#
# Here we use a medium value representing a lossy dielectric/semiconductor

sigma_medium = 1.0   # Conductivity in S/m (like sea water)
eps_r = 1.0          # Relative permittivity
mr = 1.0             # Relative permeability
sm = 0.0             # Magnetic conductivity (usually 0)

print("=" * 60)
print("3D FDTD Simulation with Lossy Medium")
print("=" * 60)
print(f"Grid size: {nx} x {ny} x {nz}")
print(f"Grid spacing: dx = dy = dz = {dx*1000:.2f} mm")
print(f"Time step: dt = {dt*1e12:.4f} ps")
print(f"Conductivity: sigma = {sigma_medium} S/m")
print(f"Relative permittivity: eps_r = {eps_r}")
print(f"Skin depth at 1 GHz: {np.sqrt(2/(2*np.pi*1e9*m0*sigma_medium))*1000:.2f} mm")
print("=" * 60)

# ============================================================================
# Field Arrays
# ============================================================================
Ex = np.zeros((nx, ny + 1, nz + 1), dtype=floattype)
Ey = np.zeros((nx + 1, ny, nz + 1), dtype=floattype)
Ez = np.zeros((nx + 1, ny + 1, nz), dtype=floattype)
Hx = np.zeros((nx + 1, ny, nz), dtype=floattype)
Hy = np.zeros((nx, ny + 1, nz), dtype=floattype)
Hz = np.zeros((nx, ny, nz + 1), dtype=floattype)

# ============================================================================
# Update Coefficient Arrays
# ============================================================================
# Material ID: 1 = lossy medium, 2 = PEC
updatecoeffsH = np.zeros((6, 6), dtype=floattype)
updatecoeffsE = np.zeros((6, 6), dtype=floattype)

# Material ID array for each field component
# ID[0] -> Ex, ID[1] -> Ey, ID[2] -> Ez
# ID[3] -> Hx, ID[4] -> Hy, ID[5] -> Hz
ID = np.zeros((6, nx + 1, ny + 1, nz + 1), dtype=np.int32)

# ============================================================================
# Update Coefficient Functions
# ============================================================================
def update_H_coeff(IDnum, dx, dy, dz, dt, sm):
    """Calculate magnetic field update coefficients for material IDnum"""
    if IDnum == 1:  # Lossy medium (or free space if sm=0)
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
    """Calculate electric field update coefficients for material IDnum"""
    if IDnum == 1:  # Lossy medium
        # For lossy medium with conductivity se (sigma)
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
    elif IDnum == 2:  # PEC (Perfect Electric Conductor)
        updatecoeffsE[IDnum, 0] = 0
        updatecoeffsE[IDnum, 1] = 0
        updatecoeffsE[IDnum, 2] = 0
        updatecoeffsE[IDnum, 3] = 0
        updatecoeffsE[IDnum, 4] = 0

# ============================================================================
# Initialize Material Properties
# ============================================================================
print("\nInitializing material properties...")

# Set all points to lossy medium (ID = 1)
for i in range(0, nx + 1):
    for j in range(0, ny + 1):
        for k in range(0, nz + 1):
            # Electric field components - lossy medium
            ID[0, i, j, k] = 1  # Ex
            ID[1, i, j, k] = 1  # Ey
            ID[2, i, j, k] = 1  # Ez
            # Magnetic field components
            ID[3, i, j, k] = 1  # Hx
            ID[4, i, j, k] = 1  # Hy
            ID[5, i, j, k] = 1  # Hz

# Calculate coefficients for lossy medium (ID = 1) with conductivity
update_E_coeff(1, dx, dy, dz, dt, sigma_medium)  # sigma_medium is the conductivity
update_E_coeff(2, dx, dy, dz, dt, 0)  # PEC coefficients (if needed)
update_H_coeff(1, dx, dy, dz, dt, sm)  # Magnetic coefficients

print(f"E-field update coefficients (CA, CBx, CBy, CBz):")
print(f"  CA  = {updatecoeffsE[1, 0]:.6f}")
print(f"  CBx = {updatecoeffsE[1, 1]:.6e}")
print(f"  CBy = {updatecoeffsE[1, 2]:.6e}")
print(f"  CBz = {updatecoeffsE[1, 3]:.6e}")

# ============================================================================
# Source Parameters (Gaussian derivative pulse)
# ============================================================================
rtau = 50.0e-12        # Pulse width parameter (s)
tau = rtau / dt        # Pulse width in time steps
ndelay = 3 * tau       # Delay to center the pulse
srcconst = -dt * 3.0e11  # Source amplitude constant

source = np.zeros(nmax + 1, dtype=floattype)
for n in range(1, nmax + 1):
    source[n] = srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / tau**2))

# ============================================================================
# Recording Array
# ============================================================================
RecordEz = np.zeros(nmax + 1, dtype=floattype)

# Observation point (slightly off-center to see propagation)
obs_i, obs_j, obs_k = 11, 11, 9

# ============================================================================
# Setup Visualization
# ============================================================================
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# ============================================================================
# Main FDTD Time-Stepping Loop
# ============================================================================
print("\nStarting 3D FDTD simulation...")
print(f"Running {nmax} time steps\n")

for n in range(1, nmax + 1):
    t_start = time.time()
    
    # ========================================================================
    # Update Electric Fields (Ex, Ey, Ez)
    # ========================================================================
    
    # Ex update - interior points
    for i in range(0, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                materialEx = ID[0, i, j, k]
                Ex[i, j, k] = (updatecoeffsE[materialEx, 0] * Ex[i, j, k] + 
                              updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - 
                              updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))
    
    # Ey update - interior points
    for i in range(1, nx):
        for j in range(0, ny):
            for k in range(1, nz):
                materialEy = ID[1, i, j, k]
                Ey[i, j, k] = (updatecoeffsE[materialEy, 0] * Ey[i, j, k] + 
                              updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - 
                              updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))
    
    # Ez update - interior points
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(0, nz):
                materialEz = ID[2, i, j, k]
                Ez[i, j, k] = (updatecoeffsE[materialEz, 0] * Ez[i, j, k] + 
                              updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - 
                              updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))
    
    # ========================================================================
    # Apply Source (soft source on Ez)
    # ========================================================================
    Ez[Is, Js, Ks] = Ez[Is, Js, Ks] + source[n]
    
    # ========================================================================
    # Update Magnetic Fields (Hx, Hy, Hz)
    # ========================================================================
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                materialHx = ID[3, i, j, k]
                materialHy = ID[4, i, j, k]
                materialHz = ID[5, i, j, k]
                
                Hx[i, j, k] = (updatecoeffsH[materialHx, 0] * Hx[i, j, k] - 
                              updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + 
                              updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k]))
                
                Hy[i, j, k] = (updatecoeffsH[materialHy, 0] * Hy[i, j, k] - 
                              updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + 
                              updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k]))
                
                Hz[i, j, k] = (updatecoeffsH[materialHz, 0] * Hz[i, j, k] - 
                              updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + 
                              updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k]))
    
    elapsed = time.time() - t_start
    
    # Record field at observation point
    RecordEz[n] = Ez[obs_i, obs_j, obs_k]
    
    # Print progress
    if n % 50 == 0 or n == 1:
        print(f"Time step: {n:4d}/{nmax}, Ez_max: {np.max(np.abs(Ez)):.6e}, "
              f"Ez[{obs_i},{obs_j},{obs_k}]: {Ez[obs_i, obs_j, obs_k]:.6e}, "
              f"Time/step: {elapsed*1000:.1f} ms")
    
    # ========================================================================
    # Visualization (update every 5 time steps)
    # ========================================================================
    if n == 1:
        # Initial plot setup
        ax.clear()
        vmax = max(np.max(np.abs(Ez[:, :, Ks])), 1e-10)
        im = ax.imshow(Ez[:, :, Ks].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, nx*dx*1000, 0, ny*dy*1000])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'3D FDTD - Lossy Medium (σ = {sigma_medium} S/m)\n'
                     f'Ez at z = {Ks*dz*1000:.1f} mm, Time Step: {n}')
        cbar = plt.colorbar(im, ax=ax, label='Ez (V/m)')
        plt.draw()
        plt.pause(0.1)
    
    if n % 5 == 0:
        ax.clear()
        
        # Plot Ez field at z = Ks (middle slice)
        vmax = max(np.max(np.abs(Ez[:, :, Ks])), 1e-10)
        im = ax.imshow(Ez[:, :, Ks].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, nx*dx*1000, 0, ny*dy*1000])
        
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'3D FDTD - Lossy Medium (σ = {sigma_medium} S/m)\n'
                     f'Ez at z = {Ks*dz*1000:.1f} mm, Time Step: {n}, t = {n*dt*1e12:.2f} ps')
        
        plt.draw()
        plt.pause(0.05)

# ============================================================================
# Post-Processing
# ============================================================================
print("\n" + "=" * 60)
print("Simulation completed!")
print(f"Maximum Ez amplitude: {np.max(np.abs(Ez)):.6e} V/m")
print("=" * 60)

# Plot time history of Ez at observation point
plt.ioff()
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time axis in picoseconds
time_axis = np.arange(nmax + 1) * dt * 1e12

ax1.plot(time_axis, source, 'b-', linewidth=1)
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Source Amplitude')
ax1.set_title('Excitation Source (Gaussian Derivative)')
ax1.grid(True)
ax1.set_xlim([0, time_axis[-1]])

ax2.plot(time_axis, RecordEz, 'r-', linewidth=1)
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Ez (V/m)')
ax2.set_title(f'Ez at Observation Point ({obs_i}, {obs_j}, {obs_k}) - Lossy Medium σ = {sigma_medium} S/m')
ax2.grid(True)
ax2.set_xlim([0, time_axis[-1]])

plt.tight_layout()
plt.show()

# Optional: Plot multiple z-slices at final time step
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
z_slices = [nz//4, nz//2, 3*nz//4, nz-1]
vmax_global = np.max(np.abs(Ez))

for idx, (ax, z_idx) in enumerate(zip(axes.flat, z_slices)):
    im = ax.imshow(Ez[:, :, z_idx].T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax_global, vmax=vmax_global,
                   extent=[0, nx*dx*1000, 0, ny*dy*1000])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Ez at z = {z_idx*dz*1000:.1f} mm')
    plt.colorbar(im, ax=ax, label='Ez (V/m)')

fig3.suptitle(f'3D FDTD Final State - Lossy Medium (σ = {sigma_medium} S/m)', fontsize=14)
plt.tight_layout()
plt.show()

