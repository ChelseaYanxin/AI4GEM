# -*- coding: utf-8 -*-
"""
2D FDTD TM Mode Simulation with Lossy Medium (Conductivity)

This script simulates electromagnetic wave propagation in a 2D domain
using the Finite-Difference Time-Domain (FDTD) method for TM polarization
(Ez, Hx, Hy components) with a lossy medium characterized by conductivity sigma.

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
nmax = 2000          # Maximum number of time steps
dx = 1.0E-3          # Grid spacing in x-direction (m)
dy = 1.0E-3          # Grid spacing in y-direction (m)

# CFL stability condition for time step
dt = 0.99 / (c0 * (1.0/dx**2 + 1.0/dy**2)**0.5)

# Grid dimensions
nx = 100
ny = 100

# Source point at the center
Is = round(nx/2 - 1)
Js = round(ny/2 - 1)

# ============================================================================
# Material Parameters - Lossy Medium
# ============================================================================
# Conductivity (S/m):
#   - Perfect insulator: sigma = 0
#   - Lossy dielectric:  sigma = 0.001 - 0.1 S/m
#   - Semiconductor:     sigma = 0.1 - 100 S/m
#   - Good conductor:    sigma > 1e6 S/m (metals)
#
# Here we use a medium value representing a lossy dielectric/semiconductor

sigma_medium = 4  # Conductivity in S/m (medium value between insulator and metal)
eps_r = 1.0          # Relative permittivity

print("=" * 60)
print("2D FDTD TM Mode Simulation with Lossy Medium")
print("=" * 60)
print(f"Grid size: {nx} x {ny}")
print(f"Grid spacing: dx = dy = {dx*1000:.2f} mm")
print(f"Time step: dt = {dt*1e12:.4f} ps")
print(f"Conductivity: sigma = {sigma_medium} S/m")
print(f"Relative permittivity: eps_r = {eps_r}")
print(f"Skin depth at 1 GHz: {np.sqrt(2/(2*np.pi*1e9*m0*sigma_medium))*1000:.2f} mm")
print("=" * 60)

# ============================================================================
# Field Arrays (TM mode: Ez, Hx, Hy)
# ============================================================================
Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)
Hx = np.zeros((nx + 1, ny), dtype=floattype)
Hy = np.zeros((nx, ny + 1), dtype=floattype)

# ============================================================================
# Coefficient Arrays for Ez Update
# ============================================================================
CA_Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)
CB_Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)

# Conductivity and permittivity arrays
sig_z = np.zeros((nx + 1, ny + 1), dtype=floattype)
eps = np.zeros((nx + 1, ny + 1), dtype=floattype)

# Magnetic field update coefficients (constant for non-magnetic media)
DA = 1.0
DB = dt / m0

# ============================================================================
# Initialize Material Properties
# ============================================================================
# Set uniform lossy medium throughout the domain
for i in range(0, nx + 1):
    for j in range(0, ny + 1):
        sig_z[i, j] = sigma_medium    # Conductivity
        eps[i, j] = eps_r * e0        # Permittivity
        
        # Calculate update coefficients for lossy medium
        # These come from the discretized Maxwell's equations:
        # dD/dt + sigma*E = curl(H)
        # With D = epsilon*E, we get:
        # epsilon*dE/dt + sigma*E = curl(H)
        
        denom = 1.0 + sig_z[i, j] * dt / (2.0 * eps[i, j])
        CA_Ez[i, j] = (1.0 - sig_z[i, j] * dt / (2.0 * eps[i, j])) / denom
        CB_Ez[i, j] = (dt / eps[i, j]) / denom

# ============================================================================
# Spatial Derivative Denominators
# ============================================================================
den_hx = np.ones(nx, dtype=floattype) / dy
den_hy = np.ones(ny, dtype=floattype) / dy
den_ex = np.ones(nx, dtype=floattype) / dx
den_ey = np.ones(ny, dtype=floattype) / dy

# ============================================================================
# Source Parameters (Gaussian derivative pulse)
# ============================================================================
rtau = 50.0e-12        # Pulse width parameter (s)
tau = rtau / dt        # Pulse width in time steps
ndelay = 3 * tau       # Delay to center the pulse
srcconst = -dt * 3.0e11  # Source amplitude constant

source = np.zeros(nmax + 1, dtype=floattype)
for n in range(1, nmax + 1):
    source[n] = (n - ndelay) * np.exp(-((n - ndelay)**2 / tau**2))

# ============================================================================
# Recording Array
# ============================================================================
RecordEz = np.zeros(nmax + 1, dtype=floattype)

# ============================================================================
# Setup Visualization
# ============================================================================
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# ============================================================================
# Main FDTD Time-Stepping Loop
# ============================================================================
print("\nStarting FDTD simulation...")
print(f"Running {nmax} time steps\n")

for n in range(1, nmax + 1):
    t_start = time.time()
    
    # ========================================================================
    # Update Magnetic Fields (Hx, Hy)
    # ========================================================================
    # Hx update: Hx = DA*Hx - DB*(dEz/dy)
    for i in range(0, nx):
        for j in range(0, ny):
            Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j + 1] - Ez[i, j]) * den_hy[j]
    
    # Hy update: Hy = DA*Hy + DB*(dEz/dx)
    for i in range(0, nx):
        for j in range(0, ny):
            Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i + 1, j] - Ez[i, j]) * den_hx[i]
    
    # ========================================================================
    # Update Electric Field (Ez) - Interior points only (PEC boundaries)
    # ========================================================================
    # Ez update: Ez = CA*Ez + CB*(dHy/dx - dHx/dy)
    for i in range(1, nx):
        for j in range(1, ny):
            Ez[i, j] = (CA_Ez[i, j] * Ez[i, j] + 
                       CB_Ez[i, j] * ((Hy[i, j] - Hy[i - 1, j]) * den_ex[i] - 
                                      (Hx[i, j] - Hx[i, j - 1]) * den_ey[j]))
    
    elapsed = time.time() - t_start
    
    # ========================================================================
    # Apply Source (soft source)
    # ========================================================================
    Ez[Is, Js] = Ez[Is, Js] + CB_Ez[Is, Js] * source[n]
    
    # Record field at source location
    RecordEz[n] = Ez[Is, Js]
    
    # Print progress every 100 steps
    if n % 100 == 0 or n == 1:
        print(f"Time step: {n:4d}/{nmax}, Ez_max: {np.max(np.abs(Ez)):.6e}, "
              f"Time/step: {elapsed*1000:.2f} ms")
    
    # ========================================================================
    # Visualization (update every 5 time steps for smoother animation)
    # ========================================================================
    if n % 5 == 0:
        ax.clear()
        
        # Plot Ez field with symmetric colormap for wave visualization
        vmax = max(np.max(np.abs(Ez)), 1e-10)  # Avoid division by zero
        im = ax.imshow(Ez[:, :].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, nx*dx*1000, 0, ny*dy*1000])
        
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'2D FDTD TM Mode - Lossy Medium (σ = {sigma_medium} S/m)\n'
                     f'Time Step: {n}, t = {n*dt*1e12:.2f} ps')
        
        # Add colorbar only once
        if n == 5:
            cbar = plt.colorbar(im, ax=ax, label='Ez (V/m)')
        else:
            im.set_clim(-vmax, vmax)
        
        plt.draw()
        plt.pause(0.01)

# ============================================================================
# Post-Processing
# ============================================================================
print("\nSimulation completed!")
print(f"Maximum Ez amplitude: {np.max(np.abs(Ez)):.6e} V/m")

# Plot time history of Ez at source point
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
ax2.set_title(f'Ez at Source Point ({Is}, {Js}) - Lossy Medium σ = {sigma_medium} S/m')
ax2.grid(True)
ax2.set_xlim([0, time_axis[-1]])

plt.tight_layout()
plt.show()

