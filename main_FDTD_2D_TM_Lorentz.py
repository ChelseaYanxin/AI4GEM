# -*- coding: utf-8 -*-
"""
2D FDTD TM Mode Simulation with Lorentz Dispersive and Lossy Medium

This script simulates electromagnetic wave propagation in a 2D domain
using the Finite-Difference Time-Domain (FDTD) method for TM polarization
(Ez, Hx, Hy components) with a Lorentz dispersive medium.

The Lorentz model describes frequency-dependent permittivity:
    ε(ω) = ε_∞ + (ε_s - ε_∞) * ω_0² / (ω_0² + jωγ - ω²)

Where:
    ε_∞  = permittivity at infinite frequency (high-frequency limit)
    ε_s  = static (DC) permittivity
    ω_0  = resonance angular frequency
    γ    = damping coefficient (collision frequency)

Implementation uses the Auxiliary Differential Equation (ADE) method.

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
nmax = 3000          # Maximum number of time steps
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
# Lorentz Dispersive Medium Parameters
# ============================================================================
# Lorentz model: ε(ω) = ε_∞ + Δε * ω_0² / (ω_0² + jωγ - ω²)
# where Δε = ε_s - ε_∞

# High-frequency (optical) relative permittivity
eps_inf = 1.0

# Static (DC) relative permittivity  
eps_s = 5.0

# Permittivity difference
delta_eps = eps_s - eps_inf

# Resonance frequency (rad/s) - example: optical phonon resonance
f0 = 10e9           # Resonance frequency in Hz (10 GHz)
omega_0 = 2 * np.pi * f0

# Damping coefficient (collision frequency) - controls loss bandwidth
gamma = 2 * np.pi * 2e9    # Damping in rad/s (2 GHz bandwidth)

# Additional DC conductivity (optional, set to 0 for pure Lorentz)
sigma_dc = 0.0      # DC conductivity in S/m

# ============================================================================
# Print Simulation Information
# ============================================================================
print("=" * 70)
print("2D FDTD TM Mode - Lorentz Dispersive Medium")
print("=" * 70)
print(f"Grid size: {nx} x {ny}")
print(f"Grid spacing: dx = dy = {dx*1000:.2f} mm")
print(f"Time step: dt = {dt*1e12:.4f} ps")
print(f"Total simulation time: {nmax*dt*1e9:.2f} ns")
print("-" * 70)
print("Lorentz Model Parameters:")
print(f"  ε_∞ (high-freq permittivity)  = {eps_inf}")
print(f"  ε_s (static permittivity)     = {eps_s}")
print(f"  Δε = ε_s - ε_∞                = {delta_eps}")
print(f"  f_0 (resonance frequency)     = {f0/1e9:.2f} GHz")
print(f"  ω_0 (resonance angular freq)  = {omega_0:.3e} rad/s")
print(f"  γ (damping coefficient)       = {gamma:.3e} rad/s")
print(f"  σ_dc (DC conductivity)        = {sigma_dc} S/m")
print("-" * 70)

# Check stability criterion for dispersive media
stability_param = omega_0 * dt
print(f"Stability check: ω_0 * dt = {stability_param:.4f} (should be << 1)")
if stability_param > 0.5:
    print("WARNING: ω_0 * dt may be too large for stable simulation!")
print("=" * 70)

# ============================================================================
# Field Arrays (TM mode: Ez, Hx, Hy)
# ============================================================================
Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)
Hx = np.zeros((nx + 1, ny), dtype=floattype)
Hy = np.zeros((nx, ny + 1), dtype=floattype)

# ============================================================================
# Auxiliary Arrays for Lorentz Dispersion (ADE Method)
# ============================================================================
# Polarization current density Jz = dPz/dt
Jz = np.zeros((nx + 1, ny + 1), dtype=floattype)

# Polarization density Pz
Pz = np.zeros((nx + 1, ny + 1), dtype=floattype)

# ============================================================================
# Pre-compute Update Coefficients
# ============================================================================
# Magnetic field update coefficients (non-dispersive)
DA = 1.0
DB = dt / m0

# ============================================================================
# Electric field update coefficients for Lorentz + conductivity
# ============================================================================
# From Maxwell: ε_0*ε_∞*dE/dt + σ*E + J_pol = curl(H)
# where J_pol = dP/dt is the polarization current
#
# Discretized (semi-implicit for σ term):
# ε_0*ε_∞*(E^n+1 - E^n)/dt + σ*(E^n+1 + E^n)/2 + J^n+1/2 = curl(H)^n+1/2
#
# Solving for E^n+1:
# E^n+1 = CE1*E^n + CE2*curl(H) - CE3*J^n+1/2

eps_inf_e0 = eps_inf * e0
alpha_e = eps_inf_e0 / dt + sigma_dc / 2.0
beta_e = eps_inf_e0 / dt - sigma_dc / 2.0

# CE1: coefficient for E^n
CE1 = beta_e / alpha_e

# CE2: coefficient for curl(H) - NOTE: spatial derivatives handled separately
CE2 = 1.0 / alpha_e

# CE3: coefficient for polarization current J^n+1/2
CE3 = 1.0 / alpha_e

# ============================================================================
# Polarization current (Jz) update coefficients - ADE for Lorentz
# ============================================================================
# Lorentz oscillator: d²P/dt² + γ*dP/dt + ω_0²*P = ε_0*Δε*ω_0²*E
# With J = dP/dt:
#   dJ/dt + γ*J + ω_0²*P = ε_0*Δε*ω_0²*E
#   dP/dt = J
#
# Semi-implicit discretization for J:
# (J^n+1/2 - J^n-1/2)/dt + γ*(J^n+1/2 + J^n-1/2)/2 + ω_0²*P^n = ε_0*Δε*ω_0²*E^n
#
# Solving for J^n+1/2:
# J^n+1/2 = CJ1*J^n-1/2 + CJ2*P^n + CJ3*E^n

alpha_j = 1.0 / dt + gamma / 2.0
beta_j = 1.0 / dt - gamma / 2.0

# CJ1: coefficient for J^n-1/2
CJ1 = beta_j / alpha_j

# CJ2: coefficient for P^n
CJ2 = -omega_0**2 / alpha_j

# CJ3: coefficient for E^n
CJ3 = e0 * delta_eps * omega_0**2 / alpha_j

# Polarization (Pz) update coefficient
# P^n+1 = P^n + dt * J^n+1/2
CP1 = dt

print("\nUpdate Coefficients:")
print(f"  CE1 = {CE1:.6f}")
print(f"  CE2 = {CE2:.6e}")
print(f"  CE3 = {CE3:.6e}")
print(f"  CJ1 = {CJ1:.6f}")
print(f"  CJ2 = {CJ2:.6e}")
print(f"  CJ3 = {CJ3:.6e}")

# Stability check for ADE-FDTD
print(f"\nStability parameters:")
print(f"  |CJ1| = {abs(CJ1):.6f} (should be < 1 for stability)")
print(f"  γ*dt/2 = {gamma*dt/2:.6f}")
print(f"  ω_0²*dt² = {(omega_0*dt)**2:.6f}")

# ============================================================================
# Spatial Derivative Factors
# ============================================================================
inv_dx = 1.0 / dx
inv_dy = 1.0 / dy

# ============================================================================
# Source Parameters (Gaussian derivative pulse - broadband)
# ============================================================================
# Use Gaussian derivative for broadband excitation to see dispersion effects
rtau = 50.0e-12        # Pulse width parameter (s)
tau = rtau / dt        # Pulse width in time steps
ndelay = 3 * tau       # Delay to center the pulse
src_amplitude = 1.0    # Source amplitude (keep moderate)

source = np.zeros(nmax + 1, dtype=floattype)
for n in range(1, nmax + 1):
    source[n] = src_amplitude * (n - ndelay) * np.exp(-((n - ndelay)**2 / tau**2))

# Source injection coefficient (for soft source)
# This ensures proper impedance matching
src_coeff = CE2  # Use the E-field curl coefficient for soft source

print(f"\nSource parameters:")
print(f"  Pulse width (rtau) = {rtau*1e12:.1f} ps")
print(f"  Pulse center delay = {ndelay*dt*1e12:.1f} ps")

# ============================================================================
# Recording Arrays
# ============================================================================
RecordEz = np.zeros(nmax + 1, dtype=floattype)
RecordPz = np.zeros(nmax + 1, dtype=floattype)

# ============================================================================
# Setup Visualization
# ============================================================================
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============================================================================
# Main FDTD Time-Stepping Loop
# ============================================================================
print("\nStarting FDTD simulation with Lorentz dispersion...")
print(f"Running {nmax} time steps\n")

for n in range(1, nmax + 1):
    t_start = time.time()
    
    # ========================================================================
    # Update Magnetic Fields (Hx, Hy) - Standard update (non-dispersive)
    # ========================================================================
    # Hx update: Hx = DA*Hx - DB*(dEz/dy)
    for i in range(0, nx):
        for j in range(0, ny):
            Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j + 1] - Ez[i, j]) * inv_dy
    
    # Hy update: Hy = DA*Hy + DB*(dEz/dx)
    for i in range(0, nx):
        for j in range(0, ny):
            Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i + 1, j] - Ez[i, j]) * inv_dx
    
    # ========================================================================
    # Update Polarization Current Jz (ADE for Lorentz)
    # ========================================================================
    # Jz^n+1/2 = CJ1*Jz^n-1/2 + CJ2*Pz^n + CJ3*Ez^n
    for i in range(1, nx):
        for j in range(1, ny):
            Jz[i, j] = CJ1 * Jz[i, j] + CJ2 * Pz[i, j] + CJ3 * Ez[i, j]
    
    # ========================================================================
    # Update Electric Field Ez (with Lorentz dispersion)
    # ========================================================================
    # Ez^n+1 = CE1*Ez^n + CE2*curl(H) - CE3*J^n+1/2
    # Note: Use J directly (polarization current), not dP
    for i in range(1, nx):
        for j in range(1, ny):
            curl_H = (Hy[i, j] - Hy[i - 1, j]) * inv_dx - (Hx[i, j] - Hx[i, j - 1]) * inv_dy
            Ez[i, j] = CE1 * Ez[i, j] + CE2 * curl_H - CE3 * Jz[i, j]
    
    # ========================================================================
    # Update Polarization Pz (after E update)
    # ========================================================================
    # Pz^n+1 = Pz^n + dt * Jz^n+1/2
    for i in range(1, nx):
        for j in range(1, ny):
            Pz[i, j] = Pz[i, j] + CP1 * Jz[i, j]
    
    elapsed = time.time() - t_start
    
    # ========================================================================
    # Apply Source (soft source with proper coefficient)
    # ========================================================================
    Ez[Is, Js] = Ez[Is, Js] + src_coeff * source[n]
    
    # Record fields at source location
    RecordEz[n] = Ez[Is, Js]
    RecordPz[n] = Pz[Is, Js]
    
    # Print progress every 200 steps
    if n % 200 == 0 or n == 1:
        print(f"Time step: {n:4d}/{nmax}, Ez_max: {np.max(np.abs(Ez)):.6e}, "
              f"Pz_max: {np.max(np.abs(Pz)):.6e}, Time/step: {elapsed*1000:.2f} ms")
    
    # ========================================================================
    # Visualization (update every 10 time steps)
    # ========================================================================
    if n % 10 == 0:
        # Plot Ez field
        axes[0].clear()
        vmax_ez = max(np.max(np.abs(Ez)), 1e-10)
        im0 = axes[0].imshow(Ez[:, :].T, origin='lower', cmap='RdBu_r',
                             vmin=-vmax_ez, vmax=vmax_ez,
                             extent=[0, nx*dx*1000, 0, ny*dy*1000])
        axes[0].set_xlabel('x (mm)')
        axes[0].set_ylabel('y (mm)')
        axes[0].set_title(f'Ez Field\nt = {n*dt*1e9:.3f} ns, Step: {n}')
        
        # Plot Pz (polarization) field
        axes[1].clear()
        vmax_pz = max(np.max(np.abs(Pz)), 1e-15)
        im1 = axes[1].imshow(Pz[:, :].T, origin='lower', cmap='PuOr',
                             vmin=-vmax_pz, vmax=vmax_pz,
                             extent=[0, nx*dx*1000, 0, ny*dy*1000])
        axes[1].set_xlabel('x (mm)')
        axes[1].set_ylabel('y (mm)')
        axes[1].set_title(f'Polarization Pz\nt = {n*dt*1e9:.3f} ns')
        
        fig.suptitle(f'2D FDTD - Lorentz Dispersive Medium\n'
                     f'ε_∞={eps_inf}, ε_s={eps_s}, f_0={f0/1e9:.1f} GHz, γ={gamma/(2*np.pi)/1e9:.1f} GHz',
                     fontsize=12)
        
        # Add colorbars only once
        if n == 10:
            plt.colorbar(im0, ax=axes[0], label='Ez (V/m)', shrink=0.8)
            plt.colorbar(im1, ax=axes[1], label='Pz (C/m²)', shrink=0.8)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

# ============================================================================
# Post-Processing
# ============================================================================
print("\n" + "=" * 70)
print("Simulation completed!")
print(f"Maximum Ez amplitude: {np.max(np.abs(Ez)):.6e} V/m")
print(f"Maximum Pz amplitude: {np.max(np.abs(Pz)):.6e} C/m²")
print("=" * 70)

# ============================================================================
# Plot Time History
# ============================================================================
plt.ioff()
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))

# Time axis
time_axis = np.arange(nmax + 1) * dt * 1e9  # in ns

# Source
axes2[0].plot(time_axis, source, 'b-', linewidth=1)
axes2[0].set_xlabel('Time (ns)')
axes2[0].set_ylabel('Source')
axes2[0].set_title(f'Excitation Source (Gaussian derivative, τ = {rtau*1e12:.0f} ps)')
axes2[0].grid(True)
axes2[0].set_xlim([0, time_axis[-1]])

# Ez at source point
axes2[1].plot(time_axis, RecordEz, 'r-', linewidth=1)
axes2[1].set_xlabel('Time (ns)')
axes2[1].set_ylabel('Ez (V/m)')
axes2[1].set_title(f'Ez at Source Point ({Is}, {Js})')
axes2[1].grid(True)
axes2[1].set_xlim([0, time_axis[-1]])

# Pz at source point
axes2[2].plot(time_axis, RecordPz, 'g-', linewidth=1)
axes2[2].set_xlabel('Time (ns)')
axes2[2].set_ylabel('Pz (C/m²)')
axes2[2].set_title(f'Polarization Pz at Source Point ({Is}, {Js})')
axes2[2].grid(True)
axes2[2].set_xlim([0, time_axis[-1]])

fig2.suptitle(f'Lorentz Dispersive Medium: ε_∞={eps_inf}, ε_s={eps_s}, '
              f'f_0={f0/1e9:.1f} GHz, γ={gamma/(2*np.pi)/1e9:.1f} GHz', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# Frequency Domain Analysis (FFT)
# ============================================================================
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))

# Compute FFT of source and Ez response
freq = np.fft.fftfreq(nmax + 1, dt)
freq_positive = freq[:nmax//2] / 1e9  # Convert to GHz

source_fft = np.abs(np.fft.fft(source))[:nmax//2]
ez_fft = np.abs(np.fft.fft(RecordEz))[:nmax//2]

# Normalize
source_fft_norm = source_fft / np.max(source_fft) if np.max(source_fft) > 0 else source_fft
ez_fft_norm = ez_fft / np.max(ez_fft) if np.max(ez_fft) > 0 else ez_fft

axes3[0].plot(freq_positive, source_fft_norm, 'b-', linewidth=1, label='Source')
axes3[0].plot(freq_positive, ez_fft_norm, 'r-', linewidth=1, label='Ez Response')
axes3[0].axvline(x=f0/1e9, color='k', linestyle='--', label=f'f_0 = {f0/1e9:.1f} GHz')
axes3[0].set_xlabel('Frequency (GHz)')
axes3[0].set_ylabel('Normalized Amplitude')
axes3[0].set_title('Frequency Spectrum')
axes3[0].legend()
axes3[0].grid(True)
axes3[0].set_xlim([0, min(50, freq_positive[-1])])

# Theoretical Lorentz permittivity
freq_theory = np.linspace(0.1e9, 30e9, 1000)
omega_theory = 2 * np.pi * freq_theory
eps_lorentz = eps_inf + delta_eps * omega_0**2 / (omega_0**2 + 1j*omega_theory*gamma - omega_theory**2)

axes3[1].plot(freq_theory/1e9, np.real(eps_lorentz), 'b-', linewidth=2, label="ε' (real)")
axes3[1].plot(freq_theory/1e9, np.imag(eps_lorentz), 'r-', linewidth=2, label="ε'' (imag)")
axes3[1].axvline(x=f0/1e9, color='k', linestyle='--', label=f'f_0 = {f0/1e9:.1f} GHz')
axes3[1].axhline(y=eps_inf, color='gray', linestyle=':', alpha=0.7, label=f'ε_∞ = {eps_inf}')
axes3[1].axhline(y=eps_s, color='gray', linestyle=':', alpha=0.7, label=f'ε_s = {eps_s}')
axes3[1].set_xlabel('Frequency (GHz)')
axes3[1].set_ylabel('Relative Permittivity')
axes3[1].set_title('Lorentz Model: ε(ω) = ε_∞ + Δε·ω_0² / (ω_0² + jωγ - ω²)')
axes3[1].legend()
axes3[1].grid(True)
axes3[1].set_xlim([0, 30])

fig3.suptitle('Frequency Domain Analysis', fontsize=14)
plt.tight_layout()
plt.show()

