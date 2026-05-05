#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance 2D FDTD (NeCLO Style - TM Mode)
Adapted from 3D version
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0

dtype_torch = torch.float32 

device = torch.device("cpu")
print(f"Running on: {device} (Fixed for compatibility)")
print(f"Precision: {dtype_torch}")

# ==========================================
# 1. Grid & Physics Setup (2D)
# ==========================================
# Grid Setup (Increased size for better 2D viz)
nx, ny = 200, 200
dx, dy = 1e-3, 1e-3
# Shape: (Batch, Channel, Height(x), Width(y))
shape_max = (1, 3, nx + 1, ny + 1) 

# Source Position (Center)
Is, Js = nx // 2, ny // 2

# Time Setup
nmax = 500
c0 = 2.99792458e8
# 2D CFL Condition
dt = 0.99 / (c0 * np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy)))

# Material Parameters
sigma = 4.0  # S/m (Lossy Medium)
eps_r = 1.0

# Coefficients (Same physics formula as 3D)
EA = (e0 * eps_r / dt) + 0.5 * sigma
EB = (e0 * eps_r / dt) - 0.5 * sigma
CA = EB / EA
CB = 1.0 / (EA * dx) # Assuming dx=dy
C_H = dt / (m0 * dx)

print(f"Lossy Coefficients: CA={CA:.6f}, CB={CB:.6e}")

# Convert to Tensors
CA = torch.tensor(CA, dtype=dtype_torch, device=device)
CB = torch.tensor(CB, dtype=dtype_torch, device=device)
C_H = torch.tensor(C_H, dtype=dtype_torch, device=device)

# ==========================================
# 2. Kernel Definition (2D TM Mode)
# ==========================================
# TM Mode Components: 
# Channel 0: Hx
# Channel 1: Hy
# Channel 2: Ez

def get_curl_kernel_2d(mode='backward'):
    # Kernel shape for Conv2d: (Out_C, In_C, kH, kW) -> (3, 3, 3, 3)
    k = torch.zeros((3, 3, 3, 3), dtype=dtype_torch, device=device)
    
    def set_k(out_c, in_c, axis, sign):
        # axis 0 = x (height/row), axis 1 = y (width/col)
        center = [1, 1]
        neighbor = [1, 1]
        
        if mode == 'backward': # For E update (Backward Difference)
            w_c, w_n = 1.0, -1.0; neighbor[axis] -= 1
        else: # For H update (Forward Difference)
            w_c, w_n = -1.0, 1.0; neighbor[axis] += 1
            
        k[out_c, in_c, center[0], center[1]] = w_c * sign
        k[out_c, in_c, neighbor[0], neighbor[1]] = w_n * sign

    # --- Maxwell's Equations (TM Mode) ---
    # Curl E -> Updates H
    # dHx/dt ~ -dEz/dy  => Hx(0) from Ez(2), axis=1(y), sign=+1 (due to forward diff logic handled inside)
    # dHy/dt ~  dEz/dx  => Hy(1) from Ez(2), axis=0(x), sign=-1
    
    # Curl H -> Updates E
    # dEz/dt ~ (dHy/dx - dHx/dy)
    # Ez(2) from Hy(1), axis=0(x), sign=+1
    # Ez(2) from Hx(0), axis=1(y), sign=-1

    if mode == 'forward': # Calculate Curl E (to update H)
        set_k(0, 2, 1, +1) # Hx terms
        set_k(1, 2, 0, -1) # Hy terms
        
    else: # Calculate Curl H (to update E)
        set_k(2, 1, 0, +1) # Ez terms from Hy
        set_k(2, 0, 1, -1) # Ez terms from Hx
        
    return k

K_E = get_curl_kernel_2d('backward')
K_H = get_curl_kernel_2d('forward')

# ==========================================
# 3. Initialization
# ==========================================
# Mask Setup (PEC Boundary for 2D)
mask = torch.ones(shape_max, dtype=dtype_torch, device=device)
# Mask boundaries for Ez (Channel 2)
mask[:, 2, 0, :] = 0; mask[:, 2, -1, :] = 0  # Top/Bottom X boundaries
mask[:, 2, :, 0] = 0; mask[:, 2, :, -1] = 0  # Left/Right Y boundaries

# Field Tensors
E = torch.zeros(shape_max, dtype=dtype_torch, device=device)
H = torch.zeros(shape_max, dtype=dtype_torch, device=device)

# Source Pulse (Gaussian Derivative)
t_steps = torch.arange(1, nmax + 2, dtype=dtype_torch, device=device) * dt
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * (3.0e11)
source = srcconst * ((t_steps/dt) - ndelay) * torch.exp(-(((t_steps/dt) - ndelay)**2 / (tau**2)))

# ==========================================
# 4. Simulation Loop
# ==========================================
print(f"Starting 2D CPU Simulation ({nmax} steps)...")
t_start = time.time()

with torch.no_grad():
    for n in range(nmax):
        # --- Update H ---
        # H(n+0.5) = H(n-0.5) - C_H * Curl(E(n))
        curl_E = F.conv2d(E, K_H, padding=1)
        H.sub_(C_H * curl_E)
        
        # --- Update E ---
        # E(n+1) = CA * E(n) + CB * Curl(H(n+0.5))
        curl_H = F.conv2d(H, K_E, padding=1)
        E.mul_(CA).add_(curl_H, alpha=CB)
        
        # Boundary & Source
        E.mul_(mask)
        E[:, 2, Is, Js] += source[n] # Inject Source into Ez channel
        
        # --- Logging ---
        if (n+1) % 50 == 0:
            t_now = time.time()
            ez_val = E[0, 2, Is, Js].item()
            rate = (n+1) / (t_now - t_start)
            print(f"Step {n+1}/{nmax}, Ez_src: {ez_val:.4e}, Rate: {rate:.1f} steps/s")

t_end = time.time()
total_time = t_end - t_start

print(f"\nSimulation finished in {total_time:.4f} s")
print(f"Average Throughput: {nmax/total_time:.2f} steps/s")

# ==========================================
# 5. Visualization
# ==========================================
# Extract Ez field (Channel 2)
# Numpy conversion for plotting
Ez_field = E[0, 2, :, :].numpy()

plt.figure(figsize=(6, 5))
vmax = np.max(np.abs(Ez_field)) + 1e-10 # Avoid div by 0
# Use 'jet' colormap to match your requested style
plt.imshow(Ez_field.T, origin='lower', cmap='jet', vmin=-vmax, vmax=vmax)
plt.title(f"2D NeCLO (FDTD) Result (Step {nmax})\nTM Mode: Ez Field")
plt.colorbar(label='Ez (V/m)')
plt.xlabel("x grid")
plt.ylabel("y grid")
plt.tight_layout()
plt.show()