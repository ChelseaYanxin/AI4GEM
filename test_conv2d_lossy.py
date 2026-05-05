#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D TMz NeCLO - Only Hy Visualization
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0, speed_of_light as c0

# ==========================================
# 0. Configuration & Physics
# ==========================================
dtype_torch = torch.float64
dtype_np = np.float64
device = torch.device("cpu") 

nx, ny = 20, 20
dx, dy = 2e-3, 2e-3
Is, Js = nx // 2, ny // 2
nmax = 100
dt = 0.99 / (c0 * np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy)))

# Material & Coefficients
sigma = 4.0
eps_r = 1.0
EA = (e0 * eps_r / dt) + 0.5 * sigma
EB = (e0 * eps_r / dt) - 0.5 * sigma
CA = EB / EA
CB = 1.0 / (EA * dx) 
HA = (m0 / dt)
HB = (m0 / dt)
DA = HB / HA
DB = (1.0 / dx) / HA

# Source Pulse
t = np.arange(1, nmax + 1, dtype=dtype_np) * dt
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * 3.0e11
source_np = srcconst * ((t/dt) - ndelay) * np.exp(-(((t/dt) - ndelay)**2 / (tau**2)))

print(f"Grid: {nx}x{ny}, Sigma: {sigma}")

# ==========================================
# 1. Solvers
# ==========================================
def run_ground_truth():
    print("Running Ground Truth (Numpy)...")
    Ez = np.zeros((nx + 1, ny + 1), dtype=dtype_np)
    Hx = np.zeros((nx + 1, ny), dtype=dtype_np)
    Hy = np.zeros((nx, ny + 1), dtype=dtype_np)
    
    for n in range(nmax):
        # Update H
        for i in range(nx + 1):
            for j in range(ny):
                Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j+1] - Ez[i, j])
        for i in range(nx):
            for j in range(ny + 1):
                Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i+1, j] - Ez[i, j])
                
        # Update E
        for i in range(1, nx):
            for j in range(1, ny):
                Ez[i, j] = CA * Ez[i, j] + CB * ((Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1]))
        
        Ez[Is, Js] += source_np[n]
    return Ez, Hx, Hy

def run_neclo():
    print("Running NeCLO (Diff Kernel)...")
    Ez = torch.zeros((1, 1, nx+1, ny+1), dtype=dtype_torch, device=device)
    Hx = torch.zeros((1, 1, nx+1, ny),   dtype=dtype_torch, device=device)
    Hy = torch.zeros((1, 1, nx,   ny+1), dtype=dtype_torch, device=device)
    source = torch.from_numpy(source_np).to(device)

    # Kernels
    def get_k(axis):
        if axis == 'x': 
            k = torch.zeros((1, 1, 2, 1), dtype=dtype_torch, device=device)
            k[0,0,0,0], k[0,0,1,0] = -1.0, 1.0
            return k
        else:
            k = torch.zeros((1, 1, 1, 2), dtype=dtype_torch, device=device)
            k[0,0,0,0], k[0,0,0,1] = -1.0, 1.0
            return k
    k_dx, k_dy = get_k('x'), get_k('y')
    slice_inner = (slice(None), slice(None), slice(1, nx), slice(1, ny))

    with torch.no_grad():
        for n in range(nmax):
            dEz_dy = F.conv2d(Ez, k_dy)
            Hx = DA * Hx - DB * dEz_dy
            dEz_dx = F.conv2d(Ez, k_dx)
            Hy = DA * Hy + DB * dEz_dx
            
            dHy_dx = F.conv2d(Hy, k_dx)
            dHx_dy = F.conv2d(Hx, k_dy)
            term_Hy = dHy_dx[:, :, :, 1:ny] 
            term_Hx = dHx_dy[:, :, 1:nx, :]
            
            Ez[slice_inner] = CA * Ez[slice_inner] + CB * (term_Hy - term_Hx)
            Ez[:, :, Is, Js] += source[n]

    return (Ez.squeeze().cpu().numpy(), Hx.squeeze().cpu().numpy(), Hy.squeeze().cpu().numpy())

# Run Simulation
Ez_gt, Hx_gt, Hy_gt = run_ground_truth()
Ez_sim, Hx_sim, Hy_sim = run_neclo()

# ==========================================
# 2. Visualization (Only Hy)
# ==========================================
print("Plotting Hy only...")

# 改动 1: 设置 rows=1, 并调整图片大小 (figsize)
rows, cols = 1, 2
fig, axes = plt.subplots(rows, cols, figsize=(10, 5)) 

def plot_field(ax, data, title, cmap='jet'):
    im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (index)')
    ax.set_ylabel('y (index)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def get_vlims(d1, d2):
    vmax = max(np.max(np.abs(d1)), np.max(np.abs(d2)))
    if vmax == 0: vmax = 1e-10
    return -vmax, vmax

# --- Only Plot Hx ---
vmin, vmax = get_vlims(Ez_gt, Ez_sim)

# 改动 2: axes 变成了一维数组，直接通过索引访问
plot_field(axes[0], Ez_gt, "Ez Ground Truth")
axes[0].images[0].set_clim(vmin, vmax)

plot_field(axes[1], Ez_sim, "Ez NeCLO (lossy)")
axes[1].images[0].set_clim(vmin, vmax)

plt.tight_layout()
plt.show()
print("Plotting Done.")