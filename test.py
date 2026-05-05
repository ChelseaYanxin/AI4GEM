#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STABLE & FAST 3D FDTD (CPU Slicing)
Fixes:
1. Corrected spatial indexing (using ':-1' instead of '1:') to match Yee Grid.
2. Prevents numerical explosion (NaN) by respecting causality.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0

# ==========================================
# 1. Setup
# ==========================================
dtype_torch = torch.float64
device = torch.device("cpu") 

print(f"Running on: {device} (Stable Slicing)")
# Grid
nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3
shape_max = (1, 3, nx + 1, ny + 1, nz + 1)

# Source
Is, Js, Ks = 9, 9, 9 

# Time
nmax = 500
c0 = 2.99792458e8
# CFL Condition Check
dt = 0.99 / (c0 * np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)))

# Material
sigma = 1.0; eps_r = 1.0
EA = (e0 * eps_r / dt) + 0.5 * sigma
EB = (e0 * eps_r / dt) - 0.5 * sigma
CA = EB / EA
CB = 1.0 / (EA * dx) 
C_H = dt / (m0 * dx)

CA = torch.tensor(CA, dtype=dtype_torch, device=device)
CB = torch.tensor(CB, dtype=dtype_torch, device=device)
C_H = torch.tensor(C_H, dtype=dtype_torch, device=device)

E = torch.zeros(shape_max, dtype=dtype_torch, device=device)
H = torch.zeros(shape_max, dtype=dtype_torch, device=device)

# Source
t = torch.arange(1, nmax + 2, dtype=dtype_torch, device=device) * dt
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * (3.0e11)
source = srcconst * ((t/dt) - ndelay) * torch.exp(-(((t/dt) - ndelay)**2 / (tau**2)))

# ==========================================
# 2. Main Loop (Stable)
# ==========================================
print(f"Starting Simulation ({nmax} steps)...")
t_start = time.time()

with torch.no_grad():
    for n in range(nmax):
        
        # --- Update H ---

        # 1. Hx (位于 y+0.5, z+0.5) -> 需要 dEz/dy 和 dEy/dz
        # dEz_dy: 沿Y求导(Y少1), X/Z 保持全长
        dEz_dy = E[:, 2, :, 1:, :] - E[:, 2, :, :-1, :] 
        # dEy_dz: 沿Z求导(Z少1), X/Y 保持全长
        dEy_dz = E[:, 1, :, :, 1:] - E[:, 1, :, :, :-1]
        
        # 对齐：X取 [:-1] (因为 Hx 只需要 nx 个), Y取与 dEz_dy 一致, Z取与 dEy_dz 一致
        # dEz_dy 缺Y，需切 Z[:-1]
        # dEy_dz 缺Z，需切 Y[:-1]
        term1 = dEz_dy[..., :-1]    # Shape: (B, X, Y_reduced, Z_reduced)
        term2 = dEy_dz[..., :-1, :] # Shape: (B, X, Y_reduced, Z_reduced)
        
        H[:, 0, :, :-1, :-1] -= C_H * (term1 - term2)

        # 2. Hy (位于 x+0.5, z+0.5) -> 需要 dEx/dz - dEz/dx
        dEx_dz = E[:, 0, :, :, 1:] - E[:, 0, :, :, :-1]
        dEz_dx = E[:, 2, 1:, :, :] - E[:, 2, :-1, :, :]
        
        term1 = dEx_dz[..., :-1, :, :] # Cut X to match dEz_dx
        term2 = dEz_dx[..., :-1]       # Cut Z to match dEx_dz
        
        H[:, 1, :-1, :, :-1] -= C_H * (term1 - term2)

        # 3. Hz (位于 x+0.5, y+0.5) -> 需要 dEy/dx - dEx/dy
        dEy_dx = E[:, 1, 1:, :, :] - E[:, 1, :-1, :, :]
        dEx_dy = E[:, 0, :, 1:, :] - E[:, 0, :, :-1, :]
        
        term1 = dEy_dx[..., :-1, :] # Cut Y to match dEx_dy
        term2 = dEx_dy[..., :-1, :, :] # Cut X to match dEy_dx
        
        H[:, 2, :-1, :-1, :] -= C_H * (term1 - term2)

        # --- Update E ---
        # Ex
        hz_term = H[:, 2, :, 1:-1, 1:-1] - H[:, 2, :, :-2, 1:-1]
        hy_term = H[:, 1, :, 1:-1, 1:-1] - H[:, 1, :, 1:-1, :-2]
        E[:, 0, :, 1:-1, 1:-1].mul_(CA).add_(hz_term - hy_term, alpha=CB)

        # Ey
        hx_term = H[:, 0, 1:-1, :, 1:-1] - H[:, 0, 1:-1, :, :-2]
        hz_term = H[:, 2, 1:-1, :, 1:-1] - H[:, 2, :-2, :, 1:-1]
        E[:, 1, 1:-1, :, 1:-1].mul_(CA).add_(hx_term - hz_term, alpha=CB)

        # Ez
        hy_term = H[:, 1, 1:-1, 1:-1, :] - H[:, 1, :-2, 1:-1, :]
        hx_term = H[:, 0, 1:-1, 1:-1, :] - H[:, 0, 1:-1, :-2, :]
        E[:, 2, 1:-1, 1:-1, :].mul_(CA).add_(hy_term - hx_term, alpha=CB)

        E[:, 2, Is, Js, Ks] += source[n]
        
        # Logging
        if (n+1) % 50 == 0:
            ez_val = E[0, 2, Is, Js, Ks].item()
            rate = (n+1) / (time.time() - t_start)
            print(f"Step {n+1}/{nmax}, Ez: {ez_val:.4e}, Rate: {rate:.1f} steps/s")

total_time = time.time() - t_start
print(f"\nFinal: {total_time:.4f} s | Rate: {nmax/total_time:.2f} steps/s")

mid_k = 9
Ez_sim_slice = E[0, 2, :, :, mid_k].numpy()
plt.figure(figsize=(6, 5))
vmax = np.max(np.abs(Ez_sim_slice))
plt.imshow(Ez_sim_slice.T, origin='lower', cmap='jet', vmin=-vmax, vmax=vmax)
plt.title("Stable Slicing FDTD Result")
plt.colorbar()
plt.show()