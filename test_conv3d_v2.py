#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Optimized 3D FDTD with PyTorch
Features:
1. Unified Tensor Shape (nx+1, ny+1, nz+1)
2. Only 2 conv3d calls per step (Full Vectorization)
3. PEC Boundary masking
"""

import os
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0

dtype_torch = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3

# 统一申请最大尺寸 (nx+1, ny+1, nz+1)
# 这样 Ex, Ey, Ez 都可以放进同一个 Tensor
shape_max = (nx + 1, ny + 1, nz + 1)

Is, Js, Ks = 9, 9, 9 

nmax = 500
c0 = 2.99792458e8
dt = dx / (2.0 * c0)

C_E = dt / (e0 * dx)
C_H = dt / (m0 * dx)

# Input: (B, 3, D, H, W) -> Output: (B, 3, D, H, W)

def get_curl_kernel(mode='backward'):
    # Kernel shape: (Out_Ch=3, In_Ch=3, D=3, H=3, W=3)
    k = torch.zeros((3, 3, 3, 3, 3), dtype=dtype_torch, device=device)
    
    def set_k(out_c, in_c, axis, sign):
        # axis: 0=x(depth), 1=y(height), 2=z(width)
        # Center index is [1,1,1]
        center = [1, 1, 1]
        neighbor = [1, 1, 1]
        
        if mode == 'backward': # For E update (H_i - H_{i-1})
            w_center, w_neigh = 1.0, -1.0
            neighbor[axis] -= 1
        else: # For H update (E_{i+1} - E_i)
            w_center, w_neigh = -1.0, 1.0
            neighbor[axis] += 1
            
        k[out_c, in_c, center[0], center[1], center[2]] = w_center * sign
        k[out_c, in_c, neighbor[0], neighbor[1], neighbor[2]] = w_neigh * sign

    # Curl formulas:
    # 0 (Ex) <- dy(Hz) - dz(Hy)
    set_k(0, 2, 1, +1); set_k(0, 1, 2, -1)
    # 1 (Ey) <- dz(Hx) - dx(Hz)
    set_k(1, 0, 2, +1); set_k(1, 2, 0, -1)
    # 2 (Ez) <- dx(Hy) - dy(Hx)
    set_k(2, 1, 0, +1); set_k(2, 0, 1, -1)
    
    return k

K_E = get_curl_kernel('backward') # H -> E
K_H = get_curl_kernel('forward')  # E -> H

# Mask = 1 (内部), Mask = 0 (边界)
mask = torch.ones((1, 3, *shape_max), dtype=dtype_torch, device=device)

# Channel 0: Ex (定义在 y, z 边缘)
# y=0, y=ny 处 Ex 切向 = 0
mask[:, 0, :, 0, :] = 0; mask[:, 0, :, -1, :] = 0
# z=0, z=nz 处 Ex 切向 = 0
mask[:, 0, :, :, 0] = 0; mask[:, 0, :, :, -1] = 0
# x=nx 处 (索引 -1) 原本 Ex 只定义到 nx-1，所以最后一位是 ghost，设为 0 安全
mask[:, 0, -1, :, :] = 0 

# Channel 1: Ey (定义在 x, z 边缘)
# x=0, x=nx 处 Ey 切向 = 0
mask[:, 1, 0, :, :] = 0; mask[:, 1, -1, :, :] = 0
# z=0, z=nz 处 Ey 切向 = 0
mask[:, 1, :, :, 0] = 0; mask[:, 1, :, :, -1] = 0
# y=ny 处是 ghost
mask[:, 1, :, -1, :] = 0

# Channel 2: Ez (定义在 x, y 边缘)
# x=0, x=nx 处 Ez 切向 = 0
mask[:, 2, 0, :, :] = 0; mask[:, 2, -1, :, :] = 0
# y=0, y=ny 处 Ez 切向 = 0
mask[:, 2, :, 0, :] = 0; mask[:, 2, :, -1, :] = 0
# z=nz 处是 ghost
mask[:, 2, :, :, -1] = 0


# (Batch=1, Ch=3, nx+1, ny+1, nz+1)
E = torch.zeros((1, 3, *shape_max), dtype=dtype_torch, device=device)
H = torch.zeros((1, 3, *shape_max), dtype=dtype_torch, device=device)

t = torch.arange(1, nmax + 2, dtype=dtype_torch, device=device)
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * (3.0e11)
source = srcconst * (t - ndelay) * torch.exp(-((t - ndelay)**2 / (tau**2)))

mat_file = '/Users/zyanxin/Documents/code/GEM_N/data/FDTD_3D_cavity_clean_data.mat'
compare_ref = False
if os.path.exists(mat_file):
    mat = scipy.io.loadmat(mat_file)
    Ez_ref_all = mat['Ez'].astype(np.float32)
    Ez_ref_gt = Ez_ref_all[:nx+1, :ny+1, :nz, :nmax] 
    Ez_ref_final = Ez_ref_gt[:, :, :, -1] 
    compare_ref = True

print("Starting Optimized Simulation...")

with torch.no_grad():
    for n in range(nmax):
       
        H_pad = F.pad(H, (1,1,1,1,1,1)) 
        curl_H = F.conv3d(H_pad, K_E)
        E += C_E * curl_H
        E *= mask
        E[0, 2, Is, Js, Ks] += source[n]
        E_pad = F.pad(E, (1,1,1,1,1,1))
        curl_E = F.conv3d(E_pad, K_H)
        H -= C_H * curl_E
        
        if n % 50 == 0:
            print(f"Step {n}/{nmax}")

print("Plotting...")

mid_k = nz // 2
Ez_sim_slice = E[0, 2, :, :, mid_k].cpu().numpy() # (nx+1, ny+1)

fig, ax = plt.subplots(1, 2 if compare_ref else 1, figsize=(10, 5))

# Plot Sim
if not compare_ref: ax = [ax]
im = ax[0].imshow(Ez_sim_slice.T, origin='lower', cmap='jet', aspect='auto')
ax[0].set_title(f"GEM Optimized (Conv3D) - Step {nmax}")
plt.colorbar(im, ax=ax[0])

# Plot Ref
if compare_ref:
    # Reference slice
    Ez_ref_slice = Ez_ref_final[:, :, mid_k]
    im2 = ax[1].imshow(Ez_ref_slice.T, origin='lower', cmap='jet', aspect='auto')
    ax[1].set_title("Ground Truth (Matlab)")
    plt.colorbar(im2, ax=ax[1])
    
    err = np.max(np.abs(Ez_sim_slice - Ez_ref_slice))
    print(f"Final Max Abs Error: {err:.4e}")

plt.tight_layout()
plt.show()