#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Optimized 3D FDTD with PyTorch (Lossy / Conductive Medium)
Features:
1. Unified Tensor Shape
2. Only 2 conv3d calls per step
3. Supports Lossy Medium (Conductivity sigma)
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0 as m0, epsilon_0 as e0

dtype_torch = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3
shape_max = (nx + 1, ny + 1, nz + 1)
Is, Js, Ks = 9, 9, 9 

nmax = 500
c0 = 2.99792458e8
dt = dx / (2.0 * c0)

# --- 1. 定义电导率 Sigma 分布 (Lossy Map) ---
# 默认为 0 (无损耗)
sigma = torch.zeros((1, 3, *shape_max), dtype=dtype_torch, device=device)

# [示例]：在网格的一半区域添加有耗介质 (例如海水 sigma=4.0)
sigma[:, :, :, :, nz//2:] = 4.0 
# 注意：这里我们简单地对 Ex, Ey, Ez 都应用了相同的 sigma 分布
# 如果要更严谨，应该像 PEC Mask 那样考虑错位，但在统一张量下这样已经足够好

# --- 2. 预计算带损耗的更新系数 ---
# 系数因子 factor = (sigma * dt) / (2 * e0)
# 为了数值稳定，e0 可以加上 epsilon_r (如果需要)
factor = (sigma * dt) / (2.0 * e0)

# CA: 衰减系数 (Lossy Decay)
# 无损耗处 sigma=0 -> CA=1
# 有损耗处 -> CA < 1
CA = (1.0 - factor) / (1.0 + factor)

# CB: 旋度更新系数 (Curl Coefficient)
# 包含了原本的 C_E 以及损耗带来的修正
# 原本 C_E = dt / (e0 * dx)
CB = (dt / (e0 * dx)) / (1.0 + factor)

# 磁场系数 (假设磁损耗 sigma_m = 0)
C_H = dt / (m0 * dx)

# --- Curl Kernels (不变) ---
def get_curl_kernel(mode='backward'):
    k = torch.zeros((3, 3, 3, 3, 3), dtype=dtype_torch, device=device)
    def set_k(out_c, in_c, axis, sign):
        center = [1, 1, 1]; neighbor = [1, 1, 1]
        if mode == 'backward': 
            w_center, w_neigh = 1.0, -1.0; neighbor[axis] -= 1
        else: 
            w_center, w_neigh = -1.0, 1.0; neighbor[axis] += 1
        k[out_c, in_c, center[0], center[1], center[2]] = w_center * sign
        k[out_c, in_c, neighbor[0], neighbor[1], neighbor[2]] = w_neigh * sign
    set_k(0, 2, 1, +1); set_k(0, 1, 2, -1)
    set_k(1, 0, 2, +1); set_k(1, 2, 0, -1)
    set_k(2, 1, 0, +1); set_k(2, 0, 1, -1)
    return k

K_E = get_curl_kernel('backward')
K_H = get_curl_kernel('forward')

# PEC Mask (不变)
mask = torch.ones((1, 3, *shape_max), dtype=dtype_torch, device=device)
mask[:, 0, :, 0, :] = 0; mask[:, 0, :, -1, :] = 0
mask[:, 0, :, :, 0] = 0; mask[:, 0, :, :, -1] = 0
mask[:, 0, -1, :, :] = 0 
mask[:, 1, 0, :, :] = 0; mask[:, 1, -1, :, :] = 0
mask[:, 1, :, :, 0] = 0; mask[:, 1, :, :, -1] = 0
mask[:, 1, :, -1, :] = 0
mask[:, 2, 0, :, :] = 0; mask[:, 2, -1, :, :] = 0
mask[:, 2, :, 0, :] = 0; mask[:, 2, :, -1, :] = 0
mask[:, 2, :, :, -1] = 0

E = torch.zeros((1, 3, *shape_max), dtype=dtype_torch, device=device)
H = torch.zeros((1, 3, *shape_max), dtype=dtype_torch, device=device)

# Source
t = torch.arange(1, nmax + 2, dtype=dtype_torch, device=device)
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * (3.0e11)
source = srcconst * (t - ndelay) * torch.exp(-((t - ndelay)**2 / (tau**2)))

print("Starting Lossy Simulation...")

with torch.no_grad():
    for n in range(nmax):
        # --- H Update (不变) ---
        E_pad = F.pad(E, (1,1,1,1,1,1))
        curl_E = F.conv3d(E_pad, K_H)
        H -= C_H * curl_E
        
        # --- E Update (修改处) ---
        H_pad = F.pad(H, (1,1,1,1,1,1)) 
        curl_H = F.conv3d(H_pad, K_E)
        
        # 原本: E += C_E * curl_H
        # 现在: E = CA * E + CB * curl_H
        # 这里的计算也是全向量化的，CA 和 CB 是张量
        E = CA * E + CB * curl_H
        
        E *= mask # 应用 PEC 边界
        E[0, 2, Is, Js, Ks] += source[n] # 硬源添加
        
        if n % 50 == 0:
            print(f"Step {n}/{nmax}, Max E: {E.max().item():.2e}")

# Plot
mid_k = nz // 2
Ez_sim_slice = E[0, 2, :, :, mid_k].cpu().numpy()

plt.figure(figsize=(6, 5))
plt.imshow(Ez_sim_slice.T, origin='lower', cmap='jet', aspect='auto')
plt.title(f"Lossy FDTD (Sigma=4.0 in right half)")
plt.colorbar()
plt.show()