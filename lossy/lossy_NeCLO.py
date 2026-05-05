#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance 3D FDTD (NeCLO v2 - Lossy Medium)
With Publication-Quality Visualization (Based on Reference Style)
"""

import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0

# --- 1. 配置与初始化 (Configuration) ---
dtype_torch = torch.float32
device = torch.device("cpu") # 改为 "cuda" 如果需要
print(f"Running on: {device}")
print(f"Precision: {dtype_torch}")

# Grid Setup
nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3
shape_max = (1, 3, nx + 1, ny + 1, nz + 1) # (Batch, Channel, x, y, z)

# Source Position
Is, Js, Ks = 9, 9, 9 

# Time Setup
nmax = 500
c0 = 2.99792458e8
dt = 0.99 / (c0 * np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)))

# Lossy Parameters
sigma = 4.0  # S/m (Lossy)
eps_r = 1.0

EA = (e0 * eps_r / dt) + 0.5 * sigma
EB = (e0 * eps_r / dt) - 0.5 * sigma
CA = EB / EA
CB = 1.0 / (EA * dx)
C_H = dt / (m0 * dx)

print(f"Lossy Coefficients: CA={CA:.6f}, CB={CB:.6e}")

# Convert to Tensors
CA = torch.tensor(CA, dtype=dtype_torch, device=device)
CB = torch.tensor(CB, dtype=dtype_torch, device=device)
C_H = torch.tensor(C_H, dtype=dtype_torch, device=device)

# --- 2. 核心求解器 (Solver: NeCLO v2) ---
def get_curl_kernel(mode='backward'):
    k = torch.zeros((3, 3, 3, 3, 3), dtype=dtype_torch, device=device)
    def set_k(out_c, in_c, axis, sign):
        center = [1, 1, 1]; neighbor = [1, 1, 1]
        if mode == 'backward': # For E update
            w_c, w_n = 1.0, -1.0; neighbor[axis] -= 1
        else: # For H update
            w_c, w_n = -1.0, 1.0; neighbor[axis] += 1
        k[out_c, in_c, center[0], center[1], center[2]] = w_c * sign
        k[out_c, in_c, neighbor[0], neighbor[1], neighbor[2]] = w_n * sign

    # Channel Mapping: 0=Hx, 1=Hy, 2=Ez
    set_k(0, 2, 1, +1); set_k(0, 1, 2, -1) # Curl E -> Hx (Out 0)
    set_k(1, 0, 2, +1); set_k(1, 2, 0, -1) # Curl E -> Hy (Out 1)
    set_k(2, 1, 0, +1); set_k(2, 0, 1, -1) # Curl E -> Ez (Out 2)
    return k

K_E = get_curl_kernel('backward')
K_H = get_curl_kernel('forward')

# Mask Setup
mask = torch.ones(shape_max, dtype=dtype_torch, device=device)
mask[:, 0, :, 0, :] = 0; mask[:, 0, :, -1, :] = 0
mask[:, 0, :, :, 0] = 0; mask[:, 0, :, :, -1] = 0; mask[:, 0, -1, :, :] = 0 
mask[:, 1, 0, :, :] = 0; mask[:, 1, -1, :, :] = 0
mask[:, 1, :, :, 0] = 0; mask[:, 1, :, :, -1] = 0; mask[:, 1, :, -1, :] = 0 
mask[:, 2, 0, :, :] = 0; mask[:, 2, -1, :, :] = 0
mask[:, 2, :, 0, :] = 0; mask[:, 2, :, -1, :] = 0; mask[:, 2, :, :, -1] = 0 

E = torch.zeros(shape_max, dtype=dtype_torch, device=device)
H = torch.zeros(shape_max, dtype=dtype_torch, device=device)

# Source Definition
t = torch.arange(1, nmax + 2, dtype=dtype_torch, device=device) * dt
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau; srcconst = -dt * (3.0e11)
source = srcconst * ((t/dt) - ndelay) * torch.exp(-(((t/dt) - ndelay)**2 / (tau**2)))

print(f"Starting NeCLO v2 Simulation ({nmax} steps)...")
t_start = time.time()

with torch.no_grad():
    for n in range(nmax):
        # Update H
        curl_E = F.conv3d(E, K_H, padding=1)
        H.sub_(C_H * curl_E)
        
        # Update E
        curl_H = F.conv3d(H, K_E, padding=1)
        E.mul_(CA).add_(curl_H, alpha=CB)
        
        # Boundary & Source
        E.mul_(mask)
        E[:, 2, Is, Js, Ks] += source[n]
        
        if (n+1) % 50 == 0:
            print(f"Step {n+1}/{nmax}...")

t_end = time.time()
print(f"Simulation finished in {t_end - t_start:.4f} s")

# --- 3. 数据后处理与 Ground Truth 加载 (Data Loading) ---

# Sim Results to CPU
E_sim_cpu = E.detach().cpu().numpy()
H_sim_cpu = H.detach().cpu().numpy()

# Path to your Ground Truth
gt_path = '/Users/zyanxin/Documents/code/GEM_N/data/FDTD_3D_lossy_clean_data.mat'
compare_ref = False

if os.path.exists(gt_path):
    print(f"Loading Ground Truth from: {gt_path}")
    gt_data = scipy.io.loadmat(gt_path)
    
    required_keys = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy']
    if all(k in gt_data for k in required_keys):
        print("Found separate field components. Padding and Stacking...")
        
        # 读取并取最后一帧
        def get_last_frame(data):
            return data[..., -1] if data.ndim == 4 else data
            
        Ex_raw = get_last_frame(gt_data['Ex'])
        Ey_raw = get_last_frame(gt_data['Ey'])
        Ez_raw = get_last_frame(gt_data['Ez'])
        Hx_raw = get_last_frame(gt_data['Hx'])
        Hy_raw = get_last_frame(gt_data['Hy'])
        Hz_raw = get_last_frame(gt_data.get('Hz', np.zeros_like(Hx_raw)))

        # 补零对齐
        target_shape = (nx + 1, ny + 1, nz + 1)
        def pad_to_target(data, target):
            pad_width = []
            for d_curr, d_tgt in zip(data.shape, target):
                p = max(0, d_tgt - d_curr)
                pad_width.append((0, p))
            return np.pad(data, pad_width, mode='constant')

        Ex_gt = pad_to_target(Ex_raw, target_shape)
        Ey_gt = pad_to_target(Ey_raw, target_shape)
        Ez_gt = pad_to_target(Ez_raw, target_shape)
        Hx_gt = pad_to_target(Hx_raw, target_shape)
        Hy_gt = pad_to_target(Hy_raw, target_shape)
        Hz_gt = pad_to_target(Hz_raw, target_shape)
        
        # 拼装堆叠 -> (1, 3, x, y, z)
        E_gt_cpu = np.stack([Ex_gt, Ey_gt, Ez_gt], axis=0)[np.newaxis, ...]
        H_gt_cpu = np.stack([Hx_gt, Hy_gt, Hz_gt], axis=0)[np.newaxis, ...]
        
        print(f"Ground Truth Loaded. Shape: {E_gt_cpu.shape}")
        compare_ref = True
    else:
        print("Error: Required keys not found in .mat file.")
else:
    print(f"Warning: File not found at {gt_path}. Skipping comparison.")

# --- 4. 可视化 (Visualization) ---
mid_k = nz // 2

# 提取切片 (Sim)
Ez_sim_slice = E_sim_cpu[0, 2, :, :, mid_k]
Hx_sim_slice = H_sim_cpu[0, 0, :, :, mid_k]
Hy_sim_slice = H_sim_cpu[0, 1, :, :, mid_k]

# 提取切片 (GT)
if compare_ref:
    Ez_gt_slice = E_gt_cpu[0, 2, :, :, mid_k]
    Hx_gt_slice = H_gt_cpu[0, 0, :, :, mid_k]
    Hy_gt_slice = H_gt_cpu[0, 1, :, :, mid_k]

# 创建画布: 3行 x (2列 或 1列)
rows, cols = 1, (2 if compare_ref else 1)
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

# 辅助绘图函数 (完全参考你的样式代码)
def plot_field(ax, data, title, cmap='jet'):
    im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (index)')
    ax.set_ylabel('y (index)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 统一 vmax 的辅助函数 (可选，为了更好对比)
def get_vlims(d1, d2):
    vmax = max(np.max(np.abs(d1)), np.max(np.abs(d2)))
    if vmax == 0: vmax = 1e-10
    return -vmax, vmax

if compare_ref:
    # --- Row 1: Ez ---
    #vmin, vmax = get_vlims(Ez_gt_slice, Ez_sim_slice)
    #plot_field(axes[0, 0], Ez_gt_slice, "Ez Ground Truth")
    #axes[0, 0].images[0].set_clim(vmin, vmax) # 强制统一色标
    
    #plot_field(axes[0, 1], Ez_sim_slice, "Ez NeCLO (lossy)")
    #axes[0, 1].images[0].set_clim(vmin, vmax)

    # --- Row 2: Hx ---
    #vmin, vmax = get_vlims(Hx_gt_slice, Hx_sim_slice)
    #plot_field(axes[1, 0], Hx_gt_slice, "Hx Ground Truth")
    #axes[1, 0].images[0].set_clim(vmin, vmax)
    
    #plot_field(axes[1, 1], Hx_sim_slice, "Hx NeCLO (lossy)")
    #axes[1, 1].images[0].set_clim(vmin, vmax)

    # --- Row 3: Hy ---
    vmin, vmax = get_vlims(Hy_gt_slice, Hy_sim_slice)
    plot_field(axes[0], Hy_gt_slice, "Hy Ground Truth")
    axes[0].images[0].set_clim(vmin, vmax)
    
    plot_field(axes[1], Hy_sim_slice, "Hy NeCLO (lossy)")
    axes[1].images[0].set_clim(vmin, vmax)

else:
    
    plot_field(axes[0], Ez_sim_slice, "Ez NeCLO (lossy)")
    plot_field(axes[1], Hx_sim_slice, "Hx NeCLO (lossy)")
    plot_field(axes[2], Hy_sim_slice, "Hy NeCLO (lossy)")

plt.tight_layout()
#plt.suptitle(f"Field Comparison (Lossy $\sigma$={sigma})", y=1.02, fontsize=16, fontweight='bold')
plt.show()
print("Plotting Done.")