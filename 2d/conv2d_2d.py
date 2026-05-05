#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeCLO 2D TM Solver (Final Fixed Version)
Fixes the index misalignment in Ez update that caused instability.
"""

import os
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0

# --- 1. 初始化 ---
dtype_torch = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

mat_file = '../data/FDTD_2D_TM_clean_data.mat' # 确认路径
if not os.path.exists(mat_file):
    raise FileNotFoundError(f"找不到 {mat_file}")

print(f"Loading Ground Truth from {mat_file}...")
mat = scipy.io.loadmat(mat_file)

# 读取参数
dx = mat['dx'][0,0]
dy = mat['dy'][0,0]
dt = mat['dt'][0,0]
nx = mat['nx'][0,0] # 100
ny = mat['ny'][0,0] # 100

Ez_GT_all = mat['Ez'] # (101, 101, T)
Hx_GT_all = mat['Hx'] # (101, 100, T)
Hy_GT_all = mat['Hy'] # (100, 101, T)
total_gt_steps = Ez_GT_all.shape[-1]
nmax = 500 



# 系数 (直接使用和 GT 完全一样的常数)
coeff_H = dt / m0
coeff_E = dt / e0
m_hx = coeff_H / dy 
m_hy = coeff_H / dx
m_ex = coeff_E / dx
m_ey = coeff_E / dy

# 转 Tensor
def to_tensor(numpy_array):
    return torch.tensor(numpy_array, dtype=dtype_torch, device=device).unsqueeze(0).unsqueeze(0)

Ez = to_tensor(Ez_GT_all[:, :, 0])
Hx = to_tensor(Hx_GT_all[:, :, 0])
Hy = to_tensor(Hy_GT_all[:, :, 0])
print("Initial state loaded.")

# --- 2. 卷积核 (方向已修正) ---
def get_diff_kernel_2d(axis):
    # Dim 2 is x (Height), Dim 3 is y (Width)
    if axis == 'x': # d/dx: 垂直差分 (2, 1)
        k = torch.zeros((1, 1, 2, 1), dtype=dtype_torch, device=device)
        k[0, 0, 0, 0] = -1.0; k[0, 0, 1, 0] = 1.0
        return k
    if axis == 'y': # d/dy: 水平差分 (1, 2)
        k = torch.zeros((1, 1, 1, 2), dtype=dtype_torch, device=device)
        k[0, 0, 0, 0] = -1.0; k[0, 0, 0, 1] = 1.0
        return k

k_dx = get_diff_kernel_2d('x')
k_dy = get_diff_kernel_2d('y')

# --- 3. 主循环 ---
print(f"Starting simulation for {nmax} steps...")
max_abs_error = 0.0

for n in range(1, nmax + 1):
    
    # === H Update ===
    # GT Logic: Hx loop 0..nx (updates 0..99), Hy loop 0..nx (updates 0..99)
    
    # 1. Hx (Update Size: 100x100)
    # Need dEz/dy. Ez(101,101) * k_dy(1,2) -> (101, 100)
    dEz_dy = F.conv2d(Ez, k_dy)
    # GT Hx loop is range(0, nx). So we update rows 0..99.
    # dEz_dy rows 0..99 correspond exactly to Ez[i, j+1]-Ez[i, j]. Correct.
    Hx[:, :, 0:nx, :] -= m_hx * dEz_dy[:, :, 0:nx, :]

    # 2. Hy (Update Size: 100x100)
    # Need dEz/dx. Ez(101,101) * k_dx(2,1) -> (100, 101)
    dEz_dx = F.conv2d(Ez, k_dx)
    # GT Hy loop is range(0, nx). So we update cols 0..99.
    Hy[:, :, :, 0:ny] += m_hy * dEz_dx[:, :, :, 0:ny]

    # === E Update ===
    # GT Logic: Ez loop 1..nx (updates 1..99). Interior only.
    
    # 1. Term dHy/dx: Need (Hy[i, j] - Hy[i-1, j])
    # Hy slice (0:nx, 0:ny+1) -> (100, 101)
    # Conv k_dx (2,1) -> Output (99, 101).
    # Output row k corresponds to Hy[k+1] - Hy[k].
    # We need index i=1..99. Let k = i-1. 
    # Row k=0 -> Hy[1]-Hy[0]. Matches Ez row i=1.
    # So we use all 99 rows.
    # We need cols j=1..99.
    # So we slice cols 1..99 (index 1:ny).
    
    dHy_dx_full = F.conv2d(Hy[:, :, 0:nx, :], k_dx) # (99, 101)
    term_dHy_dx = dHy_dx_full[:, :, :, 1:ny]        # (99, 99)

    # 2. Term dHx/dy: Need (Hx[i, j] - Hx[i, j-1])
    # Hx slice (0:nx+1, 0:ny) -> (101, 100)
    # Conv k_dy (1,2) -> Output (101, 99).
    # Output col k corresponds to Hx[k+1] - Hx[k].
    # We need index j=1..99. Let k = j-1.
    # Col k=0 -> Hx[1]-Hx[0]. Matches Ez col j=1.
    # So we use all 99 cols.
    # We need rows i=1..99.
    # So we slice rows 1..99 (index 1:nx).
    
    dHx_dy_full = F.conv2d(Hx[:, :, :, 0:ny], k_dy) # (101, 99)
    term_dHx_dy = dHx_dy_full[:, :, 1:nx, :]        # (99, 99)
    
    # 3. Apply Update
    Ez[:, :, 1:nx, 1:ny] += m_ex * term_dHy_dx - m_ey * term_dHx_dy

    # === Comparison ===
    Ez_sim_np = Ez.cpu().numpy().squeeze()
    Ez_ref_np = Ez_GT_all[:, :, n]
    
    # Compare Interior
    diff = np.abs(Ez_sim_np[1:nx, 1:ny] - Ez_ref_np[1:nx, 1:ny])
    err = np.max(diff)
    if err > max_abs_error: max_abs_error = err
    
    if n % 50 == 0:
        print(f"Step {n}/{nmax}, Max Error: {err:.6e}")

print(f"\nFinal Max Error: {max_abs_error:.6e}")

if max_abs_error < 1e-10:
    print("SUCCESS: Bit-exact match! Generating comparison plots...")
    
    # 提取最后一步的数据
    Ez_sim_final = Ez.cpu().numpy().squeeze()
    Hx_sim_final = Hx.cpu().numpy().squeeze()
    Hy_sim_final = Hy.cpu().numpy().squeeze()
    
    Ez_gt_final = Ez_GT_all[:, :, nmax]
    Hx_gt_final = Hx_GT_all[:, :, nmax]
    Hy_gt_final = Hy_GT_all[:, :, nmax]

    # 定义绘图函数 (完全复刻你想要的风格)
    def plot_comparison(gt_data, sim_data, field_name, step):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 统一颜色范围，确保对比准确
        vmin = min(gt_data.min(), sim_data.min())
        vmax = max(gt_data.max(), sim_data.max())
        
        # 1. Ground Truth
        im1 = axes[0].imshow(gt_data.T, origin='lower', cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{field_name} Ground Truth (2D)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("x (index)")
        axes[0].set_ylabel("y (index)")
        plt.colorbar(im1, ax=axes[0])
        
        # 2. NeCLO
        im2 = axes[1].imshow(sim_data.T, origin='lower', cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"{field_name} NeCLO (2D)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("x (index)")
        axes[1].set_ylabel("y (index)")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        save_name = f"Comparison_{field_name}_Step{step}.png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}")
        plt.show()

    # 生成三组对比图
    plot_comparison(Ez_gt_final, Ez_sim_final, "Ez", nmax)
    plot_comparison(Hx_gt_final, Hx_sim_final, "Hx", nmax)
    plot_comparison(Hy_gt_final, Hy_sim_final, "Hy", nmax)

else:
    print("WARNING: Still mismatch. No plots generated.")