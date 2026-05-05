#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorized GNN-based FDTD (GEM) with Visualization.
Replaces Python loops with Tensor Meshgrids for instant graph construction.
"""

import os
import numpy as np
import scipy.io
import torch
import time
import matplotlib.pyplot as plt # 引入绘图库
from scipy.constants import mu_0 as m0, epsilon_0 as e0

# -------------------------
# 1. Configuration & Constants
# -------------------------
# Default to float64 for CPU/CUDA
dtype = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Handle Mac MPS (Metal Performance Shaders)
if torch.backends.mps.is_available(): 
    device = torch.device("mps")

# MPS does not support float64, switch to float32
if device.type == 'mps':
    print("MPS detected: Switching to float32 (float64 not supported on MPS).")
    dtype = torch.float32

print(f"Running Vectorized GNN-FDTD on: {device}")

nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3
nmax = 500

# Constants
dt = dx / (2.0 * 2.99792458e8)
Is, Js, Ks = 9, 9, 9  # Source position

# Coefficients
EA = 1.0; EB = 1.0; CA = 1.0
CBx = (dt / e0) / dx; CBy = (dt / e0) / dy; CBz = (dt / e0) / dz
HA = 1.0; HB = 1.0; DA = 1.0
DBx = (dt / m0) / dx; DBy = (dt / m0) / dy; DBz = (dt / m0) / dz

# Source
rtau = 50.0e-12; tau = rtau / dt; ndelay = 3 * tau
srcconst = -dt * (3.0e11)
SOURCE_END_STEP = 83

# Load Ground Truth
mat_file = '/Users/zyanxin/Documents/code/GEM_N/data/FDTD_3D_cavity_clean_data.mat'
if not os.path.exists(mat_file):
    print("Warning: .mat file not found, skipping validation.")
    Ex_ref_np = None
else:
    mat = scipy.io.loadmat(mat_file)
    Ex_ref_np = mat['Ex']; Ey_ref_np = mat['Ey']; Ez_ref_np = mat['Ez']
    Hx_ref_np = mat['Hx']; Hy_ref_np = mat['Hy']; Hz_ref_np = mat['Hz']

# -------------------------
# 2. Vectorized Graph Construction (The Fast Way)
# -------------------------
print("Building Graph Topology (Vectorized)...")
t_build_start = time.time()

# Shapes
Nx_Ex, Ny_Ex, Nz_Ex = nx, ny+1, nz+1
Nx_Ey, Ny_Ey, Nz_Ey = nx+1, ny, nz+1
Nx_Ez, Ny_Ez, Nz_Ez = nx+1, ny+1, nz
Nx_Hx, Ny_Hx, Nz_Hx = nx+1, ny, nz
Nx_Hy, Ny_Hy, Nz_Hy = nx, ny+1, nz
Nx_Hz, Ny_Hz, Nz_Hz = nx, ny, nz+1

# Total Nodes
NEx = Nx_Ex * Ny_Ex * Nz_Ex
NEy = Nx_Ey * Ny_Ey * Nz_Ey
NEz = Nx_Ez * Ny_Ez * Nz_Ez
NHx = Nx_Hx * Ny_Hx * Nz_Hx
NHy = Nx_Hy * Ny_Hy * Nz_Hy
NHz = Nx_Hz * Ny_Hz * Nz_Hz

# --- Helper: Vectorized Indexing ---
# Instead of python functions, we define strides for direct tensor math
def get_grid_and_idx(xr, yr, zr, stride_y, stride_z):
    """
    Generates a 3D meshgrid of indices and the corresponding flat linear indices.
    xr, yr, zr: tuples (start, end)
    stride_y, stride_z: multipliers for flattening
    """
    i = torch.arange(xr[0], xr[1], device=device)
    j = torch.arange(yr[0], yr[1], device=device)
    k = torch.arange(zr[0], zr[1], device=device)
    # indexing='ij' ensures dimension order is (i, j, k)
    I, J, K = torch.meshgrid(i, j, k, indexing='ij')
    
    # Calculate flat index: i * (Ny*Nz) + j * (Nz) + k
    flat_idx = I * stride_y + J * stride_z + K
    return I, J, K, flat_idx.flatten()

# Strides for each field type
str_Ex = (Ny_Ex * Nz_Ex, Nz_Ex)
str_Ey = (Ny_Ey * Nz_Ey, Nz_Ey)
str_Ez = (Ny_Ez * Nz_Ez, Nz_Ez)
str_Hx = (Ny_Hx * Nz_Hx, Nz_Hx)
str_Hy = (Ny_Hy * Nz_Hy, Nz_Hy)
str_Hz = (Ny_Hz * Nz_Hz, Nz_Hz)

edge_lists = {}

def add_vec_edge(name, dst_flat, src_flat, weight):
    if name not in edge_lists:
        edge_lists[name] = [[], [], []]
    edge_lists[name][0].append(src_flat)
    edge_lists[name][1].append(dst_flat)
    # create weight tensor of same size
    w_tensor = torch.full_like(src_flat, weight, dtype=dtype)
    edge_lists[name][2].append(w_tensor)

# --- Build E Edges ---

# 1. Ex Update (Range: 0..nx, 1..ny, 1..nz)
range_ex = ((0, nx), (1, ny), (1, nz))
I, J, K, dst_idx = get_grid_and_idx(*range_ex, *str_Ex)

# Hz terms
src_hz1 = I * str_Hz[0] + J * str_Hz[1] + K      # Hz[i, j, k]
src_hz2 = I * str_Hz[0] + (J-1) * str_Hz[1] + K  # Hz[i, j-1, k]
add_vec_edge('Hz_Ex', dst_idx, src_hz1.flatten(), +CBy)
add_vec_edge('Hz_Ex', dst_idx, src_hz2.flatten(), -CBy)

# Hy terms
src_hy1 = I * str_Hy[0] + J * str_Hy[1] + K      # Hy[i, j, k]
src_hy2 = I * str_Hy[0] + J * str_Hy[1] + (K-1)  # Hy[i, j, k-1]
add_vec_edge('Hy_Ex', dst_idx, src_hy1.flatten(), -CBz)
add_vec_edge('Hy_Ex', dst_idx, src_hy2.flatten(), +CBz)


# 2. Ey Update (Range: 1..nx, 0..ny, 1..nz)
range_ey = ((1, nx), (0, ny), (1, nz))
I, J, K, dst_idx = get_grid_and_idx(*range_ey, *str_Ey)

# Hx terms
src_hx1 = I * str_Hx[0] + J * str_Hx[1] + K
src_hx2 = I * str_Hx[0] + J * str_Hx[1] + (K-1)
add_vec_edge('Hx_Ey', dst_idx, src_hx1.flatten(), +CBz)
add_vec_edge('Hx_Ey', dst_idx, src_hx2.flatten(), -CBz)

# Hz terms
src_hz1 = I * str_Hz[0] + J * str_Hz[1] + K
src_hz2 = (I-1) * str_Hz[0] + J * str_Hz[1] + K
add_vec_edge('Hz_Ey', dst_idx, src_hz1.flatten(), -CBx)
add_vec_edge('Hz_Ey', dst_idx, src_hz2.flatten(), +CBx)


# 3. Ez Update (Range: 1..nx, 1..ny, 0..nz)
range_ez = ((1, nx), (1, ny), (0, nz))
I, J, K, dst_idx = get_grid_and_idx(*range_ez, *str_Ez)

# Hy terms
src_hy1 = I * str_Hy[0] + J * str_Hy[1] + K
src_hy2 = (I-1) * str_Hy[0] + J * str_Hy[1] + K
add_vec_edge('Hy_Ez', dst_idx, src_hy1.flatten(), +CBx)
add_vec_edge('Hy_Ez', dst_idx, src_hy2.flatten(), -CBx)

# Hx terms
src_hx1 = I * str_Hx[0] + J * str_Hx[1] + K
src_hx2 = I * str_Hx[0] + (J-1) * str_Hx[1] + K
add_vec_edge('Hx_Ez', dst_idx, src_hx1.flatten(), -CBy)
add_vec_edge('Hx_Ez', dst_idx, src_hx2.flatten(), +CBy)


# --- Build H Edges ---

# 4. Hx Update (Range: 0..nx+1, 0..ny, 0..nz)
range_hx = ((0, nx+1), (0, ny), (0, nz))
I, J, K, dst_idx = get_grid_and_idx(*range_hx, *str_Hx)

# Ez terms
src_ez1 = I * str_Ez[0] + J * str_Ez[1] + K      # Ez[i, j, k]
src_ez2 = I * str_Ez[0] + (J+1) * str_Ez[1] + K  # Ez[i, j+1, k]
add_vec_edge('Ez_to_Hx', dst_idx, src_ez1.flatten(), +DBy)
add_vec_edge('Ez_to_Hx', dst_idx, src_ez2.flatten(), -DBy)

# Ey terms
src_ey1 = I * str_Ey[0] + J * str_Ey[1] + (K+1)  # Ey[i, j, k+1]
src_ey2 = I * str_Ey[0] + J * str_Ey[1] + K      # Ey[i, j, k]
add_vec_edge('Ey_to_Hx', dst_idx, src_ey1.flatten(), +DBz)
add_vec_edge('Ey_to_Hx', dst_idx, src_ey2.flatten(), -DBz)


# 5. Hy Update (Range: 0..nx, 0..ny+1, 0..nz)
range_hy = ((0, nx), (0, ny+1), (0, nz))
I, J, K, dst_idx = get_grid_and_idx(*range_hy, *str_Hy)

# Ex terms
src_ex1 = I * str_Ex[0] + J * str_Ex[1] + K
src_ex2 = I * str_Ex[0] + J * str_Ex[1] + (K+1)
add_vec_edge('Ex_to_Hy', dst_idx, src_ex1.flatten(), +DBz)
add_vec_edge('Ex_to_Hy', dst_idx, src_ex2.flatten(), -DBz)

# Ez terms
src_ez1 = (I+1) * str_Ez[0] + J * str_Ez[1] + K
src_ez2 = I * str_Ez[0] + J * str_Ez[1] + K
add_vec_edge('Ez_to_Hy', dst_idx, src_ez1.flatten(), +DBx)
add_vec_edge('Ez_to_Hy', dst_idx, src_ez2.flatten(), -DBx)


# 6. Hz Update (Range: 0..nx, 0..ny, 0..nz+1)
range_hz = ((0, nx), (0, ny), (0, nz+1))
I, J, K, dst_idx = get_grid_and_idx(*range_hz, *str_Hz)

# Ey terms
src_ey1 = I * str_Ey[0] + J * str_Ey[1] + K
src_ey2 = (I+1) * str_Ey[0] + J * str_Ey[1] + K
add_vec_edge('Ey_to_Hz', dst_idx, src_ey1.flatten(), +DBx)
add_vec_edge('Ey_to_Hz', dst_idx, src_ey2.flatten(), -DBx)

# Ex terms
src_ex1 = I * str_Ex[0] + (J+1) * str_Ex[1] + K
src_ex2 = I * str_Ex[0] + J * str_Ex[1] + K
add_vec_edge('Ex_to_Hz', dst_idx, src_ex1.flatten(), +DBy)
add_vec_edge('Ex_to_Hz', dst_idx, src_ex2.flatten(), -DBy)

print(f"Graph Construction Complete. Time: {time.time() - t_build_start:.4f} sec")


# -------------------------
# 3. GNN Model & Solver
# -------------------------
class FDTD_GNN(torch.nn.Module):
    def __init__(self, edges_dict):
        super().__init__()
        self.ops = {}
        for key, lists in edges_dict.items():
            # Concat the list of tensors into one large tensor
            s = torch.cat(lists[0])
            d = torch.cat(lists[1])
            w = torch.cat(lists[2])
            self.ops[key] = (s, d, w)
            
    def update_field(self, target, source1, source2, key1, key2):
        if key1 in self.ops:
            s, d, w = self.ops[key1]
            target.index_add_(0, d, source1[s] * w)
        if key2 in self.ops:
            s, d, w = self.ops[key2]
            target.index_add_(0, d, source2[s] * w)

model = FDTD_GNN(edge_lists)

# Initialize Fields (Flattened 1D)
Ex = torch.zeros(NEx, dtype=dtype, device=device)
Ey = torch.zeros(NEy, dtype=dtype, device=device)
Ez = torch.zeros(NEz, dtype=dtype, device=device)
Hx = torch.zeros(NHx, dtype=dtype, device=device)
Hy = torch.zeros(NHy, dtype=dtype, device=device)
Hz = torch.zeros(NHz, dtype=dtype, device=device)

# Source
source_val = np.zeros(nmax + 1)
for n in range(1, nmax + 1):
    source_val[n] = srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / (tau**2)))
source_gpu = torch.tensor(source_val, dtype=dtype, device=device)
# Manual index for Ez source at [Is, Js, Ks]
# Ez strides: (Ny_Ez*Nz_Ez, Nz_Ez, 1) -> ( (ny+1)*nz, nz, 1 )
src_idx_flat = Is * ((ny+1)*nz) + Js * (nz) + Ks

# -------------------------
# 4. Time Stepping
# -------------------------
print("Starting Vectorized GNN Time Stepping...")
max_abs_error = 0.0
if device.type == 'cuda': torch.cuda.synchronize()
elif device.type == 'mps': torch.mps.synchronize()
t_start = time.time()

for n in range(1, nmax + 1):
    # E Updates
    model.update_field(Ex, Hz, Hy, 'Hz_Ex', 'Hy_Ex')
    model.update_field(Ey, Hx, Hz, 'Hx_Ey', 'Hz_Ey')
    model.update_field(Ez, Hy, Hx, 'Hy_Ez', 'Hx_Ez')
    
    # Source
    if n <= SOURCE_END_STEP:
        Ez[src_idx_flat] += source_gpu[n]
        
    # H Updates
    model.update_field(Hx, Ez, Ey, 'Ez_to_Hx', 'Ey_to_Hx')
    model.update_field(Hy, Ex, Ez, 'Ex_to_Hy', 'Ez_to_Hy')
    model.update_field(Hz, Ey, Ex, 'Ey_to_Hz', 'Ex_to_Hz')
    
    # Validation
    if Ex_ref_np is not None and n > SOURCE_END_STEP:
        idx_ref = n - SOURCE_END_STEP - 1
        
        # Reshape for check (only doing Ez for speed)
        Ez_3d = Ez.cpu().numpy().reshape(Nx_Ez, Ny_Ez, Nz_Ez)
        curr_max = np.max(np.abs(Ez_3d - Ez_ref_np[..., idx_ref]))
        max_abs_error = max(max_abs_error, curr_max)
        
        if n % 50 == 0:
            print(f"Step {n}: Max Err (Ez) = {curr_max:.3e}")

if device.type == 'cuda': torch.cuda.synchronize()
elif device.type == 'mps': torch.mps.synchronize()
t_end = time.time()

print(f"\nFinal Result:")
print(f"Total Time: {t_end - t_start:.4f} sec")
print(f"Max Absolute Error: {max_abs_error:.3e}")


# =============================================================================
# 5. VISUALIZATION (Plotting End Step)
# =============================================================================
print("\nGenerating Plots...")

# 1. Prepare Data
# Need to reshape 1D tensors back to 3D: (Nx, Ny, Nz)
k_slice = nz // 2  # Slice in the middle of Z-axis

# Reshape and extract slices (Move to CPU numpy first)
Hx_sim = Hx.cpu().numpy().reshape(Nx_Hx, Ny_Hx, Nz_Hx)[:, :, k_slice]
Hy_sim = Hy.cpu().numpy().reshape(Nx_Hy, Ny_Hy, Nz_Hy)[:, :, k_slice]
Ez_sim = Ez.cpu().numpy().reshape(Nx_Ez, Ny_Ez, Nz_Ez)[:, :, k_slice]

# Reference Data Index (Final Step)
compare_ref = (Ex_ref_np is not None)

if compare_ref:
    # Calculate which index in .mat corresponds to current 'nmax'
    ref_idx = nmax - SOURCE_END_STEP - 1
    # Check bounds
    if ref_idx >= Ez_ref_np.shape[-1]:
        print(f"Warning: nmax ({nmax}) exceeds reference data length. Using last frame.")
        ref_idx = Ez_ref_np.shape[-1] - 1
    
    Hx_gt = Hx_ref_np[:, :, k_slice, ref_idx]
    Hy_gt = Hy_ref_np[:, :, k_slice, ref_idx]
    Ez_gt = Ez_ref_np[:, :, k_slice, ref_idx]

# 2. Plotting
# Layout: 3 Rows (Hx, Hy, Ez), 2 Columns (Ground Truth, GEM) if ref exists
rows, cols = 3, (2 if compare_ref else 1)
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

# Helper function to plot heatmap
def plot_field(ax, data, title, cmap='jet'):
    # Transpose for correct x-y orientation in imshow (usually x is col, y is row)
    im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (index)')
    ax.set_ylabel('y (index)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

if compare_ref:
    # Row 1: Hx
    plot_field(axes[0, 0], Hx_gt, "Hx Ground Truth")
    plot_field(axes[0, 1], Hx_sim, "Hx GEM (Vec-GNN)")
    
    # Row 2: Hy
    plot_field(axes[1, 0], Hy_gt, "Hy Ground Truth")
    plot_field(axes[1, 1], Hy_sim, "Hy GEM (Vec-GNN)")
    
    # Row 3: Ez
    plot_field(axes[2, 0], Ez_gt, "Ez Ground Truth")
    plot_field(axes[2, 1], Ez_sim, "Ez GEM (Vec-GNN)")

else:
    # Single column mode (Handle 1D axes array if cols=1)
    ax_arr = axes if rows > 1 else [axes]
    plot_field(ax_arr[0], Hx_sim, "Hx GEM (Vec-GNN)")
    plot_field(ax_arr[1], Hy_sim, "Hy GEM (Vec-GNN)")
    plot_field(ax_arr[2], Ez_sim, "Ez GEM (Vec-GNN)")

plt.tight_layout()
plt.show()