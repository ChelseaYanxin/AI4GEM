#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conv3D-based NeCLO.
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
print("Device:", device)

nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3

Is, Js, Ks = 9, 9, 9  # 10-1

nmax = 500
c0 = 2.99792458e8
dt = dx / (2.0 * c0)

rtau = 50.0e-12
tau = rtau / dt
ndelay = 3 * tau
srcconst = -dt * (3.0e11)
SOURCE_END_STEP = 83

mat_file = '/Users/zyanxin/Documents/code/GEM_N/data/FDTD_3D_cavity_clean_data.mat'
if not os.path.exists(mat_file):
    print(f"Warning: {mat_file} not found. Visualization will only show simulation results.")
    compare_ref = False
else:
    compare_ref = True

er, se, mr, sm = 1.0, 0.0, 1.0, 0.0
EA = (e0 * er / dt) + 0.5 * se
EB = (e0 * er / dt) - 0.5 * se
CA = EB / EA
CBx = (1.0 / dx) / EA
CBy = (1.0 / dy) / EA
CBz = (1.0 / dz) / EA

HA = (m0 * mr / dt) + 0.5 * sm
HB = (m0 * mr / dt) - 0.5 * sm
DA = HB / HA
DBx = (1.0 / dx) / HA
DBy = (1.0 / dy) / HA
DBz = (1.0 / dz) / HA

Ex = torch.zeros((1,1,nx, ny+1, nz+1), dtype=dtype_torch, device=device)
Ey = torch.zeros((1,1,nx+1, ny, nz+1), dtype=dtype_torch, device=device)
Ez = torch.zeros((1,1,nx+1, ny+1, nz), dtype=dtype_torch, device=device)
Hx = torch.zeros((1,1,nx+1, ny, nz), dtype=dtype_torch, device=device)
Hy = torch.zeros((1,1,nx, ny+1, nz), dtype=dtype_torch, device=device)
Hz = torch.zeros((1,1,nx, ny, nz+1), dtype=dtype_torch, device=device)


def get_diff_kernel(axis):
    if axis == 'x':
        k = torch.zeros((1,1,2,1,1), dtype=dtype_torch, device=device)
        k[0,0,0,0,0] = -1.0
        k[0,0,1,0,0] = +1.0
        return k
    if axis == 'y':
        k = torch.zeros((1,1,1,2,1), dtype=dtype_torch, device=device)
        k[0,0,0,0,0] = -1.0
        k[0,0,0,1,0] = +1.0
        return k
    if axis == 'z':
        k = torch.zeros((1,1,1,1,2), dtype=dtype_torch, device=device)
        k[0,0,0,0,0] = -1.0
        k[0,0,0,0,1] = +1.0
        return k

k_dx = get_diff_kernel('x')
k_dy = get_diff_kernel('y')
k_dz = get_diff_kernel('z')

# Source
source_np = np.zeros(nmax + 1, dtype=np.float64)
for n in range(1, nmax + 1):
    source_np[n] = srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / (tau**2)))
source = torch.from_numpy(source_np).to(device=device, dtype=dtype_torch)

if compare_ref:
    mat = scipy.io.loadmat(mat_file)
    Ex_ref_np = mat['Ex']
    Ey_ref_np = mat['Ey']
    Ez_ref_np = mat['Ez']
    Hx_ref_np = mat['Hx']
    Hy_ref_np = mat['Hy']
    Hz_ref_np = mat['Hz']

slice_Ex = (slice(None), slice(None), slice(0,nx), slice(1,ny), slice(1,nz))
slice_Ey = (slice(None), slice(None), slice(1,nx), slice(0,ny), slice(1,nz))
slice_Ez = (slice(None), slice(None), slice(1,nx), slice(1,ny), slice(0,nz))

slice_Hx = (slice(None), slice(None), slice(0,nx+1), slice(0,ny), slice(0,nz))
slice_Hy = (slice(None), slice(None), slice(0,nx), slice(0,ny+1), slice(0,nz))
slice_Hz = (slice(None), slice(None), slice(0,nx), slice(0,ny), slice(0,nz+1))

max_abs_error = -1.0

for n in range(1, nmax + 1):
    
    # --- E Updates ---
    dHz_dy = F.conv3d(Hz, k_dy) 
    dHy_dz = F.conv3d(Hy, k_dz)
    
    Ex_interior = Ex[slice_Ex]
    term_hz = dHz_dy[:, :, 0:nx, :, 1:nz] 
    term_hy = dHy_dz[:, :, 0:nx, 1:ny, :]
    Ex[slice_Ex] = CA * Ex_interior + (CBy * term_hz) + (-CBz * term_hy)

    dHx_dz = F.conv3d(Hx, k_dz)
    dHz_dx = F.conv3d(Hz, k_dx)
    
    Ey_interior = Ey[slice_Ey]
    term_hx = dHx_dz[:, :, 1:nx, 0:ny, :]
    term_hz2 = dHz_dx[:, :, :, 0:ny, 1:nz]
    Ey[slice_Ey] = CA * Ey_interior + (CBz * term_hx) + (-CBx * term_hz2)

    dHy_dx = F.conv3d(Hy, k_dx)
    dHx_dy = F.conv3d(Hx, k_dy)
    
    Ez_interior = Ez[slice_Ez]
    term_hy2 = dHy_dx[:, :, :, 1:ny, 0:nz]
    term_hx2 = dHx_dy[:, :, 1:nx, :, 0:nz]
    Ez[slice_Ez] = CA * Ez_interior + (CBx * term_hy2) + (-CBy * term_hx2)

    # Source
    if n <= SOURCE_END_STEP:
        Ez[:, :, Is, Js, Ks] += source[n]

    # --- H Updates ---
    dEz_dy = F.conv3d(Ez, k_dy)
    dEy_dz = F.conv3d(Ey, k_dz)
    Hx[slice_Hx] = DA * Hx[slice_Hx] + (-DBy * dEz_dy) + (DBz * dEy_dz)

    dEx_dz = F.conv3d(Ex, k_dz)
    dEz_dx = F.conv3d(Ez, k_dx)
    Hy[slice_Hy] = DA * Hy[slice_Hy] + (-DBz * dEx_dz) + (DBx * dEz_dx)

    dEy_dx = F.conv3d(Ey, k_dx)
    dEx_dy = F.conv3d(Ex, k_dy)
    Hz[slice_Hz] = DA * Hz[slice_Hz] + (-DBx * dEy_dx) + (DBy * dEx_dy)

    # --- Compare Error ---
    if compare_ref and n > SOURCE_END_STEP:
        idx = n - SOURCE_END_STEP - 1
        # Safety check for index bound
        if idx < Ez_ref_np.shape[-1]:
            Ez_np = Ez.cpu().numpy().reshape(nx+1, ny+1, nz)
            err = np.max(np.abs(Ez_np - Ez_ref_np[:,:,:,idx])) 
            if err > max_abs_error: max_abs_error = err
        
        if n % 50 == 0:
            print(f"Step {n}/{nmax}")

print(f"\nFinal Max Error: {max_abs_error:.20e}")
if max_abs_error < 1e-13:
    print("SUCCESS: Error is negligible.")
else:
    print("WARNING: Error is still high.")

print("Generating Plots...")

k_slice = nz // 2

# Hx shape: (1,1, nx+1, ny, nz) -> (nx+1, ny) slice    
Hx_sim = Hx.cpu().numpy().squeeze()[:, :, k_slice]
# Hy shape: (1,1, nx, ny+1, nz) -> (nx, ny+1) slice
Hy_sim = Hy.cpu().numpy().squeeze()[:, :, k_slice]
# Ez shape: (1,1, nx+1, ny+1, nz) -> (nx+1, ny+1) slice    
Ez_sim = Ez.cpu().numpy().squeeze()[:, :, k_slice]

if compare_ref:
    # Calculate which index in .mat corresponds to current 'nmax'
    ref_idx = nmax - SOURCE_END_STEP - 1
    # Check if index is valid
    if ref_idx >= Ez_ref_np.shape[-1]:
        print(f"Warning: nmax ({nmax}) exceeds reference data length. Using last frame.")
        ref_idx = Ez_ref_np.shape[-1] - 1
    
    Hx_gt = Hx_ref_np[:, :, k_slice, ref_idx]
    Hy_gt = Hy_ref_np[:, :, k_slice, ref_idx]
    Ez_gt = Ez_ref_np[:, :, k_slice, ref_idx]

#rows, cols = 3, (2 if compare_ref else 1)
#fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
rows, cols = 1, (2 if compare_ref else 1)
fig, axes = plt.subplots(rows, cols, figsize=(8, 4))

# Helper function to plot heatmap
def plot_field(ax, data, title, cmap='jet'):
    im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (index)')
    ax.set_ylabel('y (index)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

if compare_ref:
    # Row 1: Hx
    plot_field(axes[0, 0], Hx_gt, "Hx Ground Truth")
    plot_field(axes[0, 1], Hx_sim, "Hx NeCLO")
    
    # Row 2: Hy
    plot_field(axes[1, 0], Hy_gt, "Hy Ground Truth")
    plot_field(axes[1, 1], Hy_sim, "Hy NeCLO")
    
    # Row 3: Ez
    plot_field(axes[2, 0], Ez_gt, "Ez Ground Truth")
    plot_field(axes[2, 1], Ez_sim, "Ez NeCLO")

else:
    plot_field(axes[0], Hx_sim, "Hx NeCLO")
    plot_field(axes[1], Hy_sim, "Hy NeCLO")
    plot_field(axes[2], Ez_sim, "Ez NeCLO")

plt.tight_layout()
plt.show()