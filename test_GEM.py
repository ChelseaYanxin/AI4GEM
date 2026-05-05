#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original GNN-based FDTD (Loop-based Graph Construction).
Used for performance comparison against Vectorized GEM.
"""

import os
import numpy as np
import scipy.io
import torch
import time
from scipy.constants import mu_0 as m0, epsilon_0 as e0

# -------------------------
# 1. Configuration & Constants
# -------------------------
# Default to float64 for CPU/CUDA
dtype = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Note: We explicitly avoid MPS here to ensure float64 precision.
# MPS currently only supports float32, which causes significant numerical error in FDTD.
print(f"Running Loop-based GNN-FDTD on: {device}")

nx, ny, nz = 20, 20, 20
dx, dy, dz = 2e-3, 2e-3, 2e-3
nmax = 500

# Constants
dt = dx / (2.0 * 2.99792458e8)
Is, Js, Ks = 9, 9, 9

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
# 2. Graph Topology Construction (Loops)
# -------------------------
print("Building Graph Topology (Loops)...")
t_build_start = time.time()

# Helper: Linear Index Mapping
def idx_Ex(i, j, k): return i*(ny+1)*(nz+1) + j*(nz+1) + k
def idx_Ey(i, j, k): return i*ny*(nz+1) + j*(nz+1) + k
def idx_Ez(i, j, k): return i*(ny+1)*nz + j*nz + k
def idx_Hx(i, j, k): return i*ny*nz + j*nz + k
def idx_Hy(i, j, k): return i*(ny+1)*nz + j*nz + k
def idx_Hz(i, j, k): return i*ny*(nz+1) + j*(nz+1) + k

# Define total nodes
NEx = nx * (ny + 1) * (nz + 1)
NEy = (nx + 1) * ny * (nz + 1)
NEz = (nx + 1) * (ny + 1) * nz
NHx = (nx + 1) * ny * nz
NHy = nx * (ny + 1) * nz
NHz = nx * ny * (nz + 1)

edge_lists = {}

def add_edge_group(name, iter_ranges, target_idx_fn, src_idx_fn, i_off, j_off, k_off, weight):
    s_list, d_list, w_list = [], [], []
    xr, yr, zr = iter_ranges
    for i in range(xr[0], xr[1]):
        for j in range(yr[0], yr[1]):
            for k in range(zr[0], zr[1]):
                dst = target_idx_fn(i, j, k)
                src = src_idx_fn(i + i_off, j + j_off, k + k_off)
                s_list.append(src)
                d_list.append(dst)
                w_list.append(weight)
    
    if name not in edge_lists: edge_lists[name] = [[], [], []]
    edge_lists[name][0].extend(s_list)
    edge_lists[name][1].extend(d_list)
    edge_lists[name][2].extend(w_list)

# --- Ex Edges ---
r_ex = [(0, nx), (1, ny), (1, nz)]
add_edge_group('Hz_Ex', r_ex, idx_Ex, idx_Hz, 0, 0, 0, +CBy)
add_edge_group('Hz_Ex', r_ex, idx_Ex, idx_Hz, 0, -1, 0, -CBy)
add_edge_group('Hy_Ex', r_ex, idx_Ex, idx_Hy, 0, 0, 0, -CBz)
add_edge_group('Hy_Ex', r_ex, idx_Ex, idx_Hy, 0, 0, -1, +CBz)

# --- Ey Edges ---
r_ey = [(1, nx), (0, ny), (1, nz)]
add_edge_group('Hx_Ey', r_ey, idx_Ey, idx_Hx, 0, 0, 0, +CBz)
add_edge_group('Hx_Ey', r_ey, idx_Ey, idx_Hx, 0, 0, -1, -CBz)
add_edge_group('Hz_Ey', r_ey, idx_Ey, idx_Hz, 0, 0, 0, -CBx)
add_edge_group('Hz_Ey', r_ey, idx_Ey, idx_Hz, -1, 0, 0, +CBx)

# --- Ez Edges ---
r_ez = [(1, nx), (1, ny), (0, nz)]
add_edge_group('Hy_Ez', r_ez, idx_Ez, idx_Hy, 0, 0, 0, +CBx)
add_edge_group('Hy_Ez', r_ez, idx_Ez, idx_Hy, -1, 0, 0, -CBx)
add_edge_group('Hx_Ez', r_ez, idx_Ez, idx_Hx, 0, 0, 0, -CBy)
add_edge_group('Hx_Ez', r_ez, idx_Ez, idx_Hx, 0, -1, 0, +CBy)

# --- Hx Edges ---
sx, dx, wx = [], [], []; sy, dy, wy = [], [], []
for i in range(nx+1):
    for j in range(ny):
        for k in range(nz):
            d = idx_Hx(i,j,k)
            sx.append(idx_Ez(i,j,k)); dx.append(d); wx.append(DBy)
            if j+1<=ny: sx.append(idx_Ez(i,j+1,k)); dx.append(d); wx.append(-DBy)
            sy.append(idx_Ey(i,j,k)); dy.append(d); wy.append(-DBz)
            if k+1<=nz: sy.append(idx_Ey(i,j,k+1)); dy.append(d); wy.append(DBz)
edge_lists['Ez_to_Hx'] = (sx, dx, wx); edge_lists['Ey_to_Hx'] = (sy, dy, wy)

# --- Hy Edges ---
sx, dx, wx = [], [], []; sy, dy, wy = [], [], []
for i in range(nx):
    for j in range(ny+1):
        for k in range(nz):
            d = idx_Hy(i,j,k)
            sx.append(idx_Ex(i,j,k)); dx.append(d); wx.append(DBz)
            if k+1<=nz: sx.append(idx_Ex(i,j,k+1)); dx.append(d); wx.append(-DBz)
            sy.append(idx_Ez(i,j,k)); dy.append(d); wy.append(-DBx)
            if i+1<=nx: sy.append(idx_Ez(i+1,j,k)); dy.append(d); wy.append(DBx)
edge_lists['Ex_to_Hy'] = (sx, dx, wx); edge_lists['Ez_to_Hy'] = (sy, dy, wy)

# --- Hz Edges ---
sx, dx, wx = [], [], []; sy, dy, wy = [], [], []
for i in range(nx):
    for j in range(ny):
        for k in range(nz+1):
            d = idx_Hz(i,j,k)
            sx.append(idx_Ey(i,j,k)); dx.append(d); wx.append(DBx)
            if i+1 <= nx: sx.append(idx_Ey(i+1,j,k)); dx.append(d); wx.append(-DBx)
            sy.append(idx_Ex(i,j,k)); dy.append(d); wy.append(-DBy)
            if j+1 <= ny: sy.append(idx_Ex(i,j+1,k)); dy.append(d); wy.append(DBy)
edge_lists['Ey_to_Hz'] = (sx, dx, wx); edge_lists['Ex_to_Hz'] = (sy, dy, wy)

print(f"Graph Construction Complete. Time: {time.time() - t_build_start:.4f} sec")

# -------------------------
# 3. GNN Model
# -------------------------
class FDTD_GNN(torch.nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.ops = {}
        for key, (s, d, w) in edges.items():
            self.ops[key] = (
                torch.tensor(s, dtype=torch.long, device=device),
                torch.tensor(d, dtype=torch.long, device=device),
                torch.tensor(w, dtype=dtype, device=device)
            )
            
    def update_field(self, target, source1, source2, key1, key2):
        if key1 in self.ops:
            s, d, w = self.ops[key1]
            target.index_add_(0, d, source1[s] * w)
        if key2 in self.ops:
            s, d, w = self.ops[key2]
            target.index_add_(0, d, source2[s] * w)

model = FDTD_GNN(edge_lists)

# Initialize Fields
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
src_idx_flat = idx_Ez(Is, Js, Ks)

# -------------------------
# 4. Time Loop
# -------------------------
print("Starting GNN Time Stepping...")
max_abs_error = 0.0
if device.type == 'cuda': torch.cuda.synchronize()
# Only sync MPS if we were using it, but we are skipping it now.
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
    
    # Comparison
    if Ex_ref_np is not None and n > SOURCE_END_STEP:
        idx_ref = n - SOURCE_END_STEP - 1
        Ez_3d = Ez.cpu().numpy().reshape(nx+1, ny+1, nz)
        curr_max = np.max(np.abs(Ez_3d - Ez_ref_np[..., idx_ref]))
        max_abs_error = max(max_abs_error, curr_max)
        if n % 50 == 0:
            print(f"Step {n}: Max Err (Ez) = {curr_max:.3e}")

if device.type == 'cuda': torch.cuda.synchronize()
t_end = time.time()

print(f"\nFinal Result:")
print(f"Total Time: {t_end - t_start:.4f} sec")
print(f"Max Absolute Error: {max_abs_error:.3e}")