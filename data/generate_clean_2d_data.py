# -*- coding: utf-8 -*-
"""
Script 1: 2D FDTD Data Generation (Fixed)
Action: Generates 'FDTD_2D_TM_clean_data.mat' correctly.
"""

import numpy as np
import scipy.io
import os
import time
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as m0, epsilon_0 as e0, speed_of_light as c0

# --- 1. 参数配置 ---
floattype = np.float64
nx, ny = 100, 100
dx, dy = 1.0e-3, 1.0e-3

# 源位置 (中心)
Is, Js = nx // 2 - 1, ny // 2 - 1

# 时间设置 (nmax=500)
nmax = 500  
dt = 0.99 / (c0 * np.sqrt(1.0/dx**2 + 1.0/dy**2))

# 源参数
rtau = 50.0e-12
tau = rtau / dt
ndelay = 3 * tau
srcconst = -dt * (3.0e11)
SOURCE_END_STEP = int(ndelay + 4 * tau) 
if SOURCE_END_STEP < 80: SOURCE_END_STEP = 80

print(f"Generating {nmax} steps. Clean data from step {SOURCE_END_STEP+1}...")

# --- 2. 初始化场与系数 ---
Ez = np.zeros((nx, ny), dtype=floattype)
Hx = np.zeros((nx, ny), dtype=floattype)
Hy = np.zeros((nx, ny), dtype=floattype)

CA_Ez = 1.0; CB_Ez = dt / e0
DA = 1.0;    DB = dt / m0
val_DB_dy = DB / dy; val_DB_dx = DB / dx
val_CB_dx = CB_Ez / dx; val_CB_dy = CB_Ez / dy

# 预计算源
source = np.array([srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / (tau**2))) 
                   for n in range(nmax + 1)])

# 存储容器
n_save = nmax - SOURCE_END_STEP
Ez_save = np.zeros((nx, ny, n_save), dtype=floattype)
Hx_save = np.zeros((nx, ny, n_save), dtype=floattype)
Hy_save = np.zeros((nx, ny, n_save), dtype=floattype)

# --- 3. FDTD 主循环 ---
print("Running FDTD simulation...")
t_start = time.time()

for n in range(1, nmax + 1):
    # Update Hx: depends on dEz/dy (j direction)
    Hx[:, :-1] -= val_DB_dy * (Ez[:, 1:] - Ez[:, :-1])
    
    # Update Hy: depends on dEz/dx (i direction)
    Hy[:-1, :] += val_DB_dx * (Ez[1:, :] - Ez[:-1, :])
    
    # Update Ez: depends on dHy/dx - dHx/dy
    Ez[1:, 1:] += val_CB_dx * (Hy[1:, 1:] - Hy[:-1, 1:]) - \
                  val_CB_dy * (Hx[1:, 1:] - Hx[1:, :-1])
    
    # Add Source
    if n <= SOURCE_END_STEP:
        Ez[Is, Js] += CB_Ez * source[n]
    
    # Save Data
    if n > SOURCE_END_STEP:
        idx = n - SOURCE_END_STEP - 1
        Ez_save[:, :, idx] = Ez
        Hx_save[:, :, idx] = Hx
        Hy_save[:, :, idx] = Hy
        
    if n % 50 == 0:
        print(f"Step {n}/{nmax} done.")

print(f"Simulation finished in {time.time()-t_start:.2f}s")

# --- 4. 先保存文件 (关键修改) ---
output_dir = './data'
if not os.path.exists(output_dir): os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'FDTD_2D_TM_clean_data.mat')

print(f"Saving to {output_file}...")
scipy.io.savemat(output_file, {
    'Ez': Ez_save, 'Hx': Hx_save, 'Hy': Hy_save,
    't': np.arange(SOURCE_END_STEP + 1, nmax + 1) * dt,
    'dx': dx, 'dy': dy, 'dt': dt
})
print("Save successful!")

# --- 5. 最后再画图 ---
plt.figure(figsize=(5,4))
plt.imshow(Ez.T, cmap='jet', origin='lower')
plt.colorbar(label='Ez')
plt.title(f'Ground Truth at Step {nmax}')
plt.show()