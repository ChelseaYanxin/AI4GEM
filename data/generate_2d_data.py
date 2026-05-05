# -*- coding: utf-8 -*-
"""
FDTD 2D TM Mode - Ground Truth Generator (Legacy Loops)
Generates 'Clean Data' for NeCLO training.

Logic:
1. Run simulation with Source until Source decays to zero.
2. Save the field at that moment as 'Initial State'.
3. Continue running (without source) to record pure propagation data.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time
from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0
from scipy.constants import speed_of_light as c0

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
# 精度设置
floattype = np.float64

# 网格设置
nx = 100
ny = 100
dx = 1.0e-3
dy = 1.0e-3

# 时间步设置 (CFL Condition)
dt = 0.99 / (c0 * np.sqrt(1.0/dx**2 + 1.0/dy**2))
nmax = 2000  # 总模拟步数

# 源位置 (中心)
Is = int(round(nx/2 - 1))
Js = int(round(ny/2 - 1))

# 源参数 (高斯导数脉冲)
rtau = 50.0e-12
tau = rtau / dt
ndelay = 3 * tau
srcconst = -dt * (3.0e11)

# 输出文件路径
OUTPUT_FILE = "FDTD_2D_TM_clean_data.mat"

# ==========================================
# 2. 预计算源并确定截断点 (Source Calculation)
# ==========================================
print("--- Pre-calculating Source ---")
source = np.zeros(nmax + 1, dtype=floattype)
source_end_step = 0
threshold = 1e-20  # 认为源衰减为0的阈值

for n in range(1, nmax + 1):
    # 使用你指定的公式
    source[n] = srcconst * (n - ndelay) * (np.exp(-((n - ndelay)**2 / (tau**2))))
    
    # 自动检测源何时结束 (当过了峰值且数值极小时)
    if n > ndelay * 1.5 and abs(source[n]) < threshold and source_end_step == 0:
        source_end_step = n

# 如果没检测到（说明nmax太短），强制设为 ndelay * 2.5
if source_end_step == 0:
    source_end_step = int(ndelay * 2.5)

print(f"Total Steps: {nmax}")
print(f"Source Peak Step: ~{int(ndelay)}")
print(f"Source End Step (Clean Data Start): {source_end_step}")

# 绘制源波形供确认
plt.figure(figsize=(10, 4))
plt.plot(source[:source_end_step + 50])
plt.axvline(source_end_step, color='r', linestyle='--', label='Cutoff (Clean Data Start)')
plt.title("Source Waveform Check")
plt.legend()
plt.show()

# ==========================================
# 3. 场初始化 (Field Initialization)
# ==========================================
# TM Mode: Ez, Hx, Hy
Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)
Hx = np.zeros((nx + 1, ny), dtype=floattype)
Hy = np.zeros((nx, ny + 1), dtype=floattype)

# 系数 (真空/空气)
# CA = 1.0, CB = dt/eps0
CA_Ez = 1.0
CB_Ez = dt / e0
DA = 1.0
DB = dt / m0

# 空间步长倒数 (用于差分)
den_dx = 1.0 / dx
den_dy = 1.0 / dy

# 数据记录容器 (只记录 Clean Data 阶段)
# 记录从 source_end_step 开始到 nmax 的数据
record_steps = nmax - source_end_step
saved_Ez = np.zeros((nx + 1, ny + 1, record_steps), dtype=floattype)
saved_Hx = np.zeros((nx + 1, ny, record_steps), dtype=floattype)
saved_Hy = np.zeros((nx, ny + 1, record_steps), dtype=floattype)

print(f"\n--- Starting FDTD Simulation (Legacy Loops) ---")
print(f"WARNING: Pure Python loops are slow. This might take a while.")

# ==========================================
# 4. 主循环 (Three-Layer Loop FDTD)
# ==========================================
start_time = time.time()

for n in range(1, nmax + 1):
    
    # --- Update H (Magnetic Field) ---
    # Loop over x
    for i in range(0, nx + 1):
        for j in range(0, ny):
            # Hx 更新: 依赖 dEz/dy
            # Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j+1] - Ez[i, j]) / dy
            # 注意边界: Ez的j索引覆盖 0~ny
            dEz_dy = (Ez[i, j + 1] - Ez[i, j]) * den_dy
            Hx[i, j] = DA * Hx[i, j] - DB * dEz_dy

    # Loop over y
    for i in range(0, nx):
        for j in range(0, ny + 1):
            # Hy 更新: 依赖 dEz/dx
            # Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i+1, j] - Ez[i, j]) / dx
            dEz_dx = (Ez[i + 1, j] - Ez[i, j]) * den_dx
            Hy[i, j] = DA * Hy[i, j] + DB * dEz_dx

    # --- Update E (Electric Field) ---
    for i in range(1, nx):
        for j in range(1, ny):
            # Ez 更新: 依赖 dHy/dx - dHx/dy
            # Curl H: (Hy[i, j] - Hy[i-1, j])/dx - (Hx[i, j] - Hx[i, j-1])/dy
            curl_H = (Hy[i, j] - Hy[i - 1, j]) * den_dx - \
                     (Hx[i, j] - Hx[i, j - 1]) * den_dy
            
            Ez[i, j] = CA_Ez * Ez[i, j] + CB_Ez * curl_H

    # --- Source Injection (Soft Source) ---
    # 只有在 source_end_step 之前才加源
    if n <= source_end_step:
        Ez[Is, Js] = Ez[Is, Js] + CB_Ez * source[n]

    # --- Data Recording (Clean Data Only) ---
    # 当 n > source_end_step 时，源已经是0了，记录此时的波传播
    if n > source_end_step:
        idx = n - source_end_step - 1
        saved_Ez[:, :, idx] = Ez
        saved_Hx[:, :, idx] = Hx
        saved_Hy[:, :, idx] = Hy

    # --- Progress Print ---
    if n % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Step {n}/{nmax} completed. Time elapsed: {elapsed:.2f}s")
        if n == source_end_step:
            print(f"--> Source finished. Recording clean data from now on...")

# ==========================================
# 5. 保存数据 (Save to .mat)
# ==========================================
print("\nSaving data to .mat file...")

data_dict = {
    'Ez': saved_Ez,
    'Hx': saved_Hx,
    'Hy': saved_Hy,
    'dx': dx,
    'dy': dy,
    'dt': dt,
    'nx': nx,
    'ny': ny,
    'source_end_step': source_end_step,
    'description': '2D TM FDTD clean data (source removed). Contains steps AFTER source_end_step.'
}

scipy.io.savemat(OUTPUT_FILE, data_dict)
print(f"Done! Data saved to {OUTPUT_FILE}")
print(f"Recorded shape: {saved_Ez.shape}")

# ==========================================
# 6. 简单的可视化验证 (Visualization)
# ==========================================
# 画最后一张图看看
plt.figure()
plt.imshow(saved_Ez[:, :, -1].T, cmap='jet', origin='lower')
plt.colorbar()
plt.title(f"Ez Field at Step {nmax} (Clean Propagation)")
plt.show()