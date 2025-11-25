#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from scipy.constants import mu_0 as m0, epsilon_0 as e0, speed_of_light as c0
import time

dx = 1.0e-3
dy = 1.0e-3
dt = 0.99 / (c0 * (1/dx**2 + 1/dy**2)**0.5)

nx = 100
ny = 100

floattype = np.float64

mat = loadmat("data/data_FDTD_2D_cavity/2D_TM_all_processed_jiequ_200_300.mat")

Ez_gt = mat["Ez"]       # shape = (nx+1, ny+1, T)
Hx_gt = mat["Hx"]       # shape = (nx+1, ny,   T)
Hy_gt = mat["Hy"]       # shape = (nx,   ny+1, T)

T = Ez_gt.shape[2]
nsteps = T - 1  # use full GT sequence

#t = mat["t"].ravel()
#dt = t[1] - t[0]

Ez = Ez_gt[:, :, 0].copy()
Hx = Hx_gt[:, :, 0].copy()
Hy = Hy_gt[:, :, 0].copy()

eps = np.ones((nx+1, ny+1)) * e0
sig = np.zeros((nx+1, ny+1))

CA_Ez = (1 - sig * dt / (2 * eps)) / (1 + sig * dt / (2 * eps))
CB_Ez = (dt / eps) / (1 + sig * dt / (2 * eps))

DA = 1.0
DB = dt / m0

den_hx = np.ones(nx) / dy
den_hy = np.ones(ny) / dy
den_ex = np.ones(nx) / dx
den_ey = np.ones(ny) / dy

def rel_l2(pred, gt):
    num = np.linalg.norm(pred - gt)
    den = np.linalg.norm(gt)
    return num / (den + 1e-12)

print("Running FDTD TM without source...")
print("Using GT[t=0] as initial condition")
print("Total steps:", nsteps)

for n in range(1, nsteps) :
    t0 = time.time()

    # ---- update Hx, Hy ----
    for i in range(0, nx):
        for j in range(0, ny):
            Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j+1] - Ez[i, j]) * den_hy[j]
            Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i+1, j] - Ez[i, j]) * den_hx[i]

    # ---- update Ez ----
    for i in range(1, nx):
        for j in range(1, ny):
            Ez[i, j] = CA_Ez[i, j] * Ez[i, j] + CB_Ez[i, j] * (
                (Hy[i, j] - Hy[i-1, j]) * den_ex[i] -
                (Hx[i, j] - Hx[i, j-1]) * den_ey[j]
            )

    # ---- compare with GT ----
    Ez_err = rel_l2(Ez, Ez_gt[:, :, n])
    Hx_err = rel_l2(Hx, Hx_gt[:, :, n])
    Hy_err = rel_l2(Hy, Hy_gt[:, :, n])

    elapsed = time.time() - t0

    print(f"step={n:04d} | Ez_err={Ez_err:.6e} | Hx_err={Hx_err:.6e} | Hy_err={Hy_err:.6e} | dt={elapsed:.4f}s")

print("Done.")