#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import argparse


def fdtd_step(Ez, Hx, Hy, dx, dy, dt):
    Nx, Ny = Ez.shape

    # --- 更新 Hx ---
    # Hx[i,j] -= dt/dy * (Ez[i,j+1] - Ez[i,j])
    Hx[:, :-1] -= dt/dy * (Ez[:, 1:] - Ez[:, :-1])

    # --- 更新 Hy ---
    # Hy[i,j] += dt/dx * (Ez[i+1,j] - Ez[i,j])
    Hy[:-1, :] += dt/dx * (Ez[1:, :] - Ez[:-1, :])

    # --- 更新 Ez ---
    # Ez[i,j] += dt * [(Hy[i,j] - Hy[i-1,j]) / dx - (Hx[i,j] - Hx[i,j-1]) / dy]
    curl_H = np.zeros_like(Ez)

    # Hy contribution
    curl_H[1:, :] += (Hy[1:, :] - Hy[:-1, :]) / dx

    # Hx contribution
    curl_H[:, 1:] -= (Hx[:, 1:] - Hx[:, :-1]) / dy

    Ez += dt * curl_H

    return Ez, Hx, Hy


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, required=True)
    args = parser.parse_args()

    data = sio.loadmat(args.mat)
    Ez_gt = data["Ez"]      # (T, Nx, Ny)
    Hx_gt = data["Hx"]
    Hy_gt = data["Hy"]

    T, Nx, Ny = Ez_gt.shape
    print("Loaded GT:", Ez_gt.shape)

    # 初始条件 = GT[0]
    Ez = Ez_gt[0].copy()
    Hx = Hx_gt[0].copy()
    Hy = Hy_gt[0].copy()

    # dx/dy/dt（随便选，只要稳定）
    dx = dy = 1e-3
    c = 3e8
    dt = 0.99 * dx / (np.sqrt(2) * c)

    # time stepping
    for t in range(1, T):
        Ez, Hx, Hy = fdtd_step(Ez, Hx, Hy, dx, dy, dt)

        err = rel_err(Ez, Ez_gt[t])
        print(f"{t}, {err:.8e}")


if __name__ == "__main__":
    main()
