# run_fdtd.py
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from gemte_minimal import GEMTE2D

def rel_err(a, b):
    eps = 1e-12
    return np.mean(np.abs(a - b) / (np.abs(b) + eps))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, required=True)
    args = parser.parse_args()

    print("Loading GT:", args.mat)
    mat = sio.loadmat(args.mat)

    Ez_gt_all = mat["Ez"]      # shape: (steps, nx+1, ny+1)
    Hx_gt_all = mat["Hx"]
    Hy_gt_all = mat["Hy"]

    nsteps = Ez_gt_all.shape[0]
    nx = Ez_gt_all.shape[1] - 1
    ny = Ez_gt_all.shape[2] - 1

    dx = float(mat["dx"])
    dy = float(mat["dy"])
    dt = float(mat["dt"])

    eps_map = mat["eps"]
    mu_map = mat["mu"]
    sigma_map = mat["sigma"]

    # Initialize fields (zeros, same as GT)
    Ez = np.zeros((nx + 1, ny + 1))
    Hx = np.zeros((nx + 1, ny))
    Hy = np.zeros((nx, ny + 1))

    model = GEMTE2D(dx, dy, dt, eps_map, mu_map, sigma_map)

    rel_list = []

    for step in range(nsteps):
        Ez, Hx, Hy = model.step(Ez, Hx, Hy)

        # Ground truth
        Ez_gt = Ez_gt_all[step]
        Hx_gt = Hx_gt_all[step]
        Hy_gt = Hy_gt_all[step]

        # relative error
        err = (
            rel_err(Ez, Ez_gt) +
            rel_err(Hx, Hx_gt) +
            rel_err(Hy, Hy_gt)
        ) / 3.0

        rel_list.append(err)
        print(f"[Step {step+1}/{nsteps}] Relative error = {err:.3e}")

    # plot error curve
    plt.figure()
    plt.plot(rel_list)
    plt.title("Per-step Relative Error")
    plt.xlabel("Step")
    plt.ylabel("Relative Error")
    plt.grid()
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
