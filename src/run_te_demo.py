from __future__ import annotations
import numpy as np
import torch
from gem import Grid, GEMTE
import matplotlib.pyplot as plt

# Simple demo: free-space Gaussian pulse in TE 2D

def main():
    nx, nz = 200, 200
    dx = dz = 1e-3  # 1 mm
    c0 = 299_792_458.0
    eps0 = 8.854187817e-12
    mu0 = 4 * np.pi * 1e-7

    # Courant dt for 2D FDTD (safe factor 0.95)
    dt = 0.95 / (c0 * np.sqrt((1/dx**2) + (1/dz**2)))

    grid = Grid.homogeneous(nx, nz, dx, dz, eps=eps0, mu=mu0, sigma=0.0)

    # Torch tensors with [N,C,H,W] layout
    device = torch.device('cpu')
    # Convert arrays to torch tensors via torch.tensor (safe copy) to avoid ABI/type issues
    Ey = torch.tensor(np.asarray(grid.Ey, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    Hx = torch.tensor(np.asarray(grid.Hx, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    Hz = torch.tensor(np.asarray(grid.Hz, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    eps_t = torch.tensor(np.asarray(grid.eps, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    mu_t = torch.tensor(np.asarray(grid.mu, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sigma_t = torch.tensor(np.asarray(grid.sigma, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    model = GEMTE(dx=dx, dz=dz, dt=dt, eps=eps_t, mu=mu_t, sigma=sigma_t).to(device)

    # Source: Gaussian at center, added to Ey each step
    cx, cz = nx//2, nz//2
    t0 = 30
    spread = 10

    steps = 200
    snapshots = []
    for n in range(steps):
        # add source
        gauss = float(np.exp(-0.5 * ((n - t0)/spread)**2))
        Ey[0,0,cx,cz] += torch.tensor(gauss, dtype=Ey.dtype, device=device)
        Ey, Hx, Hz = model.step(Ey, Hx, Hz)
        if n % 20 == 0:
            snapshots.append(Ey[0,0].cpu().numpy().copy())

    # Plot final snapshot
    plt.figure(figsize=(5,4))
    plt.imshow(Ey[0,0].cpu().numpy().T, origin='lower', cmap='RdBu',
               extent=[0, nx*dx*1e3, 0, nz*dz*1e3])
    plt.colorbar(label='Ey (a.u.)')
    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    plt.title('GEM TE Ey snapshot')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
