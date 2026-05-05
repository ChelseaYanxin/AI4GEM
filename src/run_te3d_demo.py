from __future__ import annotations
import math
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from gem.gem_te_3d import GEMTE3D
from gem import GEMTEGraph3D


def main():
    parser = argparse.ArgumentParser(description='3D Yee-style demo with simple source and slices')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--dx', type=float, default=1e-3)
    parser.add_argument('--dy', type=float, default=1e-3)
    parser.add_argument('--dz', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--source', type=str, choices=['gauss','cw'], default='gauss')
    parser.add_argument('--amp', type=float, default=5.0)
    parser.add_argument('--t0', type=float, default=40.0)
    parser.add_argument('--spread', type=float, default=15.0)
    parser.add_argument('--freq-ghz', type=float, default=5.0)
    parser.add_argument('--n-pml', type=int, default=8)
    parser.add_argument('--sigma-max', type=float, default=5e7)
    parser.add_argument('--no-crop', action='store_true')
    parser.add_argument('--q', type=float, default=0.995)
    parser.add_argument('--backend', type=str, choices=['dense','gnn'], default='dense')
    args, _ = parser.parse_known_args()

    # Small cubic grid
    nx, ny, nz = args.nx, args.ny, args.nz
    dx, dy, dz = args.dx, args.dy, args.dz
    c0 = 299_792_458.0
    eps0 = 8.854187817e-12
    mu0 = 4 * math.pi * 1e-7

    # 3D CFL with tighter safety factor
    dt = 0.5 / (c0 * math.sqrt((1/dx**2) + (1/dy**2) + (1/dz**2)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = (1, 1, nx, ny, nz)

    # Fields
    Ex = torch.zeros(shape, dtype=torch.float32, device=device)
    Ey = torch.zeros_like(Ex)
    Ez = torch.zeros_like(Ex)
    Hx = torch.zeros_like(Ex)
    Hy = torch.zeros_like(Ex)
    Hz = torch.zeros_like(Ex)

    # Materials with simple graded damping near boundaries (poor-man's PML)
    eps = torch.full(shape, eps0, dtype=torch.float32, device=device)
    mu  = torch.full(shape, mu0, dtype=torch.float32, device=device)
    sigma = torch.zeros(shape, dtype=torch.float32, device=device)
    n_pml = int(args.n_pml)
    sigma_max = float(args.sigma_max)  # stronger absorption to reduce reflections
    if nx > 2*n_pml and ny > 2*n_pml and nz > 2*n_pml:
        for i in range(n_pml):
            w = ((n_pml - i) / n_pml)**2
            s = sigma_max * w
            sigma[:, :, i, :, :]      += s
            sigma[:, :, -1-i, :, :]   += s
            sigma[:, :, :, i, :]      += s
            sigma[:, :, :, -1-i, :]   += s
            sigma[:, :, :, :, i]      += s
            sigma[:, :, :, :, -1-i]   += s

    if args.backend == 'dense':
        model = GEMTE3D(dx=dx, dy=dy, dz=dz, dt=dt, eps=eps, mu=mu, sigma=sigma).to(device)
    else:
        model = GEMTEGraph3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt, eps=eps, mu=mu, sigma=sigma).to(device)

    # Simple soft source at grid center
    cx, cy, cz = nx//2, ny//2, nz//2
    t0, spread = float(args.t0), float(args.spread)

    def apply_bc(Ex, Ey, Ez, Hx, Hy, Hz):
        # PEC-like: clamp E at faces, leave H unconstrained
        for E in (Ex, Ey, Ez):
            E[..., 0, :, :] = 0.0
            E[..., -1, :, :] = 0.0
            E[..., :, 0, :] = 0.0
            E[..., :, -1, :] = 0.0
            E[..., :, :, 0] = 0.0
            E[..., :, :, -1] = 0.0
        return Ex, Ey, Ez, Hx, Hy, Hz

    steps = int(args.steps)
    freq = float(args.freq_ghz) * 1e9
    for n in range(steps):
        if args.source == 'gauss':
            env = math.exp(-0.5 * ((n - t0)/max(1.0, spread))**2)
            Ez[0,0,cx,cy,cz] += float(args.amp) * env
        else:
            ramp = min(1.0, n / max(1.0, t0))
            s = math.sin(2.0 * math.pi * freq * (n * dt))
            Ez[0,0,cx,cy,cz] += float(args.amp) * ramp * s

        Ex, Ey, Ez, Hx, Hy, Hz = model.step(Ex, Ey, Ez, Hx, Hy, Hz)
        Ex, Ey, Ez, Hx, Hy, Hz = apply_bc(Ex, Ey, Ez, Hx, Hy, Hz)

    # Helper to save central z-slice for a field
    def save_slice(field: torch.Tensor, title: str, out_png: str):
        sl = field[0,0,:,:,cz].detach().cpu()
        # crop out PML band to avoid edge artifacts in visualization
        crop = sl[n_pml:-n_pml, n_pml:-n_pml] if (not args.no_crop and nx>2*n_pml and ny>2*n_pml) else sl
        # upsample for smoother display (visual only)
        up = F.interpolate(crop.unsqueeze(0).unsqueeze(0), size=(512,512), mode='bilinear', align_corners=False)[0,0]
        # percentile-based vmax to enhance visibility of small signals
        a = torch.abs(up).reshape(-1)
        vmax = float(torch.quantile(a, torch.tensor(min(max(args.q, 0.5), 0.999), dtype=a.dtype)))
        if vmax <= 0.0 or not (vmax == vmax):
            vmax = float(torch.max(a))
        if vmax <= 0.0 or not (vmax == vmax):
            vmax = 1.0
        plt.figure(figsize=(5,4))
        plt.imshow(up.tolist(), origin='lower', cmap='turbo', vmin=-vmax, vmax=vmax,
                   extent=[0, 1000.0, 0, 1000.0], aspect='equal', interpolation='bilinear')
        plt.colorbar(label=f'{title} (a.u.)')
        plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
        plt.title(f'3D TM snapshot: {title} at central z-slice')
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f'Saved {out_png} with vmax={vmax:.3e}')
        plt.close()

    # Save all components
    save_slice(Ez, 'Ez', 'Ez_slice_zmid.png')
    save_slice(Ex, 'Ex', 'Ex_slice_zmid.png')
    save_slice(Ey, 'Ey', 'Ey_slice_zmid.png')
    save_slice(Hx, 'Hx', 'Hx_slice_zmid.png')
    save_slice(Hy, 'Hy', 'Hy_slice_zmid.png')
    save_slice(Hz, 'Hz', 'Hz_slice_zmid.png')

    # Print quick stats to confirm non-zero fields
    def stats(name, T):
        t = T.abs().max().item()
        print(f'{name} max|.|={t:.3e}')
    for nm, T in [('Ex', Ex), ('Ey', Ey), ('Ez', Ez), ('Hx', Hx), ('Hy', Hy), ('Hz', Hz)]:
        stats(nm, T)


if __name__ == '__main__':
    main()
