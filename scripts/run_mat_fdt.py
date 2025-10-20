import sys
import os
from glob import glob
import torch
import math
import argparse
import matplotlib.pyplot as plt

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from gem.gem_te import GEMTE

def load_mat_to_torch(path):
    
    out = {}
    try:
        from scipy.io import loadmat
        raw = loadmat(path)
        for k, v in raw.items():
            try:
                # convert numeric arrays to torch tensors
                t = torch.as_tensor(v, dtype=torch.float32)
                out[k] = t
            except Exception:
                out[k] = v
        return out
    except Exception:
        import h5py
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                try:
                    out[k] = torch.as_tensor(f[k][()], dtype=torch.float32)
                except Exception:
                    out[k] = f[k][()]
        return out

parser = argparse.ArgumentParser(description="Run 2D TMz (GEM/FDTD) from .mat and plot snapshots")
parser.add_argument('--mat', type=str, default=None)
parser.add_argument('--cfl-scale', type=float, default=0.07)
parser.add_argument('--sigma-uniform', type=float, default=1e-7)
parser.add_argument('--norm', type=str, default='sym', choices=['sym','zero-one','abs','percentile'])
parser.add_argument('--percentiles', type=str, default='1,99')
parser.add_argument('--share-scale', action='store_true')
parser.add_argument('--triptych', action='store_true')
parser.add_argument('--cmap', type=str, default=None)
parser.add_argument('--unitE', type=str, default='a.u.')
parser.add_argument('--unitH', type=str, default='a.u.')
args, _ = parser.parse_known_args()

# find .mat under data/data_FDTD_2D_cavity if not provided
if args.mat is None:
    data_dir = os.path.join(proj_root, "data", "data_FDTD_2D_cavity")
    mats = glob(os.path.join(data_dir, "*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {data_dir}")
    mat_path = mats[0]
else:
    mat_path = args.mat
print("Loading", mat_path)
mat = load_mat_to_torch(mat_path)

print("keys:", list(mat.keys()))
for k in sorted(mat.keys()):
    v = mat[k]
    try:
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape))
        else:
            print(k, type(v))
    except Exception:
        print(k, type(v))

def get(key):
    return mat.get(key, None)

Ez = get("Ez")
Hx = get("Hx")
Hy = get("Hy")
x = get("x")
y = get("y")
t = get("t")
idx_bc = get("idx_bc")

if isinstance(Ez, torch.Tensor) and Ez.dim() == 3:
    nt = Ez.shape[2]
    Ez0 = Ez[:, :, 0]
else:
    Ez0 = Ez
    nt = int(t.numel()) if (torch.is_tensor(t)) else (int(len(t)) if t is not None else 1)

# determine grid and spacing
if x is not None and y is not None:
    # x,y expected as tensors or array-like; convert to 1D torch tensors
    if torch.is_tensor(x):
        x = x.squeeze()
    else:
        x = torch.as_tensor(x).squeeze()
    if torch.is_tensor(y):
        y = y.squeeze()
    else:
        y = torch.as_tensor(y).squeeze()
    nx = int(x.numel())
    nz = int(y.numel())
    dx = float((x[1:] - x[:-1]).mean().item())
    dz = float((y[1:] - y[:-1]).mean().item())
else:
    nx, nz = Ez0.shape
    Lx = Lz = 50e-3
    dx = Lx / (nx - 1)
    dz = Lz / (nz - 1)
    x = torch.linspace(0, Lx, nx)
    y = torch.linspace(0, Lz, nz)

print(f"grid: nx={nx}, nz={nz}, dx={dx:.3e}, dz={dz:.3e}")

# material maps (defaults if missing)
eps0 = 8.854187817e-12
mu0 = 4*math.pi*1e-7
eps_map = get("eps") if get("eps") is not None else torch.full((nx, nz), eps0, dtype=torch.float32)
mu_map  = get("mu")  if get("mu")  is not None else torch.full((nx, nz), mu0, dtype=torch.float32)
sigma_map = get("sigma") if get("sigma") is not None else torch.zeros((nx, nz), dtype=torch.float32)

# optional uniform damping (S/m), set >0 to stabilize if you see blow-up
if float(args.sigma_uniform) != 0.0:
    sigma_map = sigma_map + float(args.sigma_uniform)

# convert loaded arrays to torch tensors and ensure float32
device = torch.device("cpu")
if torch.is_tensor(Ez0):
    Ez0_t = Ez0.to(torch.float32).to(device)
else:
    Ez0_t = torch.as_tensor(Ez0, dtype=torch.float32, device=device)
Ez_t = Ez0_t.unsqueeze(0).unsqueeze(0).contiguous()
Hx_t = torch.zeros_like(Ez_t)
Hy_t = torch.zeros_like(Ez_t)
if Hy is not None:
    if torch.is_tensor(Hy):
        Hy_arr = Hy
    else:
        Hy_arr = torch.as_tensor(Hy, dtype=torch.float32)
    if Hy_arr.dim() == 3:
        Hy_arr = Hy_arr[..., 0]
    Hy_t = Hy_arr.to(device).unsqueeze(0).unsqueeze(0).contiguous()
if Hx is not None:
    if torch.is_tensor(Hx):
        Hx_arr = Hx
    else:
        Hx_arr = torch.as_tensor(Hx, dtype=torch.float32)
    if Hx_arr.dim() == 3:
        Hx_arr = Hx_arr[..., 0]
    Hx_t = Hx_arr.to(device).unsqueeze(0).unsqueeze(0).contiguous()

eps_t = eps_map.to(torch.float32).to(device).unsqueeze(0).unsqueeze(0).contiguous()
mu_t  = mu_map.to(torch.float32).to(device).unsqueeze(0).unsqueeze(0).contiguous()
sigma_t = sigma_map.to(torch.float32).to(device).unsqueeze(0).unsqueeze(0).contiguous()

# CFL dt
c0 = 299792458.0
# add a CFL safety factor to avoid blow-up on non-staggered discretization
dt = float(args.cfl_scale) / (c0 * math.sqrt((1/dx**2) + (1/dz**2)))
print("dt =", dt)

model = GEMTE(dx=dx, dz=dz, dt=dt, eps=eps_t, mu=mu_t, sigma=sigma_t).to(device)

# sanity shapes
assert Ez_t.dim()==4 and Hx_t.dim()==4 and Hy_t.dim()==4, (Ez_t.shape, Hx_t.shape, Hy_t.shape)

def enforce_dirichlet_indices(field_tensor, idx_bc, nx, nz):
    # field_tensor: torch [B,C,nx,nz]
    if idx_bc is None:
        field_tensor[..., 0, :] = 0.0
        field_tensor[..., -1, :] = 0.0
        field_tensor[..., :, 0] = 0.0
        field_tensor[..., :, -1] = 0.0
        return field_tensor
    # idx_bc may include space-time linear indices from PDE dataset (often > nx*nz).
    # If so, fall back to simple spatial edge BCs.
    if torch.is_tensor(idx_bc):
        arr = idx_bc.reshape(-1).to(torch.int64)
    else:
        arr = torch.as_tensor(idx_bc).reshape(-1).to(torch.int64)
    if int(arr.numel()) > nx * nz:
        # fallback: only spatial edges
        field_tensor[..., 0, :] = 0.0
        field_tensor[..., -1, :] = 0.0
        field_tensor[..., :, 0] = 0.0
        field_tensor[..., :, -1] = 0.0
        return field_tensor
    arr0 = arr - 1
    rows = (arr0 % nx).to(torch.int64)
    cols = (arr0 // nx).to(torch.int64)
    # clamp and filter valid indices
    for r, c in zip(rows.tolist(), cols.tolist()):
        if 0 <= r < nx and 0 <= c < nz:
            field_tensor[0, 0, int(r), int(c)] = 0.0
    return field_tensor

Ez_t = enforce_dirichlet_indices(Ez_t, idx_bc, nx, nz)
Hx_t = enforce_dirichlet_indices(Hx_t, None, nx, nz)
Hy_t = enforce_dirichlet_indices(Hy_t, None, nx, nz)

# time stepping
# choose steps from Ez time dim if present, else from t, else default
if isinstance(Ez, torch.Tensor) and Ez.dim()==3:
    steps = int(Ez.shape[2])
elif t is not None:
    steps = int(t.numel()) if torch.is_tensor(t) else int(len(t))
else:
    steps = 200
snapshots = []
for n in range(steps):
    Ez_t, Hx_t, Hy_t = model.step(Ez_t, Hx_t, Hy_t)
    Ez_t = enforce_dirichlet_indices(Ez_t, idx_bc, nx, nz)
    Hx_t = enforce_dirichlet_indices(Hx_t, None, nx, nz)
    Hy_t = enforce_dirichlet_indices(Hy_t, None, nx, nz)
    # early break on numerical issues
    if (not torch.isfinite(Ez_t).all()) or (not torch.isfinite(Hx_t).all()) or (not torch.isfinite(Hy_t).all()):
        print(f"break at step {n}: non-finite values detected")
        break
    # per-step statistics using torch
    arr_cur_t = Ez_t[0,0].detach().cpu()
    if n % max(1, steps//10) == 0:
        snapshots.append(arr_cur_t.clone())
        nmin = float(torch.min(arr_cur_t).item())
        nmax = float(torch.max(arr_cur_t).item())
        nmean = float(torch.mean(arr_cur_t).item())
        nonzero = int((arr_cur_t != 0).sum().item())
        print(f"step {n}: min={nmin:.3e} max={nmax:.3e} mean={nmean:.3e} nonzero={nonzero}")

# final stats (torch)
arr_t = Ez_t[0,0].detach().cpu()
print('Ez final stats: min=%g max=%g mean=%g nonzero=%d' % (
    float(torch.min(arr_t).item()), float(torch.max(arr_t).item()), float(torch.mean(arr_t).item()), int((arr_t!=0).sum().item())
))
hx_t = Hx_t[0,0].detach().cpu()
hy_t = Hy_t[0,0].detach().cpu()
print('Hx final stats: min=%g max=%g mean=%g nonzero=%d' % (
    float(torch.min(hx_t).item()), float(torch.max(hx_t).item()), float(torch.mean(hx_t).item()), int((hx_t!=0).sum().item())
))
print('Hy final stats: min=%g max=%g mean=%g nonzero=%d' % (
    float(torch.min(hy_t).item()), float(torch.max(hy_t).item()), float(torch.mean(hy_t).item()), int((hy_t!=0).sum().item())
))
# save torch tensors for external inspection
torch.save({'Ez': Ez_t.cpu(), 'Hx': Hx_t.cpu(), 'Hy': Hy_t.cpu()}, os.path.join(proj_root, 'fields_final.pt'))
print('Saved final tensors to', os.path.join(proj_root, 'fields_final.pt'))

def _compute_scale(ft: torch.Tensor, mode: str, percent_pair=(1.0, 99.0)):
    ft = torch.nan_to_num(ft, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(torch.min(ft).item())
    mx = float(torch.max(ft).item())
    if mode == 'sym':
        vmax_abs = max(abs(mn), abs(mx)) if (not (mn == 0.0 and mx == 0.0)) else 1.0
        return -vmax_abs, vmax_abs, None
    if mode == 'abs':
        vmax_abs = max(abs(mn), abs(mx)) if (not (mn == 0.0 and mx == 0.0)) else 1.0
        return 0.0, vmax_abs, None
    if mode == 'zero-one':
        if mx == mn:
            return 0.0, 1.0, (ft*0.0)
        ft_norm = (ft - mn) / (mx - mn)
        return 0.0, 1.0, ft_norm
    if mode == 'percentile':
        lo, hi = percent_pair
        lo = max(0.0, min(100.0, float(lo)))
        hi = max(0.0, min(100.0, float(hi)))
        if hi <= lo:
            lo, hi = 1.0, 99.0
        q = torch.quantile(ft.flatten(), torch.tensor([lo/100.0, hi/100.0], dtype=ft.dtype))
        vmin = float(q[0].item()); vmax = float(q[1].item())
        if vmin == vmax:
            eps = 1e-12
            vmin -= eps; vmax += eps
        return vmin, vmax, None
    # default fallback
    vmax_abs = max(abs(mn), abs(mx)) if (not (mn == 0.0 and mx == 0.0)) else 1.0
    return -vmax_abs, vmax_abs, None

def plot_field(field_t, title, out_png, mode: str, cmap: str|None, unit_label: str, shared_v: tuple|None=None, percent_pair=(1.0,99.0)):
    ft = field_t[0,0].detach().cpu()
    # coords in mm
    try:
        x_mm_t = x.squeeze().detach().cpu() * 1e3
        y_mm_t = y.squeeze().detach().cpu() * 1e3
    except Exception:
        nx_, ny_ = ft.shape
        x_mm_t = torch.linspace(0.0, 1.0, nx_) * 1e3
        y_mm_t = torch.linspace(0.0, 1.0, ny_) * 1e3
    extent = [float(x_mm_t.min().item()), float(x_mm_t.max().item()), float(y_mm_t.min().item()), float(y_mm_t.max().item())]

    # scaling and possible pre-normalization
    if shared_v is not None:
        vmin, vmax = shared_v
        ft_plot = torch.nan_to_num(ft, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        vmin, vmax, ft_opt = _compute_scale(ft, mode, percent_pair)
        ft_plot = ft_opt if ft_opt is not None else torch.nan_to_num(ft, nan=0.0, posinf=0.0, neginf=0.0)

    # choose colormap
    cm = cmap
    if cm is None:
        cm = 'turbo' if mode in ('sym','percentile') else 'viridis'

    arr_list = ft_plot.transpose(0,1).tolist()
    plt.figure(figsize=(5,4))
    im = plt.imshow(arr_list, origin='lower', cmap=cm, extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=title + f' ({unit_label})')
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f'Saved {out_png} with vmin={vmin:.3e}, vmax={vmax:.3e}, cmap={cm}')
    plt.close()

def plot_triptych(Ez_t, Hx_t, Hy_t, mode: str, cmapE: str|None, cmapH: str|None, unitE: str, unitH: str, out_png: str, percent_pair=(1.0,99.0), share_scale=False):
    Ez_ft = Ez_t[0,0].detach().cpu(); Hx_ft = Hx_t[0,0].detach().cpu(); Hy_ft = Hy_t[0,0].detach().cpu()
    # coords in mm
    try:
        x_mm_t = x.squeeze().detach().cpu() * 1e3
        y_mm_t = y.squeeze().detach().cpu() * 1e3
    except Exception:
        nx_, ny_ = Ez_ft.shape
        x_mm_t = torch.linspace(0.0, 1.0, nx_) * 1e3
        y_mm_t = torch.linspace(0.0, 1.0, ny_) * 1e3
    extent = [float(x_mm_t.min().item()), float(x_mm_t.max().item()), float(y_mm_t.min().item()), float(y_mm_t.max().item())]

    # scales
    p = tuple(map(float, args.percentiles.split(','))) if mode=='percentile' else (1.0,99.0)
    if share_scale:
        stacked = torch.stack([torch.nan_to_num(Ez_ft), torch.nan_to_num(Hx_ft), torch.nan_to_num(Hy_ft)], dim=0)
        vmin, vmax, _ = _compute_scale(stacked, mode, p)
        scales = [(vmin, vmax)]*3
    else:
        scales = [ _compute_scale(Ez_ft, mode, p)[:2], _compute_scale(Hx_ft, mode, p)[:2], _compute_scale(Hy_ft, mode, p)[:2] ]

    fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    cmE = cmapE if cmapE is not None else ('turbo' )
    cmH = cmapH if cmapH is not None else ('turbo' )
    titles = [f'Ez ({unitE})', f'Hx ({unitH})', f'Hy ({unitH})']
    imgs = []
    for ax, ft, (vmin, vmax), cm, title in zip(axs, [Ez_ft, Hx_ft, Hy_ft], scales, [cmE, cmH, cmH], titles):
        # zero-one mode needs normalization per-panel
        if mode=='zero-one':
            mn = float(torch.min(ft)); mx = float(torch.max(ft))
            ftp = ((ft - mn) / (mx - mn)) if mx!=mn else (ft*0.0)
        else:
            ftp = torch.nan_to_num(ft)
        im = ax.imshow(ftp.transpose(0,1).tolist(), origin='lower', cmap=cm, extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('x (mm)')
        if ax is axs[0]: ax.set_ylabel('y (mm)')
        imgs.append(im)
    for ax, im in zip(axs, imgs):
        fig.colorbar(im, ax=ax)
    fig.suptitle('TMz snapshot')
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}')
    plt.close(fig)

# save three panels separately with chosen normalization
percent_pair = tuple(map(float, args.percentiles.split(','))) if args.norm=='percentile' else (1.0,99.0)
shared_v = None
if args.share_scale:
    # share scale across all three using combined stats of final fields
    stacked = torch.stack([Ez_t[0,0].detach().cpu(), Hx_t[0,0].detach().cpu(), Hy_t[0,0].detach().cpu()], dim=0)
    shared_v = _compute_scale(stacked, args.norm, percent_pair)[:2]

plot_field(Ez_t, 'Ez', os.path.join(proj_root, 'Ez_snapshot.png'), mode=args.norm, cmap=args.cmap, unit_label=args.unitE, shared_v=shared_v if args.share_scale else None, percent_pair=percent_pair)
plot_field(Hx_t, 'Hx', os.path.join(proj_root, 'Hx_snapshot.png'), mode=args.norm, cmap=args.cmap, unit_label=args.unitH, shared_v=shared_v if args.share_scale else None, percent_pair=percent_pair)
plot_field(Hy_t, 'Hy', os.path.join(proj_root, 'Hy_snapshot.png'), mode=args.norm, cmap=args.cmap, unit_label=args.unitH, shared_v=shared_v if args.share_scale else None, percent_pair=percent_pair)

if args.triptych:
    plot_triptych(Ez_t, Hx_t, Hy_t, mode=args.norm, cmapE=args.cmap, cmapH=args.cmap, unitE=args.unitE, unitH=args.unitH, out_png=os.path.join(proj_root, 'figure1_debug_side_by_side.png'), percent_pair=percent_pair, share_scale=args.share_scale)