from __future__ import annotations
import torch
from torch import Tensor


def _flatten3(x: Tensor) -> Tensor:
    # expects [B,C,nx,ny,nz] with B=C=1; returns [N]
    if x.dim() != 5:
        raise ValueError('expected [B,C,nx,ny,nz]')
    if x.size(0) != 1 or x.size(1) != 1:
        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]).mean(0, keepdim=True).mean(0)
    return x.view(-1)


def _unflatten3(v: Tensor, nx: int, ny: int, nz: int) -> Tensor:
    # returns [1,1,nx,ny,nz]
    return v.view(1, 1, nx, ny, nz)


class GEMTEGraph3DMPNN(torch.nn.Module):
    """
    3D GEM via two-step fixed-weight message passing (no global sparse matmul).
    - Step 1 (E -> H): H^{n+1/2} = H^{n-1/2} - dt/mu * curl(E^n)
    - Step 2 (H -> E): E^{n+1}   = (A-/A+) E^{n} + dt/(eps A+) * curl(H^{n+1/2})
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float, dt: float,
                 eps: Tensor, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dx = float(dx); self.dy = float(dy); self.dz = float(dz); self.dt = float(dt)

        # material maps
        self.register_buffer('eps', eps.to(torch.float32))
        self.register_buffer('mu', mu.to(torch.float32))
        self.register_buffer('sigma', sigma.to(torch.float32))

        # conductivity update factors for E
        A_plus = 1.0 + (self.sigma * self.dt) / (2.0 * self.eps)
        A_minus = 1.0 - (self.sigma * self.dt) / (2.0 * self.eps)
        self.register_buffer('A_plus', A_plus)
        self.register_buffer('A_minus', A_minus)

        # build edge lists for centered differences (interior only)
        self._build_edges()

    def _build_edges(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz

        def idx(i: int, j: int, k: int) -> int:
            return i * (ny * nz) + j * nz + k

        device = self.eps.device
        inv2dx = 1.0 / (2.0 * self.dx)
        inv2dy = 1.0 / (2.0 * self.dy)
        inv2dz = 1.0 / (2.0 * self.dz)

        # Helpers to append edges
        def make_edge_storage():
            return [], [], []  # tgt_idx, src_idx, coef

        # H updates: curl(E)
        hx_from_ez_t, hx_from_ez_s, hx_from_ez_c = make_edge_storage()
        hx_from_ey_t, hx_from_ey_s, hx_from_ey_c = make_edge_storage()

        hy_from_ex_t, hy_from_ex_s, hy_from_ex_c = make_edge_storage()
        hy_from_ez_t, hy_from_ez_s, hy_from_ez_c = make_edge_storage()

        hz_from_ey_t, hz_from_ey_s, hz_from_ey_c = make_edge_storage()
        hz_from_ex_t, hz_from_ex_s, hz_from_ex_c = make_edge_storage()

        # E updates: curl(H)
        ex_from_hz_t, ex_from_hz_s, ex_from_hz_c = make_edge_storage()
        ex_from_hy_t, ex_from_hy_s, ex_from_hy_c = make_edge_storage()

        ey_from_hx_t, ey_from_hx_s, ey_from_hx_c = make_edge_storage()
        ey_from_hz_t, ey_from_hz_s, ey_from_hz_c = make_edge_storage()

        ez_from_hy_t, ez_from_hy_s, ez_from_hy_c = make_edge_storage()
        ez_from_hx_t, ez_from_hx_s, ez_from_hx_c = make_edge_storage()

        # interior nodes only; boundaries excluded
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)

                    # Hx: dEz/dy - dEy/dz
                    hx_from_ez_t += [p, p]
                    hx_from_ez_s += [idx(i, j + 1, k), idx(i, j - 1, k)]
                    hx_from_ez_c += [ +inv2dy, -inv2dy]

                    hx_from_ey_t += [p, p]
                    hx_from_ey_s += [idx(i, j, k + 1), idx(i, j, k - 1)]
                    hx_from_ey_c += [ -inv2dz, +inv2dz]  # minus dEy/dz

                    # Hy: dEx/dz - dEz/dx
                    hy_from_ex_t += [p, p]
                    hy_from_ex_s += [idx(i, j, k + 1), idx(i, j, k - 1)]
                    hy_from_ex_c += [ +inv2dz, -inv2dz]

                    hy_from_ez_t += [p, p]
                    hy_from_ez_s += [idx(i + 1, j, k), idx(i - 1, j, k)]
                    hy_from_ez_c += [ -inv2dx, +inv2dx]  # minus dEz/dx

                    # Hz: dEy/dx - dEx/dy
                    hz_from_ey_t += [p, p]
                    hz_from_ey_s += [idx(i + 1, j, k), idx(i - 1, j, k)]
                    hz_from_ey_c += [ +inv2dx, -inv2dx]

                    hz_from_ex_t += [p, p]
                    hz_from_ex_s += [idx(i, j + 1, k), idx(i, j - 1, k)]
                    hz_from_ex_c += [ -inv2dy, +inv2dy]  # minus dEx/dy

                    # Ex: dHz/dy - dHy/dz
                    ex_from_hz_t += [p, p]
                    ex_from_hz_s += [idx(i, j + 1, k), idx(i, j - 1, k)]
                    ex_from_hz_c += [ +inv2dy, -inv2dy]

                    ex_from_hy_t += [p, p]
                    ex_from_hy_s += [idx(i, j, k + 1), idx(i, j, k - 1)]
                    ex_from_hy_c += [ -inv2dz, +inv2dz]  # minus dHy/dz

                    # Ey: dHx/dz - dHz/dx
                    ey_from_hx_t += [p, p]
                    ey_from_hx_s += [idx(i, j, k + 1), idx(i, j, k - 1)]
                    ey_from_hx_c += [ +inv2dz, -inv2dz]

                    ey_from_hz_t += [p, p]
                    ey_from_hz_s += [idx(i + 1, j, k), idx(i - 1, j, k)]
                    ey_from_hz_c += [ -inv2dx, +inv2dx]  # minus dHz/dx

                    # Ez: dHy/dx - dHx/dy
                    ez_from_hy_t += [p, p]
                    ez_from_hy_s += [idx(i + 1, j, k), idx(i - 1, j, k)]
                    ez_from_hy_c += [ +inv2dx, -inv2dx]

                    ez_from_hx_t += [p, p]
                    ez_from_hx_s += [idx(i, j + 1, k), idx(i, j - 1, k)]
                    ez_from_hx_c += [ -inv2dy, +inv2dy]  # minus dHx/dy

        def to_tensor(idxs, vals):
            if len(idxs[0]) == 0:
                return (
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                )
            edge_index = torch.tensor([idxs[0], idxs[1]], dtype=torch.long, device=device)
            coef = torch.tensor(vals, dtype=torch.float32, device=device)
            return edge_index, coef

        # Pack and register buffers
        self.register_buffer('hx_from_ez_edge', torch.tensor([hx_from_ez_t, hx_from_ez_s], dtype=torch.long, device=device) if len(hx_from_ez_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('hx_from_ez_coef', torch.tensor(hx_from_ez_c, dtype=torch.float32, device=device))
        self.register_buffer('hx_from_ey_edge', torch.tensor([hx_from_ey_t, hx_from_ey_s], dtype=torch.long, device=device) if len(hx_from_ey_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('hx_from_ey_coef', torch.tensor(hx_from_ey_c, dtype=torch.float32, device=device))

        self.register_buffer('hy_from_ex_edge', torch.tensor([hy_from_ex_t, hy_from_ex_s], dtype=torch.long, device=device) if len(hy_from_ex_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('hy_from_ex_coef', torch.tensor(hy_from_ex_c, dtype=torch.float32, device=device))
        self.register_buffer('hy_from_ez_edge', torch.tensor([hy_from_ez_t, hy_from_ez_s], dtype=torch.long, device=device) if len(hy_from_ez_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('hy_from_ez_coef', torch.tensor(hy_from_ez_c, dtype=torch.float32, device=device))

        self.register_buffer('hz_from_ey_edge', torch.tensor([hz_from_ey_t, hz_from_ey_s], dtype=torch.long, device=device) if len(hz_from_ey_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('hz_from_ey_coef', torch.tensor(hz_from_ey_c, dtype=torch.float32, device=device))
        self.register_buffer('hz_from_ex_edge', torch.tensor([hz_from_ex_t, hz_from_ex_s], dtype=torch.long, device=device) if len(hz_from_ex_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('hz_from_ex_coef', torch.tensor(hz_from_ex_c, dtype=torch.float32, device=device))

        self.register_buffer('ex_from_hz_edge', torch.tensor([ex_from_hz_t, ex_from_hz_s], dtype=torch.long, device=device) if len(ex_from_hz_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('ex_from_hz_coef', torch.tensor(ex_from_hz_c, dtype=torch.float32, device=device))
        self.register_buffer('ex_from_hy_edge', torch.tensor([ex_from_hy_t, ex_from_hy_s], dtype=torch.long, device=device) if len(ex_from_hy_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('ex_from_hy_coef', torch.tensor(ex_from_hy_c, dtype=torch.float32, device=device))

        self.register_buffer('ey_from_hx_edge', torch.tensor([ey_from_hx_t, ey_from_hx_s], dtype=torch.long, device=device) if len(ey_from_hx_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('ey_from_hx_coef', torch.tensor(ey_from_hx_c, dtype=torch.float32, device=device))
        self.register_buffer('ey_from_hz_edge', torch.tensor([ey_from_hz_t, ey_from_hz_s], dtype=torch.long, device=device) if len(ey_from_hz_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('ey_from_hz_coef', torch.tensor(ey_from_hz_c, dtype=torch.float32, device=device))

        self.register_buffer('ez_from_hy_edge', torch.tensor([ez_from_hy_t, ez_from_hy_s], dtype=torch.long, device=device) if len(ez_from_hy_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('ez_from_hy_coef', torch.tensor(ez_from_hy_c, dtype=torch.float32, device=device))
        self.register_buffer('ez_from_hx_edge', torch.tensor([ez_from_hx_t, ez_from_hx_s], dtype=torch.long, device=device) if len(ez_from_hx_t) else torch.empty((2,0), dtype=torch.long, device=device))
        self.register_buffer('ez_from_hx_coef', torch.tensor(ez_from_hx_c, dtype=torch.float32, device=device))

        # Save N for reshaping
        self.N = N

    @torch.no_grad()
    def step(self,
             Ex: Tensor, Ey: Tensor, Ez: Tensor,
             Hx: Tensor, Hy: Tensor, Hz: Tensor):
        # shape checks
        if not (Ex.dim() == Ey.dim() == Ez.dim() == Hx.dim() == Hy.dim() == Hz.dim() == 5):
            raise ValueError('All fields must be 5-D tensors [B,C,nx,ny,nz]')

        # cast to dtype of buffers
        dtype = self.eps.dtype
        Ex = Ex.to(dtype); Ey = Ey.to(dtype); Ez = Ez.to(dtype)
        Hx = Hx.to(dtype); Hy = Hy.to(dtype); Hz = Hz.to(dtype)

        nx, ny, nz, N = self.nx, self.ny, self.nz, self.N

        # flatten
        ex = _flatten3(Ex)
        ey = _flatten3(Ey)
        ez = _flatten3(Ez)
        hx = _flatten3(Hx)
        hy = _flatten3(Hy)
        hz = _flatten3(Hz)

        eps_v = _flatten3(self.eps)
        mu_v = _flatten3(self.mu)
        Aplus_v = _flatten3(self.A_plus)
        Aminus_v = _flatten3(self.A_minus)

        # scatter-add helper: aggregate messages from src to tgt using edge list and coef
        def aggregate(src_vec: Tensor, edge_idx: Tensor, coef: Tensor) -> Tensor:
            if edge_idx.numel() == 0:
                return torch.zeros(N, dtype=src_vec.dtype, device=src_vec.device)
            tgt, src = edge_idx[0], edge_idx[1]
            vals = src_vec.index_select(0, src) * coef
            out = torch.zeros(N, dtype=src_vec.dtype, device=src_vec.device)
            out.index_add_(0, tgt, vals)
            return out

        # --- Update H from E
        curlE_x = aggregate(ez, self.hx_from_ez_edge, self.hx_from_ez_coef) + \
                  aggregate(ey, self.hx_from_ey_edge, self.hx_from_ey_coef)
        curlE_y = aggregate(ex, self.hy_from_ex_edge, self.hy_from_ex_coef) + \
                  aggregate(ez, self.hy_from_ez_edge, self.hy_from_ez_coef)
        curlE_z = aggregate(ey, self.hz_from_ey_edge, self.hz_from_ey_coef) + \
                  aggregate(ex, self.hz_from_ex_edge, self.hz_from_ex_coef)

        inv_mu = torch.where(mu_v != 0, 1.0 / mu_v, torch.zeros_like(mu_v))
        hx_new = hx - self.dt * (curlE_x * inv_mu)
        hy_new = hy - self.dt * (curlE_y * inv_mu)
        hz_new = hz - self.dt * (curlE_z * inv_mu)

        # --- Update E from H
        curlH_x = aggregate(hz_new, self.ex_from_hz_edge, self.ex_from_hz_coef) + \
                  aggregate(hy_new, self.ex_from_hy_edge, self.ex_from_hy_coef)
        curlH_y = aggregate(hx_new, self.ey_from_hx_edge, self.ey_from_hx_coef) + \
                  aggregate(hz_new, self.ey_from_hz_edge, self.ey_from_hz_coef)
        curlH_z = aggregate(hy_new, self.ez_from_hy_edge, self.ez_from_hy_coef) + \
                  aggregate(hx_new, self.ez_from_hx_edge, self.ez_from_hx_coef)

        selfE = torch.where(Aplus_v != 0, Aminus_v / Aplus_v, torch.zeros_like(Aplus_v))
        scaleE = torch.where(Aplus_v != 0, self.dt / (eps_v * Aplus_v + 1e-12), torch.zeros_like(Aplus_v))

        ex_new = selfE * ex + scaleE * curlH_x
        ey_new = selfE * ey + scaleE * curlH_y
        ez_new = selfE * ez + scaleE * curlH_z

        # reshape back; boundaries should be enforced by caller
        Ex_o = _unflatten3(ex_new, nx, ny, nz)
        Ey_o = _unflatten3(ey_new, nx, ny, nz)
        Ez_o = _unflatten3(ez_new, nx, ny, nz)
        Hx_o = _unflatten3(hx_new, nx, ny, nz)
        Hy_o = _unflatten3(hy_new, nx, ny, nz)
        Hz_o = _unflatten3(hz_new, nx, ny, nz)
        return Ex_o, Ey_o, Ez_o, Hx_o, Hy_o, Hz_o
