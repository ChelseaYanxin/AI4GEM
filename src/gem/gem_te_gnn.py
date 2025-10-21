from __future__ import annotations
import torch
from torch import Tensor


def _flatten2(x: Tensor) -> Tensor:
    # expects [B,C,nx,ny] with B=C=1; returns [N]
    if x.dim() != 4:
        raise ValueError('expected [B,C,nx,ny]')
    if x.size(0) != 1 or x.size(1) != 1:
        # simple demo implementation; extend to batching if needed
        x = x.reshape(-1, x.shape[-2], x.shape[-1]).mean(0, keepdim=True).mean(0)
    return x.view(-1)


def _unflatten2(v: Tensor, nx: int, ny: int) -> Tensor:
    # returns [1,1,nx,ny]
    return v.view(1, 1, nx, ny)


class GEMTEGraph2D(torch.nn.Module):
    """
    2D TMz GEM via fixed-weight message passing (sparse operators).
    Fields: Ez out-of-plane, Hx/Hy in-plane. Shapes [B,C,nx,ny] (B=C=1 supported).
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float,
                 eps: Tensor, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.nx, self.ny = int(nx), int(ny)
        self.dx = float(dx); self.dy = float(dy); self.dt = float(dt)

        # store material maps
        self.register_buffer('eps', eps.to(torch.float32))
        self.register_buffer('mu', mu.to(torch.float32))
        self.register_buffer('sigma', sigma.to(torch.float32))

        # Precompute A+/A- at Ez positions
        A_plus = 1.0 + (self.sigma * self.dt) / (2.0 * self.eps)
        A_minus = 1.0 - (self.sigma * self.dt) / (2.0 * self.eps)
        self.register_buffer('A_plus', A_plus)
        self.register_buffer('A_minus', A_minus)

        # Build sparse centered-difference operators on interior nodes
        self._build_operators()

    def _build_operators(self):
        nx, ny = self.nx, self.ny
        N = nx * ny

        def idx(i: int, j: int) -> int:
            return i * ny + j

        # Hx update depends on dEz/dy (centered): (Ez[i,j+1]-Ez[i,j-1])/(2dy)
        hx_rows, hx_cols, hx_vals = [], [], []
        hy_rows, hy_cols, hy_vals = [], [], []
        ezhy_rows, ezhy_cols, ezhy_vals = [], [], []  # dHy/dx at Ez
        ezhx_rows, ezhx_cols, ezhx_vals = [], [], []  # dHx/dy at Ez

        inv_2dy = 1.0 / (2.0 * self.dy)
        inv_2dx = 1.0 / (2.0 * self.dx)

        # Build for interior only; boundaries left to caller BCs
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                p = idx(i, j)
                # Hx at (i,j): use Ez(i,j+1) - Ez(i,j-1)
                r_hx = p
                hx_rows += [r_hx, r_hx]
                hx_cols += [idx(i, j+1), idx(i, j-1)]
                hx_vals += [inv_2dy, -inv_2dy]

                # Hy at (i,j): use Ez(i+1,j) - Ez(i-1,j)
                r_hy = p
                hy_rows += [r_hy, r_hy]
                hy_cols += [idx(i+1, j), idx(i-1, j)]
                hy_vals += [inv_2dx, -inv_2dx]

                # Ez at (i,j): dHy/dx and dHx/dy
                # dHy/dx = (Hy(i+1,j) - Hy(i-1,j)) / (2dx)
                ezhy_rows += [p, p]
                ezhy_cols += [idx(i+1, j), idx(i-1, j)]
                ezhy_vals += [inv_2dx, -inv_2dx]
                # dHx/dy = (Hx(i,j+1) - Hx(i,j-1)) / (2dy)
                ezhx_rows += [p, p]
                ezhx_cols += [idx(i, j+1), idx(i, j-1)]
                ezhx_vals += [inv_2dy, -inv_2dy]

        device = self.eps.device
        def mk_sparse(rows, cols, vals):
            if len(rows) == 0:
                # minimal empty
                return torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long, device=device),
                                               torch.empty((0,), dtype=torch.float32, device=device),
                                               (N, N))
            i = torch.tensor([rows, cols], dtype=torch.long, device=device)
            v = torch.tensor(vals, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(i, v, (N, N)).coalesce()

        self.register_buffer('L_hx_from_ez', mk_sparse(hx_rows, hx_cols, hx_vals))
        self.register_buffer('L_hy_from_ez', mk_sparse(hy_rows, hy_cols, hy_vals))
        self.register_buffer('L_ez_from_hy', mk_sparse(ezhy_rows, ezhy_cols, ezhy_vals))
        self.register_buffer('L_ez_from_hx', mk_sparse(ezhx_rows, ezhx_cols, ezhx_vals))

    @torch.no_grad()
    def step(self, Ez: Tensor, Hx: Tensor, Hy: Tensor):
        # flatten
        ez = _flatten2(Ez.to(self.eps.dtype))
        hx = _flatten2(Hx.to(self.eps.dtype))
        hy = _flatten2(Hy.to(self.eps.dtype))

        nx, ny = self.nx, self.ny
        N = nx * ny

        eps_v = _flatten2(self.eps)
        mu_v  = _flatten2(self.mu)
        Aplus_v = _flatten2(self.A_plus)
        Aminus_v= _flatten2(self.A_minus)

        # H update (centered curl of E)
        # Hx_new = Hx - dt/mu * dEz/dy; dEz/dy = L_hx_from_ez @ ez
        dez_dy = torch.sparse.mm(self.L_hx_from_ez, ez.view(N,1)).view(-1)
        hx_new = hx - self.dt * (dez_dy / mu_v)

        # Hy_new = Hy + dt/mu * dEz/dx; sign from TMz curl
        dez_dx = torch.sparse.mm(self.L_hy_from_ez, ez.view(N,1)).view(-1)
        hy_new = hy + self.dt * (dez_dx / mu_v)

        # E update (centered curl of H)
        dHy_dx = torch.sparse.mm(self.L_ez_from_hy, hy_new.view(N,1)).view(-1)
        dHx_dy = torch.sparse.mm(self.L_ez_from_hx, hx_new.view(N,1)).view(-1)
        curl_z = dHy_dx - dHx_dy

        ez_new = (Aminus_v / Aplus_v) * ez + (self.dt / (Aplus_v * eps_v)) * curl_z

        # reshape back and leave boundaries to caller
        Ez_o = _unflatten2(ez_new, nx, ny)
        Hx_o = _unflatten2(hx_new, nx, ny)
        Hy_o = _unflatten2(hy_new, nx, ny)
        return Ez_o, Hx_o, Hy_o
