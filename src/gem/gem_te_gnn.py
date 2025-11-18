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
    """GEM 2D TMz solver using graph neural network (message passing).
    
    All field components (Ez, Hx, Hy) are co-located at the same grid nodes.
    Spatial derivatives use centered finite differences with neighbor nodes.
    Time stepping uses leapfrog: H at half-steps, E at integer steps.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float, eps: Tensor, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.nx, self.ny = int(nx), int(ny)
        self.dx = float(dx); self.dy = float(dy); self.dt = float(dt)
        dty = torch.get_default_dtype()
        self.register_buffer('eps', eps.to(dty))
        self.register_buffer('mu', mu.to(dty))
        self.register_buffer('sigma', sigma.to(dty))
        A_plus = 1.0 + (self.sigma * self.dt) / (2.0 * self.eps)
        A_minus = 1.0 - (self.sigma * self.dt) / (2.0 * self.eps)
        self.register_buffer('A_plus', A_plus)
        self.register_buffer('A_minus', A_minus)
        self._build_operators()

    def _build_operators(self):
        """Build sparse operators for Yee staggered grid (standard FDTD).
        
        Yee staggering:
        - Ez at (i, j)
        - Hx at (i, j+1/2) stored in array as Hx[i, j] representing position between Ez[i,j] and Ez[i,j+1]
        - Hy at (i+1/2, j) stored in array as Hy[i, j] representing position between Ez[i,j] and Ez[i+1,j]
        
        H update uses forward differences:
        - Hx[i,j] uses Ez[i,j+1] - Ez[i,j]
        - Hy[i,j] uses Ez[i+1,j] - Ez[i,j]
        
        E update uses backward differences (interior only):
        - Ez[i,j] uses Hy[i,j] - Hy[i-1,j] and Hx[i,j] - Hx[i,j-1]
        """
        nx, ny = self.nx, self.ny
        N = nx * ny
        
        def idx(i: int, j: int) -> int:
            """Map 2D grid index to 1D linear index."""
            return i * ny + j
        
        # Forward difference operators for H update
        hx_from_ez_rows, hx_from_ez_cols, hx_from_ez_vals = [], [], []
        hy_from_ez_rows, hy_from_ez_cols, hy_from_ez_vals = [], [], []
        
        # Backward difference operators for E update
        ez_from_hy_rows, ez_from_hy_cols, ez_from_hy_vals = [], [], []
        ez_from_hx_rows, ez_from_hx_cols, ez_from_hx_vals = [], [], []
        
        inv_dy = 1.0 / self.dy
        inv_dx = 1.0 / self.dx
        
        # H update operators (forward differences)
        # Hx at (i, j+1/2): dEz/dy = Ez(i,j+1) - Ez(i,j)
        for i in range(nx):
            for j in range(ny - 1):  # j < ny-1 to have Ez[i,j+1]
                p = idx(i, j)
                hx_from_ez_rows += [p, p]
                hx_from_ez_cols += [idx(i, j+1), idx(i, j)]
                hx_from_ez_vals += [inv_dy, -inv_dy]
        
        # Hy at (i+1/2, j): dEz/dx = Ez(i+1,j) - Ez(i,j)
        for i in range(nx - 1):  # i < nx-1 to have Ez[i+1,j]
            for j in range(ny):
                p = idx(i, j)
                hy_from_ez_rows += [p, p]
                hy_from_ez_cols += [idx(i+1, j), idx(i, j)]
                hy_from_ez_vals += [inv_dx, -inv_dx]
        
        # E update operators (backward differences, interior only)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                p = idx(i, j)
                
                # dHy/dx = Hy[i,j] - Hy[i-1,j]
                ez_from_hy_rows += [p, p]
                ez_from_hy_cols += [idx(i, j), idx(i-1, j)]
                ez_from_hy_vals += [inv_dx, -inv_dx]
                
                # dHx/dy = Hx[i,j] - Hx[i,j-1]
                ez_from_hx_rows += [p, p]
                ez_from_hx_cols += [idx(i, j), idx(i, j-1)]
                ez_from_hx_vals += [inv_dy, -inv_dy]
        
        device = self.eps.device
        
        def mk_sparse(rows, cols, vals):
            if len(rows) == 0:
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=self.eps.dtype, device=device),
                    (N, N)
                )
            i = torch.tensor([rows, cols], dtype=torch.long, device=device)
            v = torch.tensor(vals, dtype=self.eps.dtype, device=device)
            return torch.sparse_coo_tensor(i, v, (N, N)).coalesce()
        
        self.register_buffer('L_hx_from_ez', mk_sparse(hx_from_ez_rows, hx_from_ez_cols, hx_from_ez_vals))
        self.register_buffer('L_hy_from_ez', mk_sparse(hy_from_ez_rows, hy_from_ez_cols, hy_from_ez_vals))
        self.register_buffer('L_ez_from_hy', mk_sparse(ez_from_hy_rows, ez_from_hy_cols, ez_from_hy_vals))
        self.register_buffer('L_ez_from_hx', mk_sparse(ez_from_hx_rows, ez_from_hx_cols, ez_from_hx_vals))

    @torch.no_grad()
    def step(self, Ez: Tensor, Hx: Tensor, Hy: Tensor):
        """Advance fields by one time step using Yee staggered grid leapfrog FDTD.
        
        H update (n-1/2 → n+1/2) using forward differences:
          Hx^{n+1/2}(i,j+1/2) = Hx^{n-1/2}(i,j+1/2) - (dt/mu) * [Ez^n(i,j+1) - Ez^n(i,j)] / dy
          Hy^{n+1/2}(i+1/2,j) = Hy^{n-1/2}(i+1/2,j) + (dt/mu) * [Ez^n(i+1,j) - Ez^n(i,j)] / dx
        
        E update (n → n+1) using backward differences (interior only):
          Ez^{n+1}(i,j) = (A-/A+)*Ez^n(i,j) + (dt/(A+*eps)) * [
              (Hy^{n+1/2}(i+1/2,j) - Hy^{n+1/2}(i-1/2,j))/dx
            - (Hx^{n+1/2}(i,j+1/2) - Hx^{n+1/2}(i,j-1/2))/dy
          ]
        
        where A+ = 1 + sigma*dt/(2*eps), A- = 1 - sigma*dt/(2*eps)
        """
        ez = _flatten2(Ez.to(self.eps.dtype))
        hx = _flatten2(Hx.to(self.eps.dtype))
        hy = _flatten2(Hy.to(self.eps.dtype))
        
        nx, ny = self.nx, self.ny
        N = nx * ny
        
        eps_v = _flatten2(self.eps)
        mu_v = _flatten2(self.mu)
        Aplus_v = _flatten2(self.A_plus)
        Aminus_v = _flatten2(self.A_minus)
        
        # 1) Update H fields using forward differences of E
        # Hx -= dt/mu * dEz/dy_forward
        dez_dy_fwd = torch.sparse.mm(self.L_hx_from_ez, ez.view(N, 1)).view(-1)
        hx_new = hx - self.dt * (dez_dy_fwd / mu_v)
        
        # Hy += dt/mu * dEz/dx_forward
        dez_dx_fwd = torch.sparse.mm(self.L_hy_from_ez, ez.view(N, 1)).view(-1)
        hy_new = hy + self.dt * (dez_dx_fwd / mu_v)
        
        # 2) Update E field using backward differences of H (interior only)
        # curl_z = dHy/dx_backward - dHx/dy_backward
        dhy_dx_bwd = torch.sparse.mm(self.L_ez_from_hy, hy_new.view(N, 1)).view(-1)
        dhx_dy_bwd = torch.sparse.mm(self.L_ez_from_hx, hx_new.view(N, 1)).view(-1)
        curl_z = dhy_dx_bwd - dhx_dy_bwd
        
        # Ez update with lossy term
        ez_new = (Aminus_v / Aplus_v) * ez + (self.dt / (Aplus_v * eps_v)) * curl_z
        
        # Reshape back to [B,C,nx,ny]
        Ez_o = _unflatten2(ez_new, nx, ny)
        Hx_o = _unflatten2(hx_new, nx, ny)
        Hy_o = _unflatten2(hy_new, nx, ny)
        
        # Apply PEC boundary conditions on Ez (set to zero at edges)
        Ez_o[..., 0, :] = 0
        Ez_o[..., -1, :] = 0
        Ez_o[..., :, 0] = 0
        Ez_o[..., :, -1] = 0
        
        return Ez_o, Hx_o, Hy_o
