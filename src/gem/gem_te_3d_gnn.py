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


class GEMTEGraph3D(torch.nn.Module):
    """
    3D GEM via fixed-weight message passing (sparse operators with centered differences).
    Fields: Ex,Ey,Ez and Hx,Hy,Hz on the same grid interior; shapes [B,C,nx,ny,nz] (B=C=1 supported).
    Boundaries are left to the caller (e.g., PEC or PML clamping of E at faces).
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

        # build sparse centered-difference operators along x,y,z on the interior
        self._build_operators()

    def _build_operators(self):
        """Build sparse operators for Yee staggered grid with forward/backward differences.
        
        Yee grid staggering:
        - E components at cell centers
        - H components at cell edges (staggered by 1/2 cell)
        
        Forward differences (for H update from E):
        - Used when computing curl(E) for H
        - Example: dEz/dy = Ez[i,j+1,k] - Ez[i,j,k]
        
        Backward differences (for E update from H):
        - Used when computing curl(H) for E
        - Example: dHy/dx = Hy[i,j,k] - Hy[i-1,j,k]
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz

        def idx(i: int, j: int, k: int) -> int:
            return i * (ny * nz) + j * nz + k

        inv_dx = 1.0 / self.dx
        inv_dy = 1.0 / self.dy
        inv_dz = 1.0 / self.dz

        # Forward difference operators for H update: curl(E)
        # dEx/dy forward
        dEx_dy_fwd_rows, dEx_dy_fwd_cols, dEx_dy_fwd_vals = [], [], []
        for i in range(nx):
            for j in range(ny - 1):
                for k in range(nz):
                    p = idx(i, j, k)
                    dEx_dy_fwd_rows += [p, p]
                    dEx_dy_fwd_cols += [idx(i, j+1, k), idx(i, j, k)]
                    dEx_dy_fwd_vals += [inv_dy, -inv_dy]
        
        # dEx/dz forward
        dEx_dz_fwd_rows, dEx_dz_fwd_cols, dEx_dz_fwd_vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz - 1):
                    p = idx(i, j, k)
                    dEx_dz_fwd_rows += [p, p]
                    dEx_dz_fwd_cols += [idx(i, j, k+1), idx(i, j, k)]
                    dEx_dz_fwd_vals += [inv_dz, -inv_dz]
        
        # dEy/dx forward
        dEy_dx_fwd_rows, dEy_dx_fwd_cols, dEy_dx_fwd_vals = [], [], []
        for i in range(nx - 1):
            for j in range(ny):
                for k in range(nz):
                    p = idx(i, j, k)
                    dEy_dx_fwd_rows += [p, p]
                    dEy_dx_fwd_cols += [idx(i+1, j, k), idx(i, j, k)]
                    dEy_dx_fwd_vals += [inv_dx, -inv_dx]
        
        # dEy/dz forward
        dEy_dz_fwd_rows, dEy_dz_fwd_cols, dEy_dz_fwd_vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz - 1):
                    p = idx(i, j, k)
                    dEy_dz_fwd_rows += [p, p]
                    dEy_dz_fwd_cols += [idx(i, j, k+1), idx(i, j, k)]
                    dEy_dz_fwd_vals += [inv_dz, -inv_dz]
        
        # dEz/dx forward
        dEz_dx_fwd_rows, dEz_dx_fwd_cols, dEz_dx_fwd_vals = [], [], []
        for i in range(nx - 1):
            for j in range(ny):
                for k in range(nz):
                    p = idx(i, j, k)
                    dEz_dx_fwd_rows += [p, p]
                    dEz_dx_fwd_cols += [idx(i+1, j, k), idx(i, j, k)]
                    dEz_dx_fwd_vals += [inv_dx, -inv_dx]
        
        # dEz/dy forward
        dEz_dy_fwd_rows, dEz_dy_fwd_cols, dEz_dy_fwd_vals = [], [], []
        for i in range(nx):
            for j in range(ny - 1):
                for k in range(nz):
                    p = idx(i, j, k)
                    dEz_dy_fwd_rows += [p, p]
                    dEz_dy_fwd_cols += [idx(i, j+1, k), idx(i, j, k)]
                    dEz_dy_fwd_vals += [inv_dy, -inv_dy]

        # Backward difference operators for E update: curl(H)
        # Interior only (boundaries handled by caller)
        # dHx/dy backward
        dHx_dy_bwd_rows, dHx_dy_bwd_cols, dHx_dy_bwd_vals = [], [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)
                    dHx_dy_bwd_rows += [p, p]
                    dHx_dy_bwd_cols += [idx(i, j, k), idx(i, j-1, k)]
                    dHx_dy_bwd_vals += [inv_dy, -inv_dy]
        
        # dHx/dz backward
        dHx_dz_bwd_rows, dHx_dz_bwd_cols, dHx_dz_bwd_vals = [], [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)
                    dHx_dz_bwd_rows += [p, p]
                    dHx_dz_bwd_cols += [idx(i, j, k), idx(i, j, k-1)]
                    dHx_dz_bwd_vals += [inv_dz, -inv_dz]
        
        # dHy/dx backward
        dHy_dx_bwd_rows, dHy_dx_bwd_cols, dHy_dx_bwd_vals = [], [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)
                    dHy_dx_bwd_rows += [p, p]
                    dHy_dx_bwd_cols += [idx(i, j, k), idx(i-1, j, k)]
                    dHy_dx_bwd_vals += [inv_dx, -inv_dx]
        
        # dHy/dz backward
        dHy_dz_bwd_rows, dHy_dz_bwd_cols, dHy_dz_bwd_vals = [], [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)
                    dHy_dz_bwd_rows += [p, p]
                    dHy_dz_bwd_cols += [idx(i, j, k), idx(i, j, k-1)]
                    dHy_dz_bwd_vals += [inv_dz, -inv_dz]
        
        # dHz/dx backward
        dHz_dx_bwd_rows, dHz_dx_bwd_cols, dHz_dx_bwd_vals = [], [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)
                    dHz_dx_bwd_rows += [p, p]
                    dHz_dx_bwd_cols += [idx(i, j, k), idx(i-1, j, k)]
                    dHz_dx_bwd_vals += [inv_dx, -inv_dx]
        
        # dHz/dy backward
        dHz_dy_bwd_rows, dHz_dy_bwd_cols, dHz_dy_bwd_vals = [], [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    p = idx(i, j, k)
                    dHz_dy_bwd_rows += [p, p]
                    dHz_dy_bwd_cols += [idx(i, j, k), idx(i, j-1, k)]
                    dHz_dy_bwd_vals += [inv_dy, -inv_dy]

        device = self.eps.device
        def mk_sparse(rows, cols, vals):
            if len(rows) == 0:
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    (N, N)
                )
            i = torch.tensor([rows, cols], dtype=torch.long, device=device)
            v = torch.tensor(vals, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(i, v, (N, N)).coalesce()

        # Forward difference operators for H update
        self.register_buffer('dEx_dy_fwd', mk_sparse(dEx_dy_fwd_rows, dEx_dy_fwd_cols, dEx_dy_fwd_vals))
        self.register_buffer('dEx_dz_fwd', mk_sparse(dEx_dz_fwd_rows, dEx_dz_fwd_cols, dEx_dz_fwd_vals))
        self.register_buffer('dEy_dx_fwd', mk_sparse(dEy_dx_fwd_rows, dEy_dx_fwd_cols, dEy_dx_fwd_vals))
        self.register_buffer('dEy_dz_fwd', mk_sparse(dEy_dz_fwd_rows, dEy_dz_fwd_cols, dEy_dz_fwd_vals))
        self.register_buffer('dEz_dx_fwd', mk_sparse(dEz_dx_fwd_rows, dEz_dx_fwd_cols, dEz_dx_fwd_vals))
        self.register_buffer('dEz_dy_fwd', mk_sparse(dEz_dy_fwd_rows, dEz_dy_fwd_cols, dEz_dy_fwd_vals))
        
        # Backward difference operators for E update
        self.register_buffer('dHx_dy_bwd', mk_sparse(dHx_dy_bwd_rows, dHx_dy_bwd_cols, dHx_dy_bwd_vals))
        self.register_buffer('dHx_dz_bwd', mk_sparse(dHx_dz_bwd_rows, dHx_dz_bwd_cols, dHx_dz_bwd_vals))
        self.register_buffer('dHy_dx_bwd', mk_sparse(dHy_dx_bwd_rows, dHy_dx_bwd_cols, dHy_dx_bwd_vals))
        self.register_buffer('dHy_dz_bwd', mk_sparse(dHy_dz_bwd_rows, dHy_dz_bwd_cols, dHy_dz_bwd_vals))
        self.register_buffer('dHz_dx_bwd', mk_sparse(dHz_dx_bwd_rows, dHz_dx_bwd_cols, dHz_dx_bwd_vals))
        self.register_buffer('dHz_dy_bwd', mk_sparse(dHz_dy_bwd_rows, dHz_dy_bwd_cols, dHz_dy_bwd_vals))

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

        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz

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

        # convenience for sparse matmul
        def smm(M, v):
            return torch.sparse.mm(M, v.view(N, 1)).view(-1)

        # --- Update H from E: H^{n+1/2} = H^{n-1/2} - dt/mu * curl(E^n)
        # Use forward differences for H update
        dEz_dy = smm(self.dEz_dy_fwd, ez)
        dEy_dz = smm(self.dEy_dz_fwd, ey)
        hx_new = hx - self.dt * ((dEz_dy - dEy_dz) / mu_v)

        dEx_dz = smm(self.dEx_dz_fwd, ex)
        dEz_dx = smm(self.dEz_dx_fwd, ez)
        hy_new = hy - self.dt * ((dEx_dz - dEz_dx) / mu_v)

        dEy_dx = smm(self.dEy_dx_fwd, ey)
        dEx_dy = smm(self.dEx_dy_fwd, ex)
        hz_new = hz - self.dt * ((dEy_dx - dEx_dy) / mu_v)

        # --- Update E from H: E^{n+1} = (A-/A+) E^{n} + dt/(eps A+) * curl(H^{n+1/2})
        # Use backward differences for E update
        ex_new = (Aminus_v / Aplus_v) * ex
        ey_new = (Aminus_v / Aplus_v) * ey
        ez_new = (Aminus_v / Aplus_v) * ez

        dHz_dy = smm(self.dHz_dy_bwd, hz_new)
        dHy_dz = smm(self.dHy_dz_bwd, hy_new)
        ex_new = ex_new + self.dt * ((dHz_dy - dHy_dz) / (eps_v * Aplus_v))

        dHx_dz = smm(self.dHx_dz_bwd, hx_new)
        dHz_dx = smm(self.dHz_dx_bwd, hz_new)
        ey_new = ey_new + self.dt * ((dHx_dz - dHz_dx) / (eps_v * Aplus_v))

        dHy_dx = smm(self.dHy_dx_bwd, hy_new)
        dHx_dy = smm(self.dHx_dy_bwd, hx_new)
        ez_new = ez_new + self.dt * ((dHy_dx - dHx_dy) / (eps_v * Aplus_v))

        # reshape back; boundaries should be enforced by caller
        Ex_o = _unflatten3(ex_new, nx, ny, nz)
        Ey_o = _unflatten3(ey_new, nx, ny, nz)
        Ez_o = _unflatten3(ez_new, nx, ny, nz)
        Hx_o = _unflatten3(hx_new, nx, ny, nz)
        Hy_o = _unflatten3(hy_new, nx, ny, nz)
        Hz_o = _unflatten3(hz_new, nx, ny, nz)
        return Ex_o, Ey_o, Ez_o, Hx_o, Hy_o, Hz_o
