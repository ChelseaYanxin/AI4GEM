import torch
import torch.nn as nn
from torch import Tensor

class GEMTE(nn.Module):
    def __init__(self, nx, nz, dx, dz, dt, eps, mu, sigma):
        super().__init__()
        self.nx = nx
        self.nz = nz
        
        # Store as buffers for device management
        self.register_buffer('dx', torch.as_tensor(dx))
        self.register_buffer('dz', torch.as_tensor(dz))
        self.register_buffer('dt', torch.as_tensor(dt))
        self.register_buffer('eps', eps if isinstance(eps, Tensor) else torch.as_tensor(eps))
        self.register_buffer('mu', mu if isinstance(mu, Tensor) else torch.as_tensor(mu))
        self.register_buffer('sigma', sigma if isinstance(sigma, Tensor) else torch.as_tensor(sigma))

    @torch.no_grad()
    def step(self, Ez: Tensor, Hx: Tensor, Hy: Tensor):
        """
        Following main_FDTD_2D_TM.py exactly.
        Ez, Hx, Hy: [B, C, nx, ny] tensors (co-located, all same shape)
        Returns: Ez_new, Hx_new, Hy_new
        """
        if not (Ez.dim() == Hx.dim() == Hy.dim() == 4):
            raise ValueError('Ez, Hx, Hy must be 4-D tensors [B,C,nx,ny]')

        # Cast to buffer dtype
        Ez = Ez.to(self.eps.dtype)
        Hx = Hx.to(self.eps.dtype)
        Hy = Hy.to(self.eps.dtype)

        B, C, grid_nx, grid_ny = Ez.shape
        
        # GT code: nx=100, ny=100 (loop ranges), but arrays are 101×101
        nx = grid_nx - 1  # 100
        ny = grid_ny - 1  # 100
        
        # Coefficients from main_FDTD_2D_TM.py
        DA = 1.0
        DB = self.dt / self.mu[..., 0, 0]
        den_hx = 1.0 / self.dz  # 1/dy
        den_hy = 1.0 / self.dz  # 1/dy  
        den_ex = 1.0 / self.dx  # 1/dx
        den_ey = 1.0 / self.dz  # 1/dy

        # === Update H first ===
        # for i in range(0, nx):
        #     for j in range(0, ny):
        #         Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j + 1] - Ez[i, j]) * den_hy
        #         Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i + 1, j] - Ez[i, j]) * den_hx
        Hx_new = Hx.clone()
        Hy_new = Hy.clone()
        
        for i in range(nx):
            for j in range(ny):
                Hx_new[..., i, j] = DA * Hx[..., i, j] - DB * (Ez[..., i, j + 1] - Ez[..., i, j]) * den_hy
                Hy_new[..., i, j] = DA * Hy[..., i, j] + DB * (Ez[..., i + 1, j] - Ez[..., i, j]) * den_hx

        # === Update E second ===
        # for i in range(1, nx):
        #     for j in range(1, ny):
        #         Ez[i, j] = CA_Ez[i,j] * Ez[i, j] + CB_Ez[i,j] * ((Hy[i, j] - Hy[i - 1, j]) * den_ex - (Hx[i, j] - Hx[i, j - 1]) * den_ey)
        Ez_new = Ez.clone()
        
        # Compute CA_Ez and CB_Ez coefficients
        # CA_Ez[i,j] = (1 - sigma*dt/(2*eps)) / (1 + sigma*dt/(2*eps))
        # CB_Ez[i,j] = (dt/eps) / (1 + sigma*dt/(2*eps))
        A_plus = 1.0 + (self.sigma * self.dt) / (2.0 * self.eps)
        A_minus = 1.0 - (self.sigma * self.dt) / (2.0 * self.eps)
        
        for i in range(1, nx):
            for j in range(1, ny):
                CA = A_minus[..., i, j] / A_plus[..., i, j]
                CB = self.dt / A_plus[..., i, j]
                
                curl_term = (Hy_new[..., i, j] - Hy_new[..., i - 1, j]) * den_ex - (Hx_new[..., i, j] - Hx_new[..., i, j - 1]) * den_ey
                Ez_new[..., i, j] = CA * Ez[..., i, j] + CB * curl_term

        return Ez_new, Hx_new, Hy_new
