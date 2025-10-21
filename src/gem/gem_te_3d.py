from __future__ import annotations
import torch
from torch import Tensor


class GEMTE3D(torch.nn.Module):
    
    def __init__(self, dx: float, dy: float, dz: float, dt: float,
                 eps: Tensor, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.dt = float(dt)

        # register material maps as buffers
        self.register_buffer('eps', eps)
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        # conductivity update factors for E update
        self.register_buffer('A_plus', 1.0 + (sigma * dt) / (2.0 * eps))
        self.register_buffer('A_minus', 1.0 - (sigma * dt) / (2.0 * eps))

    @torch.no_grad()
    def step(self,
             Ex: Tensor, Ey: Tensor, Ez: Tensor,
             Hx: Tensor, Hy: Tensor, Hz: Tensor):
        if not (Ex.dim() == Ey.dim() == Ez.dim() == Hx.dim() == Hy.dim() == Hz.dim() == 5):
            raise ValueError('All fields must be 5-D tensors [B,C,nx,ny,nz]')

        # cast to consistent dtype
        dtype = self.eps.dtype
        Ex = Ex.to(dtype); Ey = Ey.to(dtype); Ez = Ez.to(dtype)
        Hx = Hx.to(dtype); Hy = Hy.to(dtype); Hz = Hz.to(dtype)

        B, C, nx, ny, nz = Ex.shape

        # --- Update H from E on interior using centered differences
        Hx_new = Hx.clone()
        Hy_new = Hy.clone()
        Hz_new = Hz.clone()
        if nx > 2 and ny > 2 and nz > 2:
            core = (slice(None), slice(None), slice(1, -1), slice(1, -1), slice(1, -1))

            # Hx: (∂Ez/∂y - ∂Ey/∂z)
            dEz_dy = (Ez[:, :, 1:-1, 2:, 1:-1] - Ez[:, :, 1:-1, 0:-2, 1:-1]) / (2.0 * self.dy)
            dEy_dz = (Ey[:, :, 1:-1, 1:-1, 2:] - Ey[:, :, 1:-1, 1:-1, 0:-2]) / (2.0 * self.dz)
            curl_x = dEz_dy - dEy_dz
            Hx_new[core] = Hx[core] - (self.dt / self.mu[core]) * curl_x

            # Hy: (∂Ex/∂z - ∂Ez/∂x)
            dEx_dz = (Ex[:, :, 1:-1, 1:-1, 2:] - Ex[:, :, 1:-1, 1:-1, 0:-2]) / (2.0 * self.dz)
            dEz_dx = (Ez[:, :, 2:, 1:-1, 1:-1] - Ez[:, :, 0:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
            curl_y = dEx_dz - dEz_dx
            Hy_new[core] = Hy[core] - (self.dt / self.mu[core]) * curl_y

            # Hz: (∂Ey/∂x - ∂Ex/∂y)
            dEy_dx = (Ey[:, :, 2:, 1:-1, 1:-1] - Ey[:, :, 0:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
            dEx_dy = (Ex[:, :, 1:-1, 2:, 1:-1] - Ex[:, :, 1:-1, 0:-2, 1:-1]) / (2.0 * self.dy)
            curl_z = dEy_dx - dEx_dy
            Hz_new[core] = Hz[core] - (self.dt / self.mu[core]) * curl_z

        # --- Update E from H: E^{n+1} = (A- / A+) E^{n} + dt/(eps A+) * (curl H)
        Ex_new = (self.A_minus / self.A_plus) * Ex
        Ey_new = (self.A_minus / self.A_plus) * Ey
        Ez_new = (self.A_minus / self.A_plus) * Ez

        if nx > 2 and ny > 2 and nz > 2:
            core = (slice(None), slice(None), slice(1, -1), slice(1, -1), slice(1, -1))

            # For Ex core: (∂Hz/∂y - ∂Hy/∂z)
            dHz_dy = (Hz_new[:, :, 1:-1, 2:, 1:-1] - Hz_new[:, :, 1:-1, 0:-2, 1:-1]) / (2.0 * self.dy)
            dHy_dz = (Hy_new[:, :, 1:-1, 1:-1, 2:] - Hy_new[:, :, 1:-1, 1:-1, 0:-2]) / (2.0 * self.dz)
            curl_ex = dHz_dy - dHy_dz
            Ex_new[core] = (self.A_minus[core] / self.A_plus[core]) * Ex[core] + (self.dt / (self.eps[core] * self.A_plus[core])) * curl_ex

            # For Ey core: (∂Hx/∂z - ∂Hz/∂x)
            dHx_dz = (Hx_new[:, :, 1:-1, 1:-1, 2:] - Hx_new[:, :, 1:-1, 1:-1, 0:-2]) / (2.0 * self.dz)
            dHz_dx = (Hz_new[:, :, 2:, 1:-1, 1:-1] - Hz_new[:, :, 0:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
            curl_ey = dHx_dz - dHz_dx
            Ey_new[core] = (self.A_minus[core] / self.A_plus[core]) * Ey[core] + (self.dt / (self.eps[core] * self.A_plus[core])) * curl_ey

            # For Ez core: (∂Hy/∂x - ∂Hx/∂y)
            dHy_dx = (Hy_new[:, :, 2:, 1:-1, 1:-1] - Hy_new[:, :, 0:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
            dHx_dy = (Hx_new[:, :, 1:-1, 2:, 1:-1] - Hx_new[:, :, 1:-1, 0:-2, 1:-1]) / (2.0 * self.dy)
            curl_ez = dHy_dx - dHx_dy
            Ez_new[core] = (self.A_minus[core] / self.A_plus[core]) * Ez[core] + (self.dt / (self.eps[core] * self.A_plus[core])) * curl_ez

        return Ex_new, Ey_new, Ez_new, Hx_new, Hy_new, Hz_new
