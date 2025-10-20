from __future__ import annotations
import torch
from torch import Tensor


class GEMTE(torch.nn.Module):
        """2D TMz-style update with naming: Ez (out-of-plane), Hx, Hy.

        Layout: all fields are [B, C, nx, ny]. Constructor uses (dx, dz) for
        backward compatibility; here we treat dz as dy (y spacing).
        """

        def __init__(self, dx: float, dz: float, dt: float,
                                eps: Tensor, mu: Tensor, sigma: Tensor):
                super().__init__()
                self.dx = float(dx)
                self.dz = float(dz)  # interpreted as dy
                self.dt = float(dt)

                # physical maps (registered as buffers so they move with the module)
                self.register_buffer('eps', eps)
                self.register_buffer('mu', mu)
                self.register_buffer('sigma', sigma)

                # conductive update factors (elementwise) for E update
                self.register_buffer('A_plus', 1.0 + (sigma * dt) / (2.0 * eps))
                self.register_buffer('A_minus', 1.0 - (sigma * dt) / (2.0 * eps))

        @torch.no_grad()
        def step(self, Ez: Tensor, Hx: Tensor, Hy: Tensor):
                """One Yee-staggered leapfrog step (TMz):

                - Hx, Hy updated at half step using spatial forward differences of Ez.
                - Ez updated at full step using centered curl of (Hx,Hy) at interior cells.

                All tensors keep shape [B,C,nx,ny]. Interior points are updated; edges retain
                their previous values (caller may apply boundary conditions afterward).
                """
                if not (Ez.dim() == Hx.dim() == Hy.dim() == 4):
                        raise ValueError('Ez, Hx, Hy must be 4-D tensors [B,C,nx,ny]')

                # cast to buffer dtype
                Ez = Ez.to(self.eps.dtype)
                Hx = Hx.to(self.eps.dtype)
                Hy = Hy.to(self.eps.dtype)

                B, C, nx, ny = Ez.shape

                # --- Update H from Ez (half-step in time, forward difference in space)
                # Hx(i, j+1/2)^{n+1/2} = Hx(i, j+1/2)^{n-1/2} - dt/(mu*dy) * (Ez(i, j+1)^n - Ez(i, j)^n)
                Hx_new = Hx.clone()
                if ny > 1:
                        dEz_dy = Ez[..., :, 1:] - Ez[..., :, :-1]  # [B,C,nx,ny-1]
                        Hx_new[..., :, :-1] = Hx[..., :, :-1] - (self.dt / self.dz) * (dEz_dy / self.mu[..., :, :-1])
                        # last column (j=ny-1) unchanged here; boundary handler will fix

                # Hy(i+1/2, j)^{n+1/2} = Hy(i+1/2, j)^{n-1/2} + dt/(mu*dx) * (Ez(i+1, j)^n - Ez(i, j)^n)
                Hy_new = Hy.clone()
                if nx > 1:
                        dEz_dx = Ez[..., 1:, :] - Ez[..., :-1, :]  # [B,C,nx-1,ny]
                        Hy_new[..., :-1, :] = Hy[..., :-1, :] + (self.dt / self.dx) * (dEz_dx / self.mu[..., :-1, :])
                        # last row (i=nx-1) unchanged

                # --- Update Ez from H (full step using centered curl at interior cells)
                Ez_new = (self.A_minus / self.A_plus) * Ez
                if nx > 2 and ny > 2:
                        # Centered curl aligned to Ez interior [1:-1,1:-1]
                        # dHy/dx at centers: Hy[i,j] - Hy[i-1,j]
                        dHy_dx_center = Hy_new[..., 1:-1, 1:-1] - Hy_new[..., 0:-2, 1:-1]  # [B,C,nx-2,ny-2]
                        # dHx/dy at centers: Hx[i,j] - Hx[i,j-1]
                        dHx_dy_center = Hx_new[..., 1:-1, 1:-1] - Hx_new[..., 1:-1, 0:-2]  # [B,C,nx-2,ny-2]

                        curl_z = (dHy_dx_center / (self.dx * self.eps[..., 1:-1, 1:-1])) \
                                 - (dHx_dy_center / (self.dz * self.eps[..., 1:-1, 1:-1]))

                        Ez_new[..., 1:-1, 1:-1] = (self.A_minus[..., 1:-1, 1:-1] / self.A_plus[..., 1:-1, 1:-1]) * Ez[..., 1:-1, 1:-1] \
                                + (self.dt / self.A_plus[..., 1:-1, 1:-1]) * curl_z

                return Ez_new, Hx_new, Hy_new

