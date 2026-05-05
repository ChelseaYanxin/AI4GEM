"""
Traditional FDTD implementation using explicit loops (to match GT data exactly).

This implementation uses the same loop structure as the MATLAB/Python FDTD code
that generated the ground truth data, ensuring numerical equivalence.
"""
from __future__ import annotations
import torch
from torch import Tensor


class GEMTELoop(torch.nn.Module):
    """Traditional FDTD with explicit loops (matches GT generation code).
    
    This is intentionally NOT vectorized to ensure exact numerical match with
    the ground truth FDTD code that uses nested loops.
    """

    def __init__(self, dx: float, dz: float, dt: float,
                 eps: Tensor, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.dx = float(dx)
        self.dz = float(dz)  # dy in 2D
        self.dt = float(dt)

        # Physical parameter maps
        self.register_buffer('eps', eps)
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        # Pre-compute update coefficients (matching GT code structure)
        self.register_buffer('A_plus', 1.0 + (sigma * dt) / (2.0 * eps))
        self.register_buffer('A_minus', 1.0 - (sigma * dt) / (2.0 * eps))

    @torch.no_grad()
    def step(self, Ez: Tensor, Hx: Tensor, Hy: Tensor):
        """FDTD time step using explicit loops (matches GT code).
        
        Update order (standard Yee leapfrog):
        1. Update H fields using E at current time (forward differences)
        2. Update E fields using H at new time (backward differences)
        """
        if not (Ez.dim() == Hx.dim() == Hy.dim() == 4):
            raise ValueError('Ez, Hx, Hy must be 4-D tensors [B,C,nx,ny]')

        # Cast to buffer dtype
        Ez = Ez.to(self.eps.dtype)
        Hx = Hx.to(self.eps.dtype)
        Hy = Hy.to(self.eps.dtype)

        B, C, nx, ny = Ez.shape
        
        # Remove batch/channel dimensions for loop processing
        ez = Ez[0, 0]  # [nx, ny]
        hx = Hx[0, 0]  # [nx, ny]
        hy = Hy[0, 0]  # [nx, ny]
        
        eps_map = self.eps[0, 0]
        mu_map = self.mu[0, 0]
        A_plus_map = self.A_plus[0, 0]
        A_minus_map = self.A_minus[0, 0]
        
        # Create output tensors (clone to avoid in-place modification)
        hx_new = hx.clone()
        hy_new = hy.clone()
        ez_new = ez.clone()
        
        # ====================================================================
        # Update H fields (matching GT code: forward differences)
        # ====================================================================
        
        # Hx update: dEz/dy forward
        # GT code: Hx[i, j, k] = DA * Hx[i, j, k] - DBy * (Ez[i, j+1, k] - Ez[i, j, k])
        for i in range(nx):
            for j in range(ny - 1):  # j < ny-1 to have Ez[j+1]
                dEz_dy = (ez[i, j + 1] - ez[i, j]) / self.dz
                hx_new[i, j] = hx[i, j] - self.dt / mu_map[i, j] * dEz_dy
        
        # Hy update: dEz/dx forward  
        # GT code: Hy[i, j, k] = DA * Hy[i, j, k] + DBx * (Ez[i+1, j, k] - Ez[i, j, k])
        for i in range(nx - 1):  # i < nx-1 to have Ez[i+1]
            for j in range(ny):
                dEz_dx = (ez[i + 1, j] - ez[i, j]) / self.dx
                hy_new[i, j] = hy[i, j] + self.dt / mu_map[i, j] * dEz_dx
        
        # ====================================================================
        # Update E fields (matching GT code: backward differences, interior only)
        # ====================================================================
        
        # Ez update: curl_z = dHy/dx - dHx/dy
        # GT code: Ez[i, j, k] = CA * Ez[i, j, k] + CBx * (Hy[i, j, k] - Hy[i-1, j, k])
        #                                          - CBy * (Hx[i, j, k] - Hx[i, j-1, k])
        for i in range(1, nx):  # Start from 1 to have Hy[i-1]
            for j in range(1, ny):  # Start from 1 to have Hx[j-1]
                dHy_dx = (hy_new[i, j] - hy_new[i - 1, j]) / self.dx
                dHx_dy = (hx_new[i, j] - hx_new[i, j - 1]) / self.dz
                curl_z = dHy_dx - dHx_dy
                
                # Ez update with loss (sigma)
                ca = A_minus_map[i, j] / A_plus_map[i, j]
                cb = self.dt / (eps_map[i, j] * A_plus_map[i, j])
                ez_new[i, j] = ca * ez[i, j] + cb * curl_z
        
        # Restore batch/channel dimensions
        Ez_out = ez_new.unsqueeze(0).unsqueeze(0)
        Hx_out = hx_new.unsqueeze(0).unsqueeze(0)
        Hy_out = hy_new.unsqueeze(0).unsqueeze(0)
        
        return Ez_out, Hx_out, Hy_out


# Verification test
if __name__ == "__main__":
    print("Testing GEMTELoop (traditional FDTD with loops)")
    
    nx, ny = 51, 51
    dx = dy = 0.01
    dt = 2.335e-11
    
    # Free space parameters
    eps = torch.ones(1, 1, nx, ny)
    mu = torch.ones(1, 1, nx, ny)
    sigma = torch.zeros(1, 1, nx, ny)
    
    model = GEMTELoop(dx, dy, dt, eps, mu, sigma)
    
    # Initialize with Gaussian pulse
    x = torch.linspace(0, (nx-1)*dx, nx)
    y = torch.linspace(0, (ny-1)*dy, ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    x0, y0 = (nx//2)*dx, (ny//2)*dy
    Ez = torch.exp(-((xx-x0)**2 + (yy-y0)**2) / (0.05)**2).unsqueeze(0).unsqueeze(0)
    Hx = torch.zeros(1, 1, nx, ny)
    Hy = torch.zeros(1, 1, nx, ny)
    
    print(f"Grid: {nx}x{ny}, dt={dt}")
    print(f"Initial Ez: min={Ez.min():.3f}, max={Ez.max():.3f}")
    
    # Run a few steps
    for n in range(5):
        Ez, Hx, Hy = model.step(Ez, Hx, Hy)
        print(f"Step {n+1}: Ez max={Ez.abs().max():.6f}")
    
    print("\n✅ GEMTELoop test completed")
    print("This implementation uses explicit loops to match GT FDTD code exactly.")
