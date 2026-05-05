"""
GEM 2D TE mode solver with fully vectorized sparse matrix construction.

This implementation eliminates ALL explicit loops:
- Sparse matrix construction: vectorized using meshgrid and reshape
- Time stepping: sparse matrix multiplication only

Compared to gem_te_gnn.py which uses loops to build sparse matrices,
this version constructs them purely through tensor operations.
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple


class GEMTEVectorized2D(torch.nn.Module):
    """Fully vectorized GEM 2D TMz solver.
    
    Key innovation: sparse matrices are built without ANY explicit loops,
    using only tensor operations (meshgrid, reshape, stack, etc.)
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float,
                 eps: Tensor, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.nx, self.ny = int(nx), int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        
        dty = torch.get_default_dtype()
        self.register_buffer('eps', eps.to(dty))
        self.register_buffer('mu', mu.to(dty))
        self.register_buffer('sigma', sigma.to(dty))
        
        # Conductivity factors
        A_plus = 1.0 + (self.sigma * self.dt) / (2.0 * self.eps)
        A_minus = 1.0 - (self.sigma * self.dt) / (2.0 * self.eps)
        self.register_buffer('A_plus', A_plus)
        self.register_buffer('A_minus', A_minus)
        
        # Build sparse operators without loops
        self._build_operators_vectorized()
    
    def _build_operators_vectorized(self):
        """Build sparse difference operators using pure tensor operations.
        
        No explicit loops - everything is vectorized using meshgrid and masking.
        
        Yee grid staggering:
        - Ez at (i, j)
        - Hx at (i, j+1/2): needs dEz/dy = [Ez(i,j+1) - Ez(i,j)] / dy
        - Hy at (i+1/2, j): needs dEz/dx = [Ez(i+1,j) - Ez(i,j)] / dx
        """
        nx, ny = self.nx, self.ny
        N = nx * ny
        device = self.eps.device
        
        # Create index grids (vectorized alternative to nested loops)
        i_grid = torch.arange(nx, device=device).view(nx, 1).expand(nx, ny)
        j_grid = torch.arange(ny, device=device).view(1, ny).expand(nx, ny)
        
        def idx_tensor(i: Tensor, j: Tensor) -> Tensor:
            """Vectorized 2D to 1D index mapping."""
            return i * ny + j
        
        inv_dx = 1.0 / self.dx
        inv_dy = 1.0 / self.dy
        
        # ============ H update operators (forward differences) ============
        
        # Hx update: dEz/dy forward
        # Valid for j < ny-1 (need j+1 neighbor)
        mask_hx = j_grid < (ny - 1)
        i_hx = i_grid[mask_hx]
        j_hx = j_grid[mask_hx]
        
        tgt_hx = idx_tensor(i_hx, j_hx)
        src_hx_plus = idx_tensor(i_hx, j_hx + 1)
        src_hx_zero = idx_tensor(i_hx, j_hx)
        
        # Stack: [source, target] pairs
        hx_edge_index = torch.stack([
            torch.cat([src_hx_plus, src_hx_zero]),
            torch.cat([tgt_hx, tgt_hx])
        ], dim=0)
        hx_edge_weight = torch.cat([
            torch.full_like(tgt_hx, inv_dy, dtype=self.eps.dtype),
            torch.full_like(tgt_hx, -inv_dy, dtype=self.eps.dtype)
        ])
        
        # Hy update: dEz/dx forward
        # Valid for i < nx-1 (need i+1 neighbor)
        mask_hy = i_grid < (nx - 1)
        i_hy = i_grid[mask_hy]
        j_hy = j_grid[mask_hy]
        
        tgt_hy = idx_tensor(i_hy, j_hy)
        src_hy_plus = idx_tensor(i_hy + 1, j_hy)
        src_hy_zero = idx_tensor(i_hy, j_hy)
        
        hy_edge_index = torch.stack([
            torch.cat([src_hy_plus, src_hy_zero]),
            torch.cat([tgt_hy, tgt_hy])
        ], dim=0)
        hy_edge_weight = torch.cat([
            torch.full_like(tgt_hy, inv_dx, dtype=self.eps.dtype),
            torch.full_like(tgt_hy, -inv_dx, dtype=self.eps.dtype)
        ])
        
        # ============ E update operators (backward differences, interior only) ============
        
        # Ez update: needs interior points only (1 <= i < nx-1, 1 <= j < ny-1)
        mask_ez = (i_grid >= 1) & (i_grid < nx - 1) & (j_grid >= 1) & (j_grid < ny - 1)
        i_ez = i_grid[mask_ez]
        j_ez = j_grid[mask_ez]
        tgt_ez = idx_tensor(i_ez, j_ez)
        
        # dHy/dx backward: Hy[i,j] - Hy[i-1,j]
        src_hy_0 = idx_tensor(i_ez, j_ez)
        src_hy_m = idx_tensor(i_ez - 1, j_ez)
        
        ez_from_hy_index = torch.stack([
            torch.cat([src_hy_0, src_hy_m]),
            torch.cat([tgt_ez, tgt_ez])
        ], dim=0)
        ez_from_hy_weight = torch.cat([
            torch.full_like(tgt_ez, inv_dx, dtype=self.eps.dtype),
            torch.full_like(tgt_ez, -inv_dx, dtype=self.eps.dtype)
        ])
        
        # dHx/dy backward: Hx[i,j] - Hx[i,j-1] (with negative sign for curl)
        src_hx_0 = idx_tensor(i_ez, j_ez)
        src_hx_m = idx_tensor(i_ez, j_ez - 1)
        
        ez_from_hx_index = torch.stack([
            torch.cat([src_hx_0, src_hx_m]),
            torch.cat([tgt_ez, tgt_ez])
        ], dim=0)
        ez_from_hx_weight = torch.cat([
            torch.full_like(tgt_ez, -inv_dy, dtype=self.eps.dtype),
            torch.full_like(tgt_ez, +inv_dy, dtype=self.eps.dtype)
        ])
        
        # Convert edge lists to sparse COO tensors
        def make_sparse(edge_index, edge_weight):
            return torch.sparse_coo_tensor(
                edge_index.long(),
                edge_weight.to(self.eps.dtype),  # Match dtype with eps/mu
                (N, N),
                device=device
            ).coalesce()
        
        self.register_buffer('L_hx_from_ez', make_sparse(hx_edge_index, hx_edge_weight))
        self.register_buffer('L_hy_from_ez', make_sparse(hy_edge_index, hy_edge_weight))
        self.register_buffer('L_ez_from_hy', make_sparse(ez_from_hy_index, ez_from_hy_weight))
        self.register_buffer('L_ez_from_hx', make_sparse(ez_from_hx_index, ez_from_hx_weight))
    
    def _flatten(self, x: Tensor) -> Tensor:
        """Flatten [B,C,nx,ny] to [N]."""
        if x.dim() != 4:
            raise ValueError('Expected [B,C,nx,ny]')
        if x.size(0) != 1 or x.size(1) != 1:
            x = x.reshape(-1, x.shape[-2], x.shape[-1]).mean(0, keepdim=True).mean(0)
        return x.view(-1)
    
    def _unflatten(self, v: Tensor) -> Tensor:
        """Reshape [N] to [1,1,nx,ny]."""
        return v.view(1, 1, self.nx, self.ny)
    
    @torch.no_grad()
    def step(self, Ez: Tensor, Hx: Tensor, Hy: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Advance fields by one time step.
        
        Pure matrix operations - no loops anywhere!
        """
        # Flatten to vectors
        ez = self._flatten(Ez.to(self.eps.dtype))
        hx = self._flatten(Hx.to(self.eps.dtype))
        hy = self._flatten(Hy.to(self.eps.dtype))
        
        N = self.nx * self.ny
        eps_v = self._flatten(self.eps)
        mu_v = self._flatten(self.mu)
        Aplus_v = self._flatten(self.A_plus)
        Aminus_v = self._flatten(self.A_minus)
        
        # ========== H update: sparse matrix multiplication ==========
        # Hx -= dt/mu * dEz/dy
        dez_dy = torch.sparse.mm(self.L_hx_from_ez, ez.view(N, 1)).view(-1)
        hx_new = hx - self.dt * (dez_dy / mu_v)
        
        # Hy += dt/mu * dEz/dx
        dez_dx = torch.sparse.mm(self.L_hy_from_ez, ez.view(N, 1)).view(-1)
        hy_new = hy + self.dt * (dez_dx / mu_v)
        
        # ========== E update: sparse matrix multiplication ==========
        # curl_z = dHy/dx - dHx/dy
        dhy_dx = torch.sparse.mm(self.L_ez_from_hy, hy_new.view(N, 1)).view(-1)
        dhx_dy = torch.sparse.mm(self.L_ez_from_hx, hx_new.view(N, 1)).view(-1)
        curl_z = dhy_dx + dhx_dy  # Note: dhx_dy weights already negative
        
        # Ez update with loss
        ez_new = (Aminus_v / Aplus_v) * ez + (self.dt / (eps_v * Aplus_v)) * curl_z
        
        # Reshape back
        Ez_o = self._unflatten(ez_new)
        Hx_o = self._unflatten(hx_new)
        Hy_o = self._unflatten(hy_new)
        
        # PEC boundary conditions
        Ez_o[..., 0, :] = 0
        Ez_o[..., -1, :] = 0
        Ez_o[..., :, 0] = 0
        Ez_o[..., :, -1] = 0
        
        return Ez_o, Hx_o, Hy_o


# Example demonstrating the vectorization
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("GEM 2D Fully Vectorized Implementation Demo")
    print("=" * 70)
    
    # Setup
    nx, ny = 101, 101
    dx = dy = 0.01
    dt = dx / (2 * 3e8 * (2**0.5)) * 0.99
    
    eps = torch.ones(1, 1, nx, ny)
    mu = torch.ones(1, 1, nx, ny)
    sigma = torch.zeros(1, 1, nx, ny)
    
    print(f"\nGrid: {nx}x{ny}, dt={dt:.3e}s")
    print("\nBuilding sparse operators (fully vectorized - no loops)...")
    
    start = time.time()
    model = GEMTEVectorized2D(nx, ny, dx, dy, dt, eps, mu, sigma)
    build_time = time.time() - start
    
    print(f"Build time: {build_time*1000:.2f}ms")
    print("\nSparse operator statistics:")
    print(f"  L_hx_from_ez: {model.L_hx_from_ez._nnz()} non-zeros")
    print(f"  L_hy_from_ez: {model.L_hy_from_ez._nnz()} non-zeros")
    print(f"  L_ez_from_hy: {model.L_ez_from_hy._nnz()} non-zeros")
    print(f"  L_ez_from_hx: {model.L_ez_from_hx._nnz()} non-zeros")
    
    # Initialize fields
    x = torch.linspace(0, (nx-1)*dx, nx)
    y = torch.linspace(0, (ny-1)*dy, ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    x0, y0 = (nx//2)*dx, (ny//2)*dy
    Ez = torch.exp(-((xx-x0)**2 + (yy-y0)**2) / (0.05)**2).unsqueeze(0).unsqueeze(0)
    Hx = torch.zeros(1, 1, nx, ny)
    Hy = torch.zeros(1, 1, nx, ny)
    
    # Run simulation
    print("\nRunning simulation (pure matrix ops - no loops)...")
    num_steps = 100
    start = time.time()
    
    for n in range(num_steps):
        Ez, Hx, Hy = model.step(Ez, Hx, Hy)
        
        if (n + 1) % 20 == 0:
            ez_max = Ez.abs().max().item()
            print(f"Step {n+1:3d}: |Ez|_max = {ez_max:.6e}")
    
    elapsed = time.time() - start
    print(f"\nCompleted {num_steps} steps in {elapsed:.3f}s")
    print(f"Time per step: {elapsed/num_steps*1000:.2f}ms")
    print("\n" + "=" * 70)
    print("✅ All operations completed without explicit loops!")
    print("=" * 70)
