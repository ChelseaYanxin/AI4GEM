"""
GEM 2D TE mode solver using pure graph operations (Message Passing).

This implementation represents the FDTD Yee grid as a graph where:
- Nodes: field components (Ez, Hx, Hy) at each spatial location
- Edges: spatial derivative relationships (forward/backward differences)
- Message Passing: spatial derivatives are computed by aggregating neighbor values

Compared to gem_te_gnn.py which uses sparse matrices, this version uses
explicit graph operations (message passing) which is more flexible for
irregular grids and easier to extend with learned components.
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple


class GEMTEGraph2D(torch.nn.Module):
    """GEM 2D TMz solver using graph message passing.
    
    Graph structure:
    - 3 node types: Ez, Hx, Hy (each has nx*ny nodes)
    - Edge types:
      * Ez -> Hx: forward difference in y direction
      * Ez -> Hy: forward difference in x direction  
      * Hx -> Ez: backward difference in y direction
      * Hy -> Ez: backward difference in x direction
    
    Message passing:
    1. H update: H receives messages from E neighbors via forward diff edges
    2. E update: E receives messages from H neighbors via backward diff edges
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
        
        # Build graph connectivity
        self._build_graph()
    
    def _build_graph(self):
        """Build graph edges for message passing on Yee staggered grid.
        
        Yee grid staggering (physical positions):
        - Ez at (i, j)
        - Hx at (i, j+1/2)  [stored as Hx[i,j], represents midpoint]
        - Hy at (i+1/2, j)  [stored as Hy[i,j], represents midpoint]
        
        Edge types:
        1. Ez -> Hx (forward diff y): Hx[i,j] needs Ez[i,j+1] - Ez[i,j]
        2. Ez -> Hy (forward diff x): Hy[i,j] needs Ez[i+1,j] - Ez[i,j]
        3. Hy -> Ez (backward diff x): Ez[i,j] needs Hy[i,j] - Hy[i-1,j]
        4. Hx -> Ez (backward diff y): Ez[i,j] needs Hx[i,j] - Hx[i,j-1]
        """
        nx, ny = self.nx, self.ny
        device = self.eps.device
        
        def idx(i: int, j: int) -> int:
            """Convert 2D grid index to 1D node index."""
            return i * ny + j
        
        # Edge lists: [source_node, target_node, edge_weight]
        ez_to_hx_edges = []  # For H update: forward diff in y
        ez_to_hy_edges = []  # For H update: forward diff in x
        hy_to_ez_edges = []  # For E update: backward diff in x
        hx_to_ez_edges = []  # For E update: backward diff in y
        
        inv_dx = 1.0 / self.dx
        inv_dy = 1.0 / self.dy
        
        # Build edges for H update (forward differences)
        # Hx[i,j] at (i, j+1/2) needs dEz/dy = [Ez(i,j+1) - Ez(i,j)] / dy
        for i in range(nx):
            for j in range(ny - 1):
                tgt = idx(i, j)  # Hx[i,j]
                src_plus = idx(i, j + 1)  # Ez[i, j+1]
                src_zero = idx(i, j)      # Ez[i, j]
                
                ez_to_hx_edges.append([src_plus, tgt, +inv_dy])
                ez_to_hx_edges.append([src_zero, tgt, -inv_dy])
        
        # Hy[i,j] at (i+1/2, j) needs dEz/dx = [Ez(i+1,j) - Ez(i,j)] / dx
        for i in range(nx - 1):
            for j in range(ny):
                tgt = idx(i, j)  # Hy[i,j]
                src_plus = idx(i + 1, j)  # Ez[i+1, j]
                src_zero = idx(i, j)      # Ez[i, j]
                
                ez_to_hy_edges.append([src_plus, tgt, +inv_dx])
                ez_to_hy_edges.append([src_zero, tgt, -inv_dx])
        
        # Build edges for E update (backward differences, interior only)
        # Ez[i,j] needs dHy/dx = [Hy(i,j) - Hy(i-1,j)] / dx
        #             and dHx/dy = [Hx(i,j) - Hx(i,j-1)] / dy
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                tgt = idx(i, j)  # Ez[i,j]
                
                # dHy/dx contribution
                src_hy_0 = idx(i, j)      # Hy[i, j]
                src_hy_m = idx(i - 1, j)  # Hy[i-1, j]
                hy_to_ez_edges.append([src_hy_0, tgt, +inv_dx])
                hy_to_ez_edges.append([src_hy_m, tgt, -inv_dx])
                
                # dHx/dy contribution
                src_hx_0 = idx(i, j)      # Hx[i, j]
                src_hx_m = idx(i, j - 1)  # Hx[i, j-1]
                hx_to_ez_edges.append([src_hx_0, tgt, -inv_dy])  # Note: negative!
                hx_to_ez_edges.append([src_hx_m, tgt, +inv_dy])
        
        # Convert to tensors
        def make_edge_index_and_weight(edges):
            if len(edges) == 0:
                return (torch.empty((2, 0), dtype=torch.long, device=device),
                        torch.empty((0,), dtype=torch.float32, device=device))
            edges_tensor = torch.tensor(edges, dtype=torch.float32, device=device)
            edge_index = edges_tensor[:, :2].t().long()  # [2, num_edges]
            edge_weight = edges_tensor[:, 2]              # [num_edges]
            return edge_index, edge_weight
        
        # Store edge indices and weights
        self.ez_to_hx_edge_index, self.ez_to_hx_edge_weight = \
            make_edge_index_and_weight(ez_to_hx_edges)
        self.ez_to_hy_edge_index, self.ez_to_hy_edge_weight = \
            make_edge_index_and_weight(ez_to_hy_edges)
        self.hy_to_ez_edge_index, self.hy_to_ez_edge_weight = \
            make_edge_index_and_weight(hy_to_ez_edges)
        self.hx_to_ez_edge_index, self.hx_to_ez_edge_weight = \
            make_edge_index_and_weight(hx_to_ez_edges)
        
        self.register_buffer('_ez_to_hx_ei', self.ez_to_hx_edge_index)
        self.register_buffer('_ez_to_hx_ew', self.ez_to_hx_edge_weight)
        self.register_buffer('_ez_to_hy_ei', self.ez_to_hy_edge_index)
        self.register_buffer('_ez_to_hy_ew', self.ez_to_hy_edge_weight)
        self.register_buffer('_hy_to_ez_ei', self.hy_to_ez_edge_index)
        self.register_buffer('_hy_to_ez_ew', self.hy_to_ez_edge_weight)
        self.register_buffer('_hx_to_ez_ei', self.hx_to_ez_edge_index)
        self.register_buffer('_hx_to_ez_ew', self.hx_to_ez_edge_weight)
    
    def _message_passing(self, src_values: Tensor, edge_index: Tensor, 
                        edge_weight: Tensor, num_nodes: int) -> Tensor:
        """Perform message passing: aggregate neighbor values with edge weights.
        
        Args:
            src_values: [num_nodes] source node values
            edge_index: [2, num_edges] edge connectivity [source, target]
            edge_weight: [num_edges] edge weights (derivative coefficients)
            num_nodes: number of target nodes
            
        Returns:
            aggregated: [num_nodes] aggregated messages at each target node
        """
        if edge_index.size(1) == 0:
            return torch.zeros(num_nodes, device=src_values.device, dtype=src_values.dtype)
        
        # Get source values for each edge
        src_idx = edge_index[0]  # [num_edges]
        tgt_idx = edge_index[1]  # [num_edges]
        
        # Compute messages: src_value * edge_weight
        messages = src_values[src_idx] * edge_weight  # [num_edges]
        
        # Aggregate messages at target nodes (sum)
        aggregated = torch.zeros(num_nodes, device=src_values.device, dtype=src_values.dtype)
        aggregated.scatter_add_(0, tgt_idx, messages)
        
        return aggregated
    
    def _flatten(self, x: Tensor) -> Tensor:
        """Flatten [B,C,nx,ny] to [N] for graph operations."""
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
        """Advance fields by one time step using graph message passing.
        
        Message passing flow:
        1. H update: H <-- aggregate messages from E neighbors (forward diff)
        2. E update: E <-- aggregate messages from H neighbors (backward diff)
        
        Args:
            Ez, Hx, Hy: [B,C,nx,ny] field tensors
            
        Returns:
            Ez_new, Hx_new, Hy_new: updated fields
        """
        # Flatten to node features
        ez = self._flatten(Ez.to(self.eps.dtype))
        hx = self._flatten(Hx.to(self.eps.dtype))
        hy = self._flatten(Hy.to(self.eps.dtype))
        
        N = self.nx * self.ny
        eps_v = self._flatten(self.eps)
        mu_v = self._flatten(self.mu)
        Aplus_v = self._flatten(self.A_plus)
        Aminus_v = self._flatten(self.A_minus)
        
        # ========== Step 1: Update H using messages from E ==========
        # Hx update: Hx^{n+1/2} = Hx^{n-1/2} - dt/mu * dEz/dy
        dez_dy = self._message_passing(ez, self._ez_to_hx_ei, self._ez_to_hx_ew, N)
        hx_new = hx - self.dt * (dez_dy / mu_v)
        
        # Hy update: Hy^{n+1/2} = Hy^{n-1/2} + dt/mu * dEz/dx
        dez_dx = self._message_passing(ez, self._ez_to_hy_ei, self._ez_to_hy_ew, N)
        hy_new = hy + self.dt * (dez_dx / mu_v)
        
        # ========== Step 2: Update E using messages from H ==========
        # Ez update: curl_z = dHy/dx - dHx/dy
        dhy_dx = self._message_passing(hy_new, self._hy_to_ez_ei, self._hy_to_ez_ew, N)
        dhx_dy = self._message_passing(hx_new, self._hx_to_ez_ei, self._hx_to_ez_ew, N)
        curl_z = dhy_dx + dhx_dy  # Note: dhx_dy edges already have negative weight
        
        # Ez^{n+1} = (A-/A+) Ez^n + dt/(eps*A+) * curl_z
        ez_new = (Aminus_v / Aplus_v) * ez + (self.dt / (eps_v * Aplus_v)) * curl_z
        
        # Reshape back to [B,C,nx,ny]
        Ez_o = self._unflatten(ez_new)
        Hx_o = self._unflatten(hx_new)
        Hy_o = self._unflatten(hy_new)
        
        # Apply PEC boundary conditions
        Ez_o[..., 0, :] = 0
        Ez_o[..., -1, :] = 0
        Ez_o[..., :, 0] = 0
        Ez_o[..., :, -1] = 0
        
        return Ez_o, Hx_o, Hy_o


# Example usage and comparison
if __name__ == "__main__":
    import time
    
    # Setup
    nx, ny = 101, 101
    dx = dy = 0.01
    dt = dx / (2 * 3e8 * (2**0.5)) * 0.99
    
    eps = torch.ones(1, 1, nx, ny)
    mu = torch.ones(1, 1, nx, ny)
    sigma = torch.zeros(1, 1, nx, ny)
    
    # Initialize model
    model = GEMTEGraph2D(nx, ny, dx, dy, dt, eps, mu, sigma)
    
    # Initialize fields with Gaussian pulse
    x = torch.linspace(0, (nx-1)*dx, nx)
    y = torch.linspace(0, (ny-1)*dy, ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    x0, y0 = (nx//2)*dx, (ny//2)*dy
    Ez = torch.exp(-((xx-x0)**2 + (yy-y0)**2) / (0.05)**2).unsqueeze(0).unsqueeze(0)
    Hx = torch.zeros(1, 1, nx, ny)
    Hy = torch.zeros(1, 1, nx, ny)
    
    # Run simulation
    print("Running GEM graph-based 2D FDTD simulation...")
    print(f"Grid: {nx}x{ny}, dt={dt:.3e}s")
    
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
    print("\nGraph statistics:")
    print(f"  Ez->Hx edges: {model._ez_to_hx_ei.size(1)}")
    print(f"  Ez->Hy edges: {model._ez_to_hy_ei.size(1)}")
    print(f"  Hy->Ez edges: {model._hy_to_ez_ei.size(1)}")
    print(f"  Hx->Ez edges: {model._hx_to_ez_ei.size(1)}")
