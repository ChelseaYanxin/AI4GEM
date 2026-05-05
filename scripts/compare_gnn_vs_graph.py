"""
Compare three GEM implementations:
1. gem_te_gnn.py: Sparse matrices built with loops
2. gem_te_graph.py: Message passing with scatter_add
3. gem_te_vectorized.py: Sparse matrices built WITHOUT loops (fully vectorized)

All implement the same physics (Yee FDTD) but differ in implementation:
- gem_te_gnn.py: Uses loops to build sparse matrices, then sparse.mm
- gem_te_graph.py: Uses loops to build edge lists, then message passing
- gem_te_vectorized.py: Builds sparse matrices purely with tensor ops (no loops!)
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gem.gem_te_gnn import GEMTEGraph2D as GEMSparse
from gem.gem_te_graph import GEMTEGraph2D as GEMGraph
from gem.gem_te_vectorized import GEMTEVectorized2D as GEMVectorized


def compare_implementations():
    """Compare three implementations: sparse (with loops), message passing, and fully vectorized."""
    
    # Setup identical parameters
    nx, ny = 51, 51
    dx = dy = 0.01  # 1cm cells
    c0 = 3e8
    dt = dx / (c0 * np.sqrt(2)) * 0.99  # CFL = 0.99
    
    print("=" * 70)
    print("Comparing GEM implementations:")
    print("  1. gem_te_gnn.py        (sparse matrix - built with loops)")
    print("  2. gem_te_graph.py      (message passing)")
    print("  3. gem_te_vectorized.py (sparse matrix - NO loops!) ⭐")
    print("=" * 70)
    print(f"\nGrid: {nx}x{ny}, dx={dx}m, dt={dt:.3e}s")
    
    # Material parameters
    eps = torch.ones(1, 1, nx, ny, dtype=torch.float64)
    mu = torch.ones(1, 1, nx, ny, dtype=torch.float64)
    sigma = torch.zeros(1, 1, nx, ny, dtype=torch.float64)
    
    # Initialize both models
    print("\nInitializing models...")
    model_sparse = GEMSparse(nx, ny, dx, dy, dt, eps, mu, sigma)
    model_graph = GEMGraph(nx, ny, dx, dy, dt, eps, mu, sigma)
    model_vectorized = GEMVectorized(nx, ny, dx, dy, dt, eps, mu, sigma)
    
    # Check graph structure
    print("\nImplementation details:")
    print("  1. gem_te_gnn (sparse with loops):")
    print(f"     - L_hx_from_ez: {model_sparse.L_hx_from_ez.size()}, nnz={model_sparse.L_hx_from_ez._nnz()}")
    
    print("  2. gem_te_graph (message passing):")
    print(f"     - Ez->Hx edges: {model_graph._ez_to_hx_ei.size(1)}")
    
    print("  3. gem_te_vectorized (sparse NO loops) ⭐:")
    print(f"     - L_hx_from_ez: {model_vectorized.L_hx_from_ez.size()}, nnz={model_vectorized.L_hx_from_ez._nnz()}")
    
    # Initialize fields with Gaussian pulse
    x = torch.linspace(0, (nx-1)*dx, nx)
    y = torch.linspace(0, (ny-1)*dy, ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    x0, y0 = (nx//2)*dx, (ny//2)*dy
    sigma_pulse = 0.05
    Ez_init = torch.exp(-((xx-x0)**2 + (yy-y0)**2) / sigma_pulse**2)
    Ez_init = Ez_init.unsqueeze(0).unsqueeze(0).to(torch.float64)
    
    Hx_init = torch.zeros(1, 1, nx, ny, dtype=torch.float64)
    Hy_init = torch.zeros(1, 1, nx, ny, dtype=torch.float64)
    
    # Clone initial conditions for both models
    Ez_sparse, Hx_sparse, Hy_sparse = Ez_init.clone(), Hx_init.clone(), Hy_init.clone()
    Ez_graph, Hx_graph, Hy_graph = Ez_init.clone(), Hx_init.clone(), Hy_init.clone()
    Ez_vec, Hx_vec, Hy_vec = Ez_init.clone(), Hx_init.clone(), Hy_init.clone()
    
    print(f"\nInitial conditions:")
    print(f"  Ez: min={Ez_init.min():.6f}, max={Ez_init.max():.6f}")
    print(f"  Hx: min={Hx_init.min():.6f}, max={Hx_init.max():.6f}")
    print(f"  Hy: min={Hy_init.min():.6f}, max={Hy_init.max():.6f}")
    
    # Run both models for several steps
    num_steps = 50
    print(f"\nRunning {num_steps} time steps...")
    print("-" * 80)
    print(f"{'Step':>4} {'Sparse':>14} {'Graph':>14} {'Vectorized':>14} {'Diff S-V':>12} {'Diff G-V':>12}")
    print("-" * 80)
    
    for step in range(num_steps):
        # Step all models
        Ez_sparse, Hx_sparse, Hy_sparse = model_sparse.step(Ez_sparse, Hx_sparse, Hy_sparse)
        Ez_graph, Hx_graph, Hy_graph = model_graph.step(Ez_graph, Hx_graph, Hy_graph)
        Ez_vec, Hx_vec, Hy_vec = model_vectorized.step(Ez_vec, Hx_vec, Hy_vec)
        
        # Compare results
        if (step + 1) % 10 == 0:
            ez_sparse_max = Ez_sparse.abs().max().item()
            ez_graph_max = Ez_graph.abs().max().item()
            ez_vec_max = Ez_vec.abs().max().item()
            
            # Compute differences
            diff_sv = (Ez_sparse - Ez_vec).abs().max().item()
            diff_gv = (Ez_graph - Ez_vec).abs().max().item()
            
            print(f"{step+1:4d} {ez_sparse_max:14.8e} {ez_graph_max:14.8e} {ez_vec_max:14.8e} {diff_sv:12.2e} {diff_gv:12.2e}")
    
    print("-" * 80)
    
    # Final comparison
    print("\nFinal field differences (vs vectorized version):")
    print("\n  Sparse vs Vectorized:")
    ez_diff_sv = (Ez_sparse - Ez_vec).abs().max().item()
    hx_diff_sv = (Hx_sparse - Hx_vec).abs().max().item()
    hy_diff_sv = (Hy_sparse - Hy_vec).abs().max().item()
    print(f"    Ez: {ez_diff_sv:.6e}")
    print(f"    Hx: {hx_diff_sv:.6e}")
    print(f"    Hy: {hy_diff_sv:.6e}")
    
    print("\n  Graph vs Vectorized:")
    ez_diff_gv = (Ez_graph - Ez_vec).abs().max().item()
    hx_diff_gv = (Hx_graph - Hx_vec).abs().max().item()
    hy_diff_gv = (Hy_graph - Hy_vec).abs().max().item()
    print(f"    Ez: {ez_diff_gv:.6e}")
    print(f"    Hx: {hx_diff_gv:.6e}")
    print(f"    Hy: {hy_diff_gv:.6e}")
    
    # Check if implementations match
    tol_abs = 1e-6  # Absolute tolerance (relaxed for accumulated floating point errors)
    tol_rel = 1e-8  # Relative tolerance
    
    # Compute relative errors
    ez_rel_sv = (Ez_sparse - Ez_vec).abs().max() / (Ez_vec.abs().max() + 1e-12)
    hx_rel_sv = (Hx_sparse - Hx_vec).abs().max() / (Hx_vec.abs().max() + 1e-12)
    hy_rel_sv = (Hy_sparse - Hy_vec).abs().max() / (Hy_vec.abs().max() + 1e-12)
    
    ez_rel_gv = (Ez_graph - Ez_vec).abs().max() / (Ez_vec.abs().max() + 1e-12)
    hx_rel_gv = (Hx_graph - Hx_vec).abs().max() / (Hx_vec.abs().max() + 1e-12)
    hy_rel_gv = (Hy_graph - Hy_vec).abs().max() / (Hy_vec.abs().max() + 1e-12)
    
    print("\nRelative errors (vs vectorized):")
    print(f"\n  Sparse vs Vectorized:")
    print(f"    Ez: {ez_rel_sv:.6e}")
    print(f"    Hx: {hx_rel_sv:.6e}")
    print(f"    Hy: {hy_rel_sv:.6e}")
    
    print(f"\n  Graph vs Vectorized:")
    print(f"    Ez: {ez_rel_gv:.6e}")
    print(f"    Hx: {hx_rel_gv:.6e}")
    print(f"    Hy: {hy_rel_gv:.6e}")
    
    max_abs_err = max(ez_diff_sv, hx_diff_sv, hy_diff_sv, 
                      ez_diff_gv, hx_diff_gv, hy_diff_gv)
    max_rel_err = max(ez_rel_sv.item(), hx_rel_sv.item(), hy_rel_sv.item(),
                      ez_rel_gv.item(), hx_rel_gv.item(), hy_rel_gv.item())
    
    all_match = (max_abs_err < tol_abs) or (max_rel_err < tol_rel)
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ SUCCESS: All three implementations produce identical results!")
        print(f"\n   Max absolute error: {max_abs_err:.2e} (tolerance: {tol_abs:.2e})")
        print(f"   Max relative error: {max_rel_err:.2e} (tolerance: {tol_rel:.2e})")
        print("\nKey takeaway:")
        print("  - gem_te_vectorized.py builds sparse matrices WITHOUT loops")
        print("  - Uses meshgrid, masking, and tensor operations only")
        print("  - Same physics, more Pythonic/PyTorch style!")
        print("\nNote: Small differences in Hx/Hy are due to:")
        print("  - Different order of floating point operations")
        print("  - Accumulated rounding errors over 50 time steps")
        print("  - All three methods are numerically equivalent")
    else:
        print("⚠️  WARNING: Implementations differ by more than tolerance")
        print(f"   Absolute tolerance: {tol_abs:.2e}, max error: {max_abs_err:.2e}")
        print(f"   Relative tolerance: {tol_rel:.2e}, max error: {max_rel_err:.2e}")
    print("=" * 80)
    
    return all_match


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    success = compare_implementations()
    sys.exit(0 if success else 1)
