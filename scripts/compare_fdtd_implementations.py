"""
Compare three FDTD implementations against ground truth data:
1. gem_te.py (original vectorized version)
2. gem_te_loop.py (explicit loops matching GT code)
3. Ground truth data

This will help determine which implementation matches the GT data exactly.
"""

import sys
import torch
import scipy.io as sio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gem.gem_te import GEMTE as GEMTEVectorized
from gem.gem_te_loop import GEMTELoop


def load_gt_data():
    """Load ground truth FDTD data."""
    data_path = Path(__file__).parent.parent / "data/data_FDTD_2D_cavity/2D_TM_all_processed_jiequ_200_300.mat"
    data = sio.loadmat(str(data_path))
    
    Ez_gt = torch.as_tensor(data['Ez'], dtype=torch.float64)  # [nx, ny, nt]
    Hx_gt = torch.as_tensor(data['Hx'], dtype=torch.float64)
    Hy_gt = torch.as_tensor(data['Hy'], dtype=torch.float64)
    
    # Get spatial grid
    x = torch.as_tensor(data['x'], dtype=torch.float64).squeeze()
    y = torch.as_tensor(data['y'], dtype=torch.float64).squeeze()
    
    # Check if we need to transpose (xx_2d variance test)
    xx_2d = data['xx_2d']
    yy_2d = data['yy_2d']
    col_var = float(xx_2d[:, 0].var())
    row_var = float(xx_2d[0, :].var())
    
    if col_var >= row_var:
        print("Detected [y,x] order - transposing to [x,y]")
        Ez_gt = Ez_gt.transpose(0, 1)
        Hx_gt = Hx_gt.transpose(0, 1)
        Hy_gt = Hy_gt.transpose(0, 1)
    
    # Get time step
    t = torch.as_tensor(data['t'], dtype=torch.float64).squeeze()
    if len(t) >= 2:
        dt = float((t[1:] - t[:-1]).mean())
    else:
        dt = 2.335e-11
    
    # Get grid spacing
    dx = float((x[1:] - x[:-1]).mean())
    dy = float((y[1:] - y[:-1]).mean())
    
    return Ez_gt, Hx_gt, Hy_gt, dx, dy, dt


def compare_implementations():
    """Compare vectorized vs loop implementations against GT."""
    
    print("=" * 80)
    print("Comparing FDTD Implementations vs Ground Truth")
    print("=" * 80)
    
    # Load GT data
    Ez_gt, Hx_gt, Hy_gt, dx, dy, dt = load_gt_data()
    nx, ny, nt = Ez_gt.shape
    
    print(f"\nGround Truth Data:")
    print(f"  Grid: {nx}x{ny}, {nt} time steps")
    print(f"  dx={dx:.3e}m, dy={dy:.3e}m, dt={dt:.3e}s")
    
    # Material parameters (free space)
    eps = torch.ones(1, 1, nx, ny, dtype=torch.float64)
    mu = torch.ones(1, 1, nx, ny, dtype=torch.float64)
    sigma = torch.zeros(1, 1, nx, ny, dtype=torch.float64)
    
    # Initialize models
    model_vec = GEMTEVectorized(dx, dy, dt, eps, mu, sigma)
    model_loop = GEMTELoop(dx, dy, dt, eps, mu, sigma)
    
    print("\nModels:")
    print("  1. gem_te.py (vectorized)")
    print("  2. gem_te_loop.py (explicit loops)")
    
    # Initialize fields from GT frame 0
    Ez_vec = Ez_gt[:, :, 0].unsqueeze(0).unsqueeze(0).clone()
    Hx_vec = Hx_gt[:, :, 0].unsqueeze(0).unsqueeze(0).clone()
    Hy_vec = Hy_gt[:, :, 0].unsqueeze(0).unsqueeze(0).clone()
    
    Ez_loop = Ez_vec.clone()
    Hx_loop = Hx_vec.clone()
    Hy_loop = Hy_vec.clone()
    
    # Run simulation and compare
    num_steps = min(10, nt - 1)
    print(f"\nRunning {num_steps} steps...")
    print("-" * 80)
    print(f"{'Step':>4} {'Vec Ez Err':>12} {'Loop Ez Err':>12} {'Vec Hx Err':>12} {'Loop Hx Err':>12}")
    print("-" * 80)
    
    for n in range(num_steps):
        # Step both models
        Ez_vec, Hx_vec, Hy_vec = model_vec.step(Ez_vec, Hx_vec, Hy_vec)
        Ez_loop, Hx_loop, Hy_loop = model_loop.step(Ez_loop, Hx_loop, Hy_loop)
        
        # Apply PEC boundary conditions (matching GT)
        Ez_vec[..., 0, :] = 0
        Ez_vec[..., -1, :] = 0
        Ez_vec[..., :, 0] = 0
        Ez_vec[..., :, -1] = 0
        
        Ez_loop[..., 0, :] = 0
        Ez_loop[..., -1, :] = 0
        Ez_loop[..., :, 0] = 0
        Ez_loop[..., :, -1] = 0
        
        # Compare with GT frame n+1
        gt_idx = n + 1
        ez_gt = Ez_gt[:, :, gt_idx]
        hx_gt = Hx_gt[:, :, gt_idx]
        
        # Compute relative errors
        eps_rel = 1e-12
        
        # Vectorized version errors
        ez_vec_2d = Ez_vec[0, 0]
        rel_ez_vec = (ez_vec_2d - ez_gt).abs() / (ez_gt.abs() + eps_rel)
        rel_ez_vec_mean = rel_ez_vec.mean().item()
        
        hx_vec_2d = Hx_vec[0, 0]
        rel_hx_vec = (hx_vec_2d - hx_gt).abs() / (hx_gt.abs() + eps_rel)
        rel_hx_vec_mean = rel_hx_vec.mean().item()
        
        # Loop version errors
        ez_loop_2d = Ez_loop[0, 0]
        rel_ez_loop = (ez_loop_2d - ez_gt).abs() / (ez_gt.abs() + eps_rel)
        rel_ez_loop_mean = rel_ez_loop.mean().item()
        
        hx_loop_2d = Hx_loop[0, 0]
        rel_hx_loop = (hx_loop_2d - hx_gt).abs() / (hx_gt.abs() + eps_rel)
        rel_hx_loop_mean = rel_hx_loop.mean().item()
        
        print(f"{n+1:4d} {rel_ez_vec_mean:12.3e} {rel_ez_loop_mean:12.3e} "
              f"{rel_hx_vec_mean:12.3e} {rel_hx_loop_mean:12.3e}")
    
    print("-" * 80)
    print("\n" + "=" * 80)
    print("Conclusion:")
    if rel_ez_loop_mean < 0.01:
        print("✅ gem_te_loop.py (explicit loops) matches GT very well!")
    else:
        print("⚠️  gem_te_loop.py still has errors - further debugging needed")
    
    if rel_ez_vec_mean < rel_ez_loop_mean:
        print("✅ gem_te.py (vectorized) is actually MORE accurate than loop version")
    else:
        print("⚠️  gem_te.py (vectorized) has higher errors than loop version")
    print("=" * 80)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    compare_implementations()
