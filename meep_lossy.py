import meep as mp
import time
import numpy as np

# ==========================================
# 1. Configuration (Matching your PyTorch setup)
# ==========================================
# Grid dimensions: Increased to 100x100x100 to stress test CPU vs GPU
# Old: 20x20x20 (Too small, CPU wins due to low overhead)
# New: 100x100x100 (1 Million points, GPU should shine)
nx, ny, nz = 100, 100, 100
resolution = 1  # 1 pixel per unit length
cell_size = mp.Vector3(nx, ny, nz)

# Material: Lossy Medium
material_conductivity = 4.0 

geometry = [
    mp.Block(
        mp.Vector3(nx, ny, nz),
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(index=1.0, D_conductivity=material_conductivity)
    )
]

# Source: Gaussian Pulse at center
sources = [
    mp.Source(
        mp.GaussianSource(frequency=0.15, fwidth=0.1),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 0)
    )
]

# ==========================================
# 2. Setup Simulation
# ==========================================
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=[], # No PML, acts like a cavity
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# ==========================================
# 3. Run Benchmark
# ==========================================
print(f"Initializing MEEP with Grid: {nx}x{ny}x{nz} (1 Million points)...")

print("Starting Simulation...")
n_steps = 100 # Reduced steps slightly because CPU might be slow with 1M points
# But enough to measure speed
run_until_time = n_steps * 0.5 # MEEP dt is usually 0.5

start_time = time.time()

# CORRECT WAY: Use sim.run()
sim.run(until=run_until_time)

end_time = time.time()
total_time = end_time - start_time
avg_time_per_step = (total_time / n_steps) * 1000  # ms

print("="*40)
print(f"MEEP Benchmark Result (CPU)")
print(f"Grid Size: {nx}x{ny}x{nz}")
print("="*40)
print(f"Total Time: {total_time:.4f} s")
print(f"Time/Step : {avg_time_per_step:.2f} ms")
print("="*40)