#!/usr/bin/env python3
"""
Analyze the source excitation signal and generate clean 3D FDTD data
without the source excitation period.

Steps:
1. Plot the source signal to find when it becomes ~0
2. Run FDTD with source until source period ends
3. Save the fields at that time as initial conditions
4. Continue simulation without source for remaining time steps
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Define floattype (use float64 for precision)
floattype = np.float64

# ============================================================================
# PART 1: Analyze source signal
# ============================================================================

def analyze_source_signal():
    """Plot source signal and find when it becomes negligible."""
    
    # Parameters from original code
    c0 = 2.99792458e8
    dx = dy = dz = 2e-3
    dt = dx / (2.0 * c0)
    
    rtau = 50.0e-12
    tau = rtau / dt
    ndelay = 3 * tau
    srcconst = -dt * (3.0e11)
    
    nmax = 500
    source = np.zeros(nmax + 1, dtype=floattype)
    
    for n in range(1, nmax + 1):
        source[n] = srcconst * (n - ndelay) * np.exp(-((n - ndelay)**2 / (tau**2)))
    
    # Find when source becomes negligible (< 1% of peak)
    peak_val = np.max(np.abs(source))
    threshold = 0.01 * peak_val
    
    # Find first time when source drops below threshold AND stays below
    source_end_idx = nmax
    for n in range(int(ndelay), nmax):
        if np.all(np.abs(source[n:]) < threshold):
            source_end_idx = n
            break
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full signal
    axes[0].plot(source, 'b-', linewidth=1.5, label='Source signal')
    axes[0].axhline(y=threshold, color='r', linestyle='--', label=f'1% threshold = {threshold:.3e}')
    axes[0].axhline(y=-threshold, color='r', linestyle='--')
    axes[0].axvline(x=source_end_idx, color='g', linestyle='--', linewidth=2, 
                    label=f'Source end at step {source_end_idx}')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Source amplitude')
    axes[0].set_title('Source Excitation Signal (Gaussian pulse derivative)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Zoomed view around source end
    zoom_start = max(0, source_end_idx - 50)
    zoom_end = min(nmax, source_end_idx + 100)
    axes[1].plot(range(zoom_start, zoom_end), source[zoom_start:zoom_end], 'b-', linewidth=1.5)
    axes[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    axes[1].axhline(y=-threshold, color='r', linestyle='--')
    axes[1].axvline(x=source_end_idx, color='g', linestyle='--', linewidth=2, 
                    label=f'Source end = {source_end_idx}')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Source amplitude')
    axes[1].set_title(f'Zoomed view near source end (steps {zoom_start}-{zoom_end})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('/Users/zyanxin/Documents/code/GEM_N/data/source_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved source analysis plot to: data/source_analysis.png")
    plt.show()
    
    print("\n" + "="*70)
    print("SOURCE SIGNAL ANALYSIS")
    print("="*70)
    print(f"Time step dt: {dt:.6e} seconds")
    print(f"tau (pulse width): {tau:.2f} time steps = {tau*dt:.6e} seconds")
    print(f"ndelay (pulse center): {ndelay:.2f} time steps")
    print(f"Peak source amplitude: {peak_val:.6e}")
    print(f"Threshold (1% of peak): {threshold:.6e}")
    print(f"Source becomes negligible at step: {source_end_idx}")
    print(f"Source duration: {source_end_idx * dt:.6e} seconds = {source_end_idx * dt * 1e12:.3f} ps")
    print(f"\nRECOMMENDATION: Discard first {source_end_idx} time steps")
    print(f"                Start saving data from step {source_end_idx + 1}")
    print("="*70)
    
    return source_end_idx, dt


if __name__ == '__main__':
    source_end_step, dt = analyze_source_signal()
    
    print("\nNext steps:")
    print(f"1. Run FDTD simulation for {source_end_step} steps with source")
    print(f"2. Save fields (Ex, Ey, Ez, Hx, Hy, Hz) at step {source_end_step} as initial conditions")
    print(f"3. Continue simulation from step {source_end_step + 1} WITHOUT source")
    print(f"4. Save the remaining time steps as clean training data")
