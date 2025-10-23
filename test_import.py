"""
Quick import test to verify all modules load correctly.
This script doesn't run the simulation, just checks that all imports work.
"""

print("Testing imports...")

try:
    import taichi as ti
    print("✓ Taichi imported")
except ImportError as e:
    print(f"✗ Taichi import failed: {e}")
    exit(1)

try:
    import numpy as np
    print("✓ NumPy imported")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    exit(1)

try:
    from config import N, DOMAIN_SIZE, R_MIN, R_MAX, GRID_RES, PBD_PASSES
    print(f"✓ Config imported (N={N}, GRID_RES={GRID_RES})")
except ImportError as e:
    print(f"✗ Config import failed: {e}")
    exit(1)

try:
    from grid import (
        clear_grid, count_particles_per_cell, prefix_sum, copy_cell_pointers,
        scatter_particles, count_neighbors, update_colors, periodic_delta
    )
    print("✓ Grid kernels imported (7 kernels + helper)")
except ImportError as e:
    print(f"✗ Grid import failed: {e}")
    exit(1)

try:
    from dynamics import update_radii, project_overlaps
    print("✓ Dynamics kernels imported (2 kernels)")
except ImportError as e:
    print(f"✗ Dynamics import failed: {e}")
    exit(1)

print("\n" + "="*60)
print("ALL IMPORTS SUCCESSFUL!")
print("="*60)
print("\nConfiguration summary:")
print(f"  Particles: {N}")
print(f"  Domain: {DOMAIN_SIZE}³")
print(f"  Radius: [{R_MIN}, {R_MAX}]")
print(f"  Grid: {GRID_RES}³ cells")
print(f"  PBD: {PBD_PASSES} passes")
print("\nReady to run: python run.py")

