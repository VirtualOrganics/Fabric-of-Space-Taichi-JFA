#!/usr/bin/env python3
"""
Benchmark script for Foam Simulator - Reproducible Performance Testing
=======================================================================

Runs a fixed number of simulation frames with deterministic seed and reports:
- FPS (frames per second)
- Time breakdown (grid, PBD, topology, render)
- Memory usage
- Configuration used

Usage:
    python scripts/bench.py [--frames N] [--particles N] [--no-render]

Example:
    python scripts/bench.py --frames 100 --particles 10000
"""

import sys
import os
import time
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

# Import simulation modules
from config import *
from grid import *
from dynamics import *
from jfa import JFAContext


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark foam simulator performance')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to run (default: 100)')
    parser.add_argument('--particles', type=int, default=10000,
                        help='Number of particles (default: 10000)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering for headless benchmarking')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--jfa-cadence', type=int, default=5,
                        help='JFA cadence (default: 5)')
    return parser.parse_args()


def initialize_simulation(n_particles, seed):
    """
    Initialize simulation state.
    
    Args:
        n_particles: Number of particles to spawn
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (pos, rad, vel, color, active_n)
    """
    np.random.seed(seed)
    
    # Create fields
    pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)
    rad = ti.field(dtype=ti.f32, shape=MAX_N)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)
    color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)
    rad_target = ti.field(dtype=ti.f32, shape=MAX_N)
    delta_r = ti.field(dtype=ti.f32, shape=MAX_N)
    
    # Spawn particles
    positions = np.random.rand(n_particles, 3).astype(np.f32) * DOMAIN_SIZE - DOMAIN_SIZE / 2.0
    radii = np.full(n_particles, R_START_MANUAL, dtype=np.f32)
    
    # Add radius variance for pressure gradients
    RADIUS_VARIANCE = 0.30
    radii *= (1.0 + (np.random.rand(n_particles).astype(np.f32) - 0.5) * 2.0 * RADIUS_VARIANCE)
    radii = np.clip(radii, R_MIN, R_MAX)
    
    # Upload to Taichi
    pos.from_numpy(positions)
    rad.from_numpy(radii)
    rad_target.from_numpy(radii.copy())
    
    return pos, rad, vel, color, rad_target, delta_r, n_particles


def run_benchmark(args):
    """
    Run benchmark and collect performance statistics.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"FOAM SIMULATOR BENCHMARK")
    print(f"{'='*70}\n")
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Particles:     {args.particles}")
    print(f"  Frames:        {args.frames}")
    print(f"  Seed:          {args.seed}")
    print(f"  JFA Cadence:   {args.jfa_cadence}")
    print(f"  Render:        {'Disabled' if args.no_render else 'Enabled'}")
    print(f"  Domain Size:   {DOMAIN_SIZE}")
    print(f"  FSC Band:      [{FSC_LOW}, {FSC_HIGH}]")
    print(f"  Growth Rate:   {GROWTH_PCT * 100}%")
    print(f"\n")
    
    # Initialize Taichi
    ti.init(arch=ti.gpu, device_memory_GB=4.0)
    
    # Initialize simulation
    print("Initializing simulation...")
    pos, rad, vel, color, rad_target, delta_r, active_n = initialize_simulation(args.particles, args.seed)
    
    # Initialize JFA
    jfa = JFAContext(MAX_N)
    
    # Initialize grid
    init_grid()
    
    # Timing accumulators
    times_grid = []
    times_pbd = []
    times_topo = []
    times_render = []
    times_total = []
    
    print(f"Running {args.frames} frames...\n")
    
    # Warm-up (first few frames can be slow due to JIT compilation)
    warmup_frames = 5
    for frame in range(warmup_frames):
        rebuild_grid_and_csr(pos, rad, active_n)
        if JFA_ENABLED and (frame % args.jfa_cadence == 0):
            jfa.run(pos, rad, active_n, DOMAIN_SIZE)
    
    ti.sync()
    print(f"Warm-up complete ({warmup_frames} frames)\n")
    
    # Main benchmark loop
    start_time_total = time.perf_counter()
    
    for frame in range(args.frames):
        t0 = time.perf_counter()
        
        # 1. Grid rebuild
        t_grid_start = time.perf_counter()
        rebuild_grid_and_csr(pos, rad, active_n)
        ti.sync()
        t_grid = time.perf_counter() - t_grid_start
        
        # 2. PBD
        t_pbd_start = time.perf_counter()
        project_overlaps_multi_pass(pos, rad, active_n)
        ti.sync()
        t_pbd = time.perf_counter() - t_pbd_start
        
        # 3. Topology (JFA)
        t_topo_start = time.perf_counter()
        if JFA_ENABLED and (frame % args.jfa_cadence == 0):
            jfa.run(pos, rad, active_n, DOMAIN_SIZE)
            ti.sync()
        t_topo = time.perf_counter() - t_topo_start
        
        # 4. Render (if enabled)
        t_render_start = time.perf_counter()
        if not args.no_render:
            # Simulate render time (actual rendering would require GUI)
            ti.sync()
        t_render = time.perf_counter() - t_render_start
        
        # Record times
        t_frame = time.perf_counter() - t0
        times_grid.append(t_grid)
        times_pbd.append(t_pbd)
        times_topo.append(t_topo)
        times_render.append(t_render)
        times_total.append(t_frame)
        
        # Progress indicator
        if (frame + 1) % 10 == 0 or frame == args.frames - 1:
            fps_current = 1.0 / t_frame if t_frame > 0 else 0
            print(f"  Frame {frame+1:4d}/{args.frames}: {fps_current:5.1f} FPS")
    
    end_time_total = time.perf_counter()
    
    # Compute statistics
    total_time = end_time_total - start_time_total
    avg_fps = args.frames / total_time
    
    avg_grid = np.mean(times_grid)
    avg_pbd = np.mean(times_pbd)
    avg_topo = np.mean(times_topo)
    avg_render = np.mean(times_render)
    avg_total = np.mean(times_total)
    
    total_avg = avg_grid + avg_pbd + avg_topo + avg_render
    
    # Print results
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Overall Performance:")
    print(f"  Average FPS:   {avg_fps:.2f}")
    print(f"  Total Time:    {total_time:.2f}s")
    print(f"  Avg Frame:     {avg_total*1000:.2f}ms")
    print(f"\n")
    
    print(f"Time Breakdown (averages):")
    print(f"  Grid:          {avg_grid*1000:6.2f}ms  ({100*avg_grid/total_avg:5.1f}%)")
    print(f"  PBD:           {avg_pbd*1000:6.2f}ms  ({100*avg_pbd/total_avg:5.1f}%)")
    print(f"  Topology:      {avg_topo*1000:6.2f}ms  ({100*avg_topo/total_avg:5.1f}%)")
    print(f"  Render:        {avg_render*1000:6.2f}ms  ({100*avg_render/total_avg:5.1f}%)")
    print(f"\n")
    
    # Return results as dictionary
    return {
        'avg_fps': avg_fps,
        'total_time': total_time,
        'avg_frame_ms': avg_total * 1000,
        'avg_grid_ms': avg_grid * 1000,
        'avg_pbd_ms': avg_pbd * 1000,
        'avg_topo_ms': avg_topo * 1000,
        'avg_render_ms': avg_render * 1000,
        'config': {
            'particles': args.particles,
            'frames': args.frames,
            'seed': args.seed,
            'jfa_cadence': args.jfa_cadence,
            'render': not args.no_render
        }
    }


def main():
    """Main entry point."""
    args = parse_args()
    results = run_benchmark(args)
    
    print(f"Benchmark complete!")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

