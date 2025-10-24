"""
Configuration parameters for Fabric of Space - Custom Taichi Grid.

This module defines all simulation parameters:
- Particle properties (count, radii)
- Grid parameters (cell size, resolution)
- Degree adaptation (grow/shrink thresholds)
- PBD separation (passes, gap, displacement cap)
- Rendering (FPS, visualization scale)

All units are consistent. Domain is a periodic cubic box [0, DOMAIN_SIZE)³.
"""

import math

# ==============================================================================
# Particle properties
# ==============================================================================

N = 5000                    # Total number of particles (target mean degree ~5-6)
DOMAIN_SIZE = 0.15          # Cubic domain side length [0, 0.15)³ (reduced density for PBD breathing room)
R_MIN = 0.0020              # Minimum radius (hard lower bound) - 4x spread
R_MAX = 0.0080              # Maximum radius (hard upper bound) - 4x spread

# Periodic Boundary Conditions (PBC)
PBC_ENABLED = True          # Toggle periodic boundaries (compile-time via ti.static)
                            # True: particles wrap at domain edges (toroidal topology)
                            # False: bounded domain [0, L)³ (particles can hit walls)

# Precomputed constants (Python scope, compile-time folded by Taichi)
HALF_L = 0.5 * DOMAIN_SIZE  # Half domain size for centered coordinates
INV_L = 1.0 / DOMAIN_SIZE   # Inverse domain size (avoid repeated division)

# ==============================================================================
# Grid parameters (spatial hashing for neighbor search)
# ==============================================================================

CELL_SIZE = 2 * R_MAX       # Conservative cell size = 0.016 (updated for new R_MAX)
                            # Any pair in adjacent cells can potentially touch

# Grid resolution: ceil ensures last cell covers box edge, max(3,...) ensures
# 27-stencil is always valid (no edge cases with 1x1x1 or 2x2x2 grids)
GRID_RES = max(3, int(math.ceil(DOMAIN_SIZE / CELL_SIZE)))  # = 13 cells per axis

CONTACT_TOL = 0.015         # Contact tolerance: 1.5% beyond touching
                            # Matches PBD GAP_FRACTION. Particles at (1+TOL)*(r_i+r_j) 
                            # are counted as neighbors.

EPS = 1e-8                  # Small epsilon for numerical safety

# ==============================================================================
# Degree adaptation (grow/shrink rule) - Track 1 (Fast Path)
# ==============================================================================

DEG_LOW = 3                 # Below this: grow by GAIN_GROW (geometric PCC)
DEG_HIGH = 5                # Above this: shrink by GAIN_SHRINK
                            # In [DEG_LOW, DEG_HIGH]: no change

GAIN_GROW = 0.03            # Growth rate: 3% per step (slightly faster, still stable)
GAIN_SHRINK = 0.03          # Shrink rate: 3% per step

# ==============================================================================
# PBD separation (overlap projection) - Track 1 (Fast Path)
# ==============================================================================

# Adaptive PBD passes (scales with overlap depth)
PBD_BASE_PASSES = 4         # Base: 4 passes (normal case)
PBD_MAX_PASSES = 8          # Max: 8 passes (tight cap for speed)
PBD_ADAPTIVE_SCALE = 20.0   # Passes added per unit depth
                            # Formula: passes = base + scale * (max_depth / 0.2*R_MAX)
PBD_SUBSTEPS = 2            # PBD substeps per pass (improves convergence)

# Correction clamping (anti-tunneling)
MAX_DISPLACEMENT_FRAC = 0.2 # 20% of radius per pass
DISPLACEMENT_MULTIPLIER = 2.0  # Allow more for multi-neighbor cases
                            # Total clamp = FRAC * rad * MULTIPLIER

GAP_FRACTION = 0.015        # Target gap: 1.5% of (r_i + r_j) (soft breathing room)
                            # Slightly lower than CONTACT_TOL for PBD cushion

# ==============================================================================
# Deep overlap force fallback (rescue mode) - Track 1 (Re-enabled)
# ==============================================================================

RESCUE_ENABLED = False      # DISABLED: Taichi compiler bug with atomic operations
RESCUE_STRENGTH = 0.25      # Strength multiplier (0.2-0.4, tune for stability)
                            # NOTE: Rescue forces trigger Taichi internal compiler error
                            # Relying on adaptive PBD alone (4-8 passes) for separation

DEEP_OVERLAP_THRESHOLD = 0.10  # Trigger rescue at 10% of R_MAX
DEEP_OVERLAP_EXIT = 0.07    # Exit rescue when below 7% (hysteresis)
FORCE_STIFFNESS_MULTIPLIER = 1.0  # Tune 0.5–2.0 (higher = more aggressive)
FORCE_SUBSTEPS_MIN = 2      # Minimum substeps when rescue triggers
FORCE_SUBSTEPS_MAX = 4      # Maximum substeps for extreme overlaps
FORCE_DAMPING = 0.9         # Velocity damping per substep (10% energy loss)
GLOBAL_DAMPING = 0.995      # Per-frame global damping (0.5% energy loss)
                            # Prevents slow energy accumulation

# Squared threshold (optimization: avoid sqrt in comparisons)
DEEP_OVERLAP_THRESHOLD_SQ = (DEEP_OVERLAP_THRESHOLD * R_MAX) ** 2

# ==============================================================================
# XPBD radius adaptation (frame-rate independent)
# ==============================================================================

RADIUS_COMPLIANCE = 0.02    # Softness of constraint (0.02 often smoother than 0.01)
                            # Lower = stiffer (faster response)
                            # Higher = softer (smoother, more damped)
RADIUS_RATE_LIMIT = 0.015   # 1.5% max change per frame (slightly looser for faster adaptation)
                            # Prevents "radius shocks" that inject overlaps

DT = 0.016                  # Timestep for XPBD (≈60 FPS, adjust for target FPS)
                            # Used in compliance calculation: α / dt²

# ==============================================================================
# XSPH velocity smoothing (anti-jitter)
# ==============================================================================

XSPH_EPSILON = 0.03         # 3% blend with neighbors per frame
                            # Smooths high-frequency oscillations after PBD
XSPH_ENABLED = True         # Set False to disable XSPH smoothing

# ==============================================================================
# Dynamic grid (adaptive cell size)
# ==============================================================================

GRID_UPDATE_THRESHOLD = 0.05  # Rebuild if max_radius drifts >5%
                              # Currently not used (R_MAX is hard-clamped)
                              # Safeguard for future enhancements

# ==============================================================================
# Rendering
# ==============================================================================

FPS_TARGET = 20             # Target frames per second for GUI

VIS_SCALE = 1.0             # Radius multiplier for rendering
                            # Increase if particles appear too small
                            # Decrease if they appear too large

# ==============================================================================
# Important notes
# ==============================================================================

# Degree semantics:
# deg[i] = count of neighbors within (1 + CONTACT_TOL) * (r_i + r_j)
# This is "near-contact" (not exact touching), matching PBD gap semantics.
# Particles with deg=0 will grow (isolated particles expand).

# Grid coverage:
# GRID_RES = max(3, ceil(L / cell_size)) ensures:
#   - Last cell covers domain edge (no particles fall outside grid)
#   - Minimum 3x3x3 grid (27-stencil always valid, no edge cases)

# Precision:
# All Taichi fields use f32 for GPU performance. Consistent types = predictable.

# ==============================================================================
# Phase B: Topological Neighbor Counting (Gabriel Graph)
# ==============================================================================

# Core settings
TOPO_ENABLED = False            # Enable topological degree for radius control (DISABLED for speed)
TOPO_UPDATE_CADENCE = 60        # Update topological degree every N frames (expensive)
TOPO_EMA_ALPHA = 0.1            # EMA smoothing factor (0.1 = 10% new, 90% old)

# One-shot topology analysis (batched, on-demand)
TOPO_BATCHES = 12               # Spread topology work over N frames (reduces FPS impact)
TOPO_PAIR_SUBSAMPLE_Q = 3       # Process ~1/Q pairs (Q=3 → ~33% of pairs)
TOPO_WRITE_TO_EMA = True        # Commit results to EMA when analysis completes

# Target degree
TOPO_DEG_LOW = 10               # Grow below this threshold
TOPO_DEG_HIGH = 18              # Shrink above this threshold

# Candidate pruning
TOPO_MAX_RADIUS_MULTIPLE = 2.2  # Only test pairs within this * (r_i + r_j)
MAX_TOPO_NEIGHBORS = 28         # Pre-allocate space for candidate pairs per particle

# Performance optimizations (Phase B)
TOPO_PAIR_SUBSAMPLING_STRIDE = 3  # Process 1/3 of pairs per update (hash-based)
TOPO_EARLY_EXIT_HIGH_DEGREE = True  # Skip witness test if both endpoints are high

# Telemetry (ON by default)
TOPO_TRUNCATION_WARNING_THRESHOLD = 0.02 # Warn if >2% of particles truncate candidate list

# Blend settings (Phase B.1 stability)
TOPO_BLEND_FRAMES = 200         # Frames to blend topo + geom degrees
TOPO_BLEND_LAMBDA_START = 0.8   # Start with 80% topo, 20% geom
TOPO_BLEND_LAMBDA_END = 1.0     # Fade to 100% topo

# Advanced (Phase B.2+)
USE_LAGUERRE_GABRIEL = False    # True: power diagram (polydisperse), False: Euclidean

# k-NN fast proxy (cheap alternative to Gabriel graph)
USE_KNN_TOPO = False            # Use k-nearest neighbors instead of Gabriel graph
KNN_TOPO_K = 14                 # Target number of nearest neighbors

# ==============================================================================
# Brownian Motion / Gentle Drift (Track 1 Option B: add visual interest)
# ==============================================================================

JITTER_ENABLED = True           # Enable smooth Brownian drift (OU noise)
JITTER_RMS = 0.10               # RMS drift as fraction of mean radius per second
                                # 0.10 = gentle meander, 0.20 = more active
JITTER_TAU = 1.0                # OU time scale (seconds): higher = smoother
                                # 0.5 = jittery, 1.0 = smooth, 2.0 = very smooth
MAX_DRIFT_FRACTION = 0.20       # Cap per-step drift to 20% of GAP
                                # Protects PBD from destabilizing

# ==============================================================================
# Growth Rhythm (defaults only; GUI edits runtime copies)
# ==============================================================================
# These control the discrete pulse/relax cycle for radius adaptation.

GROWTH_RATE_DEFAULT = 0.04      # 4% per pulse (both grow and shrink)
GROWTH_INTERVAL_DEFAULT = 20    # Frames between pulses
RELAX_INTERVAL_DEFAULT = 10     # Relax frames after a pulse (PBD + Lévy only)

# ==============================================================================
# Lévy Positional Diffusion (Track 2: topological regularization)
# ==============================================================================
# This kernel balances positional irregularities through neighbor degree coupling,
# leading to smoother foam topology (Lévy centroidal relaxation approximation).
# NOTE: Lévy now runs ONLY during relax frames (controlled by growth rhythm).

LEVY_ENABLED = True             # Toggle Lévy diffusion (Track 2)
LEVY_ALPHA = 0.04               # Diffusion gain (0.02-0.05 typical)
                                # Higher = faster convergence but risk of jitter
LEVY_DEG_SPAN = 10.0            # Normalize degree difference: Δd / span
                                # Typical range for geometric degree: 10-15
LEVY_STEP_FRAC = 0.15           # Cap step at 15% of mean radius per frame
                                # Prevents large position jumps
LEVY_USE_TOPO_DEG = False       # True: use topological degree (topo_deg_ema)
                                # False: use smoothed geometric degree (deg_smoothed)
                                # Switch to True once Gabriel topology is restored

# NOTE: Keep RADIUS_RATE_LIMIT >= GROWTH_RATE_DEFAULT to avoid masking pulses

