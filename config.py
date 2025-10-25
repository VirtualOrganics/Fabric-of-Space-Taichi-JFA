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

N = 20000                    # Total number of particles (target mean degree ~5-6)
DOMAIN_SIZE = 0.15          # Cubic domain side length [0, 0.15)³ (reduced density for PBD breathing room)

# ==============================================================================
# Density-based radius auto-scaling (keeps look consistent across N)
# ==============================================================================

AUTO_SCALE_RADII = False    # If True, compute R_MIN/R_MAX from N and φ at startup (MANUAL by default)
PHI_TARGET = 0.30           # Target packing fraction (0.10=loose foam, 0.60=dense)
R_MIN_FACTOR = 0.1          # r_min = R_MIN_FACTOR * r_ref (10x smaller than ref)
R_MAX_FACTOR = 10.0         # r_max = R_MAX_FACTOR * r_ref (10x larger than ref)
                            # Spread = R_MAX_FACTOR / R_MIN_FACTOR = 100x

# Grid cell size override (None = auto-compute from r_max, or set manually e.g. 0.01)
CELL_SIZE_OVERRIDE = None

# Manual bounds (used when AUTO_SCALE_RADII = False, which is the default)
R_MIN_MANUAL = 0.0005       # Minimum particle radius
R_MAX_MANUAL = 0.0500       # Maximum particle radius (100x larger than min)

# Periodic Boundary Conditions (PBC)
PBC_ENABLED = True          # Toggle periodic boundaries (compile-time via ti.static)
                            # True: particles wrap at domain edges (toroidal topology)
                            # False: bounded domain [0, L)³ (particles can hit walls)

# Precomputed constants (Python scope, compile-time folded by Taichi)
HALF_L = 0.5 * DOMAIN_SIZE  # Half domain size for centered coordinates
INV_L = 1.0 / DOMAIN_SIZE   # Inverse domain size (avoid repeated division)

# ==============================================================================
# Helper: Compute radius bounds from packing fraction
# ==============================================================================

def compute_radius_bounds(N, phi_target, domain_size, r_min_factor, r_max_factor):
    """
    Compute reference radius and bounds from target packing fraction.
    
    Formula: r_ref = ((3 φ V_box) / (4π N))^(1/3)
    
    Args:
        N: particle count
        phi_target: target packing fraction (0.1–0.6)
        domain_size: cubic domain side length
        r_min_factor, r_max_factor: multipliers around r_ref
    
    Returns:
        (r_ref, r_min, r_max, suggested_cell_size)
    """
    V_box = domain_size ** 3
    r_ref = ((3.0 * phi_target * V_box) / (4.0 * math.pi * max(N, 1))) ** (1.0 / 3.0)
    
    r_min = r_min_factor * r_ref
    r_max = r_max_factor * r_ref
    
    # Suggest cell size: ~2× r_max with buffer (for 27-stencil neighbor search)
    suggested_cell_size = 2.2 * r_max
    
    return r_ref, r_min, r_max, suggested_cell_size

# ==============================================================================
# Apply auto-scaling (or use manual bounds)
# ==============================================================================

if AUTO_SCALE_RADII:
    R_REF, R_MIN, R_MAX, SUGGESTED_CELL_SIZE = compute_radius_bounds(
        N, PHI_TARGET, DOMAIN_SIZE, R_MIN_FACTOR, R_MAX_FACTOR
    )
    
    # Use override if provided, otherwise use suggested cell size
    if CELL_SIZE_OVERRIDE is None:
        CELL_SIZE = SUGGESTED_CELL_SIZE
    else:
        CELL_SIZE = CELL_SIZE_OVERRIDE
    
    # Startup telemetry (shows computed values)
    print(f"[Auto-Scale] N={N}, φ={PHI_TARGET:.2f} → r_ref={R_REF:.6f}, "
          f"R∈[{R_MIN:.6f}, {R_MAX:.6f}] (spread×{R_MAX/R_MIN:.1f})")
    print(f"[Auto-Scale] CELL_SIZE={CELL_SIZE:.6f} (override={CELL_SIZE_OVERRIDE is not None})")
else:
    # Use manual bounds
    R_MIN = R_MIN_MANUAL
    R_MAX = R_MAX_MANUAL
    CELL_SIZE = 2.0 * R_MAX  # Conservative default
    R_REF = None  # Not computed
    print(f"[Manual] R∈[{R_MIN:.6f}, {R_MAX:.6f}], CELL_SIZE={CELL_SIZE:.6f}")

# ==============================================================================
# Grid parameters (spatial hashing for neighbor search)
# ==============================================================================

# Grid resolution: ceil ensures last cell covers box edge, max(3,...) ensures
# 27-stencil is always valid (no edge cases with 1x1x1 or 2x2x2 grids)
GRID_RES = max(3, int(math.ceil(DOMAIN_SIZE / CELL_SIZE)))

# Safety check: warn if r_max is too large vs cell size
if R_MAX > 0.5 * CELL_SIZE:
    print(f"[Auto-Scale][WARN] R_MAX ({R_MAX:.6f}) > 0.5×CELL_SIZE ({CELL_SIZE:.6f}). "
          f"Consider increasing CELL_SIZE or GRID_RES for efficiency.")

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
# Decision Stability (hysteresis + streaks)
# ==============================================================================
# Prevents decision flip-flop on noise, enabling sustained growth/shrink runs.

HYSTERESIS = 0.6                # Degree units added around band edges
                                # Prevents chattering when deg ~ DEG_LOW or DEG_HIGH
STREAK_LOCK = 3                 # Pulses to keep a decision before reconsidering
                                # Enables multi-pulse growth/shrink runs (e.g., grow ×5)
MOMENTUM = 0.0                  # Streak momentum (0.10 = 10% gain per streak unit)
                                # Start at 0.0 (disabled), tune upward for compounding
STREAK_CAP = 4                  # Cap for momentum amplification
                                # Limits exponential explosion from long streaks

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

