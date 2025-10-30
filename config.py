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

N = 10000                   # Total number of particles (target mean degree ~5-6)
DOMAIN_SIZE = 0.189         # Cubic domain side length (SCALE WITH N - see table below)

# ═══════════════════════════════════════════════════════════════════════════════
# SCALING TABLE: Match DOMAIN_SIZE to N (keeps same density/degree)
# ═══════════════════════════════════════════════════════════════════════════════
#   N = 5,000   → DOMAIN_SIZE = 0.150
#   N = 10,000  → DOMAIN_SIZE = 0.189  (2x particles  → 1.26x bigger box)
#   N = 20,000  → DOMAIN_SIZE = 0.238  (4x particles  → 1.59x bigger box)
#   N = 25,000  → DOMAIN_SIZE = 0.257  (5x particles  → 1.71x bigger box)
#   N = 50,000  → DOMAIN_SIZE = 0.324  (10x particles → 2.16x bigger box)
#
# Formula: New_Size = 0.15 × (N / 5000)^(1/3)
# ═══════════════════════════════════════════════════════════════════════════════

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
R_MIN_MANUAL = 0.002        # Minimum particle radius
R_MAX_MANUAL = 0.010        # Maximum particle radius
R_START_MANUAL = 0.0045     # Starting radius for all particles (raised to ensure initial contact)
                            # System will naturally create size distribution via growth/shrink
                            # Tuning: Increase if particles don't touch initially (no growth)
                            #         Decrease if too much overlap at startup (slow PBD)

# Periodic Boundary Conditions (PBC)
PBC_ENABLED = True          # Toggle periodic boundaries (compile-time via ti.static)
                            # True: particles wrap at domain edges (toroidal topology)
                            # False: bounded domain [0, L)³ (particles can hit walls)

# ==============================================================================
# JFA Power Diagram (Experimental)
# ==============================================================================

# === MANUAL CONTROL MODE ===
# When True: Sliders directly control growth/shrink every frame (±5%)
# No phases, no auto-calibration, no hidden transitions
# Pure FSC-based pump: FSC < low → grow, FSC > high → shrink
# PHASE 2: Manual pump with immediate feedback (no gates, no smoothing)
FSC_MANUAL_MODE = True

# Feature flag: Enable JFA-based Face-Sharing Count (FSC) for neighbor detection
# When enabled, JFA runs in parallel with the existing grid system for validation
# FSC replaces degree-based coloring, but PBD still uses the spatial hash grid
JFA_ENABLED = True  # Re-enabled - with adaptive grid + gating, should work now

# JFA run frequency: run JFA every N measurement frames (0 = every frame)
# Higher values reduce overhead but delay FSC updates  
# Manual mode: run every frame for immediate feedback
JFA_RUN_INTERVAL = 1        # Run JFA every measurement frame (1 = every frame for manual mode)
JFA_EMA_ALPHA = 0.33         # EMA smoothing factor for FSC (1/cadence for stable convergence)

# ==============================================================================
# FSC-Only Controller (Phase 2)
# ==============================================================================
# Controller uses Face-Sharing Count (FSC) from JFA Power Diagram to control
# particle sizing. All control is derived solely from FSC—no geometric degree.

# --- Core FSC control ---
FSC_LOW = 8                 # FSC lower bound (grow below this)
FSC_HIGH = 20               # FSC upper bound (shrink above this)
GROWTH_PCT = 0.10           # 10% per measurement cycle (increased for faster response)
ADJUSTMENT_FRAMES = 15      # frames to ~95% (adaptive EMA, reduced for faster convergence)

# --- Safety rails ---
MAX_STEP_PCT = 0.12         # Per-frame cap on |ΔR|/R
MAX_STEP_PCT_RANGE = (0.05, 0.20)  # GUI slider range for advanced users

# --- Appendix A (opt-in refinements) ---
FSC_DEADBAND = 1.0          # ±FSC units near band edges (smoothstep damping)
BACKPRESSURE_MODE = "local" # "local" | "global" | "off"
RUN_SAFETY_TESTS = False    # Enable radius bounds assertions during testing

# ==============================================================================
# PRESSURE EQUILIBRATION
# ==============================================================================
# Volume-conserving pressure diffusion across FSC neighbors.
# Complements FSC controller: FSC drives long-term topology, pressure balances
# local mechanics. Equilibration is the primary driver of foam dynamics.

PRESSURE_EQUILIBRATION_ENABLED = True   # Master switch for pressure equilibration
PRESSURE_K = 0.10                       # Base diffusion coefficient (increased for visible dynamics)
PRESSURE_EXP = 3.0                      # Volume exponent (3 for 3D, 2 for 2D)
PRESSURE_PAIR_CAP = 0.02                # Per-pair ΔV cap (fraction of min(V_i, V_j))
MAX_EQ_NEI = 10                         # Max neighbors equilibrated per site per frame
EQ_MICRO_PBD_ITERS = 1                  # Micro-PBD iterations after equilibration (if needed)
EQ_OVERLAP_THRESHOLD = 0.05             # Normalized penetration depth to trigger micro-PBD

# ==============================================================================
# BROWNIAN MOTION (keeps foam "breathing" at equilibrium)
# ==============================================================================
BROWNIAN_ENABLED = True                 # Enable thermal jitter to prevent static equilibrium
BROWNIAN_STRENGTH = 0.0002              # Velocity noise strength (adjust for visible motion)
BROWNIAN_DAMPING = 0.95                 # Velocity damping per frame (0.95 = 5% friction)

# ==============================================================================
# WARM-START & DEBUG (prevents FSC=0 runaway during startup)
# ==============================================================================
WARMSTART_FRAMES = 30                   # Frames to let JFA stabilize before controller acts
FSC_ZERO_RATE_THRESH = 0.10             # If >10% have FSC==0, keep growth off

# Equilibration debug (prints every N frames)
EQ_DEBUG_EVERY = 30                     # Show [EQ Debug] telemetry every N frames
                                        # Formula: max(0, (Ra+Rb-d)/(Ra+Rb)) > threshold

# JFA Configuration (Phase 2 fixes)
# ==============================================================================

# Power diagram weight: distance metric = d² - (β·r)²
# β = 1.0 gives standard Power diagram (radius-weighted Voronoi)
# β > 1.0 inflates particle influence (more aggressive weighting)
# β < 1.0 deflates particle influence (closer to unweighted Voronoi)
POWER_BETA = 1.0            # Start with standard Power diagram

# Minimum face voxel count to accept a neighbor relationship
# This threshold filters out spurious 1-voxel "faces" caused by label noise
# Typical values: 8-16 voxels for 128³ resolution with r_mean ~ 0.005
# Lower = more permissive (risk of false positives)
# Higher = more strict (risk of missing true neighbors)
# Manual mode: very permissive to guarantee non-zero FSC during debugging
MIN_FACE_VOXELS = 2         # Threshold for accepting face-sharing neighbors (permissive for manual mode)

# Dynamic resolution bounds: JFA_RES will be computed as L / voxel_size
# where voxel_size ≈ 2.5-3.0 × r_mean (mean particle radius)
# This ensures voxels are sized appropriately relative to particles
JFA_RES_MIN = 192           # Minimum grid resolution (per Perplexity: need ≥10 voxels/diameter)
JFA_RES_MAX = 320           # Maximum grid resolution (per Perplexity: 16 voxels/diameter = reliable)
JFA_VOXEL_SCALE = 2.8       # Voxel size = VOXEL_SCALE × r_mean (tune 2.5-3.0)

# Adaptive resolution: target number of voxels across mean particle diameter
# Higher = more accurate but slower. Lower = faster but less accurate.
# Typical: 10-16 voxels/diameter (Perplexity recommendation)
JFA_ADAPTIVE_ENABLED = True  # Enable adaptive resolution based on r_mean
JFA_VOXELS_PER_DIAMETER = 12.0  # Target voxels across particle diameter (balance speed/accuracy)

# ==============================================================================
# JFA OPTIMIZATION: Multi-Rate Loop (Decimation)
# ==============================================================================
# Run JFA less frequently to reduce frame time (JFA = 77% of frame cost)
# Safety: Warm-start + watchdog to ensure topology doesn't drift

JFA_CADENCE = 5                   # Run JFA every N frames (after warm-start)
JFA_WARMSTART_FRAMES = 30         # Force every-frame JFA during first N frames (topology stabilization)
JFA_WATCHDOG_INTERVAL = 30        # Force full refresh every N JFA runs (catch drift)

# ==============================================================================
# JFA OPTIMIZATION: Spatial Decimation (Dirty Tiles) - Phase A: Instrumentation
# ==============================================================================
# Only re-compute JFA in regions where particles moved or radii changed
# Expected gain: 2-6× on top of cadence + adaptive res (15-30 FPS total)
JFA_DIRTY_TILES_ENABLED = False      # Master switch (Phase A: instrumentation only)
JFA_TILE_SIZE = 16                   # Voxels per tile (sweet spot for cache locality)
JFA_DIRTY_HALO = 1                   # Tile halo width (use 2 during warm-start)
JFA_DIRTY_WARMSTART = True           # Disable dirty tiles during WARMSTART_FRAMES
JFA_DIRTY_WATCHDOG_INTERVAL = 30     # Force full refresh every N frames
JFA_DIRTY_ESCALATION_THRESHOLD = 0.6 # Promote to full if dirty% > 60%

# Dirty criteria thresholds (relative to voxel size)
JFA_DIRTY_POS_THRESHOLD = 0.5        # Mark dirty if |Δpos| > 0.5 * voxel_size
JFA_DIRTY_RAD_THRESHOLD = 0.25       # Mark dirty if |Δr| > 0.25 * voxel_size
JFA_DIRTY_BOUNDARY_HYSTERESIS = 0.1  # Buffer before re-marking tile edge oscillations

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

# ==============================================================================
# Contact tolerance (must be defined before CELL_SIZE calculation)
# ==============================================================================

CONTACT_TOL = 0.035         # Contact tolerance: 3.5% beyond touching
                            # Set to 3.5% based on measured miss margin of 2.73% (with margin for safety).
                            # Particles at (1+TOL)*(r_i+r_j) are counted as neighbors.

# ==============================================================================
# Apply auto-scaling or use manual bounds (with contact-aware cell sizing)
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
    
    # Cell size for dynamic reach stencil neighbor search
    # Strategy: Choose CELL_SIZE small enough to ensure reach=2 (125 cells) at ALL sizes
    # Formula: r_cut = (1 + CONTACT_TOL) × 2 × r_max
    #          reach = ceil(r_cut / CELL_SIZE)
    # 
    # At startup, particles are ~0.003 (R_START)
    # At steady state, typical size is ~R_TYPICAL ≈ 0.005-0.010
    # At maximum, particles can reach R_MAX = 0.050
    # 
    # Coarse grid for efficient neighbor detection
    # Target: cell_size ≈ r_cut so reach stays small (2-3, not 10+)
    R_TYPICAL = 0.015  # Typical max particle size during simulation (updated from telemetry)
    r_cut_typical = 2.0 * (1.0 + CONTACT_TOL) * R_TYPICAL
    CELL_SIZE = r_cut_typical  # cell_size ≈ r_cut → GRID_RES ≈ 6³, reach ≈ 2
    R_REF = None  # Not computed
    
    print(f"[DEBUG] CELL_SIZE={CELL_SIZE:.6f}, r_cut_typical={r_cut_typical:.6f}")
    
    # Diagnostic printout
    grid_res_est = int(DOMAIN_SIZE / CELL_SIZE) + 1
    r_cut_at_typical = 2.0 * (1.0 + CONTACT_TOL) * R_TYPICAL
    r_cut_at_max = 2.0 * (1.0 + CONTACT_TOL) * R_MAX
    reach_at_typical = int(math.ceil(r_cut_at_typical / CELL_SIZE))
    reach_at_max = int(math.ceil(r_cut_at_max / CELL_SIZE))
    print(f"[Manual] R∈[{R_MIN:.6f}, {R_MAX:.6f}], R_TYPICAL={R_TYPICAL:.6f}")
    print(f"[Manual] CELL_SIZE={CELL_SIZE:.6f}")
    print(f"[Manual] Grid: {grid_res_est}×{grid_res_est}×{grid_res_est} = {grid_res_est**3} cells")
    print(f"[Manual] Expected reach: typical→{reach_at_typical} ({(2*reach_at_typical+1)**3} cells), max→{reach_at_max} ({(2*reach_at_max+1)**3} cells)")

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

EPS = 1e-8                  # Small epsilon for numerical safety

# ==============================================================================
# Degree adaptation (grow/shrink rule) - Track 1 (Fast Path)
# ==============================================================================

DEG_LOW = 3                 # Below this: grow by GAIN_GROW (geometric PCC)
DEG_HIGH = 5                # Above this: shrink by GAIN_SHRINK (μ≈5.8 should shrink!)
                            # In [DEG_LOW, DEG_HIGH]: no change

GAIN_GROW = 0.05            # Growth rate: 5% per step (increased for visibility)
GAIN_SHRINK = 0.05          # Shrink rate: 5% per step (increased for visibility)

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
# Geometric Repacking (push-pull to maintain tight packing during size changes)
# ==============================================================================
# When particles shrink, they create gaps. When particles grow, they create pressure.
# This system propagates expansion pressure into shrink gaps through local push/pull forces.

REPACK_ENABLED = False         # DISABLED - standard PBD push only (no pull forces)
REPACK_DEADZONE_TAU = 0.05     # Dead-zone: don't pull if gap < tau * mean_radius (prevents jitter)
REPACK_BETA_PUSH = 1.0         # Push strength on overlaps (standard PBD)
REPACK_BETA_PULL = 0.2         # Pull strength on gaps beyond dead-zone (weaker than push)
REPACK_EXTRA_PASSES = 6        # Extra PBD passes during repacking to propagate pressure
REPACK_STEP_CAP_FRAC = 0.25    # Maximum step size as fraction of mean radius (stability)

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
                            # TUNING: Bump to 0.05 if motion looks jittery
                            #         Drop to 0.02 if motion looks too "glidey"
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
ADJUSTMENT_FRAMES_DEFAULT = 30   # Frames to smoothly adjust to target size (per cycle)

# ==============================================================================
# Decision Stability (hysteresis + streaks)
# ==============================================================================
# Prevents decision flip-flop on noise, enabling sustained growth/shrink runs.

HYSTERESIS = 0.0                # Degree units added around band edges (DISABLED)
                                # Set to 0 so GUI sliders work exactly as shown
                                # Growth Rhythm cadence already prevents flip-flopping
STREAK_LOCK = 1                 # Pulses to keep a decision (1 = no locking)
                                # Growth Rhythm handles timing - no extra locking needed
MOMENTUM = 0.0                  # Streak momentum (0.10 = 10% gain per streak unit)
                                # Start at 0.0 (disabled), tune upward for compounding
STREAK_CAP = 4                  # Cap for momentum amplification
                                # Limits exponential explosion from long streaks

# ==============================================================================
# Lévy Positional Diffusion (DISABLED - Not Needed)
# ==============================================================================
# Lévy diffusion moves particles toward degree-balanced positions (degree-gradient flow).
# This is useful for equilibrium foam research, but NOT needed for our system because:
#
# 1. PBD ALREADY repositions particles after size changes (overlap projection)
# 2. Degree-based radius adaptation CREATES the target topology (not Lévy)
# 3. Lévy creates non-physical drift (particles teleport toward high-degree neighbors)
# 4. Expensive: 2-3x slower relax frames
#
# Our physically-grounded loop relies on:
#   - PBD overlap projection (repositioning after size changes)
#   - XSPH velocity smoothing (smooth local flow)
#   - Brownian drift (gentle visual motion)
#
# Code kept in repo for research/testing only. Can be manually triggered if needed.

LEVY_ENABLED = False            # Toggle Lévy diffusion (KEEP OFF for production)
LEVY_ALPHA = 0.04               # Diffusion gain (if enabled for research)
LEVY_DEG_SPAN = 10.0            # Normalize degree difference
LEVY_STEP_FRAC = 0.15           # Cap step at 15% of mean radius per frame
LEVY_USE_TOPO_DEG = False       # Use geometric degree (deg_smoothed)

