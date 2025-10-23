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

CONTACT_TOL = 0.02          # Contact tolerance: 2% beyond touching
                            # Matches PBD gap. Particles at (1+TOL)*(r_i+r_j) 
                            # are counted as neighbors.

EPS = 1e-8                  # Small epsilon for numerical safety

# ==============================================================================
# Degree adaptation (grow/shrink rule)
# ==============================================================================

DEG_LOW = 5                 # Below this: grow by GAIN_GROW (5%)
DEG_HIGH = 6                # Above this: shrink by GAIN_SHRINK (5%)
                            # In [DEG_LOW, DEG_HIGH]: no change

GAIN_GROW = 0.05            # Growth rate: 5% per step
GAIN_SHRINK = 0.05          # Shrink rate: 5% per step

# ==============================================================================
# PBD separation (overlap projection) - Phase A Enhanced
# ==============================================================================

# Adaptive PBD passes (scales with overlap depth)
PBD_BASE_PASSES = 4         # Base: 4 passes (normal case)
PBD_MAX_PASSES = 24         # Max: 24 passes (rescue mode)
PBD_ADAPTIVE_SCALE = 20.0   # Passes added per unit depth
                            # Formula: passes = base + scale * (max_depth / 0.2*R_MAX)

# Correction clamping (anti-tunneling)
MAX_DISPLACEMENT_FRAC = 0.2 # 20% of radius per pass
DISPLACEMENT_MULTIPLIER = 2.0  # Allow more for multi-neighbor cases
                            # Total clamp = FRAC * rad * MULTIPLIER

GAP_FRACTION = 0.02         # Target gap: 2% of (r_i + r_j)
                            # Matches CONTACT_TOL for consistency

# ==============================================================================
# Deep overlap force fallback (rescue mode)
# ==============================================================================

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
RADIUS_RATE_LIMIT = 0.02    # 2% max change per frame (regardless of FPS)
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

