"""
Dynamics kernels for Fabric of Space - Phase A Enhanced.

This module provides:
1. Radius adaptation (XPBD-compliant, frame-rate independent)
2. PBD overlap projection (adaptive passes with deep-overlap force fallback)
3. Force-based rescue for extreme overlaps
4. Velocity smoothing (XSPH anti-jitter)
5. Max overlap detection (two-pass reduction)

All kernels handle periodic boundaries and variable radii.
"""

import taichi as ti
import taichi.math as tm
from config import (
    R_MIN, R_MAX, DEG_LOW, DEG_HIGH, GAIN_GROW, GAIN_SHRINK,
    GAP_FRACTION, MAX_DISPLACEMENT_FRAC, DISPLACEMENT_MULTIPLIER,
    DOMAIN_SIZE, CELL_SIZE, GRID_RES, EPS,
    DEEP_OVERLAP_THRESHOLD, FORCE_STIFFNESS_MULTIPLIER, FORCE_DAMPING, GLOBAL_DAMPING,
    RADIUS_COMPLIANCE, RADIUS_RATE_LIMIT, DT,
    XSPH_EPSILON,
    PBC_ENABLED, HALF_L, INV_L
)
from grid import pdelta, wrap_cell, cell_id, wrapP

# ==============================================================================
# Kernel 1: Radius adaptation (simple threshold rule)
# ==============================================================================

@ti.kernel
def update_radii(rad: ti.template(), deg: ti.template(), n: ti.i32):
    """
    Adapt radii based on degree (neighbor count).
    
    Rule:
      - deg < DEG_LOW (5):  GROW by GAIN_GROW (5%)
      - deg > DEG_HIGH (6): SHRINK by GAIN_SHRINK (5%)
      - deg in [DEG_LOW, DEG_HIGH]: NO CHANGE
    
    After update, radii are hard-clamped to [R_MIN, R_MAX].
    
    This is a simple, robust rule that drives particles toward target degree.
    """
    for i in range(n):
        d = deg[i]
        r_old = rad[i]
        r_new = r_old
        
        # Threshold logic
        if d < DEG_LOW:
            # GROW: increase radius by 5%
            r_new = r_old * (1.0 + GAIN_GROW)
        elif d > DEG_HIGH:
            # SHRINK: decrease radius by 5%
            r_new = r_old * (1.0 - GAIN_SHRINK)
        # else: NO CHANGE (d in [DEG_LOW, DEG_HIGH])
        
        # Hard clamp to [R_MIN, R_MAX]
        r_new = ti.max(R_MIN, ti.min(R_MAX, r_new))
        
        # Write back
        rad[i] = r_new


# ==============================================================================
# Kernel 2: PBD overlap projection (multi-pass separation)
# ==============================================================================

@ti.kernel
def project_overlaps(pos: ti.template(), rad: ti.template(),
                     cell_start: ti.template(), cell_count: ti.template(),
                     cell_indices: ti.template(), n: ti.i32):
    """
    Resolve overlaps using Position-Based Dynamics (PBD).
    
    For each pair of overlapping particles:
      1. Compute overlap depth: overlap = (r_i + r_j) - dist + gap
      2. Split correction 50/50 between particles
      3. Cap per-particle displacement to MAX_DISPLACEMENT_FRAC * r_i
      4. Apply correction along minimum-image direction
      5. Use atomic add for thread-safe position updates
    
    This kernel should be called PBD_PASSES times (e.g., 4) per frame.
    Multiple passes allow overlap corrections to propagate through the system.
    
    Periodic boundaries: Uses periodic_delta for distance, wraps positions after update.
    """
    for i in range(n):
        # Accumulate correction for particle i
        correction = ti.Vector([0.0, 0.0, 0.0])
        
        # My cell coordinate (PBC-aware, same as scatter_particles)
        p_wrapped = wrapP(pos[i])
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        # Check 27 neighboring cells
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    # Neighbor cell
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    
                    # Wrap cell indices (PBC-aware)
                    nc = wrap_cell(nc)
                    
                    # Linear index (centralized function)
                    nc_id = cell_id(nc)
                    
                    # Iterate particles in neighbor cell
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    
                    for k in range(start, start + count):
                        j = cell_indices[k]
                        
                        if i != j:  # Don't process self
                            # Minimum-image vector from i to j (PBC-aware)
                            delta = pdelta(pos[i], pos[j])
                            dist = ti.sqrt(delta.dot(delta) + 1e-12)  # Add epsilon to avoid div-by-zero
                            
                            # Target separation: (r_i + r_j) * (1 + GAP_FRACTION)
                            target_sep = (rad[i] + rad[j]) * (1.0 + GAP_FRACTION)
                            
                            if dist < target_sep:
                                # Overlap depth
                                overlap = target_sep - dist
                                
                                # Direction: push i away from j
                                direction = delta / dist
                                
                                # Correction magnitude (split 50/50 between i and j)
                                correction_mag = overlap * 0.5
                                
                                # Cap displacement to prevent tunneling
                                max_displacement = MAX_DISPLACEMENT_FRAC * rad[i]
                                correction_mag = ti.min(correction_mag, max_displacement)
                                
                                # Correction vector: push i away from j (negative of delta direction)
                                correction_vec = -direction * correction_mag
                                
                                # Clamp per-axis to prevent large jumps
                                for d in ti.static(range(3)):
                                    correction_vec[d] = ti.max(-max_displacement, 
                                                                 ti.min(max_displacement, correction_vec[d]))
                                
                                # Accumulate correction for particle i
                                # (Each particle will process its own corrections when it's their turn)
                                correction += correction_vec
        
        # CRITICAL: Clamp total accumulated correction to prevent tunneling
        # Even though we clamp per-pair, accumulating 100+ pairs can still be huge
        max_total_displacement = MAX_DISPLACEMENT_FRAC * rad[i] * DISPLACEMENT_MULTIPLIER
        correction_magnitude = ti.sqrt(correction.dot(correction))
        if correction_magnitude > max_total_displacement:
            correction = correction * (max_total_displacement / correction_magnitude)
        
        # Apply accumulated correction atomically
        for d in ti.static(range(3)):
            ti.atomic_add(pos[i][d], correction[d])
        
        # Wrap position (PBC-aware, always-wrapped invariant)
        pos[i] = wrapP(pos[i])


# ==============================================================================
# Phase A: Additional Kernels for Stability
# ==============================================================================

# ------------------------------------------------------------------------------
# Velocity initialization
# ------------------------------------------------------------------------------

@ti.kernel
def init_velocities(vel: ti.template(), n: ti.i32):
    """Initialize all velocities to zero at start."""
    for i in range(n):
        vel[i] = ti.Vector([0.0, 0.0, 0.0])


# ------------------------------------------------------------------------------
# Max overlap detection (two-pass reduction)
# ------------------------------------------------------------------------------

@ti.kernel
def compute_local_max_overlaps(pos: ti.template(), rad: ti.template(),
                                cell_start: ti.template(), cell_count: ti.template(),
                                cell_indices: ti.template(),
                                local_max_depth: ti.template(), n: ti.i32):
    """
    Computes per-particle maximum overlap depth.
    First pass: local max per particle (no contention).
    """
    for i in range(n):
        pi = pos[i]
        ri = rad[i]
        
        # My cell coordinate (PBC-aware)
        p_wrapped = wrapP(pi)
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        local_max = 0.0
        
        # Check 27 neighboring cells
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    
                    # Wrap cell indices (PBC-aware)
                    nc = wrap_cell(nc)
                    
                    # Linear index (centralized function)
                    nc_id = cell_id(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    for k in range(start, start + count):
                        j = cell_indices[k]
                        if i < j:  # Check each pair once
                            pj = pos[j]
                            rj = rad[j]
                            
                            delta = pdelta(pi, pj)
                            dist = ti.sqrt(delta.dot(delta) + EPS)
                            
                            target_dist = ri + rj
                            if dist < target_dist:
                                depth = target_dist - dist
                                local_max = ti.max(local_max, depth)
        
        local_max_depth[i] = local_max


@ti.kernel
def reduce_max_depth(local_max_depth: ti.template(), n: ti.i32) -> ti.f32:
    """
    Second pass: global reduction of per-particle maxima.
    Avoids atomic contention from all threads.
    """
    global_max = 0.0
    for i in range(n):
        global_max = ti.max(global_max, local_max_depth[i])
    return global_max


def compute_max_overlap(pos, rad, cell_start, cell_count, cell_indices, local_max_depth, n):
    """
    Wrapper: two-pass reduction for robust max_depth.
    """
    compute_local_max_overlaps(pos, rad, cell_start, cell_count, cell_indices, local_max_depth, n)
    return reduce_max_depth(local_max_depth, n)


# ------------------------------------------------------------------------------
# Deep overlap force fallback (rescue mode)
# ------------------------------------------------------------------------------

@ti.kernel
def apply_repulsive_forces(pos: ti.template(), rad: ti.template(), vel: ti.template(),
                            cell_start: ti.template(), cell_count: ti.template(),
                            cell_indices: ti.template(), dt_substep: ti.f32, n: ti.i32):
    """
    Applies repulsive forces for deep overlaps using velocity integration.
    Runs for 2-4 substeps when deep overlaps are detected.
    
    CRITICAL: Uses i < j to process each pair once, applies equal-and-opposite
    impulses to both particles symmetrically.
    """
    for i in range(n):
        pi = pos[i]
        ri = rad[i]
        mi = 1.0  # Assume unit mass
        
        # My cell coordinate (PBC-aware)
        p_wrapped = wrapP(pi)
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        # Check 27 neighboring cells
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    
                    # Wrap cell indices (PBC-aware)
                    nc = wrap_cell(nc)
                    
                    # Linear index (centralized function)
                    nc_id = cell_id(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    for k in range(start, start + count):
                        j = cell_indices[k]
                        
                        # CRITICAL: Use i < j to avoid double-applying pairs
                        if i < j:
                            pj = pos[j]
                            rj = rad[j]
                            mj = 1.0
                            
                            delta = pdelta(pi, pj)
                            dist = ti.sqrt(delta.dot(delta) + EPS)
                            
                            target_dist = ri + rj
                            if dist < target_dist:
                                depth = target_dist - dist
                                
                                # Only apply force if deep overlap
                                threshold = DEEP_OVERLAP_THRESHOLD * ti.min(ri, rj)
                                if depth > threshold:
                                    # Repulsive force proportional to depth
                                    direction = delta / dist
                                    
                                    # Effective mass
                                    m_eff = (mi * mj) / (mi + mj)
                                    k = ti.cast(FORCE_STIFFNESS_MULTIPLIER * m_eff / (dt_substep * dt_substep), ti.f32)
                                    
                                    f_mag = k * depth
                                    
                                    # Clamp force to prevent explosion
                                    max_f = 0.5 * depth / dt_substep  # Max impulse = 50% depth
                                    f_mag = ti.min(f_mag, max_f)
                                    
                                    # Compute impulses (equal and opposite)
                                    impulse_i = -direction * f_mag * dt_substep / mi
                                    impulse_j = direction * f_mag * dt_substep / mj
                                    
                                    # Apply atomically to both particles
                                    ti.atomic_add(vel[i], impulse_i)
                                    ti.atomic_add(vel[j], impulse_j)


@ti.kernel
def integrate_velocities(pos: ti.template(), vel: ti.template(), n: ti.i32, dt_substep: ti.f32):
    """
    Integrate velocities → positions for force fallback substep.
    Separate kernel for clarity.
    """
    for i in range(n):
        pos[i] += vel[i] * dt_substep
        vel[i] *= FORCE_DAMPING  # Per-substep damping
        
        # Wrap to periodic domain
        for d in ti.static(range(3)):
            if pos[i][d] < 0.0:
                pos[i][d] += DOMAIN_SIZE
            elif pos[i][d] >= DOMAIN_SIZE:
                pos[i][d] -= DOMAIN_SIZE


@ti.kernel
def apply_global_damping(vel: ti.template(), n: ti.i32):
    """
    Apply gentle global damping to prevent slow energy accumulation.
    Call once per frame (not per substep).
    """
    for i in range(n):
        vel[i] *= GLOBAL_DAMPING  # 0.5% damping per frame


# ------------------------------------------------------------------------------
# XPBD radius adaptation (frame-rate independent)
# ------------------------------------------------------------------------------

@ti.kernel
def update_radii_xpbd(rad: ti.template(), deg: ti.template(), n: ti.i32, dt: ti.f32,
                       r_min: ti.f32, r_max: ti.f32):
    """
    XPBD-compliant radius adaptation.
    Frame-rate independent, rate-limited.
    Accepts runtime-adjustable min/max radius limits.
    """
    for i in range(n):
        d = deg[i]
        r_old = rad[i]
        
        # Compute desired change (5% rule)
        desired_change = 0.0
        if d < DEG_LOW:
            desired_change = 0.05 * r_old  # Grow 5%
        elif d > DEG_HIGH:
            desired_change = -0.05 * r_old  # Shrink 5%
        # else: no change (d in [DEG_LOW, DEG_HIGH])
        
        # XPBD constraint resolution
        # Compliance α makes this frame-rate independent
        alpha = RADIUS_COMPLIANCE
        delta_r = desired_change / (1.0 + alpha / (dt * dt))
        
        # Rate limit: max 2% change per frame
        max_delta = RADIUS_RATE_LIMIT * r_old
        delta_r = ti.max(-max_delta, ti.min(max_delta, delta_r))
        
        # Apply
        r_new = r_old + delta_r
        
        # Hard clamp to runtime-adjustable [r_min, r_max]
        rad[i] = ti.max(r_min, ti.min(r_max, r_new))


# ------------------------------------------------------------------------------
# XSPH velocity smoothing (anti-jitter)
# ------------------------------------------------------------------------------

@ti.kernel
def apply_xsph_smoothing(pos: ti.template(), vel: ti.template(), vel_temp: ti.template(),
                         rad: ti.template(),
                         cell_start: ti.template(), cell_count: ti.template(),
                         cell_indices: ti.template(), n: ti.i32):
    """
    Smooth velocities using XSPH: blend with neighbors.
    Reduces jitter after PBD.
    
    vel_temp: temporary buffer for smoothed velocities (must be pre-allocated)
    """
    for i in range(n):
        pi = pos[i]
        
        # My cell coordinate (PBC-aware)
        p_wrapped = wrapP(pi)
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        # Initialize average velocity (default to own velocity)
        v_avg = vel[i]
        
        # Accumulate neighbor velocities
        v_sum = ti.Vector([0.0, 0.0, 0.0])
        neighbor_count = 0
        
        # Check 27 neighboring cells
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    
                    # Wrap cell indices (PBC-aware)
                    nc = wrap_cell(nc)
                    
                    # Linear index (centralized function)
                    nc_id = cell_id(nc)
                    
                    start = cell_start[nc_id]
                    cnt = cell_count[nc_id]
                    for k in range(start, start + cnt):
                        j = cell_indices[k]
                        if i != j:
                            # Simple neighbor criterion: within 2*R_MAX
                            pj = pos[j]
                            delta = pdelta(pi, pj)
                            dist_sq = delta.dot(delta)
                            
                            if dist_sq < (2.0 * R_MAX) ** 2:
                                v_sum += vel[j]
                                neighbor_count += 1
        
        # Compute average neighbor velocity (only if we found neighbors)
        if neighbor_count > 0:
            v_avg = v_sum / ti.cast(neighbor_count, ti.f32)
        
        # Blend and store in temp
        vel_temp[i] = (1.0 - XSPH_EPSILON) * vel[i] + XSPH_EPSILON * v_avg
    
    # Copy back from temp to vel
    for i in range(n):
        vel[i] = vel_temp[i]

