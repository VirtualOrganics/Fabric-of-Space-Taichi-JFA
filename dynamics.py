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
    PBC_ENABLED, HALF_L, INV_L,
    JITTER_ENABLED, JITTER_RMS, JITTER_TAU, MAX_DRIFT_FRACTION,
    LEVY_ENABLED, LEVY_ALPHA, LEVY_DEG_SPAN, LEVY_STEP_FRAC, LEVY_USE_TOPO_DEG
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
                       r_min: ti.f32, r_max: ti.f32, deg_low: ti.i32, deg_high: ti.i32,
                       gain_grow: ti.f32, gain_shrink: ti.f32, rate_limit: ti.f32):
    """
    XPBD-compliant radius adaptation.
    Frame-rate independent, rate-limited.
    Accepts runtime-adjustable parameters:
      - r_min, r_max: radius bounds
      - deg_low, deg_high: degree thresholds
      - gain_grow, gain_shrink: growth/shrink rates
      - rate_limit: max fractional change per frame (runtime, auto-enforced >= gain)
    """
    for i in range(n):
        d = deg[i]
        r_old = rad[i]
        
        # Compute desired change (configurable gain)
        desired_change = 0.0
        if d < deg_low:
            desired_change = gain_grow * r_old  # Grow by gain_grow
        elif d > deg_high:
            desired_change = -gain_shrink * r_old  # Shrink by gain_shrink
        # else: no change (d in [deg_low, deg_high])
        
        # XPBD constraint resolution
        # Compliance α makes this frame-rate independent
        alpha = RADIUS_COMPLIANCE
        delta_r = desired_change / (1.0 + alpha / (dt * dt))
        
        # Rate limit: max rate_limit% change per frame (runtime parameter)
        max_delta = rate_limit * r_old
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


# ==============================================================================
# Kernel 7: Brownian Motion (Ornstein-Uhlenbeck smooth drift)
# ==============================================================================

@ti.kernel
def init_jitter_velocities(v_jit: ti.template(), n: ti.i32):
    """Initialize jitter velocities to zero."""
    for i in range(n):
        v_jit[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def apply_brownian(v_jit: ti.template(), rad: ti.template(), 
                   mean_radius: ti.template(), n: ti.i32, dt: ti.f32):
    """
    Apply smooth Brownian drift using Ornstein-Uhlenbeck (OU) noise.
    
    Updates:
        v_jit: per-particle jitter velocity (vec3 field)
    
    This creates gentle, smooth motion without atomics or PBD destabilization.
    OU process: dv = -(v/τ)*dt + σ*sqrt(dt)*ξ
    where ξ ~ U[-0.5,0.5]³ (cheap approx to Gaussian)
    """
    if ti.static(JITTER_ENABLED):
        mean_r = mean_radius[None]
        
        # Target RMS step size: JITTER_RMS * mean_radius per second
        # OU formula: σ = target_rms * sqrt(2 / τ)
        sigma = JITTER_RMS * mean_r * ti.sqrt(2.0 / ti.max(1e-6, JITTER_TAU))
        
        for i in range(n):
            # Random vector: U[-0.5, 0.5]³ (cheap approximation to Gaussian)
            r = ti.Vector([
                ti.random(ti.f32) - 0.5,
                ti.random(ti.f32) - 0.5,
                ti.random(ti.f32) - 0.5
            ])
            
            # OU update: dv = -(v/τ)*dt + σ*sqrt(dt)*ξ
            v = v_jit[i]
            v += (- v / ti.max(1e-6, JITTER_TAU)) * dt + sigma * ti.sqrt(dt) * r
            
            # Cap per-step drift to protect PBD
            # Allow ≤ MAX_DRIFT_FRACTION * GAP * local_size
            cap = MAX_DRIFT_FRACTION * GAP_FRACTION * (rad[i] + rad[i])
            L = ti.sqrt(v.dot(v)) * dt  # Step length
            if L > cap:
                v *= (cap / ti.max(1e-6, L)) / dt  # Scale back
            
            v_jit[i] = v


@ti.kernel
def integrate_jitter(pos: ti.template(), v_jit: ti.template(), n: ti.i32, dt: ti.f32):
    """
    Integrate jitter velocities into positions with PBC wrapping.
    
    Updates:
        pos: particle positions (wrapped to [-L/2, L/2)³)
    """
    if ti.static(JITTER_ENABLED):
        for i in range(n):
            # Update position and wrap (PBC-safe)
            pos[i] = wrapP(pos[i] + v_jit[i] * dt)


# ==============================================================================
# Kernel 8: Lévy Positional Diffusion (Track 2 - Topological Regularization)
# ==============================================================================
# These kernels implement the Lévy centroidal relaxation approximation.
# Particles diffuse toward the spatial average of better-connected neighbors,
# creating smooth, self-organizing foam structure.
#
# References:
# - Lévy, B. et al. "Centroidal Voronoi Tesselations" (2010)
# - Phase B blueprint: Lévy Diffusion Addendum

@ti.kernel
def compute_mean_radius(rad: ti.template(), n: ti.i32, mean_radius: ti.template()):
    """
    Compute global mean radius for step size normalization.
    
    Updates:
        mean_radius[None]: scalar field with mean radius value
    
    Note:
        This is a simple sequential reduction. For N > 50k, consider
        a parallel reduction kernel for better performance.
    """
    acc = 0.0
    for i in range(n):
        acc += rad[i]
    mean_radius[None] = acc / ti.max(1, n)


@ti.kernel
def smooth_degree(deg: ti.template(), deg_smoothed: ti.template(), 
                  n: ti.i32, alpha: ti.f32):
    """
    Apply Exponential Moving Average (EMA) smoothing to degree values.
    
    Smoothing reduces high-frequency noise in degree counts, leading to
    more stable Lévy diffusion behavior.
    
    Formula:
        deg_smoothed[i] = (1 - α) * deg_smoothed[i] + α * deg[i]
    
    Typical α values:
        - 0.1: Very smooth, slow response
        - 0.25: Good balance (recommended)
        - 0.5: Fast response, less smoothing
    
    Updates:
        deg_smoothed: smoothed degree field (EMA state)
    """
    for i in range(n):
        # EMA update
        deg_smoothed[i] = (1.0 - alpha) * deg_smoothed[i] + alpha * ti.cast(deg[i], ti.f32)


@ti.kernel
def levy_position_diffusion(pos: ti.template(), deg_smoothed: ti.template(),
                            cell_start: ti.template(), cell_count: ti.template(),
                            cell_indices: ti.template(), mean_radius: ti.template(),
                            n: ti.i32, alpha_diffusion: ti.f32, 
                            degree_span: ti.f32, max_step_frac: ti.f32):
    """
    Lévy positional diffusion: particles shift toward better-connected neighbors.
    
    This kernel implements a discrete approximation of Lévy's centroidal power
    diagram relaxation. Each particle moves toward the spatial average of neighbors
    weighted by their degree difference.
    
    Algorithm:
        1. For each particle i, iterate over geometric neighbors j
        2. Compute weight w = clamp((deg_j - deg_i) / span, -1, 1)
        3. Accumulate weighted displacement: Δp_i += w * (p_j - p_i)
        4. Normalize by neighbor count: Δp_i /= count
        5. Clamp step size to max_step_frac * mean_radius
        6. Update position with PBC wrapping: p_i += α * Δp_i
    
    Parameters:
        pos: particle positions (updated in-place)
        deg_smoothed: smoothed degree values (geometric or topological)
        cell_start, cell_count, cell_indices: spatial grid structure
        mean_radius: global mean radius (for step size clamping)
        n: number of active particles
        alpha_diffusion: diffusion gain (typical: 0.04)
        degree_span: normalization constant for degree differences (typical: 10.0)
        max_step_frac: max step as fraction of mean radius (typical: 0.15)
    
    Notes:
        - Uses PBC-aware distance calculation (pdelta)
        - Step size clamping prevents large jumps that could destabilize PBD
        - No atomics required (pure per-particle computation)
        - Cost: O(27 * avg_neighbors_per_cell * N) per call
    
    Future:
        Once Phase B's Gabriel topology is restored, swap deg_smoothed source
        from geometric to topological degree (topo_deg_ema) by setting
        LEVY_USE_TOPO_DEG = True in config.py
    """
    if ti.static(LEVY_ENABLED):
        mean_r = mean_radius[None]
        
        for i in range(n):
            pi = pos[i]
            deg_i = deg_smoothed[i]
            
            # Accumulate weighted shift from neighbors
            shift = ti.Vector([0.0, 0.0, 0.0])
            count = 0
            
            # --- Compute my cell coordinate (PBC-aware) ---
            p_wrapped = wrapP(pi)
            q = (p_wrapped + HALF_L) * INV_L
            my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
            
            # --- Iterate over 3x3x3 neighborhood ---
            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    for dz in ti.static(range(-1, 2)):
                        # Neighbor cell (wrapped)
                        nc = my_cell + ti.Vector([dx, dy, dz])
                        nc = wrap_cell(nc)
                        nc_id = cell_id(nc)
                        
                        # Iterate particles in this cell
                        start = cell_start[nc_id]
                        cell_particle_count = cell_count[nc_id]
                        for k_idx in range(start, start + cell_particle_count):
                            j = cell_indices[k_idx]
                            
                            # Skip self
                            if j == i:
                                continue
                            
                            # Neighbor degree and position
                            deg_j = deg_smoothed[j]
                            pj = pos[j]
                            
                            # Compute weight: normalized degree difference
                            # w > 0 if j has more neighbors (attract toward j)
                            # w < 0 if j has fewer neighbors (repel from j)
                            w = tm.clamp((deg_j - deg_i) / degree_span, -1.0, 1.0)
                            
                            # Skip negligible weights (optimization)
                            if abs(w) < 1e-4:
                                continue
                            
                            # PBC-aware displacement vector (j → i)
                            dir_vec = pdelta(pj, pi)
                            
                            # Accumulate weighted shift
                            shift += w * dir_vec
                            count += 1
            
            # --- Apply diffusion step ---
            if count > 0:
                # Average shift
                shift /= ti.cast(count, ti.f32)
                
                # Clamp step size to prevent large jumps
                max_step = max_step_frac * mean_r
                L = shift.norm()
                if L > max_step:
                    shift = shift.normalized() * max_step
                
                # Update position with diffusion gain and PBC wrapping
                pos[i] = wrapP(pi + alpha_diffusion * shift)


# ==============================================================================
# Visualization: Size-based colormaps and band highlighting
# ==============================================================================

@ti.func
def clamp01(x: ti.f32) -> ti.f32:
    """Clamp value to [0, 1] range."""
    return ti.min(1.0, ti.max(0.0, x))

@ti.func
def colormap_viridis(t: ti.f32) -> ti.types.vector(3, ti.f32):
    """
    Viridis colormap (perceptually uniform, colorblind-friendly).
    Polynomial approximation for lightweight GPU evaluation.
    """
    t = clamp01(t)
    r = 0.2803 + 0.2331*t + 0.1533*t*t - 0.3130*t*t*t
    g = 0.0040 + 0.8090*t - 0.1520*t*t - 0.1140*t*t*t
    b = 0.3340 + 1.1960*t - 1.2460*t*t + 0.3620*t*t*t
    return ti.Vector([clamp01(r), clamp01(g), clamp01(b)])

@ti.func
def colormap_turbo(t: ti.f32) -> ti.types.vector(3, ti.f32):
    """
    Google Turbo colormap (high contrast, punchy).
    Polynomial fit for GPU evaluation.
    """
    t = clamp01(t)
    r = 0.13572138 + 4.61539260*t - 42.66032258*t*t + 132.13108234*t*t*t - 152.94239396*t*t*t*t + 59.28637943*t*t*t*t*t
    g = 0.09140261 + 2.19418839*t +   4.84296658*t*t -  14.18503333*t*t*t +   4.27729857*t*t*t*t +  2.82956604*t*t*t*t*t
    b = 0.10667330 + 8.40954615*t -  33.31800624*t*t +  60.19473600*t*t*t -  56.29973286*t*t*t*t +  20.03382602*t*t*t*t*t
    return ti.Vector([clamp01(r), clamp01(g), clamp01(b)])

@ti.func
def colormap_inferno(t: ti.f32) -> ti.types.vector(3, ti.f32):
    """
    Inferno colormap (warm, good for heat maps).
    Lightweight approximation.
    """
    t = clamp01(t)
    r = clamp01(0.000 + 2.0*t - 0.5*t*t)
    g = clamp01(-0.1 + 2.8*t - 2.1*t*t + 0.3*t*t*t)
    b = clamp01(0.2 + 0.5*t + 1.2*t*t - 1.1*t*t*t)
    return ti.Vector([r, g, b])

@ti.func
def pick_palette(palette_id: ti.i32, t: ti.f32) -> ti.types.vector(3, ti.f32):
    """
    Select colormap by ID:
      0 = Viridis (perceptually uniform)
      1 = Turbo (high contrast)
      2 = Inferno (warm)
    """
    result = ti.Vector([0.5, 0.5, 0.5])  # Default gray
    if palette_id == 0:
        result = colormap_viridis(t)
    elif palette_id == 1:
        result = colormap_turbo(t)
    elif palette_id == 2:
        result = colormap_inferno(t)
    return result

@ti.kernel
def update_colors_by_size(rad: ti.template(), deg: ti.template(), color: ti.template(),
                          n: ti.i32, rad_min: ti.f32, rad_max: ti.f32,
                          deg_low: ti.i32, deg_high: ti.i32,
                          viz_mode: ti.i32, band_min: ti.f32, band_max: ti.f32,
                          hide_out: ti.i32, palette: ti.i32, dim_alpha: ti.f32):
    """
    Update particle colors based on visualization mode:
      Mode 0: Degree-based (original, red/green/blue by neighbor count)
      Mode 1: Size heatmap (colormap by radius)
      Mode 2: Size band highlight (brighten in-band, dim/hide out-of-band)
    
    Args:
        rad: particle radii
        deg: particle degrees (neighbor counts)
        color: output RGB colors
        n: number of active particles
        rad_min, rad_max: observed radius range for normalization
        deg_low, deg_high: degree band thresholds
        viz_mode: 0=degree, 1=heatmap, 2=band
        band_min, band_max: radius band for mode 2
        hide_out: 0=dim, 1=hide out-of-band
        palette: 0=viridis, 1=turbo, 2=inferno
        dim_alpha: brightness for out-of-band particles
    """
    span = ti.max(1e-9, rad_max - rad_min)  # Avoid division by zero
    
    for i in range(n):
        r = rad[i]
        d = deg[i]
        
        # Normalize radius to [0, 1] for colormap
        t = (r - rad_min) / span
        t = clamp01(t)
        
        # Default color (gray)
        c = ti.Vector([0.5, 0.5, 0.5])
        
        # Mode 0: Degree-based colors (original logic)
        if viz_mode == 0:
            if d < deg_low:
                c = ti.Vector([1.0, 0.2, 0.2])  # Red: low degree (growing)
            elif d > deg_high:
                c = ti.Vector([0.2, 0.2, 1.0])  # Blue: high degree (shrinking)
            else:
                c = ti.Vector([0.2, 1.0, 0.2])  # Green: in-band (stable)
        
        # Mode 1: Size heatmap (colormap by radius)
        elif viz_mode == 1:
            c = pick_palette(palette, t)
        
        # Mode 2: Size band highlight
        else:  # viz_mode == 2
            in_band = (r >= band_min) and (r <= band_max)
            if in_band:
                # In-band: full brightness with colormap
                c = pick_palette(palette, t)
            else:
                # Out-of-band: dim or hide
                if hide_out == 1:
                    # Hide: nearly black (renderer still draws, but invisible)
                    c = ti.Vector([0.0, 0.0, 0.0])
                else:
                    # Dim: soft gray based on dim_alpha
                    c = ti.Vector([dim_alpha, dim_alpha, dim_alpha])
        
        color[i] = c


@ti.kernel
def filter_particles_by_band(pos: ti.template(), rad: ti.template(), color: ti.template(),
                             pos_render: ti.template(), rad_render: ti.template(), 
                             color_render: ti.template(), render_count: ti.template(),
                             n: ti.i32, band_min: ti.f32, band_max: ti.f32):
    """
    Copy only in-band particles to render buffers.
    Used when hiding out-of-band particles for true transparency.
    
    Args:
        pos, rad, color: source particle data
        pos_render, rad_render, color_render: destination render buffers
        render_count: output count of in-band particles (0D field)
        n: total active particles
        band_min, band_max: radius band to filter
    
    Note: This uses a serial loop with atomic counter for simplicity.
    For large N (>50K), could optimize with parallel prefix sum.
    """
    # Reset render count
    render_count[None] = 0
    
    # Serial pass: copy in-band particles
    for i in range(n):
        r = rad[i]
        if r >= band_min and r <= band_max:
            # This particle is in-band, copy to render buffer
            idx = ti.atomic_add(render_count[None], 1)
            pos_render[idx] = pos[i]
            rad_render[idx] = r
            color_render[idx] = color[i]

