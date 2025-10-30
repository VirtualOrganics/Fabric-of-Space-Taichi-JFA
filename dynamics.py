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
    LEVY_ENABLED, LEVY_ALPHA, LEVY_DEG_SPAN, LEVY_STEP_FRAC, LEVY_USE_TOPO_DEG,
    # FSC-Only Controller
    FSC_DEADBAND, CONTACT_TOL,
    # Pressure Equilibration
    PRESSURE_EQUILIBRATION_ENABLED, PRESSURE_K, PRESSURE_EXP,
    PRESSURE_PAIR_CAP, MAX_EQ_NEI, N
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


@ti.kernel
def project_with_pull(
    pos: ti.template(), rad: ti.template(),
    cell_start: ti.template(), cell_count: ti.template(),
    cell_indices: ti.template(), n: ti.i32,
    mean_r: ti.f32, beta_push: ti.f32, beta_pull: ti.f32,
    deadzone_tau: ti.f32, step_cap_frac: ti.f32,
    push_count: ti.template(), pull_count: ti.template()
):
    """
    Signed-gap PBD: Push overlaps apart AND pull gaps together.
    
    This enables pressure propagation through the foam:
    - Expanding particles create overlaps → push pressure
    - Shrinking particles create gaps → pull pressure
    - Pressure propagates through intermediate particles
    
    Args:
        pos, rad: particle data
        cell_*: spatial hashing grid
        n: active particle count
        mean_r: mean radius (for dead-zone and step clamping)
        beta_push: correction strength for overlaps (typically 1.0)
        beta_pull: correction strength for gaps (typically 0.2, weaker)
        deadzone_tau: don't pull if gap < tau * mean_r (prevents jitter)
        step_cap_frac: maximum step as fraction of mean_r
        push_count, pull_count: telemetry counters (0D fields)
    """
    # Process all particles in parallel (atomics make counters thread-safe)
    for i in range(n):
        correction = ti.Vector([0.0, 0.0, 0.0])
        
        # My cell (PBC-aware)
        p_wrapped = wrapP(pos[i])
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        # Check 27 neighboring cells
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    nc = wrap_cell(nc)
                    nc_id = cell_id(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    
                    for k in range(start, start + count):
                        j = cell_indices[k]
                        
                        if i != j:
                            # Minimum-image vector from i to j
                            delta = pdelta(pos[i], pos[j])
                            dist = ti.sqrt(delta.dot(delta) + 1e-12)
                            direction = delta / dist
                            
                            # Target separation (contact distance)
                            target_sep = rad[i] + rad[j]
                            
                            # Signed gap: negative = overlap, positive = gap
                            gap = dist - target_sep
                            
                            # Initialize correction magnitude
                            correction_mag = 0.0
                            
                            # Decide action based on signed gap
                            if gap < 0.0:
                                # OVERLAP: push apart (standard PBD)
                                overlap_depth = -gap
                                correction_mag = beta_push * overlap_depth * 0.5
                                ti.atomic_add(push_count[None], 1)
                                
                            elif gap > deadzone_tau * mean_r:
                                # GAP BEYOND DEAD-ZONE: pull together
                                gap_excess = gap - deadzone_tau * mean_r
                                correction_mag = -beta_pull * gap_excess * 0.5  # Negative = pull
                                ti.atomic_add(pull_count[None], 1)
                                
                            else:
                                # DEAD-ZONE: do nothing
                                correction_mag = 0.0
                            
                            if correction_mag != 0.0:
                                # Apply correction
                                # Positive mag = push away (negative direction)
                                # Negative mag = pull together (positive direction)
                                correction_vec = -direction * correction_mag
                                
                                # Cap per-pair displacement
                                max_displacement = step_cap_frac * mean_r
                                for d in ti.static(range(3)):
                                    correction_vec[d] = ti.max(-max_displacement,
                                                                ti.min(max_displacement, correction_vec[d]))
                                
                                correction += correction_vec
        
        # Cap total accumulated correction
        max_total_displacement = step_cap_frac * mean_r * 2.0  # Allow some accumulation
        correction_magnitude = ti.sqrt(correction.dot(correction))
        if correction_magnitude > max_total_displacement:
            correction = correction * (max_total_displacement / correction_magnitude)
        
        # Apply correction atomically
        for d in ti.static(range(3)):
            ti.atomic_add(pos[i][d], correction[d])
        
        # Wrap position (PBC)
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
def update_colors_by_size(rad: ti.template(), color: ti.template(),
                          n: ti.i32, rad_min: ti.f32, rad_max: ti.f32,
                          viz_mode: ti.i32, band_min: ti.f32, band_max: ti.f32,
                          hide_out: ti.i32, palette: ti.i32, dim_alpha: ti.f32):
    """
    Update particle colors based on visualization mode (FSC-Only, no degree visualization).
      Mode 0: Size heatmap (colormap by radius) - default
      Mode 1: Size heatmap (same as mode 0)
      Mode 2: Size band highlight (brighten in-band, dim/hide out-of-band)
    
    Args:
        rad: particle radii
        color: output RGB colors
        n: number of active particles
        rad_min, rad_max: observed radius range for normalization
        viz_mode: 0/1=heatmap, 2=band
        band_min, band_max: radius band for mode 2
        hide_out: 0=dim, 1=hide out-of-band
        palette: 0=viridis, 1=turbo, 2=inferno
        dim_alpha: brightness for out-of-band particles
    """
    span = ti.max(1e-9, rad_max - rad_min)  # Avoid division by zero
    
    for i in range(n):
        r = rad[i]
        
        # Normalize radius to [0, 1] for colormap
        t = (r - rad_min) / span
        t = clamp01(t)
        
        # Default color (gray)
        c = ti.Vector([0.5, 0.5, 0.5])
        
        # Mode 0 or 1: Size heatmap (colormap by radius)
        if viz_mode <= 1:
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
def filter_write_indices(rad: ti.template(), idx_render: ti.template(), 
                         render_count: ti.template(), max_render: ti.i32,
                         mode: ti.i32, band_min: ti.f32, band_max: ti.f32,
                         n: ti.i32, r_min_obs: ti.f32, r_max_obs: ti.f32):
    """
    Surgical index-only filter: write indices, not full data copies.
    
    Modes:
      0 = NORMAL: Use band_min/band_max from GUI
      1 = ALL: Keep all particles (tests render path)
      2 = EVERY_OTHER: Keep even indices (tests 50% filtering)
      3 = MIDDLE_THIRD: Keep middle 33% of observed range (ignores GUI band)
    
    Args:
        rad: particle radii
        idx_render: output index buffer
        render_count: output count (0D field)
        max_render: maximum render buffer size
        mode: filter mode selector
        band_min, band_max: radius band from GUI (used in mode 0)
        n: total active particles
        r_min_obs, r_max_obs: observed radius range (used in mode 3)
    """
    render_count[None] = 0
    
    lo = ti.min(band_min, band_max)
    hi = ti.max(band_min, band_max)
    eps = 1e-8
    lo -= eps
    hi += eps
    
    # Middle third for debug mode 3
    mid_lo = r_min_obs + (r_max_obs - r_min_obs) / 3.0
    mid_hi = r_min_obs + 2.0 * (r_max_obs - r_min_obs) / 3.0
    
    for i in range(n):
        keep = False
        
        if mode == 1:  # ALL
            keep = True
        elif mode == 2:  # EVERY_OTHER
            keep = (i & 1) == 0
        elif mode == 3:  # MIDDLE_THIRD (observed)
            r = rad[i]
            keep = (r >= mid_lo) and (r <= mid_hi)
        else:  # NORMAL (band_min/band_max)
            r = rad[i]
            keep = (r >= lo) and (r <= hi)
        
        if keep:
            j = ti.atomic_add(render_count[None], 1)
            if j < max_render:
                idx_render[j] = i


@ti.kernel
def gather_filtered_to_render(
    pos: ti.template(),
    rad: ti.template(),
    color: ti.template(),
    idx_render: ti.template(),
    pos_render: ti.template(),
    rad_render: ti.template(),
    col_render: ti.template(),
    n_render: ti.i32,
    active_n: ti.i32
):
    """
    GPU gather kernel: copies filtered particles to render buffers.
    
    Copies first n_render filtered particles from source fields to render fields.
    Pushes remaining particles off-screen so they don't render.
    
    This avoids shape mismatch issues with .from_numpy() and keeps everything
    as Taichi fields for Metal compatibility.
    """
    # Copy filtered particles into render fields [0 .. n_render-1]
    for i in range(n_render):
        j = idx_render[i]  # Original particle index
        pos_render[i] = pos[j]
        rad_render[i] = rad[j]
        col_render[i] = color[j]  # Vector field, copy all components
    
    # Push remainder off-screen (beyond camera view) with zero radius
    for i in range(n_render, active_n):
        pos_render[i] = ti.Vector([1e6, 1e6, 1e6])
        rad_render[i] = 0.0
        col_render[i] = ti.Vector([0.0, 0.0, 0.0])


# ==============================================================================
# Decision Logic (hysteresis + streak locking)
# ==============================================================================

@ti.kernel
def decide_action(
    deg_s: ti.template(),           # deg_smoothed (f32)
    action: ti.template(),          # i32
    lock_pulses: ti.template(),     # i32
    streak: ti.template(),          # i32
    n: ti.i32,
    deg_low: ti.f32,
    deg_high: ti.f32,
    hysteresis: ti.f32,
    streak_lock: ti.i32
):
    """
    Decide per-particle action (grow/shrink/hold) with hysteresis + streak locking.
    
    Hysteresis prevents flip-flop at band boundaries:
    - Grow starts below (deg_low - hysteresis), stops above (deg_low + hysteresis)
    - Shrink starts above (deg_high + hysteresis), stops below (deg_high - hysteresis)
    
    Streak locking keeps a decision for `streak_lock` pulses once made,
    enabling sustained multi-pulse runs (e.g., grow ×5).
    
    Args:
        deg_s: smoothed degree field (EMA)
        action: output action field (-1=shrink, 0=hold, +1=grow)
        lock_pulses: countdown to unlock (int)
        streak: signed streak accumulator (grows on grow, shrinks on shrink, decays on hold)
        n: number of active particles
        deg_low, deg_high: target degree band
        hysteresis: band padding to prevent chattering
        streak_lock: pulses to lock a decision once made
    """
    # Compute decision thresholds with hysteresis
    grow_start = deg_low - hysteresis
    grow_stop = deg_low + hysteresis
    shrink_start = deg_high + hysteresis
    shrink_stop = deg_high - hysteresis
    
    for i in range(n):
        if lock_pulses[i] > 0:
            # Keep previous action; decrement lock counter
            lock_pulses[i] -= 1
        else:
            # Make new decision based on degree
            a = 0
            d = deg_s[i]
            if d < grow_start:
                a = 1
            elif d > shrink_start:
                a = -1
            else:
                a = 0
            
            action[i] = a
            
            # Lock decision if non-zero (grow or shrink)
            if a != 0:
                lock_pulses[i] = streak_lock
        
        # Update streak counter (momentum source)
        # Grows on grow, shrinks on shrink, decays toward 0 on hold
        if action[i] > 0:
            streak[i] = ti.min(streak[i] + 1, 32767)  # Cap at max int16
        elif action[i] < 0:
            streak[i] = ti.max(streak[i] - 1, -32768)  # Cap at min int16
        else:
            # Decay toward 0 during hold
            if streak[i] > 0:
                streak[i] -= 1
            elif streak[i] < 0:
                streak[i] += 1


@ti.kernel
def update_radii_with_actions(
    rad: ti.template(),
    action: ti.template(),      # -1, 0, +1
    streak: ti.template(),      # signed
    n: ti.i32,
    base_rate: ti.f32,          # grow_rate_rt[None]
    rate_limit: ti.f32,         # max per pulse, runtime
    rmin: ti.f32,
    rmax: ti.f32,
    momentum: ti.f32,           # MOMENTUM (0.0 = disabled)
    streak_cap: ti.i32
):
    """
    Update radii based on per-particle actions with optional momentum.
    
    Applies geometric growth/shrink: r_new = r_old * (1 + effective_rate)
    effective_rate = base_rate * (1 + momentum * min(|streak|, streak_cap))
    
    Enables sustained exponential compounding during multi-pulse streaks.
    
    Args:
        rad: particle radii
        action: per-particle decision (-1=shrink, 0=hold, +1=grow)
        streak: signed streak counter (for momentum)
        n: number of active particles
        base_rate: base growth/shrink rate per pulse
        rate_limit: max rate per pulse (runtime safety clamp)
        rmin, rmax: hard radius bounds
        momentum: momentum gain per streak unit (0.0 = disabled)
        streak_cap: max streak for momentum calculation
    """
    for i in range(n):
        a = action[i]
        if a == 0:
            continue  # Hold: no change
        
        # Compute momentum gain (optional)
        s_abs = ti.cast(ti.abs(streak[i]), ti.f32)
        s_abs = ti.min(s_abs, ti.cast(streak_cap, ti.f32))
        gain = base_rate * (1.0 + momentum * s_abs)
        
        # Apply action sign, clamp to rate_limit
        eff = ti.min(gain, rate_limit)
        eff = eff * ti.cast(a, ti.f32)
        
        # Geometric update: r_new = r_old * (1 + eff)
        r_old = rad[i]
        r_new = r_old * (1.0 + eff)
        
        # Clamp to hard bounds
        r_new = ti.max(rmin, ti.min(rmax, r_new))
        rad[i] = r_new

@ti.kernel
def set_radius_targets(
    rad: ti.template(),
    rad_target: ti.template(),
    action: ti.template(),      # -1, 0, +1
    n: ti.i32,
    growth_rate: ti.f32,        # e.g., 0.04 = 4% change
    rmin: ti.f32,
    rmax: ti.f32
):
    """
    Set target radii based on per-particle actions.
    Target = current * (1 + action * growth_rate)
    
    This is called once per measurement cycle to set new targets.
    Then nudge_radii_to_targets() is called each frame to smoothly approach targets.
    """
    for i in range(n):
        a = action[i]
        r_current = rad[i]
        
        if a == 0:
            # Hold: target = current (no change)
            rad_target[i] = r_current
        else:
            # Grow (+1) or shrink (-1)
            target = r_current * (1.0 + ti.cast(a, ti.f32) * growth_rate)
            # Clamp target to bounds
            target = ti.max(rmin, ti.min(rmax, target))
            rad_target[i] = target

@ti.kernel
def nudge_radii_to_targets(
    rad: ti.template(),
    rad_target: ti.template(),
    n: ti.i32,
    adjustment_frames: ti.i32  # Total frames in adjustment phase
):
    """
    Smoothly nudge current radii toward targets.
    Called every frame during adjustment phase.
    
    Step size = (target - current) / remaining_frames
    This ensures we reach the target exactly by the end of the adjustment phase.
    """
    step_frac = 1.0 / ti.cast(adjustment_frames, ti.f32)
    
    for i in range(n):
        r_now = rad[i]
        r_tgt = rad_target[i]
        delta = r_tgt - r_now
        rad[i] = r_now + delta * step_frac

# ==============================================================================
# FSC-Only Controller Kernels (Phase 2)
# ==============================================================================

@ti.func
def smoothstep(e0: ti.f32, e1: ti.f32, x: ti.f32) -> ti.f32:
    """
    Smoothstep function for damping band near FSC boundaries.
    
    Args:
        e0: Lower edge
        e1: Upper edge
        x: Value to interpolate
    
    Returns:
        Smoothed value in [0, 1]
    """
    t = ti.min(1.0, ti.max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)

@ti.kernel
def set_fsc_targets(
    rad: ti.template(),
    rad_target: ti.template(),
    fsc: ti.template(),
    fsc_ema: ti.template(),
    n: ti.i32,
    f_low: ti.i32,
    f_high: ti.i32,
    growth_pct: ti.f32,
    r_min: ti.f32,
    r_max: ti.f32
):
    """
    Set radius targets based on FSC band with hysteresis + EMA lag to prevent deadband lock.
    
    Hysteresis Strategy (prevents "frozen equilibrium"):
    - Use EMA (smoothed FSC) for decisions, not raw FSC → adds temporal lag
    - Hysteresis gap (±1): full action outside [f_low-1, f_high+1], reduced action inside
    - Continuous micro-nudge when in-band → keeps dynamics alive even at equilibrium
    
    This ensures the foam never fully "freezes" even when everyone is nominally in-band.
    
    Args:
        rad: Current radii
        rad_target: Target radii (output)
        fsc: Raw Face-Sharing Count from JFA (for fallback only)
        fsc_ema: Smoothed FSC (used for decisions)
        n: Number of particles
        f_low: FSC lower bound (grow below this)
        f_high: FSC upper bound (shrink above this)
        growth_pct: Fractional size change per cycle (e.g., 0.05 = 5%)
        r_min: Minimum radius bound
        r_max: Maximum radius bound
    """
    db = ti.cast(FSC_DEADBAND, ti.f32)
    hysteresis_gap = 1.0  # Hysteresis buffer (±1 FSC) to prevent deadband lock
    idle_rate = 0.003     # Micro-nudge rate when in-band (0.3% of growth_pct)
    
    for i in range(n):
        r0 = rad[i]
        
        # Use EMA for decisions (temporal lag prevents rapid oscillation)
        # Fallback to raw FSC if EMA is zero (startup condition)
        f_decision = fsc_ema[i] if fsc_ema[i] > 0.0 else ti.cast(fsc[i], ti.f32)
        
        f_low_f = ti.cast(f_low, ti.f32)
        f_high_f = ti.cast(f_high, ti.f32)
        
        # Initialize gain (for Taichi scoping)
        gain = 0.0
        
        if f_decision < f_low_f - hysteresis_gap:
            # Far below band → full growth
            gain = 1.0
            rad_target[i] = ti.min(r0 * (1.0 + growth_pct * gain), r_max)
        elif f_decision < f_low_f:
            # Hysteresis zone (below band) → reduced growth with smoothstep
            gain = 1.0 - smoothstep(f_low_f - hysteresis_gap - db, f_low_f, f_decision)
            rad_target[i] = ti.min(r0 * (1.0 + growth_pct * gain), r_max)
        elif f_decision > f_high_f + hysteresis_gap:
            # Far above band → full shrink
            gain = 1.0
            rad_target[i] = ti.max(r0 * (1.0 - growth_pct * gain), r_min)
        elif f_decision > f_high_f:
            # Hysteresis zone (above band) → reduced shrink with smoothstep
            gain = smoothstep(f_high_f, f_high_f + hysteresis_gap + db, f_decision)
            rad_target[i] = ti.max(r0 * (1.0 - growth_pct * gain), r_min)
        else:
            # Within band [f_low, f_high] → apply continuous micro-nudge
            # This prevents complete freeze at equilibrium
            # Nudge direction: gently push toward band center
            band_center = 0.5 * (f_low_f + f_high_f)
            if f_decision < band_center:
                # Below center → tiny growth
                rad_target[i] = ti.min(r0 * (1.0 + growth_pct * idle_rate), r_max)
            else:
                # Above center → tiny shrink
                rad_target[i] = ti.max(r0 * (1.0 - growth_pct * idle_rate), r_min)

@ti.kernel
def compute_controller_stats(
    rad: ti.template(),
    rad_target: ti.template(),
    fsc_ema: ti.template(),
    n: ti.i32,
    f_low: ti.i32,
    f_high: ti.i32
) -> (ti.i32, ti.i32, ti.i32, ti.f32, ti.f32):
    """
    Compute diagnostic statistics for the FSC controller.
    
    Returns: (grow_count, shrink_count, idle_count, mean_delta_r, max_delta_r)
    """
    grow_cnt = ti.cast(0, ti.i32)
    shrink_cnt = ti.cast(0, ti.i32)
    idle_cnt = ti.cast(0, ti.i32)
    sum_delta = 0.0
    max_delta = 0.0
    
    f_low_f = ti.cast(f_low, ti.f32)
    f_high_f = ti.cast(f_high, ti.f32)
    
    for i in range(n):
        f = fsc_ema[i]
        delta_r = rad_target[i] - rad[i]
        
        # Classify action based on FSC
        if f < f_low_f:
            grow_cnt += 1
        elif f > f_high_f:
            shrink_cnt += 1
        else:
            idle_cnt += 1
        
        # Accumulate delta statistics
        sum_delta += delta_r
        if ti.abs(delta_r) > max_delta:
            max_delta = ti.abs(delta_r)
    
    mean_delta = sum_delta / ti.max(1, n)
    return grow_cnt, shrink_cnt, idle_cnt, mean_delta, max_delta


@ti.kernel
def nudge_radii_adaptive_ema(
    rad: ti.template(),
    rad_target: ti.template(),
    local_max_depth: ti.template(),
    n: ti.i32,
    alpha: ti.f32,
    max_step_pct: ti.f32,
    r_min: ti.f32,
    r_max: ti.f32,
    back_global: ti.f32,
    mode: ti.i32
):
    """
    Nudge radii toward targets using adaptive EMA with per-frame cap and backpressure.
    
    Implements Appendix A refinements:
    - Adaptive EMA smoothing for gradual convergence
    - Per-frame cap (MAX_STEP_PCT) to prevent shocks
    - Per-particle backpressure based on local overlap depth
    
    Args:
        rad: Current radii
        rad_target: Target radii
        local_max_depth: Per-particle maximum overlap depth
        n: Number of particles
        alpha: EMA factor (computed from ADJUSTMENT_FRAMES)
        max_step_pct: Per-frame cap on |ΔR|/R
        r_min: Minimum radius bound
        r_max: Maximum radius bound
        back_global: Global backpressure factor (for HUD display)
        mode: Backpressure mode (0=off, 1=global, 2=local)
    """
    for i in range(n):
        r0 = rad[i]
        rt = ti.min(r_max, ti.max(r_min, rad_target[i]))
        
        # Compute EMA step
        d = alpha * (rt - r0)
        
        # Apply backpressure (initialize b first for Taichi scoping)
        b = 1.0  # Default: no backpressure
        
        if mode == 1:
            # Global backpressure
            b = back_global
        elif mode == 2:
            # Per-particle backpressure (Appendix A refinement)
            b = (CONTACT_TOL - local_max_depth[i]) / CONTACT_TOL
            b = ti.min(1.0, ti.max(0.25, b))
        
        d = d * b
        
        # Apply per-frame cap
        cap = max_step_pct * r0
        if d > cap:
            d = cap
        if d < -cap:
            d = -cap
        
        # Apply and clamp
        rad[i] = ti.min(r_max, ti.max(r_min, r0 + d))


# ==============================================================================
# PRESSURE EQUILIBRATION
# ==============================================================================
# Volume-conserving pressure diffusion across FSC neighbors.
# Two-channel control: FSC controller (slow, topological) + pressure equilibration (fast, mechanical)

@ti.func
def hash_func(i: ti.i32, frame: ti.i32) -> ti.i32:
    """
    Deterministic hash for stochastic neighbor selection.
    
    Uses Linear Congruential Generator (LCG) for GPU determinism.
    Returns a positive integer for use as offset in neighbor rotation.
    
    Args:
        i: Particle index
        frame: Current frame number
    
    Returns:
        Hash value (positive integer)
    """
    return (1103515245 * (i + 12345 * frame) + 12345) & 0x7fffffff


@ti.kernel
def equilibrate_pressure(
    n: ti.i32,
    frame: ti.i32,
    pos: ti.template(),
    rad: ti.template(),
    delta_r: ti.template(),
    jfa_face_ids: ti.template(),
    jfa_fsc: ti.template(),
    k: ti.f32,
    P_exp: ti.f32,
    pair_cap: ti.f32,
    max_nei: ti.i32,
    r_min: ti.f32,
    r_max: ti.f32
) -> (ti.f32, ti.i32):
    """
    Volume-conserving pressure equilibration across FSC neighbors.
    
    Uses Jacobi iteration to avoid race conditions:
    1. Compute volume differences across neighbor pairs
    2. Apply capped volume exchange
    3. Update radii from new volumes
    
    Args:
        n: Number of active particles
        frame: Current frame number (for hash)
        pos: Particle positions (for PBC distance if needed)
        rad: Particle radii (modified in-place)
        delta_r: Temporary buffer for Jacobi updates
        jfa_face_ids: JFA neighbor IDs [N, MAX_NEIGHBORS]
        jfa_fsc: FSC count per particle [N]
        k: Diffusion coefficient
        P_exp: Volume exponent (3.0 for 3D)
        pair_cap: Max ΔV as fraction of min(V_i, V_j)
        max_nei: Max neighbors to process per particle
        r_min: Minimum radius (for clamping)
        r_max: Maximum radius (for clamping)
    
    Returns:
        (max_abs_dr, changed_count): Maximum radius change and count of changed particles
    """
    
    # Step 1: Initialize deltas to zero
    for i in range(n):
        delta_r[i] = 0.0
    
    # Step 2: Compute volume exchanges (Jacobi style)
    for i in range(n):
        # Get current volume (P ∝ r^P_exp)
        Vi = rad[i] ** P_exp
        
        # Get FSC neighbor count
        fsc_count = jfa_fsc[i]
        if fsc_count <= 0:
            continue
        
        # Budget neighbors via hashed rotation
        actual_nei = ti.min(fsc_count, max_nei)
        start_offset = hash_func(i, frame) % fsc_count
        
        # Process budgeted neighbors
        for k_idx in range(actual_nei):
            # Rotate through neighbor list
            nei_idx = (start_offset + k_idx) % fsc_count
            
            # Get neighbor ID from JFA structure
            j = jfa_face_ids[i, nei_idx]
            
            if j < 0 or j >= n or j == i:
                # Skip invalid or self
                continue
            
            # Get neighbor volume
            Vj = rad[j] ** P_exp
            
            # Compute volume difference (pressure gradient proxy)
            delta_V_raw = k * (Vi - Vj)
            
            # Cap by minimum volume (stability)
            V_min = ti.min(Vi, Vj)
            delta_V = ti.max(-pair_cap * V_min, ti.min(pair_cap * V_min, delta_V_raw))
            
            # Compute volume change for particle i only
            # (Particle j will compute its own change when it processes i as its neighbor)
            Vi_new = Vi - delta_V
            
            # Convert back to radius change for particle i
            # New radius: r_new = V_new^(1/P_exp)
            delta_r[i] += (Vi_new ** (1.0 / P_exp)) - rad[i]
    
    # Step 3: Apply deltas with hard clamps and track stats
    max_abs_dr = ti.cast(0.0, ti.f32)
    changed = ti.cast(0, ti.i32)
    
    for i in range(n):
        d = delta_r[i]
        if ti.abs(d) > 1e-12:
            changed += 1
            if ti.abs(d) > max_abs_dr:
                max_abs_dr = ti.abs(d)
        rad[i] = ti.max(r_min, ti.min(r_max, rad[i] + d))
    
    return max_abs_dr, changed


@ti.kernel
def compute_pressure_stats(n: ti.i32, rad: ti.template(), P_exp: ti.f32, r_min: ti.f32, r_max: ti.f32) -> (ti.f32, ti.f32):
    """
    Compute radius range for telemetry.
    
    Note: Pressure variance (σ(P)) is computed on CPU side in fp64 due to GPU fp64 limitations.
    This kernel only returns radius min/max for efficiency.
    
    Args:
        n: Number of active particles
        rad: Particle radii
        P_exp: Pressure exponent (unused here, kept for API compatibility)
        r_min: Minimum radius threshold (for range check)
        r_max: Maximum radius threshold (for range check)
    
    Returns:
        (rmin, rmax): Radius min, radius max
    """
    rmin = 1e9
    rmax = -1e9
    
    # Track radius range
    for i in range(n):
        r = rad[i]
        if r < rmin:
            rmin = r
        if r > rmax:
            rmax = r
    
    return rmin, rmax


# ==============================================================================
# BROWNIAN MOTION (thermal jitter to keep foam active)
# ==============================================================================

@ti.kernel
def apply_brownian_motion(
    pos: ti.template(),
    n: ti.i32,
    strength: ti.f32,
    domain_size: ti.f32
):
    """
    Apply random Brownian motion to keep foam "breathing" at equilibrium.
    
    Adds small random displacement to each particle, creating continuous
    micro-reorganization even when FSC and pressure are in perfect balance.
    This prevents the "frozen equilibrium" problem.
    
    Args:
        pos: Particle positions (modified in-place)
        n: Number of particles
        strength: Displacement magnitude (e.g., 0.0002)
        domain_size: Size of periodic domain
    """
    half_L = domain_size * 0.5
    
    for i in range(n):
        # Generate random displacement in each axis
        dx = (ti.random() - 0.5) * 2.0 * strength
        dy = (ti.random() - 0.5) * 2.0 * strength
        dz = (ti.random() - 0.5) * 2.0 * strength
        
        # Apply displacement
        pos[i][0] += dx
        pos[i][1] += dy
        pos[i][2] += dz
        
        # Wrap PBC
        for axis in ti.static(range(3)):
            if pos[i][axis] < -half_L:
                pos[i][axis] += domain_size
            elif pos[i][axis] >= half_L:
                pos[i][axis] -= domain_size

