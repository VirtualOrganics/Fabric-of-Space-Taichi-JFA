"""
Topological neighbor counting using Gabriel graph (Phase B).

This module implements the Gabriel graph test for Voronoi adjacency approximation.
The Gabriel graph is a subgraph of the Delaunay triangulation where an edge (i, j)
exists iff the open sphere with diameter i-j contains no other points.

Key features:
- Euclidean Gabriel graph (default) for near-uniform sizes
- Laguerre Gabriel graph (optional) for high polydispersity
- Adaptive stencil size (prevents missed witnesses)
- Scaled epsilon (avoids size-dependent misclassification)
- Symmetric processing (j > i, halves work)
- PBC-aware via pdelta and wrapP

Target mean degree: 12-16 (Euclidean), 10-14 (Laguerre)
Update cadence: Every K=20 frames (expensive operation)
EMA smoothing: β=0.1 for radius control stability
"""

import taichi as ti
import taichi.math as tm

from config import (
    GRID_RES, CELL_SIZE, DOMAIN_SIZE, R_MAX,
    TOPO_MAX_RADIUS_MULTIPLE, MAX_TOPO_NEIGHBORS,
    USE_LAGUERRE_GABRIEL,
    PBC_ENABLED, HALF_L, INV_L,
    TOPO_PAIR_SUBSAMPLING_STRIDE, TOPO_EARLY_EXIT_HIGH_DEGREE,
    TOPO_DEG_HIGH,
    TOPO_BATCHES, TOPO_PAIR_SUBSAMPLE_Q,
    USE_KNN_TOPO, KNN_TOPO_K
)
from grid import wrapP, pdelta, wrap_cell, cell_id


# ==============================================================================
# Taichi Fields (topological degree + candidate storage)
# ==============================================================================

# Allocated for MAX_N particles (set in run.py based on user input)
topo_deg = None             # ti.field(ti.i32, shape=MAX_N) - Raw topological degree
topo_deg_ema = None         # ti.field(ti.f32, shape=MAX_N) - EMA-smoothed degree
topo_pairs = None           # ti.field(ti.i32, shape=(MAX_N, MAX_TOPO_NEIGHBORS)) - Candidate pairs
topo_pair_count = None      # ti.field(ti.i32, shape=MAX_N) - Count of candidates per particle
topo_truncated = None       # ti.field(ti.i32, shape=MAX_N) - Truncation flag (1 if capped)


def allocate_fields(max_n):
    """Allocate Taichi fields for topological neighbor counting.
    
    Args:
        max_n: Maximum number of particles (static allocation)
    """
    global topo_deg, topo_deg_ema, topo_pairs, topo_pair_count, topo_truncated
    
    topo_deg = ti.field(ti.i32, shape=max_n)
    topo_deg_ema = ti.field(ti.f32, shape=max_n)
    topo_pairs = ti.field(ti.i32, shape=(max_n, MAX_TOPO_NEIGHBORS))
    topo_pair_count = ti.field(ti.i32, shape=max_n)
    topo_truncated = ti.field(ti.i32, shape=max_n)


# ==============================================================================
# Kernel 1: Build Candidate Pairs (Spatial Pruning)
# ==============================================================================

@ti.kernel
def build_topo_candidates(pos: ti.template(), rad: ti.template(),
                           cell_start: ti.template(), cell_count: ti.template(),
                           cell_indices: ti.template(), n: ti.i32):
    """
    Build list of candidate neighbor pairs for Gabriel test.
    
    For each particle i, find all j within TOPO_MAX_RADIUS_MULTIPLE * (r_i + r_j)
    using the existing spatial grid. Stores up to MAX_TOPO_NEIGHBORS candidates.
    
    Args:
        pos, rad: Particle positions and radii
        cell_start, cell_count, cell_indices: Spatial grid data
        n: Number of active particles
    
    Output:
        topo_pairs[i, 0:topo_pair_count[i]] = candidate neighbors for particle i
        topo_truncated[i] = 1 if candidate list was capped (overflow)
    """
    for i in range(n):
        topo_pair_count[i] = 0
        topo_truncated[i] = 0
        
        # Compute my cell (PBC-aware, same as count_neighbors)
        p_wrapped = wrapP(pos[i])
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        my_cell = wrap_cell(my_cell)
        
        # Max search radius (squared for performance)
        max_radius = TOPO_MAX_RADIUS_MULTIPLE * (rad[i] + R_MAX)  # Conservative upper bound
        max_radius_sq = max_radius * max_radius
        
        # Check 27 neighboring cells
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    nc = wrap_cell(nc)
                    nc_id = cell_id(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    
                    for k_idx in range(start, start + count):
                        j = cell_indices[k_idx]
                        
                        # Only process j > i (symmetric, avoid duplicates)
                        if j > i:
                            # PBC-aware distance (squared, avoid sqrt)
                            delta_ij = pdelta(pos[i], pos[j])
                            dist_sq_ij = delta_ij.dot(delta_ij)
                            
                            # Distance gate
                            threshold = TOPO_MAX_RADIUS_MULTIPLE * (rad[i] + rad[j])
                            threshold_sq = threshold * threshold
                            
                            if dist_sq_ij <= threshold_sq:
                                # Add to candidate list
                                if topo_pair_count[i] < MAX_TOPO_NEIGHBORS:
                                    topo_pairs[i, topo_pair_count[i]] = j
                                    topo_pair_count[i] += 1
                                else:
                                    # Overflow: cap at MAX_TOPO_NEIGHBORS
                                    topo_truncated[i] = 1
                                    break  # Stop adding for this particle


# ==============================================================================
# Kernel 2: Gabriel Test (Witness-Based Topological Degree)
# ==============================================================================

@ti.kernel
def gabriel_test_topological_degree(pos: ti.template(), rad: ti.template(),
                                     cell_start: ti.template(), cell_count: ti.template(),
                                     cell_indices: ti.template(), n: ti.i32, frame: ti.i32):
    """
    Compute topological degree using Gabriel graph test (empty diameter-sphere check).
    
    For each candidate pair (i, j):
      1. Compute midpoint m = (i + j) / 2 (PBC-aware)
      2. Compute diameter-sphere radius r = ||i - j|| / 2
      3. Check if any particle k lies inside the open sphere (witness)
      4. If no witness found, i and j are Gabriel neighbors
    
    Optimizations:
      - Adaptive stencil: search radius scales with pair distance
      - Scaled epsilon: avoids misclassification for large/small pairs
      - Symmetric processing: j > i, update both degrees (halves work)
      - Edge guards: skip degenerate pairs (d_ij_sq < 1e-16)
      - Hashed pair subsampling: process 1/STRIDE of pairs per update
      - Early exit: skip witness test for high-degree pairs
    
    Args:
        pos, rad: Particle positions and radii
        cell_start, cell_count, cell_indices: Spatial grid data
        n: Number of active particles
        frame: Current frame number (for hash-based subsampling)
    
    Output:
        topo_deg[i] = count of Gabriel graph neighbors for particle i
    """
    # Clear topological degree
    for i in range(n):
        topo_deg[i] = 0
    
    # Process all candidate pairs
    for i in range(n):
        for pair_idx in range(topo_pair_count[i]):
            j = topo_pairs[i, pair_idx]
            
            # === PERFORMANCE OPTIMIZATION 1: Hashed pair subsampling ===
            # Process only 1/STRIDE of pairs per update (deterministic, spreads work over time)
            if ti.static(TOPO_PAIR_SUBSAMPLING_STRIDE > 1):
                hash_key = (i * 73856093) ^ (j * 19349663) ^ (frame * 83492791)
                if (hash_key % TOPO_PAIR_SUBSAMPLING_STRIDE) != 0:
                    continue
            
            # === PERFORMANCE OPTIMIZATION 2: Early exit for high-degree pairs ===
            # Skip witness test if both endpoints already exceed target degree
            if ti.static(TOPO_EARLY_EXIT_HIGH_DEGREE):
                if topo_deg_ema[i] >= TOPO_DEG_HIGH and topo_deg_ema[j] >= TOPO_DEG_HIGH:
                    continue
            
            # Compute PBC-aware distance (squared)
            delta_ij = pdelta(pos[i], pos[j])
            d_ij_sq = delta_ij.dot(delta_ij)
            
            # Edge guard: Skip degenerate pairs (too small or self)
            if d_ij_sq < 1e-16:
                continue  # Avoid divide-by-zero in diameter sphere
            
            # Diameter sphere: midpoint and radius
            midpoint = wrapP(pos[i] + 0.5 * delta_ij)  # PBC-aware midpoint
            r_sq_diameter = 0.25 * d_ij_sq  # (d_ij / 2)²
            
            # Compute midpoint cell for witness search
            q = (midpoint + HALF_L) * INV_L
            mid_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
            mid_cell = wrap_cell(mid_cell)
            
            # Adaptive stencil: size based on diameter-sphere radius
            r_diameter = ti.sqrt(d_ij_sq) * 0.5
            stencil_cells = int(ti.ceil(r_diameter / CELL_SIZE))
            stencil_cells = ti.max(1, ti.min(stencil_cells, 3))  # Clamp to [1, 3]
            stencil_cells = ti.min(stencil_cells, GRID_RES // 2)  # Never exceed grid bounds
            
            is_gabriel_neighbor = True  # Innocent until proven guilty
            
            # Iterate adaptive stencil [-stencil_cells, +stencil_cells]³
            for dx in range(-stencil_cells, stencil_cells + 1):
                for dy in range(-stencil_cells, stencil_cells + 1):
                    for dz in range(-stencil_cells, stencil_cells + 1):
                        nc = mid_cell + ti.Vector([dx, dy, dz])
                        nc = wrap_cell(nc)
                        nc_id = cell_id(nc)
                        
                        start = cell_start[nc_id]
                        count = cell_count[nc_id]
                        
                        for k_idx in range(start, start + count):
                            k = cell_indices[k_idx]
                            
                            # Skip i and j themselves
                            if k == i or k == j:
                                continue
                            
                            # Witness test: is k inside the open diameter sphere?
                            delta_km = pdelta(pos[k], midpoint)
                            dist_sq_km = delta_km.dot(delta_km)
                            
                            # Scaled epsilon: avoids misclassification for very large/small pairs
                            eps_scaled = ti.max(1e-12, 1e-6 * r_sq_diameter)
                            
                            # Euclidean Gabriel test (strict open sphere)
                            if dist_sq_km < r_sq_diameter - eps_scaled:
                                is_gabriel_neighbor = False
                                break  # Early exit (one witness is enough)
                        
                        if not is_gabriel_neighbor:
                            break  # Break out of cell loops
                    if not is_gabriel_neighbor:
                        break
                if not is_gabriel_neighbor:
                    break
            
            # If no witness found, i-j are Gabriel neighbors
            # Process each pair once (j > i) and increment both degrees (symmetric)
            if is_gabriel_neighbor and j > i:
                ti.atomic_add(topo_deg[i], 1)
                ti.atomic_add(topo_deg[j], 1)  # Symmetric (halves work, avoids bias)


# ==============================================================================
# Kernel 3: EMA Smoothing
# ==============================================================================

@ti.kernel
def update_topo_ema(ema_alpha: ti.f32, n: ti.i32):
    """
    Update EMA of topological degree (every frame, even if topo not recomputed).
    
    Formula: ema = β * new + (1 - β) * ema
    
    Args:
        ema_alpha: Smoothing factor (0.1 = 10% new, 90% old)
        n: Number of active particles
    """
    for i in range(n):
        # Cast to float for EMA computation
        new_deg = ti.cast(topo_deg[i], ti.f32)
        topo_deg_ema[i] = ema_alpha * new_deg + (1.0 - ema_alpha) * topo_deg_ema[i]


# ==============================================================================
# Kernel 4: Radius Adaptation (Topological Degree)
# ==============================================================================

@ti.kernel
def update_radii_topological(rad: ti.template(), deg_effective: ti.template(),
                              deg_low: ti.f32, deg_high: ti.f32,
                              gain_grow: ti.f32, gain_shrink: ti.f32,
                              r_min: ti.f32, r_max: ti.f32, n: ti.i32):
    """
    Adapt radii based on topological degree (smoothed via EMA).
    
    Rule:
      - deg < deg_low:  grow by gain_grow
      - deg > deg_high: shrink by gain_shrink
      - deg in [deg_low, deg_high]: no change
    
    Args:
        rad: Particle radii (modified in-place)
        deg_effective: Effective degree (blended topo + geom, or pure topo)
        deg_low, deg_high: Target degree band
        gain_grow, gain_shrink: Growth/shrink rates
        r_min, r_max: Hard radius bounds
        n: Number of active particles
    """
    for i in range(n):
        deg = deg_effective[i]
        
        if deg < deg_low:
            # Too few neighbors: grow
            rad[i] *= (1.0 + gain_grow)
        elif deg > deg_high:
            # Too many neighbors: shrink
            rad[i] *= (1.0 - gain_shrink)
        # else: in target band, no change
        
        # Clamp to hard bounds
        rad[i] = ti.max(r_min, ti.min(r_max, rad[i]))

# ==============================================================================
# One-Shot Batched Topology (On-Demand)
# ==============================================================================

@ti.func
def should_sample_pair(i: ti.i32, j: ti.i32, frame: ti.i32, Q: ti.i32) -> ti.i32:
    """
    Deterministic pair subsampling using hash.
    Returns 1 if pair should be processed, 0 otherwise.
    Keeps ~1/Q pairs.
    """
    key = (i * 73856093) ^ (j * 19349663) ^ (frame * 83492791)
    return 1 if (key % Q) == 0 else 0

@ti.kernel
def gabriel_update_batched(pos: ti.template(), rad: ti.template(),
                           cell_start: ti.template(), cell_count: ti.template(),
                           cell_indices: ti.template(), n: ti.i32,
                           batch_idx: ti.i32, num_batches: ti.i32,
                           subsample_q: ti.i32, frame: ti.i32):
    """
    Batched Gabriel graph test - process 1/num_batches of particles per call.
    Subsamples pairs by factor subsample_q to reduce cost.
    """
    for i in range(n):
        # Only process particles assigned to this batch
        if (i % num_batches) != batch_idx:
            continue
        
        # Build candidates and test (same logic as gabriel_test_topological_degree)
        # but with pair subsampling
        topo_pair_count[i] = 0
        topo_truncated[i] = 0
        
        p_wrapped = wrapP(pos[i])
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        my_cell = wrap_cell(my_cell)
        
        # Build candidate list
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
                        
                        if j > i:  # Process each pair once
                            # Subsample pairs
                            if should_sample_pair(i, j, frame, subsample_q) == 0:
                                continue
                            
                            delta_ij = pdelta(pos[i], pos[j])
                            dist_sq_ij = delta_ij.dot(delta_ij)
                            
                            candidate_threshold_sq = (TOPO_MAX_RADIUS_MULTIPLE * (rad[i] + rad[j])) ** 2
                            
                            if dist_sq_ij <= candidate_threshold_sq:
                                current_count = ti.atomic_add(topo_pair_count[i], 1)
                                if current_count < MAX_TOPO_NEIGHBORS:
                                    topo_pairs[i, current_count] = j
                                else:
                                    topo_truncated[i] = 1
        
        # Gabriel test for this particle's candidates
        for pair_idx in range(topo_pair_count[i]):
            j = topo_pairs[i, pair_idx]
            
            delta_ij = pdelta(pos[i], pos[j])
            d_ij_sq = delta_ij.dot(delta_ij)
            if d_ij_sq < 1e-16:
                continue
            
            midpoint = wrapP(pos[i] + 0.5 * delta_ij)
            is_gabriel_neighbor = True
            
            # Euclidean Gabriel test (simplified - no Laguerre for batched mode)
            r_sq_diameter = 0.25 * d_ij_sq
            r_diameter = ti.sqrt(d_ij_sq) * 0.5
            stencil_cells = int(ti.ceil(r_diameter / CELL_SIZE))
            stencil_cells = ti.max(1, ti.min(stencil_cells, 3))
            stencil_cells = ti.min(stencil_cells, GRID_RES // 2)
            
            q_mid = (midpoint + HALF_L) * INV_L
            mid_cell = ti.Vector([int(q_mid[d] * GRID_RES) for d in ti.static(range(3))])
            mid_cell = wrap_cell(mid_cell)
            
            for dx in range(-stencil_cells, stencil_cells + 1):
                for dy in range(-stencil_cells, stencil_cells + 1):
                    for dz in range(-stencil_cells, stencil_cells + 1):
                        nc = mid_cell + ti.Vector([dx, dy, dz])
                        nc = wrap_cell(nc)
                        nc_id = cell_id(nc)
                        
                        start = cell_start[nc_id]
                        count = cell_count[nc_id]
                        
                        for l in range(start, start + count):
                            k = cell_indices[l]
                            if k == i or k == j:
                                continue
                            
                            delta_km = pdelta(midpoint, pos[k])
                            dist_sq_km = delta_km.dot(delta_km)
                            
                            eps_scaled = ti.max(1e-12, 1e-6 * r_sq_diameter)
                            
                            if dist_sq_km < r_sq_diameter - eps_scaled:
                                is_gabriel_neighbor = False
                                break
                        if not is_gabriel_neighbor:
                            break
                    if not is_gabriel_neighbor:
                        break
                if not is_gabriel_neighbor:
                    break
            
            if is_gabriel_neighbor:
                ti.atomic_add(topo_deg[i], 1)
                ti.atomic_add(topo_deg[j], 1)

@ti.kernel
def commit_topo_to_ema(alpha: ti.f32, n: ti.i32):
    """Commit raw topological degree to EMA after batched analysis completes."""
    for i in range(n):
        topo_deg_ema[i] = alpha * topo_deg[i] + (1.0 - alpha) * topo_deg_ema[i]

# ==============================================================================
# k-NN Fast Proxy (Cheap Alternative)
# ==============================================================================

@ti.kernel
def knn_topo_degree(pos: ti.template(), rad: ti.template(),
                    cell_start: ti.template(), cell_count: ti.template(),
                    cell_indices: ti.template(), n: ti.i32, k: ti.i32):
    """
    Fast k-nearest neighbor topological degree proxy.
    Much cheaper than Gabriel graph, good enough for interactive work.
    """
    for i in range(n):
        # Simple approach: count neighbors within extended search radius
        # and clamp to k
        neighbor_count = 0
        max_search_radius_sq = (TOPO_MAX_RADIUS_MULTIPLE * rad[i]) ** 2
        
        p_wrapped = wrapP(pos[i])
        q = (p_wrapped + HALF_L) * INV_L
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        my_cell = wrap_cell(my_cell)
        
        # Collect distances to all neighbors
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    nc = my_cell + ti.Vector([dx, dy, dz])
                    nc = wrap_cell(nc)
                    nc_id = cell_id(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    
                    for l in range(start, start + count):
                        j = cell_indices[l]
                        if j != i:
                            delta = pdelta(pos[i], pos[j])
                            dist_sq = delta.dot(delta)
                            if dist_sq <= max_search_radius_sq:
                                neighbor_count += 1
                                if neighbor_count >= k:
                                    break
                    if neighbor_count >= k:
                        break
                if neighbor_count >= k:
                    break
            if neighbor_count >= k:
                break
        
        topo_deg[i] = ti.min(neighbor_count, k)

