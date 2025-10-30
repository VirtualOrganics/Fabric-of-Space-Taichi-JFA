"""
Spatial grid for neighbor search in Fabric of Space.

This module provides efficient neighbor detection using a spatial hash grid:
- Fixed cell size = 2 * R_MAX (conservative for 27-stencil)
- Periodic boundaries (minimum-image convention)
- Rebuild every frame to handle dynamic radii

Grid pipeline:
1. clear_grid: Zero counts
2. count_particles_per_cell: Histogram particles → cells (atomic)
3. prefix_sum: Exclusive scan for cell offsets
4. copy_cell_pointers: Duplicate offsets for scatter
5. scatter_particles: Write particle IDs to sorted array (atomic)
6. count_neighbors: 27-cell stencil with periodic wrap
7. update_colors: Map degree → RGB for rendering

All kernels use f32 for positions/radii, i32 for counts/indices.
"""

import taichi as ti
import taichi.math as tm
from config import (GRID_RES, CELL_SIZE, DOMAIN_SIZE, CONTACT_TOL, EPS,
                    PBC_ENABLED, HALF_L, INV_L)

# ==============================================================================
# PBC Helper Functions (Periodic Boundary Conditions)
# ==============================================================================

@ti.func
def wrapP(p: ti.math.vec3) -> ti.math.vec3:
    """
    Wrap position into primary cell using centered floor.
    
    Maps any point to [-L/2, L/2)³ where L = DOMAIN_SIZE.
    Uses floor(p/L + 0.5) to avoid floating-point tie issues with round().
    
    Args:
        p: Position vector (can be outside domain)
    
    Returns:
        Wrapped position in primary cell [-L/2, L/2)³
    """
    if ti.static(PBC_ENABLED):
        # Centered modulo: avoids ULP/banker's rounding issues at boundaries
        return p - DOMAIN_SIZE * ti.floor(p * INV_L + 0.5)
    else:
        return p  # No-op if PBC disabled


@ti.func
def pdelta(a: ti.math.vec3, b: ti.math.vec3) -> ti.math.vec3:
    """
    Compute minimum-image displacement from a to b.
    
    This is the ONLY way to compute particle-particle vectors in PBC.
    Ensures shortest distance across periodic boundaries.
    
    Args:
        a, b: Particle positions
    
    Returns:
        Shortest vector from a to b (wraps if PBC enabled)
    """
    if ti.static(PBC_ENABLED):
        return wrapP(b - a)
    else:
        return b - a


@ti.func
def wrap_cell(c: ti.math.ivec3) -> ti.math.ivec3:
    """
    Wrap cell indices into valid range [0, GRID_RES).
    
    Uses double-modulo to handle negative indices correctly:
      (c % N + N) % N
    
    CRITICAL: Use component-wise modulo for Taichi GPU compatibility.
    Citation [20] from Perplexity: Python's % on vectors doesn't work correctly in Taichi.
    
    Args:
        c: Cell indices (can be negative or >= GRID_RES)
    
    Returns:
        Wrapped cell indices in [0, GRID_RES)³
    """
    if ti.static(PBC_ENABLED):
        # Component-wise double-modulo for PBC wrapping
        # Handle each component separately to ensure correct behavior on GPU
        return ti.math.ivec3([
            (c[0] % GRID_RES + GRID_RES) % GRID_RES,
            (c[1] % GRID_RES + GRID_RES) % GRID_RES,
            (c[2] % GRID_RES + GRID_RES) % GRID_RES
        ])
    else:
        # Clamp to valid range for bounded domain
        return ti.math.ivec3([
            ti.max(0, ti.min(GRID_RES - 1, c[0])),
            ti.max(0, ti.min(GRID_RES - 1, c[1])),
            ti.max(0, ti.min(GRID_RES - 1, c[2]))
        ])


@ti.func
def power_dist2(x: ti.math.vec3, p: ti.math.vec3, R: ti.f32) -> ti.f32:
    """
    Power distance (squared) for future Laguerre/power diagram use.
    
    Power distance = ||x - p||² - R²
    Uses PBC-aware displacement via pdelta.
    
    Args:
        x: Query point
        p: Particle center
        R: Particle radius
    
    Returns:
        Squared power distance (can be negative if x inside sphere)
    """
    d = pdelta(x, p)
    return d.dot(d) - R * R


@ti.func
def cell_id(c: ti.math.ivec3) -> ti.i32:
    """
    Convert 3D cell indices to linear index.
    
    Centralizes the formula for easy generalization to non-cubic grids.
    
    Args:
        c: Cell indices [cx, cy, cz]
    
    Returns:
        Linear cell ID in [0, GRID_RES³)
    """
    return c[0] * GRID_RES * GRID_RES + c[1] * GRID_RES + c[2]


# ==============================================================================
# Kernel 1: Clear grid
# ==============================================================================

@ti.kernel
def clear_grid(cell_count: ti.template()):
    """
    Reset all cell counts to zero.
    
    Must be called at the start of each frame before rebuilding grid.
    """
    for i in range(GRID_RES**3):
        cell_count[i] = 0


# ==============================================================================
# Kernel 2: Clear all particle data
# ==============================================================================

@ti.kernel
def clear_all_particles(pos: ti.template(), rad: ti.template(), vel: ti.template(), 
                        color: ti.template(), n: ti.i32):
    """
    Clear all particle data by moving particles far outside the domain.
    
    This is called before reseeding to ensure stale data from previous
    runs (with potentially more particles) doesn't show up.
    
    Moves particles to position (-1000, -1000, -1000) which is:
    - Outside domain [0, DOMAIN_SIZE)³
    - Outside periodic wrap range
    - Won't be rendered or counted
    """
    for i in range(n):
        # Move particle far outside domain
        pos[i] = ti.Vector([-1000.0, -1000.0, -1000.0])
        # Reset radius to minimum
        rad[i] = 0.001
        # Zero velocity
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        # Set color to black (invisible)
        color[i] = ti.Vector([0.0, 0.0, 0.0])


# ==============================================================================
# Kernel 3: Count particles per cell (histogram)
# ==============================================================================

@ti.kernel
def count_particles_per_cell(pos: ti.template(), cell_count: ti.template(), n: ti.i32):
    """
    Count how many particles fall into each cell.
    
    For each particle:
      1. Wrap position (PBC-aware)
      2. Compute cell coordinate using precomputed constants
      3. Convert to linear index
      4. Atomic increment cell count
    
    Uses atomic add to handle collisions (multiple particles → same cell).
    """
    for i in range(n):
        # Wrap position (PBC-aware)
        p_wrapped = wrapP(pos[i])
        
        # Normalize to [0,1)³ using precomputed INV_L
        # Add tiny epsilon to prevent boundary jitter (particles on cell edges)
        eps_hash = 1e-7
        q = (p_wrapped + HALF_L + eps_hash) * INV_L
        
        # Cell coordinate (CORRECT Taichi indexing)
        # Use ti.cast(ti.floor(...), ti.i32) per Perplexity citation [1]
        c = ti.cast(ti.floor(q * GRID_RES), ti.i32)
        
        # Wrap cell indices (handles edge cases)
        c = wrap_cell(c)
        
        # Linear index (centralized function)
        c_id = cell_id(c)
        
        # Atomic increment (thread-safe)
        ti.atomic_add(cell_count[c_id], 1)


# ==============================================================================
# Kernel 3: Prefix sum (exclusive scan)
# ==============================================================================

@ti.kernel
def prefix_sum(cell_count: ti.template(), cell_start: ti.template()):
    """
    Compute exclusive prefix sum over cell counts.
    
    cell_start[i] = sum(cell_count[0..i-1])
    
    This gives the starting index for each cell's particles in the sorted array.
    Serial implementation (GRID_RES³ ≈ 2197 for 13³, negligible cost).
    
    NOTE: Taichi guarantees deterministic execution order for serial loops.
    """
    cell_start[0] = 0
    for i in range(1, GRID_RES**3):
        cell_start[i] = cell_start[i - 1] + cell_count[i - 1]


# ==============================================================================
# Kernel 4: Copy cell pointers (for scatter)
# ==============================================================================

@ti.kernel
def copy_cell_pointers(cell_start: ti.template(), cell_write: ti.template()):
    """
    Copy cell_start to cell_write.
    
    During scatter, we use atomic fetch-add on cell_write to get write positions.
    This mutates cell_write, but we need cell_start intact for iteration.
    
    Separation of read (cell_start) and write (cell_write) pointers avoids bugs.
    """
    for i in range(GRID_RES**3):
        cell_write[i] = cell_start[i]


# ==============================================================================
# Kernel 5: Scatter particles (build sorted index array)
# ==============================================================================

@ti.kernel
def scatter_particles(pos: ti.template(), cell_write: ti.template(),
                      cell_indices: ti.template(), n: ti.i32):
    """
    Write particle IDs to sorted array.
    
    For each particle:
      1. Wrap position (PBC) and store back
      2. Compute cell ID using precomputed constants
      3. Atomic fetch-add on cell_write[cell] to get write position
      4. Write particle ID to cell_indices[write_pos]
    
    After this, cell_indices[cell_start[c]:cell_start[c+1]] contains all particles in cell c.
    
    NOTE: cell_write is mutated. cell_start remains intact for iteration.
    NOTE: Positions are always wrapped after this call (invariant maintained).
    """
    for i in range(n):
        # Wrap position first (PBC-aware, use precomputed constants)
        p_wrapped = wrapP(pos[i])
        pos[i] = p_wrapped  # Store back (keep positions always wrapped)
        
        # Normalize to [0,1)³ using precomputed INV_L
        # Add tiny epsilon to prevent boundary jitter (particles on cell edges)
        eps_hash = 1e-7
        q = (p_wrapped + HALF_L + eps_hash) * INV_L
        
        # Cell coordinate (CORRECT Taichi indexing)
        # Use ti.cast(ti.floor(...), ti.i32) per Perplexity citation [1]
        c = ti.cast(ti.floor(q * GRID_RES), ti.i32)
        
        # Wrap cell indices (handles edge cases)
        c = wrap_cell(c)
        
        # Linear index (centralized function)
        c_id = cell_id(c)
        
        # Atomic fetch-add: get write position and increment pointer
        write_pos = ti.atomic_add(cell_write[c_id], 1)
        
        # Write particle ID
        cell_indices[write_pos] = i


# ==============================================================================
# Kernel 6: Update colors (FSC → RGB)
# ==============================================================================

@ti.kernel
def update_colors(deg: ti.template(), color: ti.template(), n: ti.i32, deg_low: ti.i32, deg_high: ti.i32):
    """
    Map degree to color for visualization with runtime-adjustable thresholds.
    
    Color scheme:
      deg < deg_low:            Red (below band, will grow)
      deg_low <= deg <= deg_high: Green (in target band, stable)
      deg > deg_high:            Blue (above band, will shrink)
    
    Uses simple 3-color scheme for clear visual feedback.
    """
    for i in range(n):
        d = deg[i]
        
        # Threshold-based coloring
        if d < deg_low:
            # Red: below target band (needs to grow)
            color[i] = ti.Vector([1.0, 0.0, 0.0])
        elif d <= deg_high:
            # Green: in target band (stable)
            color[i] = ti.Vector([0.0, 1.0, 0.0])
        else:
            # Blue: above target band (needs to shrink)
            color[i] = ti.Vector([0.2, 0.5, 1.0])


# ==============================================================================
# Validation: Grid accuracy check (brute-force comparison)
# ==============================================================================

@ti.kernel
def validate_grid_neighbors(pos: ti.template(), rad: ti.template(), 
                            deg_grid: ti.template(), deg_brute: ti.template(),
                            cell_start: ti.template(), cell_count: ti.template(),
                            cell_indices: ti.template(), 
                            sample_indices: ti.template(), n_samples: ti.i32, active_n: ti.i32,
                            cross_pbc_miss_count: ti.template(),
                            miss_margin_sum: ti.template(), miss_margin_count: ti.template()):
    """
    Validate grid-based neighbor counting against brute-force for a sample of particles.
    
    For each particle in sample_indices:
      - deg_grid[i]: already computed via count_neighbors (uses 27-cell stencil)
      - deg_brute[i]: computed here via brute force (checks ONLY active_n particles)
      - Tracks cross-PBC misses and distance-to-cutoff for diagnostic purposes
    
    This is O(n_samples × active_n), so only run on small samples (~200 particles).
    
    CRITICAL: Must iterate only over active_n particles, not the entire allocated field size!
    """
    cross_pbc_miss_count[None] = 0
    miss_margin_sum[None] = 0.0
    miss_margin_count[None] = 0
    
    for idx in range(n_samples):
        i = sample_indices[idx]
        
        grid_deg = deg_grid[i]
        
        # First pass: count brute-force neighbors and collect margins
        brute_count = 0
        neighbor_margins = ti.Vector([0.0] * 50)  # Store margins for up to 50 neighbors per particle
        neighbor_count_local = 0
        
        for j in range(active_n):
            if i != j and neighbor_count_local < 50:
                # Minimum-image distance (PBC-aware)
                delta = pdelta(pos[i], pos[j])
                dist_sq = delta.dot(delta)
                
                # Contact threshold: touching + tolerance (MUST match grid predicate exactly)
                r_sum = rad[i] + rad[j]
                touch_thresh = (1.0 + CONTACT_TOL) * r_sum
                touch_thresh_sq = touch_thresh * touch_thresh
                
                if dist_sq <= touch_thresh_sq:
                    brute_count += 1
                    
                    # Store margin for this neighbor (only compute sqrt for diagnostic)
                    dist = ti.sqrt(dist_sq)
                    margin_frac = (touch_thresh - dist) / r_sum
                    if neighbor_count_local < 50:
                        neighbor_margins[neighbor_count_local] = margin_frac
                        neighbor_count_local += 1
                    
                    # Check if this is a cross-PBC pair
                    is_cross_pbc = (ti.abs(delta.x) > DOMAIN_SIZE * 0.25 or 
                                   ti.abs(delta.y) > DOMAIN_SIZE * 0.25 or 
                                   ti.abs(delta.z) > DOMAIN_SIZE * 0.25)
                    
                    if is_cross_pbc:
                        ti.atomic_add(cross_pbc_miss_count[None], 1)
        
        deg_brute[i] = brute_count
        
        # If this particle has misses, track the margins of the last few neighbors
        # (This is an approximation - we're assuming the grid misses the marginal neighbors)
        miss_count_local = brute_count - grid_deg
        if miss_count_local > 0 and neighbor_count_local > 0:
            # Track margins of the last 'miss_count' neighbors (likely the marginal ones)
            for k in range(ti.max(0, neighbor_count_local - miss_count_local), neighbor_count_local):
                ti.atomic_add(miss_margin_sum[None], neighbor_margins[k])
                ti.atomic_add(miss_margin_count[None], 1)


# ==============================================================================
# Diagnostic: Stencil iteration counter
# ==============================================================================

@ti.kernel
def count_stencil_iterations(reach: ti.i32) -> ti.i32:
    """
    Diagnostic kernel: Count how many cells are actually checked in the stencil loops.
    
    Returns the total number of (dx, dy, dz) iterations executed.
    Expected: (2*reach+1)³
    
    If this returns 27 when reach=3, the loops are NOT using dynamic reach!
    """
    total = 0
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                total += 1
    return total


def validate_neighbor_counts(pos, rad, deg_grid, cell_start, cell_count, cell_indices, active_n):
    """
    Python wrapper: Validate grid-based neighbor counting.
    
    Samples ~1000 random particles, compares grid vs brute-force neighbor counts.
    Prints diagnostic info to console (no performance impact when not called).
    """
    import numpy as np
    
    # Sample size (balance accuracy vs performance)
    # Increased to 1000 to smooth out statistical noise in miss rate
    n_samples = min(1000, active_n)
    
    # Pick random particles to validate
    sample_indices_np = np.random.choice(active_n, size=n_samples, replace=False).astype(np.int32)
    sample_indices = ti.field(dtype=ti.i32, shape=n_samples)
    sample_indices.from_numpy(sample_indices_np)
    
    # Allocate brute-force degree array and diagnostic counters
    deg_brute = ti.field(dtype=ti.i32, shape=pos.shape[0])
    cross_pbc_miss_count = ti.field(dtype=ti.i32, shape=())
    miss_margin_sum = ti.field(dtype=ti.f32, shape=())
    miss_margin_count = ti.field(dtype=ti.i32, shape=())
    
    # Run brute-force validation kernel
    # CRITICAL: Pass active_n so brute-force only checks active particles, not entire field
    validate_grid_neighbors(pos, rad, deg_grid, deg_brute, 
                           cell_start, cell_count, cell_indices,
                           sample_indices, n_samples, active_n,
                           cross_pbc_miss_count, miss_margin_sum, miss_margin_count)
    
    # Extract results
    grid_deg_np = deg_grid.to_numpy()[sample_indices_np]
    brute_deg_np = deg_brute.to_numpy()[sample_indices_np]
    cross_pbc_total = cross_pbc_miss_count[None]
    margin_sum = miss_margin_sum[None]
    margin_count = miss_margin_count[None]
    
    # Compute statistics
    diff = brute_deg_np - grid_deg_np  # Positive = grid missed neighbors
    total_neighbors = brute_deg_np.sum()
    missed_neighbors = np.maximum(diff, 0).sum()  # Only count misses, not extra
    miss_rate_pct = 100.0 * missed_neighbors / max(1, total_neighbors)
    
    mean_grid = grid_deg_np.mean()
    mean_brute = brute_deg_np.mean()
    mean_err = mean_brute - mean_grid
    
    max_diff = np.abs(diff).max()
    
    # Compute cross-PBC percentage and average miss margin
    cross_pbc_pct = 100.0 * cross_pbc_total / max(1, total_neighbors)
    avg_miss_margin_pct = 100.0 * margin_sum / max(1, margin_count) if margin_count > 0 else 0.0
    
    # Print validation report
    print(f"[Grid Check] sampled {n_samples}/{active_n} particles")
    print(f"             grid_deg: μ={mean_grid:.2f} [{grid_deg_np.min()},{grid_deg_np.max()}]")
    print(f"             brute_deg: μ={mean_brute:.2f} [{brute_deg_np.min()},{brute_deg_np.max()}]")
    print(f"             missed: {missed_neighbors}/{total_neighbors} neighbors ({miss_rate_pct:.1f}%)")
    print(f"             cross-PBC: {cross_pbc_total}/{total_neighbors} ({cross_pbc_pct:.1f}%)")
    if margin_count > 0:
        print(f"             miss margin: avg={avg_miss_margin_pct:.2f}% of r_sum (n={margin_count})")
    print(f"             max |error|: {max_diff} | mean_err: {mean_err:+.2f}")
    
    # Pass/fail threshold
    if miss_rate_pct < 1.0:
        print(f"             ✓ PASS (miss_rate < 1.0%)")
    else:
        print(f"             ⚠️  FAIL - Grid too coarse or stencil incomplete")
        if cross_pbc_pct > 50.0:
            print(f"             💡 Most neighbors are cross-PBC → likely PBC wrap mismatch")
        elif avg_miss_margin_pct < 2.0 and margin_count > 0:
            print(f"             💡 Misses clustered near threshold ({avg_miss_margin_pct:.2f}%) → floating-point or <= vs < mismatch")
    
    return miss_rate_pct

