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
    
    Args:
        c: Cell indices (can be negative or >= GRID_RES)
    
    Returns:
        Wrapped cell indices in [0, GRID_RES)³
    """
    if ti.static(PBC_ENABLED):
        # Double-modulo handles negatives: (-1 % 10 + 10) % 10 = 9
        return (c % GRID_RES + GRID_RES) % GRID_RES
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
                        deg: ti.template(), color: ti.template(), n: ti.i32):
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
        # Zero degree
        deg[i] = 0
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
        q = (p_wrapped + HALF_L) * INV_L
        
        # Cell coordinate (3D index)
        c = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
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
        q = (p_wrapped + HALF_L) * INV_L
        
        # Cell coordinate (3D index)
        c = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        # Wrap cell indices (handles edge cases)
        c = wrap_cell(c)
        
        # Linear index (centralized function)
        c_id = cell_id(c)
        
        # Atomic fetch-add: get write position and increment pointer
        write_pos = ti.atomic_add(cell_write[c_id], 1)
        
        # Write particle ID
        cell_indices[write_pos] = i


# ==============================================================================
# Kernel 6: Count neighbors (27-cell stencil)
# ==============================================================================

@ti.kernel
def count_neighbors(pos: ti.template(), rad: ti.template(), deg: ti.template(),
                    cell_start: ti.template(), cell_count: ti.template(),
                    cell_indices: ti.template(), n: ti.i32):
    """
    Count neighbors within (1 + CONTACT_TOL) * (r_i + r_j).
    This is "near-contact" (not exact touching), matching PBD gap semantics.
    
    For each particle i:
      1. Find my cell
      2. Check 27 neighboring cells (3x3x3 stencil including self)
      3. For each neighbor j in those cells:
         - Compute minimum-image distance
         - Check if within contact threshold
         - Increment deg[i] if yes
    
    Periodic boundaries are handled via modulo wrapping and periodic_delta.
    """
    for i in range(n):
        deg[i] = 0  # Reset degree
        
        # My cell coordinate (PBC-aware, same as scatter_particles)
        p_wrapped = wrapP(pos[i])
        q = (p_wrapped + HALF_L) * INV_L  # Normalize to [0,1)³
        my_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
        
        # Check 27 neighboring cells (-1, 0, +1 in each dimension)
        for dx in ti.static([-1, 0, 1]):
            for dy in ti.static([-1, 0, 1]):
                for dz in ti.static([-1, 0, 1]):
                    # Neighbor cell coordinate
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
                        
                        if i != j:
                            # Minimum-image distance (PBC-aware)
                            delta = pdelta(pos[i], pos[j])
                            dist_sq = delta.dot(delta)
                            
                            # Contact threshold: touching + tolerance (matches PBD gap)
                            touch_thresh = (1.0 + CONTACT_TOL) * (rad[i] + rad[j])
                            
                            if dist_sq <= touch_thresh * touch_thresh:
                                deg[i] += 1


# ==============================================================================
# Kernel 7: Update colors (degree → RGB)
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

