# Fabric of Space - Custom Taichi Grid (Option C)

**Goal:** Minimal, clean neighbor detection system that handles per-loop radius changes.

---

## Architecture Overview

**Core Principle:** 
- Conservative cell size (CELL_SIZE = 2 × R_MAX) handles any radius within [R_MIN, R_MAX]
- Grid rebuilt every frame = no stale data, ever
- Narrow-phase uses CURRENT radii from field
- No caching, no Verlet lists, no assumptions

**Expected Size:** ~200-300 lines total across 3 files

---

## Data Structures

### Taichi Fields (GPU)
```python
# Particle state
pos: ti.Vector.field(3, dtype=ti.f32, shape=N)     # Positions [0, DOMAIN_SIZE)
rad: ti.field(dtype=ti.f32, shape=N)                # Radii [R_MIN, R_MAX]
deg: ti.field(dtype=ti.i32, shape=N)                # Degree counts
color: ti.Vector.field(3, dtype=ti.f32, shape=N)   # RGB color for rendering (degree-based)

# Spatial grid (rebuilt every frame)
cell_count: ti.field(dtype=ti.i32, shape=GRID_RES**3)    # Particles per cell
cell_start: ti.field(dtype=ti.i32, shape=GRID_RES**3)    # Prefix sum (exclusive scan) - READ ONLY after scan
cell_write: ti.field(dtype=ti.i32, shape=GRID_RES**3)    # WRITE pointer for scatter (copy of cell_start)
cell_indices: ti.field(dtype=ti.i32, shape=N)            # Sorted particle IDs
```

### Constants (Python scope)
```python
import math

N = 5000                    # Particle count
DOMAIN_SIZE = 0.15          # Cubic domain [0, 0.15)³
R_MIN = 0.0015              # Minimum radius
R_MAX = 0.0060              # Maximum radius (HARD CLAMP - never exceed)
CELL_SIZE = 2 * R_MAX       # = 0.012 (conservative for 27-stencil)

# Grid resolution: ceil ensures last cell covers box edge, min 3 for valid 27-stencil
GRID_RES = max(3, int(math.ceil(DOMAIN_SIZE / CELL_SIZE)))  # = 13 cells per axis

CONTACT_TOL = 0.02          # Contact tolerance: 2% beyond touching (matches PBD gap)
EPS = 1e-8                  # Small epsilon for numerical safety
```

**Grid indexing formula:**
- Cell coordinate: `c = ti.Vector([int(pos[i][d] / CELL_SIZE) for d in range(3)])`
- Linear index: `id = c[0] + c[1]*GRID_RES + c[2]*GRID_RES**2`
- Periodic wrap: `c_wrapped = (c % GRID_RES + GRID_RES) % GRID_RES`

**Grid coverage guarantee:**
- `GRID_RES = max(3, ceil(L / cell_size))` ensures:
  - Last cell covers domain edge (no particles fall outside grid)
  - Minimum 3x3x3 grid (27-stencil always valid, no edge cases)

**Key Insight:** CELL_SIZE = 2 × R_MAX means any pair of particles in adjacent cells (27-stencil) can potentially touch. No particles missed.

---

## Kernel Signatures (7 kernels total)

### Helper: `periodic_delta(p1, p2, domain_size)`
```python
@ti.func
def periodic_delta(p1: ti.math.vec3, p2: ti.math.vec3, L: ti.f32) -> ti.math.vec3:
    """
    Minimum-image convention for periodic boundaries.
    Wraps each axis to [-L/2, L/2].
    """
    delta = p2 - p1
    for d in ti.static(range(3)):
        if delta[d] > L * 0.5:
            delta[d] -= L
        elif delta[d] < -L * 0.5:
            delta[d] += L
    return delta
```

### 1. `clear_grid()`
```python
@ti.kernel
def clear_grid(cell_count: ti.template()):
    """Zero out cell_count array."""
    for i in cell_count:
        cell_count[i] = 0
```

### 2. `count_particles_per_cell(pos, cell_count)`
```python
@ti.kernel
def count_particles_per_cell(pos: ti.template(), cell_count: ti.template()):
    """
    For each particle:
      - Compute cell coordinate: c = int(pos[i] / CELL_SIZE)
      - Linear index: id = c[0] + c[1]*GRID_RES + c[2]*GRID_RES**2
      - Atomic add to cell_count[id]
    """
```

### 3. `prefix_sum(cell_count, cell_start)`
```python
@ti.kernel
def prefix_sum(cell_count: ti.template(), cell_start: ti.template()):
    """
    Exclusive scan: cell_start[i] = sum(cell_count[0..i-1])
    Serial implementation (GRID_RES**3 ≈ 2197 for 13^3, negligible cost).
    
    NOTE: Taichi guarantees deterministic execution order for serial loops.
    
    cell_start[0] = 0
    for i in range(1, GRID_RES**3):
        cell_start[i] = cell_start[i-1] + cell_count[i-1]
    """
```

### 4. `copy_cell_pointers(cell_start, cell_write)`
```python
@ti.kernel
def copy_cell_pointers(cell_start: ti.template(), cell_write: ti.template()):
    """
    Copy cell_start → cell_write before scatter.
    CRITICAL: Scatter will mutate cell_write, NOT cell_start.
    """
    for i in cell_start:
        cell_write[i] = cell_start[i]
```

### 5. `scatter_particles(pos, cell_write, cell_indices)`
```python
@ti.kernel
def scatter_particles(pos: ti.template(), cell_write: ti.template(), 
                      cell_indices: ti.template()):
    """
    For each particle:
      - Compute cell (linear index)
      - idx = ti.atomic_add(cell_write[cell], 1)  # Fetch-and-increment
      - cell_indices[idx] = particle_id
    
    NOTE: cell_write is mutated. cell_start remains intact for iteration.
    """
```

### 6. `count_neighbors(pos, rad, deg, cell_start, cell_count, cell_indices)`
```python
@ti.kernel
def count_neighbors(pos: ti.template(), rad: ti.template(), deg: ti.template(),
                    cell_start: ti.template(), cell_count: ti.template(),
                    cell_indices: ti.template()):
    """
    Count neighbors within (1 + CONTACT_TOL) * (r_i + r_j).
    This is "near-contact" (not exact touching), matching PBD gap semantics.
    
    For each particle i:
      deg[i] = 0
      my_cell = cell_of(pos[i])  # 3D coord
      
      # Check 27 neighboring cells (3x3x3 stencil)
      for offset in [-1,0,1]³:
          nc = my_cell + offset
          
          # Periodic wrapping per axis
          for d in ti.static(range(3)):
              nc[d] = (nc[d] % GRID_RES + GRID_RES) % GRID_RES
          
          nc_id = nc[0] + nc[1]*GRID_RES + nc[2]*GRID_RES**2
          
          # Iterate particles in neighbor cell
          start = cell_start[nc_id]
          count = cell_count[nc_id]
          for k in range(start, start + count):
              j = cell_indices[k]
              if i != j:
                  # Minimum-image distance
                  delta = periodic_delta(pos[i], pos[j], DOMAIN_SIZE)
                  dist_sq = delta.dot(delta)
                  
                  # Contact threshold: touching + tolerance (matches PBD gap)
                  touch_thresh = (1.0 + CONTACT_TOL) * (rad[i] + rad[j])
                  
                  if dist_sq <= touch_thresh * touch_thresh:
                      deg[i] += 1
    """
```

### 7. `update_colors(deg, color)`
```python
@ti.kernel
def update_colors(deg: ti.template(), color: ti.template()):
    """
    Map degree to color (simple viridis-like gradient).
    Taichi UI requires explicit RGB field.
    """
    for i in deg:
        # Normalize degree to [0, 1] (assume max degree ≈ 30)
        t = ti.min(deg[i] / 30.0, 1.0)
        # Simple gradient: blue → cyan → green → yellow → red
        color[i] = ti.Vector([t, 1.0 - ti.abs(t - 0.5)*2.0, 1.0 - t])
```

**Note:** All kernels use current frame radii. No caching. No Verlet lists.

---

## Main Loop Pseudocode

```python
# Initialization (once)
ti.init(arch=ti.metal)
pos, rad, deg, color = allocate_fields(N)  # All f32 for speed
cell_count, cell_start, cell_write, cell_indices = allocate_grid(GRID_RES)
initialize_particles(pos, rad)  # Random positions, radii ∈ [R_MIN, R_MAX]

# WARMUP: Initial de-overlap (prevents "sticky blob" at start)
for _ in range(4):
    rebuild_grid()  # Defined below
    project_overlaps(pos, rad, cell_start, cell_count, cell_indices, 
                     GAP_FRACTION=0.02, DOMAIN_SIZE)

# GUI setup
window = ti.ui.Window("Fabric of Space", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.make_camera_3d()
camera = ti.ui.Camera()
camera.position(0.075, 0.075, 0.3)  # Center on domain
camera.lookat(0.075, 0.075, 0.075)

# Helper: Grid rebuild (used multiple times)
def rebuild_grid():
    clear_grid(cell_count)
    count_particles_per_cell(pos, cell_count)
    prefix_sum(cell_count, cell_start)
    copy_cell_pointers(cell_start, cell_write)  # CRITICAL: copy before scatter
    scatter_particles(pos, cell_write, cell_indices)

# Main loop
frame = 0
while window.running:
    # 1. REBUILD GRID (every frame, using current radii)
    rebuild_grid()
    
    # 2. COUNT NEIGHBORS (with CONTACT_TOL for gap-aware detection)
    count_neighbors(pos, rad, deg, cell_start, cell_count, cell_indices)
    
    # 3. ADAPT RADII (5% rule, per-particle)
    update_radii(rad, deg, DEG_LOW=5, DEG_HIGH=6, GAIN_GROW=0.05, GAIN_SHRINK=0.05)
    # Inside update_radii: clamp to [R_MIN, R_MAX] HARD
    
    # 4. PBD OVERLAP RESOLUTION (4-8 passes, using min-image)
    for _ in range(PBD_PASSES):
        project_overlaps(pos, rad, cell_start, cell_count, cell_indices, 
                         GAP_FRACTION=0.02, DOMAIN_SIZE)
    
    # 5. UPDATE COLORS (degree → RGB)
    update_colors(deg, color)
    
    # 6. RENDER (Taichi UI requires explicit radius and color fields)
    scene.particles(pos, radius=rad * VIS_SCALE, color=color)
    camera.track_user_inputs(window)
    scene.set_camera(camera)
    canvas.scene(scene)
    window.show()
    
    frame += 1
```

**Loop time budget (5k particles @ 20 FPS = 50ms):**
- Grid rebuild: ~10ms
- Neighbor count: ~15ms
- Radius update: ~2ms
- PBD (4 passes): ~20ms
- Render: ~3ms

**Key correctness points:**
1. `cell_write` is mutated during scatter; `cell_start` stays intact for iteration.
2. `CONTACT_TOL` in neighbor count matches `GAP_FRACTION` in PBD (both ~2%).
3. `periodic_delta` used consistently in both neighbor count and PBD.
4. All fields are `f32` (no mixing with `f64`).
5. Radii HARD CLAMPED to `[R_MIN, R_MAX]` every frame.

---

## File Structure

```
Cursor_FoS-Custom-Grid/
├── BLUEPRINT.md              (this file)
├── README.md                 (quick start)
├── requirements.txt          (taichi, numpy)
├── config.py                 (constants: N, R_MIN, R_MAX, CONTACT_TOL, etc.)
├── grid.py                   (7 kernels: clear, count, scan, copy, scatter, neighbors, colors)
├── dynamics.py               (radius update, PBD)
├── run.py                    (main loop + GUI + warmup)
└── tests/
    ├── test_grid_accuracy.py         (brute-force vs grid - exact match)
    ├── test_radius_individuality.py  (3 isolated + 3 crowded → per-particle growth/shrinkage)
    ├── test_pbd_separation.py        (overlaps decrease)
    └── test_periodicity.py           (boundary wrapping)
```

**Total LOC estimate:**
- `config.py`: 25 lines (added CONTACT_TOL, EPS, comments)
- `grid.py`: 150 lines (7 kernels + periodic_delta helper)
- `dynamics.py`: 70 lines (radius update + PBD with min-image)
- `run.py`: 90 lines (main loop + GUI + warmup)
- **Total: ~335 lines** (still minimal)

---

## Configuration Parameters

```python
# config.py
import math

N = 5000                    # Particle count
DOMAIN_SIZE = 0.15          # Cubic domain side length
R_MIN = 0.0015              # Minimum radius
R_MAX = 0.0060              # Maximum radius (HARD CLAMP - never exceed)
CELL_SIZE = 2 * R_MAX       # Conservative cell size (0.012) for 27-stencil
GRID_RES = max(3, int(math.ceil(DOMAIN_SIZE / CELL_SIZE)))  # = 13 cells per axis (ceil + min 3)
CONTACT_TOL = 0.02          # Contact tolerance: 2% beyond touching (matches PBD gap)
EPS = 1e-8                  # Small epsilon for numerical safety

# Degree adaptation
DEG_LOW = 5                 # Below this: grow 5%
DEG_HIGH = 6                # Above this: shrink 5%
GAIN_GROW = 0.05            # Growth rate (5% per step)
GAIN_SHRINK = 0.05          # Shrink rate (5% per step)

# PBD separation
PBD_PASSES = 4              # Overlap projection iterations (4-8 recommended)
GAP_FRACTION = 0.02         # Target gap: 2% of sum of radii (matches CONTACT_TOL)
MAX_DISPLACEMENT_FRAC = 0.2 # Cap per-particle displacement to 20% of overlap (split 50/50, per-axis clamp)

# Rendering
FPS_TARGET = 20
VIS_SCALE = 1.0             # Radius multiplier for rendering (adjust if particles too small/large)

# Degree semantics note:
# deg[i] = count of neighbors within (1 + CONTACT_TOL) * (r_i + r_j)
# This is "near-contact", not exact touching. Particles with deg=0 will grow.
```

---

## Acceptance Tests (10 minutes total)

### Test 1: Grid Accuracy (brute-force comparison)
```python
# Generate 1000 random particles (fixed seed)
# Count neighbors via:
#   A) Brute-force O(N²) with min-image and CONTACT_TOL
#   B) Grid method with same parameters
# Assert: deg_A[i] == deg_B[i] for all i
# (No ±1 slack needed if min-image and tolerance are consistent)
```

**Pass criteria:** 100% exact match

---

### Test 2: Radius Individuality (no global averaging)
```python
# Create 6 particles:
#   - 3 isolated (far apart, deg = 0)
#   - 3 crowded (tight cluster, deg > 10)
# Run 10 steps
# Assert:
#   - Isolated particles grow 5% per step: r_final ≈ r_initial × 1.05^10 ≈ 1.63x
#   - Crowded particles shrink 5% per step: r_final ≈ r_initial × 0.95^10 ≈ 0.60x
# This ensures no global degree averaging bug (each particle acts on its own degree)
```

**Pass criteria:** Growth/shrinkage within 1% of theoretical, per particle

---

### Test 3: Radius Growth (isolated particles)
```python
# Create 10 particles far apart (no neighbors, deg = 0)
# Run 10 steps
# Assert: All particles grow 5% per step
# Expected: r_final ≈ r_initial × 1.05^10 ≈ 1.63 × r_initial
```

**Pass criteria:** Growth within 1% of theoretical

---

### Test 4: Radius Shrinkage (crowded particles)
```python
# Create 10 particles in tight cluster (deg > 10)
# Run 10 steps
# Assert: All particles shrink 5% per step
# Expected: r_final ≈ r_initial × 0.95^10 ≈ 0.60 × r_initial
```

**Pass criteria:** Shrinkage within 1% of theoretical

---

### Test 5: PBD Overlap Removal
```python
# Create 100 particles with 50% overlapping
# Run PBD for 8 passes
# Count overlaps before and after
# Assert: Overlaps reduced by at least 80%
```

**Pass criteria:** Overlap count drops significantly each pass

---

### Test 6: Periodic Boundaries
```python
# Place particle at (0.001, 0.5, 0.5) with radius 0.003
# Place particle at (0.149, 0.5, 0.5) with radius 0.003
# Distance across boundary: 0.004 (should touch)
# Assert: Both particles see each other as neighbors
```

**Pass criteria:** Boundary wrapping works in all 3 dimensions

---

### Test 7: Scale Test (5k → 10k)
```python
# Run simulation with N=5000 for 100 frames
# Measure: FPS, mean degree, degree stability
# Repeat with N=10000
# Assert: FPS(10k) >= 10, degree distribution similar
```

**Pass criteria:** 
- N=5k: FPS >= 20
- N=10k: FPS >= 10
- Mean degree stable (± 0.5 over 100 frames)

---

## Migration from GeoTaichi Work

**What to keep:**
- `src/config_geotaichi.py` → adapt parameters to `config.py`
- `src/levy_controller.py` → port `simple_radius_update` and `project_overlaps` to `dynamics.py`
- Initial state generation logic

**What to discard:**
- All GeoTaichi imports/classes
- Lévy diffusion logic (position-only, not implemented yet)
- Old SPH/grid code from `Cursor_FoS-SPH-Taichi`

---

## Known Limitations & Future Work

### Limitations (acceptable for now)
1. **No Verlet lists:** Rebuild every frame (brute-force broad phase)
   - Cost: ~10ms for 5k particles (fine for 20 FPS)
2. **Serial prefix sum:** Could be parallelized with Taichi's reduction
   - Cost: ~1ms (negligible)
3. **Fixed grid resolution:** Recompute `GRID_RES` if `R_MAX` changes dynamically
   - Not needed if radii clamped to [R_MIN, R_MAX]

### Future Enhancements (Phase 2)
1. **Lévy positional relaxation:** Add kernel to reposition particles based on degree gradient
2. **Adaptive cell size:** Recompute `CELL_SIZE` if radii drift beyond initial bounds
3. **2D/3D toggle:** Currently 3D only; add `DIM` parameter for 2D mode
4. **Export/replay:** Save state to disk for debugging/visualization

---

## Performance Targets

| Particle Count | FPS (Metal M1) | Grid Build | Neighbor Count | PBD (4 pass) | Total/Frame |
|----------------|----------------|------------|----------------|--------------|-------------|
| 1,000          | 60+            | 2ms        | 3ms            | 5ms          | ~17ms       |
| 5,000          | 20+            | 10ms       | 15ms           | 20ms         | ~50ms       |
| 10,000         | 10+            | 20ms       | 30ms           | 40ms         | ~100ms      |

---

## Correctness Checklist (User Feedback)

✅ **1. Scatter pointer separation:** `cell_write` (mutated) vs. `cell_start` (read-only after scan)  
✅ **2. Contact tolerance:** `CONTACT_TOL = 0.02` matches `GAP_FRACTION = 0.02`  
✅ **3. Periodic min-image:** Single `periodic_delta()` helper used consistently  
✅ **4. Loop order:** Count neighbors → grow/shrink 5% → PBD separate  
✅ **5. Grid indexing:** Linear index `id = x + y*R + z*R**2`, proper wrapping  
✅ **6. Types & precision:** All `f32` (no mixing with `f64`)  
✅ **7. Rendering API:** Explicit `color` field (no `color_by`/`colormap` in Taichi UI)  
✅ **8. Initial warmup:** 2-4 PBD passes after seeding to prevent sticky blob  
✅ **9. PBD stability:** Cap per-particle displacement (split overlap 50/50), clamp per-axis, use min-image direction  
✅ **10. Hard clamp:** Radii never exceed `[R_MIN, R_MAX]`

---

## Next Steps (After Review)

1. **You review this blueprint** → confirm parameters, ask questions, request changes
2. **I implement file-by-file** (no surprises):
   - `config.py` (constants)
   - `grid.py` (5 kernels, well-commented)
   - `dynamics.py` (radius update + PBD)
   - `run.py` (main loop + GUI)
3. **Run acceptance tests** (order: 1, 2, 3, 4, 5, 6)
4. **Tune parameters** if needed (PBD passes, gap fraction, etc.)
5. **Capture baseline** (FPS, degree distribution, screenshot)

---

## Design Rationale (Why This Works)

**Q: Why no Verlet lists?**  
A: Radii change every *loop* → neighbor list invalidated every loop. Rebuilding from scratch is simpler and only ~10ms for 5k particles.

**Q: Why 27-cell stencil instead of adaptive?**  
A: `CELL_SIZE = 2 × R_MAX` guarantees any particle in adjacent cells can touch. Simple, predictable, no edge cases. No missed pairs.

**Q: Why PBD instead of forces?**  
A: Stable, fast, no time-step tuning. Overlaps resolved in 4-8 iterations without physics baggage. Allows large radius changes without instability.

**Q: Why periodic boundaries?**  
A: Avoids edge effects, keeps particle density uniform. Matches your Shadertoy reference. Enables min-image convention for efficient distance calc.

**Q: Why clamp radii to [R_MIN, R_MAX]?**  
A: Prevents runaway growth/shrinkage. Keeps grid size predictable (no need to recompute `GRID_RES`). Matches initial config and prevents crashes.

**Q: Why CONTACT_TOL = GAP_FRACTION?**  
A: If PBD enforces a 2% gap, neighbor detection must use the same tolerance or particles with zero neighbors won't grow (degrees would be stuck at 0).

**Q: Why separate cell_write from cell_start?**  
A: Scatter uses atomic fetch-add, which mutates the pointer. If we mutate `cell_start`, the prefix sum is destroyed and iteration breaks. `cell_write` is the sacrificial copy.

**Q: Why f32 everywhere (no f64)?**  
A: GPU is fastest with f32. Mixed precision can cause subtle bugs and slowdowns. Consistent types = predictable performance.

**Q: Why GRID_RES = max(3, ceil(...)) instead of int(...)?**  
A: `ceil` ensures last cell covers domain edge (no particles fall outside). `max(3, ...)` ensures 27-stencil is always valid (no edge cases with tiny grids).

**Q: What does "degree" mean exactly?**  
A: `deg[i]` = count of neighbors within `(1 + CONTACT_TOL) * (r_i + r_j)`. This is "near-contact" (not exact touching), matching PBD gap semantics. Particles with `deg=0` will grow.

---

**Blueprint complete. Ready for your review.**

**What to check:**
- Parameters (N, R_MIN, R_MAX, DEG_LOW, DEG_HIGH, etc.)
- Kernel signatures (do they make sense?)
- Main loop order (rebuild → count → adapt → PBD → render)
- Acceptance tests (sufficient? too much?)
- File structure (clean? missing anything?)

**No code written yet.** Once you approve, I'll build it file-by-file with clear comments.

