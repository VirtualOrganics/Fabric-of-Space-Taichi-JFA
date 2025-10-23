# Periodic Boundary Conditions (PBC) Implementation Blueprint

## Executive Summary

**Goal**: Add periodic boundary conditions to enable seamless wrapping at domain edges, eliminating boundary artifacts and mimicking an infinite, homogeneous medium.

**Approach**: Implement minimum-image convention for all geometric operations (distances, neighbor search, corrections, integration) using wrap helpers.

**Impact**: 
- ✅ No edge effects / boundary artifacts
- ✅ Stable densities near boundaries
- ✅ True bulk behavior for foam/packing systems
- ✅ Option to toggle PBC on/off for comparison

---

## Core Concept

### Minimum-Image Convention

```python
# Wrap any position back into primary cell [-L/2, L/2)
wrapP(p) = p - L * round(p / L)

# Minimum-image displacement (shortest distance across periodic boundaries)
pdelta(a, b) = wrapP(a - b)

# Distance using minimum-image
dist(a, b) = ||pdelta(a, b)||
```

**Key insight**: Never compute raw `pi - pj` for geometry. Always use `pdelta(pi, pj)`.

---

## Critical Implementation Details ⚠️

### 1. Use Floor-Based Centered Wrap (Not Round)
- **Problem**: `ti.round()` has banker's rounding and ULP issues at half-cell boundaries
- **Solution**: Use `ti.floor(p/L + 0.5)` for deterministic, centered wrapping
- **Result**: Maps to `[-L/2, L/2)` cleanly without tie ambiguity

### 2. Consistency Audit: Kill ALL Raw `pos[i] - pos[j]`
**Critical**: Every particle-particle displacement MUST use `pdelta()`. Search and replace:
- ❌ `delta = pos[i] - pos[j]`
- ✅ `delta = pdelta(pos[i], pos[j])`

**Locations to audit**:
- `count_neighbors` (neighbor detection)
- `project_overlaps` (PBD corrections)
- `apply_repulsive_forces` (force calculation)
- `compute_max_overlap` (depth measurement)
- `apply_xsph_smoothing` (velocity smoothing)
- Any debug/telemetry distance calculations

### 3. Post-Move Wrapping Strategy
- **Wrap once per pass**, not inside pair loop (avoid bias + expense)
- **Best location**: After PBD pass loop completes, before grid rebuild
- **Pattern**: `pos[i] = wrapP(pos[i] + accumulated_correction)`

### 4. Cell Hashing: Use Centered Coordinates
Since domain is `[-L/2, L/2)`, shift to `[0, 1)` before hashing:
```python
q = (wrapP(pos[i]) + 0.5 * DOMAIN_SIZE) / DOMAIN_SIZE  # in [0,1)³
idx = ti.floor(q * GRID_RES).cast(ti.i32)
```

### 5. Grid Must Handle Current GRID_RES³ Structure
- Current code uses single `GRID_RES` for cubic grid
- Keep `GRID_RES` for now (all axes equal)
- Future-proof: Could generalize to `GRID_NX, GRID_NY, GRID_NZ` later

### 6. Float Atomics Caveat
If `compute_max_overlap()` uses `ti.atomic_max()` on floats:
- Verify Taichi 1.7.4 supports it (it should on Metal)
- If issues arise, use per-thread local max + reduction

### 7. Seeding Strategy with PBC
```python
if PBC_ENABLED:
    # Seed in [-L/2, L/2) for symmetric initial condition
    pos[i] = (ti.random() - 0.5) * DOMAIN_SIZE
else:
    # Seed in [0, L) for bounded domain
    pos[i] = ti.random() * DOMAIN_SIZE
```

### 8. Restart/Initialization Order
1. Clear ALL particle data (positions, radii, velocities, degrees)
2. Clear cell counts/starts
3. Seed new particles (using PBC-aware seeding)
4. Initialize velocities
5. Warmup PBD (with wrapping after each pass)

### 9. Keep Positions Always Wrapped
**Critical invariant**: After **every** position modification, wrap immediately.

**Locations**:
- After seeding: `pos[i] = wrapP(pos[i])`
- After PBD pass loop: `pos[i] = wrapP(pos[i])`
- After force integration: `pos[i] = wrapP(pos[i])`
- After manual test modifications

**Why**:
- Keeps float32 precision high (numbers stay small near origin)
- Simplifies debugging (all positions in `[-L/2, L/2)`)
- Prevents slow drift beyond domain boundaries

### 10. Neighbor Sweep Edge Case
**Current**: 27-cell sweep (3×3×3) with `CELL_SIZE = 2*R_MAX`

**Edge case**: If `R_MAX` grows beyond `CELL_SIZE/2` (e.g., via radius adaptation), we might miss neighbors in diagonal cells.

**Fix**: Either:
- **Option A** (recommended): Ensure `CELL_SIZE >= 2 * R_MAX_ALLOWED` (hard upper bound from config)
- **Option B**: Increase sweep radius to 2 cells (5×5×5 = 125 neighbors) if `R_MAX > CELL_SIZE/2`

**Implementation**: Add assertion in `rebuild_grid()`:
```python
assert CELL_SIZE >= 2.0 * R_MAX, f"Cell size {CELL_SIZE} too small for R_MAX={R_MAX}"
```

### 11. Atomics Fallback (if `ti.atomic_max` on float is flaky)
**Issue**: Metal backend may have issues with `ti.atomic_max(float_field[0], value)`

**Symptom**: Nondeterministic `max_depth` values or crashes

**Fix**: Two-phase reduction:
1. Each thread computes local `max_depth_thread`
2. Use `ti.atomic_max` on a small per-block buffer (size 32-256)
3. Final kernel reduces block maxes → single global max

**Fallback code** (add to `dynamics.py` if needed):
```python
max_depth_blocks = ti.field(dtype=ti.f32, shape=32)  # 32 blocks

@ti.kernel
def compute_max_overlap_robust(...) -> ti.f32:
    # Phase 1: Per-thread max → block buffer
    for i in range(n):
        # ... compute depth_i ...
        block_id = i // (n // 32)
        ti.atomic_max(max_depth_blocks[block_id], depth_i)
    
    # Phase 2: Reduce blocks
    result = 0.0
    for b in range(32):
        result = ti.max(result, max_depth_blocks[b])
    return result
```

### 12. Squared Thresholds Everywhere
**Optimization**: Avoid `sqrt()` where possible by comparing squared distances.

**Check**: All distance comparisons in:
- Neighbor detection: Already done (`dist_sq < (ri + rj + tol)²`)
- PBD overlap: Change `if depth > DEEP_OVERLAP_THRESHOLD` → `if depth_sq > threshold_sq²`
- Max overlap: Already returns squared depth

**Example**:
```python
# Before
if depth > DEEP_OVERLAP_THRESHOLD * R_MAX:
    ...

# After (precompute in config.py)
DEEP_OVERLAP_THRESHOLD_SQ = (DEEP_OVERLAP_THRESHOLD * R_MAX) ** 2

# In kernel
depth_sq = delta.dot(delta) - (ri + rj)**2
if depth_sq > DEEP_OVERLAP_THRESHOLD_SQ:
    ...
```

---

## Implementation Plan

### 1. Configuration (`config.py`)

**Add**:
```python
# Periodic Boundary Conditions
PBC_ENABLED = True  # Toggle periodic boundaries on/off (compile-time)

# Precomputed constants (Python scope, compile-time folded)
HALF_L = 0.5 * DOMAIN_SIZE  # For cell hashing
INV_L = 1.0 / DOMAIN_SIZE   # Avoid repeated division
```

**Why**: Allows easy A/B testing of PBC vs. bounded domain.

---

### 2. Core Geometry Helpers (`grid.py`)

**Add three Taichi functions** at the top of the file (after imports, before kernels):

```python
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
        return p - DOMAIN_SIZE * ti.floor(p / DOMAIN_SIZE + 0.5)
    else:
        return p  # No-op if PBC disabled


@ti.func
def pdelta(a: ti.math.vec3, b: ti.math.vec3) -> ti.math.vec3:
    """
    Compute minimum-image displacement from b to a.
    
    Returns the shortest vector connecting b to a, considering
    periodic images. This is THE fundamental operation for all
    geometry in a periodic system.
    
    Args:
        a: Position of particle i
        b: Position of particle j
    
    Returns:
        Displacement vector a - b (wrapped)
    """
    if ti.static(PBC_ENABLED):
        return wrapP(a - b)
    else:
        return a - b  # Raw displacement if PBC disabled


@ti.func
def wrap_cell(c: ti.math.ivec3) -> ti.math.ivec3:
    """
    Wrap cell indices to periodic grid.
    
    Maps any cell index to [0, GRID_RES)³ using modulo arithmetic.
    Handles negative indices correctly with double-modulo trick: (c % N + N) % N.
    
    Future-proof: Can generalize to per-axis GRID_NX, GRID_NY, GRID_NZ.
    
    Args:
        c: Cell indices (can be negative or >= GRID_RES)
    
    Returns:
        Wrapped cell indices in valid range [0, GRID_RES)³ if PBC enabled,
        clamped to [0, GRID_RES-1]³ if PBC disabled
    """
    if ti.static(PBC_ENABLED):
        return ti.Vector([
            (c[0] % GRID_RES + GRID_RES) % GRID_RES,
            (c[1] % GRID_RES + GRID_RES) % GRID_RES,
            (c[2] % GRID_RES + GRID_RES) % GRID_RES
        ])
    else:
        # Clamp to valid range if PBC disabled (bounded domain)
        return ti.Vector([
            ti.max(0, ti.min(GRID_RES - 1, c[0])),
            ti.max(0, ti.min(GRID_RES - 1, c[1])),
            ti.max(0, ti.min(GRID_RES - 1, c[2]))
        ])
```

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
```

**Critical details**:
- Use `ti.static(PBC_ENABLED)` for compile-time branching (zero overhead)
- `wrapP` uses `floor(p/L + 0.5)` not `round(p/L)` to avoid ULP/tie issues
- `pdelta` is the ONLY way to compute particle-particle vectors
- `wrap_cell` double-modulo handles negative indices correctly: `(c % N + N) % N`
- `power_dist2` ready for Phase B (topological neighbors via Laguerre)
- `cell_id` centralizes linear indexing for future per-axis grids

---

### 3. Update All Geometry Operations

#### A. Grid Scatter (`scatter_particles` kernel in `grid.py`)

**Current**:
```python
# Compute which cell this particle belongs to
idx = ti.floor((pos[i] / DOMAIN_SIZE) * GRID_RES).cast(ti.i32)
```

**Change to**:
```python
# Wrap position first, then compute cell (use precomputed constants)
p_wrapped = wrapP(pos[i])
pos[i] = p_wrapped  # Store back (keep positions always wrapped)
q = (p_wrapped + HALF_L) * INV_L  # Normalize to [0,1)³
idx = ti.floor(q * GRID_RES).cast(ti.i32)
idx = wrap_cell(idx)  # Ensure valid cell index
lin_id = cell_id(idx)  # Convert to linear index
```

**Why**: 
- Particles can be outside `[-L/2, L/2)` after PBD/forces
- Storing wrapped position back keeps numbers small (good for float32 precision)
- Using `HALF_L` and `INV_L` avoids repeated arithmetic

---

#### B. Neighbor Counting (`count_neighbors` kernel in `grid.py`)

**Current distance check**:
```python
for j in range(start, start + count):
    jj = cell_indices[j]
    if i != jj:
        delta = pos[i] - pos[jj]
        dist_sq = delta.dot(delta)
        r_sum = rad[i] + rad[jj] + CONTACT_TOL
        if dist_sq < r_sum * r_sum:
            deg[i] += 1
```

**Change to**:
```python
for j in range(start, start + count):
    jj = cell_indices[j]
    if i != jj:
        delta = pdelta(pos[i], pos[jj])  # Use minimum-image
        dist_sq = delta.dot(delta)
        r_sum = rad[i] + rad[jj] + CONTACT_TOL
        if dist_sq < r_sum * r_sum:
            deg[i] += 1
```

**Also update neighbor cell loop**:
```python
for ox, oy, oz in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
    neighbor_cell = wrap_cell(my_cell + ti.Vector([ox, oy, oz]))  # Wrap neighbor cell
    cell_id = neighbor_cell[0] * GRID_RES * GRID_RES + neighbor_cell[1] * GRID_RES + neighbor_cell[2]
    # ... rest of logic
```

**Why**: Must use wrapped displacement for correct distance across boundaries.

---

#### C. PBD Overlap Projection (`project_overlaps` kernel in `dynamics.py`)

**Current correction**:
```python
if i != j:
    delta = pos[i] - pos[j]
    dist_sq = delta.dot(delta)
    if dist_sq > EPS:
        r_sum = rad[i] + rad[j] + GAP_FRACTION * (rad[i] + rad[j])
        if dist_sq < r_sum * r_sum:
            dist = ti.sqrt(dist_sq)
            overlap = r_sum - dist
            dir = delta / dist
            # ... apply correction
```

**Change to**:
```python
if i != j:
    delta = pdelta(pos[i], pos[j])  # Minimum-image displacement
    dist_sq = delta.dot(delta)
    if dist_sq > EPS:
        r_sum = rad[i] + rad[j] + GAP_FRACTION * (rad[i] + rad[j])
        if dist_sq < r_sum * r_sum:
            dist = ti.sqrt(dist_sq)
            overlap = r_sum - dist
            dir = delta / dist
            # ... apply correction (same as before)
```

**After PBD, wrap positions back**:
```python
# At end of project_overlaps kernel (or in main loop after PBD)
pos[i] = wrapP(pos[i])
```

**Why**: Corrections must use shortest distance. After moving, particles may exit domain and need wrapping.

---

#### D. Force Fallback (`apply_repulsive_forces` kernel in `dynamics.py`)

**Same pattern as PBD**:
```python
delta = pdelta(pos[i], pos[j])  # Use minimum-image
# ... rest of force calculation
```

**After integration, wrap positions**:
```python
# In integrate_velocities or after apply_repulsive_forces
pos[i] = wrapP(pos[i] + vel[i] * dt)
```

**Why**: Forces must act along shortest path. Integration can push particles outside domain.

---

#### E. Max Overlap Computation (`compute_max_overlap` kernel in `dynamics.py`)

**Change distance to**:
```python
delta = pdelta(pos[i], pos[j])
dist_sq = delta.dot(delta)
# ... rest of overlap calculation
```

**Why**: Must measure true overlap, not across-boundary artifact.

---

#### F. XSPH Smoothing (`apply_xsph_smoothing` kernel in `dynamics.py`)

**Change displacement to**:
```python
delta = pdelta(pos[i], pos[j])
# ... rest of velocity smoothing
```

**Why**: Velocity smoothing should use nearby particles in wrapped space.

---

### 4. Update Seeding (`seed_particles` in `run.py`)

**Current**:
```python
pos[i] = ti.Vector([ti.random() * DOMAIN_SIZE, 
                     ti.random() * DOMAIN_SIZE, 
                     ti.random() * DOMAIN_SIZE])
```

**With PBC, center around origin and wrap**:
```python
if ti.static(PBC_ENABLED):
    # Seed in [-L/2, L/2) for symmetric initial condition
    p = ti.Vector([
        (ti.random() - 0.5) * DOMAIN_SIZE,
        (ti.random() - 0.5) * DOMAIN_SIZE,
        (ti.random() - 0.5) * DOMAIN_SIZE
    ])
    pos[i] = wrapP(p)  # Wrap immediately (should be no-op, but ensures invariant)
else:
    # Seed in [0, L) for bounded domain
    pos[i] = ti.Vector([
        ti.random() * DOMAIN_SIZE,
        ti.random() * DOMAIN_SIZE,
        ti.random() * DOMAIN_SIZE
    ])
```

**Why**: 
- With PBC, domain is conceptually centered at origin
- Wrapping immediately after seeding establishes the invariant: **positions are always wrapped**
- This keeps float32 precision high near boundaries

---

### 5. Rendering Considerations (`run.py`)

**Option A: Show primary cell only** (simplest):
```python
# Positions are already wrapped, just render as-is
scene.particles(pos, radius=0.001, per_vertex_radius=rad, per_vertex_color=color)
```

**Option B: Show periodic tiles** (for continuity):
```python
# Render 27 copies (3x3x3) for seamless tiling
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            offset = ti.Vector([dx, dy, dz]) * DOMAIN_SIZE
            pos_shifted = pos + offset
            scene.particles(pos_shifted, radius=0.001, per_vertex_radius=rad, per_vertex_color=color)
```

**Recommendation**: Start with Option A. Add Option B later if requested (adds 27x rendering cost).

---

### 6. GUI Toggle (`run.py`)

**Add checkbox in control panel**:
```python
window.GUI.begin("Control Panel", 0.01, 0.01, 0.32, 0.75)
# ... existing controls ...
window.GUI.text(f"=== Boundary Conditions ===")
PBC_ENABLED = window.GUI.checkbox("Periodic Boundaries (PBC)", PBC_ENABLED)
if window.GUI.is_hovered():
    window.GUI.tooltip("Toggle periodic wrapping at domain edges")
window.GUI.end()
```

**⚠️ Caveat**: Changing `PBC_ENABLED` at runtime requires kernel recompilation. Simpler to require restart.

**Better approach**: Make it read-only display, require restart to change:
```python
window.GUI.text(f"PBC: {'ON' if PBC_ENABLED else 'OFF'}")
window.GUI.text("(restart to change)")
```

---

## Testing Strategy

### Phase 0: Startup Self-Check (Deterministic)

**Add to `run.py` initialization** (after first seeding):
```python
# Startup self-check for PBC correctness
if PBC_ENABLED:
    pos_np = pos.to_numpy()[:active_n]
    
    # Check 1: All positions wrapped
    assert (-DOMAIN_SIZE * 0.5 <= pos_np).all() and (pos_np < DOMAIN_SIZE * 0.5).all(), \
        f"Particles outside [-L/2, L/2): min={pos_np.min()}, max={pos_np.max()}"
    
    # Check 2: Cross-boundary distance (place two test particles manually)
    pos[0] = ti.Vector([+0.49 * DOMAIN_SIZE, 0.0, 0.0])  # Near +X edge
    pos[1] = ti.Vector([-0.49 * DOMAIN_SIZE, 0.0, 0.0])  # Near -X edge
    delta_test = pdelta(pos[0], pos[1])
    dist_test = ti.sqrt(delta_test.dot(delta_test))
    expected_dist = 0.02 * DOMAIN_SIZE  # Short wrap distance, not 0.98*L
    assert abs(dist_test - expected_dist) < 0.001, \
        f"Cross-boundary distance wrong: {dist_test:.6f} (expected {expected_dist:.6f})"
    
    print("[PBC Self-Check] ✅ All checks passed")
    print(f"  - Positions wrapped: [{pos_np.min():.4f}, {pos_np.max():.4f}]")
    print(f"  - Cross-boundary dist: {dist_test:.6f} (expected {expected_dist:.6f})")
```

**Why**: Catches wrapping bugs immediately on startup, before they propagate to simulation.

---

### Phase 1: Validation (N=500, PBC ON)

1. **Seed particles near boundaries**:
   - Some particles at `x ≈ -DOMAIN_SIZE/2`, others at `x ≈ +DOMAIN_SIZE/2`
   - They should see each other as neighbors (wrap distance < no-wrap distance)

2. **Check degree counts**:
   - Particles near edges should have similar degrees as interior particles
   - No "cold edges" (deg ≈ 0 at boundaries)

3. **Push particle across boundary**:
   - Manually set `pos[0] = (0.076, 0, 0)` (just outside domain if L=0.15)
   - After wrap, should appear at `(-0.074, 0, 0)`
   - Neighbor detection should still work

### Phase 2: Comparison (N=5000, PBC ON vs OFF)

1. **Run with PBC OFF**:
   - Expect lower mean degree
   - Expect edge particles to have `deg < 3`

2. **Run with PBC ON**:
   - Expect higher, more uniform mean degree
   - Expect edge particles to have `deg ≈ 5-6` (same as interior)

3. **Metric**: Standard deviation of degree distribution should be lower with PBC.

### Phase 3: Stability (N=5000, PBC ON, 1000 frames)

1. **Check for explosions**:
   - Particles shouldn't fly off to infinity
   - All positions should stay within `[-L, L]` after wrapping

2. **Check PBD convergence**:
   - `MaxDepth` should stay low (<0.01)
   - `Passes` should stay reasonable (4-12)

3. **Check degree stability**:
   - Mean degree should converge and stabilize
   - No sudden spikes or crashes

---

## Success Criteria

✅ **Correctness**:
- [ ] Particles wrap seamlessly across boundaries
- [ ] Neighbor detection works across periodic images
- [ ] Degree counts are uniform (no edge effects)
- [ ] PBD corrections use minimum-image distances

✅ **Stability**:
- [ ] No position explosions (all `pos` finite)
- [ ] MaxDepth stays low (<0.01) with PBC
- [ ] Mean degree converges to target (5-6)

✅ **Flexibility**:
- [ ] Can toggle PBC on/off via config
- [ ] Non-PBC mode still works (backward compatible)

✅ **Performance**:
- [ ] No slowdown vs. non-PBC (wrap is fast)
- [ ] Grid rebuild time unchanged

---

## Potential Gotchas

### 1. Grid Cell Size
**Issue**: With PBC, particles can wrap to opposite side instantly. If `CELL_SIZE < 2*R_MAX`, we might miss neighbors.

**Fix**: Already using `CELL_SIZE = 2 * R_MAX`, which is conservative enough.

---

### 2. Mixing Wrapped/Unwrapped Positions
**Issue**: If some code uses raw `pos[i] - pos[j]` and other code uses `pdelta()`, results will be inconsistent.

**Fix**: **Strict rule**: Use `pdelta()` for ALL particle-particle geometry. Only exception: `wrapP()` itself.

---

### 3. Visualization Discontinuities
**Issue**: Particle at `x=0.074` and particle at `x=-0.074` are neighbors (distance 0.002 with L=0.15), but appear far apart on screen.

**Fix**: Either accept it (most common), or render 27 tiles (expensive).

---

### 4. Initial Overlaps at Boundaries
**Issue**: Seeding uniformly in `[0, L)` can create overlaps across wrap boundary.

**Fix**: Warmup PBD will resolve this. Or seed in `[-L/2, L/2)` for symmetry.

---

### 5. Taichi `ti.static()` Scope
**Issue**: `PBC_ENABLED` must be Python-scope constant for `ti.static()` to work.

**Fix**: Import from `config.py` at module level in `grid.py` and `dynamics.py`.

---

## Implementation Order (v2.0 - Production-Ready)

### Step 1: Config & Constants (`config.py`)
- [ ] Add `PBC_ENABLED = True` (compile-time flag)
- [ ] Add `HALF_L = 0.5 * DOMAIN_SIZE` (precomputed)
- [ ] Add `INV_L = 1.0 / DOMAIN_SIZE` (precomputed)
- [ ] Add `DEEP_OVERLAP_THRESHOLD_SQ = (...)²` if using force fallback

### Step 2: Core Geometry Helpers (`grid.py`)
- [ ] Add `wrapP(p)` with `floor(p/L + 0.5)` (not `round`)
- [ ] Add `pdelta(a, b)` returning `wrapP(a - b)`
- [ ] Add `wrap_cell(c)` with double-modulo
- [ ] Add `power_dist2(x, p, R)` for future Phase B
- [ ] Add `cell_id(c)` to centralize linear indexing

### Step 3: Audit ALL Geometry (`grid.py` + `dynamics.py`)
Replace every `pos[i] - pos[j]` with `pdelta(pos[i], pos[j])`:
- [ ] `count_neighbors` kernel
- [ ] `project_overlaps` kernel
- [ ] `apply_repulsive_forces` kernel (if re-enabled)
- [ ] `compute_max_overlap` kernel
- [ ] `apply_xsph_smoothing` kernel

### Step 4: Grid Scatter (Always-Wrapped Invariant)
- [ ] Update `scatter_particles` to use `HALF_L` and `INV_L`
- [ ] **Critical**: Store wrapped position back: `pos[i] = wrapP(pos[i])`
- [ ] Use `cell_id()` for linearization

### Step 5: Seeding (`run.py`)
- [ ] Update `seed_particles` to seed in `[-L/2, L/2)` if `PBC_ENABLED`
- [ ] Wrap immediately after seeding: `pos[i] = wrapP(p)`

### Step 6: Main Loop Wrapping (`run.py`)
- [ ] After PBD pass loop: `pos[i] = wrapP(pos[i])` (in `project_overlaps` or as separate kernel)
- [ ] After force integration: `pos[i] = wrapP(pos[i])` (if forces are used)
- [ ] After warmup PBD: wrap positions

### Step 7: Startup Self-Check (`run.py`)
- [ ] Add Phase 0 self-check after first seeding (assertions + cross-boundary test)
- [ ] Print "✅ PBC Self-Check passed" if all good

### Step 8: GUI & Documentation
- [ ] Add read-only PBC status: `"PBC: ON (restart to change)"`
- [ ] Update help text / control instructions

### Step 9: Optional Optimizations (Nice-to-Have)
- [ ] Add assertion: `CELL_SIZE >= 2.0 * R_MAX` in `rebuild_grid()`
- [ ] Replace distance comparisons with squared distances where possible
- [ ] If atomics are flaky: implement two-phase reduction for `max_depth`

### Step 10: Testing
- [ ] N=500, PBC ON (validation)
- [ ] N=5000, PBC ON vs OFF (comparison)
- [ ] N=5000, 1000 frames (stability)
- [ ] Check edge vs interior degree uniformity

**Estimated time**: 40-60 min (v2.0 includes all micro-optimizations)

---

## Minimal Test Vector 🧪

**Purpose**: Quick sanity checks to verify PBC is working correctly. Run these immediately after implementation.

### Test 1: Cross-Boundary Neighbor Detection

**Setup**:
```python
# Place two particles near opposite faces
pos[0] = ti.Vector([+0.49 * DOMAIN_SIZE, 0.0, 0.0])  # Near +X face
pos[1] = ti.Vector([-0.49 * DOMAIN_SIZE, 0.0, 0.0])  # Near -X face
rad[0] = rad[1] = 0.01  # Large enough to overlap across boundary
```

**Expected**:
- `pdelta(pos[0], pos[1])` should return `≈ [0.02*L, 0, 0]` (short distance, not `0.98*L`)
- Both particles should count each other as neighbors
- **Pass criterion**: `deg[0] >= 1` and `deg[1] >= 1`

### Test 2: Position Wrapping

**Setup**:
```python
# Push one particle across +L/2 boundary
pos[0] = ti.Vector([+0.5 * DOMAIN_SIZE + 0.01, 0.0, 0.0])  # Outside domain
# Apply wrapP
pos[0] = wrapP(pos[0])
```

**Expected**:
- `pos[0][0]` should be `≈ -0.5*L + 0.01` (wrapped to opposite side)
- **Pass criterion**: `-L/2 <= pos[0][0] < +L/2` for all three axes

### Test 3: Edge vs Interior Degree Uniformity

**Setup**:
```python
# Run simulation to equilibrium (e.g., 1000 frames)
# Sample particles near edges vs center
```

**Expected**:
- Particles within 2*R_MAX of domain boundary should have **same mean degree** as interior particles (within ±10%)
- Without PBC: edge particles have ~50% lower degree
- With PBC: edge and interior degrees match
- **Pass criterion**: `mean(deg_edge) / mean(deg_center) > 0.9`

### Quick Test Script

Add to `run.py` (temporary, remove after testing):
```python
if frame == 10 and PBC_ENABLED:
    # Test 1: Cross-boundary neighbors
    test_i, test_j = 0, 1
    delta = pdelta(pos[test_i], pos[test_j])
    dist_sq = delta.dot(delta)
    print(f"[PBC Test 1] Cross-boundary distance: {ti.sqrt(dist_sq):.6f} (should be << L={DOMAIN_SIZE})")
    
    # Test 2: Wrapping
    pos_before = ti.Vector([0.5 * DOMAIN_SIZE + 0.01, 0.0, 0.0])
    pos_after = wrapP(pos_before)
    print(f"[PBC Test 2] Wrapped {pos_before} → {pos_after} (should be ≈ [-L/2, 0, 0])")
```

---

## Post-Implementation Checklist

After implementation, verify:

- [ ] All files compile without errors
- [ ] Simulation runs with `PBC_ENABLED = True`
- [ ] Simulation runs with `PBC_ENABLED = False`
- [ ] Degree distribution is more uniform with PBC ON
- [ ] No particle positions are NaN or Inf
- [ ] MaxDepth stays low in both modes
- [ ] FPS is similar in both modes
- [ ] Restart functionality still works

---

## Future Enhancements (Optional Niceties) 🚀

These are **not** part of the initial implementation, but can be added later:

### 1. Per-Axis PBC Toggles
**Use case**: Simulate 2D sheets (XY periodic, Z walls) or slabs (Z periodic only)

**Implementation**:
```python
# config.py
PBC_X = True
PBC_Y = True
PBC_Z = False  # Bounded in Z

# grid.py
@ti.func
def wrapP_anisotropic(p: ti.math.vec3) -> ti.math.vec3:
    result = p
    if ti.static(PBC_X):
        result[0] -= DOMAIN_SIZE * ti.floor(p[0] / DOMAIN_SIZE + 0.5)
    if ti.static(PBC_Y):
        result[1] -= DOMAIN_SIZE * ti.floor(p[1] / DOMAIN_SIZE + 0.5)
    if ti.static(PBC_Z):
        result[2] -= DOMAIN_SIZE * ti.floor(p[2] / DOMAIN_SIZE + 0.5)
    return result
```

### 2. Tiled Preview Rendering (3×3×3)
**Use case**: Visualize periodic continuity (see particles "wrap around")

**Implementation**:
- Render primary cell + 26 ghost copies
- Shift each ghost by `±DOMAIN_SIZE` in each axis
- Optional GUI toggle: "Show periodic tiles"

### 3. Lees-Edwards Shear Boundaries
**Use case**: Simulate flow, measure rheology

**Implementation**:
- Modify `pdelta()` to add shear offset: `Δy += shear_rate * time * sign(Δz)`
- Only if you venture into fluid dynamics (not needed for static foam)

### 4. Non-Cubic Domains
**Use case**: Hexagonal crystals, spherical droplets

**Complexity**: High. Requires non-Cartesian grid + custom wrapping logic.
**Recommendation**: Stick with cubic for now.

---

## Design Decisions (Resolved) ✅

1. **Coordinate convention**: `[-L/2, L/2)` (centered)
   - **Rationale**: Symmetric around origin, easier wrapping logic, natural for centered physics

2. **Wrapping function**: `floor(p/L + 0.5)` (not `round(p/L)`)
   - **Rationale**: Avoids ULP/banker's rounding issues at half-cell boundaries

3. **Warmup PBD with PBC**: Yes, wrap after each pass
   - **Rationale**: Ensures consistency, prevents particle drift across boundaries

4. **Rendering**: Primary cell only (no tiled preview initially)
   - **Rationale**: Simpler implementation, tiles can be added later as optional enhancement

5. **Runtime toggle**: No, config-only (compile-time flag)
   - **Rationale**: `ti.static(PBC_ENABLED)` requires kernel recompilation, zero overhead when disabled

6. **Grid structure**: Keep single `GRID_RES` for cubic grid
   - **Rationale**: Matches current implementation, can generalize to `GRID_NX/NY/NZ` later

7. **Post-move wrapping**: Once after PBD pass loop, not inside pair loop
   - **Rationale**: Cheaper, avoids directional bias, cleaner logic

---

## References

- Minimum-image convention: Frenkel & Smit, "Understanding Molecular Simulation" (2002), Ch. 12.1
- Periodic boundary conditions: Allen & Tildesley, "Computer Simulation of Liquids" (2017), Ch. 1.4
- Your initial proposal: Excellent! Already includes all key points (wrap, pdelta, grid wrapping)

---

---

## Revision History

### v1.0 (Initial Draft)
- Core PBC concept (wrap, pdelta, wrap_cell)
- Implementation plan for all 6 geometry touchpoints
- Testing strategy

### v2.0 (User Review Corrections) ✅
**All user-provided corrections incorporated:**
1. ✅ Changed `round()` → `floor(p/L + 0.5)` for centered wrap
2. ✅ Added "Critical Implementation Details" section with 8 watch-outs
3. ✅ Updated cell hashing to use centered coordinates
4. ✅ Clarified post-move wrapping strategy (once per pass)
5. ✅ Added strict `pdelta()` audit checklist
6. ✅ Added seeding strategy for `[-L/2, L/2)` domain
7. ✅ Added minimal test vector (3 sanity checks)
8. ✅ Added float atomics caveat for `compute_max_overlap`
9. ✅ Future-proofed grid structure (noted per-axis generalization)
10. ✅ Added optional enhancements (per-axis PBC, tiled rendering, Lees-Edwards)

### v2.1 (Last-Mile Micro-Optimizations) ✅✅
**All production-ready tweaks incorporated:**
1. ✅ Precomputed constants (`HALF_L`, `INV_L`) for compile-time folding
2. ✅ Always-wrapped invariant (wrap after seed, PBD, forces, scatter)
3. ✅ Squared thresholds everywhere (avoid `sqrt()` in comparisons)
4. ✅ `power_dist2()` helper for future Phase B (Laguerre)
5. ✅ `cell_id()` helper to centralize linear indexing
6. ✅ Store wrapped position back in scatter (`pos[i] = wrapP(...)`)
7. ✅ Two-phase atomic reduction fallback (if Metal backend is flaky)
8. ✅ Startup self-check (Phase 0 deterministic test)
9. ✅ Neighbor sweep edge case (CELL_SIZE >= 2*R_MAX assertion)
10. ✅ GUI read-only PBC status ("ON - restart to change")
11. ✅ Updated implementation order (10 detailed steps)
12. ✅ Expanded "Critical Implementation Details" to 12 sections

---

**Status**: Blueprint production-ready ✅✅✅  
**Next step**: Implement in order (Steps 1-10), ~40-60 min  
**Green lights**: User confirmed all sharp edges addressed

