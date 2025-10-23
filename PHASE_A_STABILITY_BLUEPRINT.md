# Phase A: Stability Blueprint
**Goal:** Fix PBD collapse and achieve stable, no-overlap simulation with variable particle radii.

---

## Problem Statement

**Current State:**
- Degrees explode to 4997/5000 (catastrophic overlap)
- PBD cannot resolve deep overlaps → oscillation or collapse
- Particles tunnel across domain when corrections accumulate
- Visual result: uniform blob, no individuality

**Root Cause:**
- PBD attempts to fix 100+ overlaps per particle in one pass
- Per-pair corrections are clamped, but their sum is not → tunneling
- No fallback for deep overlaps → PBD thrashes
- Radius changes inject new overlaps faster than PBD can resolve

---

## The 4 Core Fixes

### Fix 1: Adaptive PBD Passes
**Problem:** Fixed 4 passes can't handle variable overlap depths.

**Solution:** Scale passes based on worst-case overlap.

```python
# Compute adaptive pass count
max_depth = max(all overlaps)  # Worst penetration in current frame
passes = clamp(4 + 20 * (max_depth / (0.2 * R_MAX)), 4, 24)
```

**Parameters:**
- Base: 4 passes (normal case)
- Max: 24 passes (rescue mode)
- Scaling: +20 passes when `max_depth = 0.2 * R_MAX` (20% penetration)

**Implementation:**
- Add `compute_max_overlap()` kernel to scan all pairs
- Call before PBD loop in `run.py`
- Log passes used for debugging

---

### Fix 2: PBD Correction Clamp (Anti-Tunneling)
**Problem:** Accumulating 100+ corrections → particle moves across entire domain.

**Solution:** Clamp total accumulated correction per particle, not just per-pair.

```python
# In project_overlaps kernel:
# (Existing code computes per-pair correction_vec and accumulates)

# NEW: After accumulating all neighbors, clamp the total
max_total_displacement = MAX_DISPLACEMENT_FRAC * rad[i] * 2.0  # 2x for multiple neighbors
correction_magnitude = sqrt(correction.dot(correction))
if correction_magnitude > max_total_displacement:
    correction = correction * (max_total_displacement / correction_magnitude)

# Then apply
ti.atomic_add(pos[i], correction)
```

**Parameters:**
- `MAX_DISPLACEMENT_FRAC = 0.2` (20% of particle radius per pass)
- Multiplier: `2.0` (allow larger moves when many neighbors)

**Why this works:**
- Prevents individual particles from teleporting
- Allows gradual separation over multiple passes
- Bounded per-pass movement → stable convergence

---

### Fix 3: Deep-Overlap Force Fallback
**Problem:** PBD can't handle overlaps > 10% in one shot → oscillates.

**Solution:** Switch to smooth force-based separation for deep overlaps, then hand back to PBD.

```python
# In project_overlaps kernel (or separate force kernel):
depth = (ri + rj) - dist

# Threshold for "deep" overlap
DEEP_OVERLAP_THRESHOLD = 0.1 * min(ri, rj)

if depth > DEEP_OVERLAP_THRESHOLD:
    # Apply repulsive force instead of direct projection
    F = k * depth * direction
    # Integrate with small substeps (2-4)
    # (Details below)
else:
    # Normal PBD projection
    # (Existing code)
```

**Force Parameters:**
- Trigger: `depth > 0.1 * min(Ri, Rj)` (10% penetration)
- Stiffness: `k = (1.0 * m_eff) / dt^2` where `m_eff = (mi * mj) / (mi + mj)`
  - For equal masses: `k = 0.5 * m / dt^2`
  - Tune: 0.5–2.0 multiplier
- Substeps: 2–4 micro-integrations per frame when triggered
- Cap impulse: `max_impulse = 0.5 * depth` per substep

**Substep Integration:**
```python
# Pseudocode for force fallback (runs only when max_depth > threshold)
for substep in range(2, 4):  # Adaptive: more substeps if depth is extreme
    for each deep overlap (i, j):
        F = k * depth * direction
        impulse = F * (dt / substeps)
        impulse = clamp(impulse, max_magnitude=0.5*depth)
        vel[i] -= impulse / m[i]
        vel[j] += impulse / m[j]
    
    # Apply velocity (damped)
    for i in range(N):
        pos[i] += vel[i] * (dt / substeps)
        vel[i] *= 0.9  # Damping to prevent oscillation
```

**When to trigger:**
- Check `max_depth` from Fix 1
- If `max_depth > 0.1 * R_MAX`, run force fallback before PBD passes
- Output should reduce `max_depth < 0.05 * R_MAX`, then PBD takes over

---

### Fix 4: XPBD Radius Adaptation (Frame-Rate Independence)
**Problem:** Current radius update is direct: `rad *= 1.05`, frame-rate dependent.

**Solution:** Use XPBD-style soft constraint with compliance.

**XPBD Theory:**
- Constraint: `C = (deg_i - deg_target)`
- Compliance: `α = 1 / (k * dt^2)` where `k` is "stiffness"
- Update: `Δrad = -C / (1/α + 1/m) = -α * C * m / (1 + α*m)`

**Simplified for radius (massless):**
```python
# XPBD radius constraint
desired_change = 0.05 * rad[i]  # 5% growth/shrink
compliance = RADIUS_COMPLIANCE  # Tunable

# XPBD update (frame-rate independent)
Δrad = desired_change / (1.0 + compliance / dt^2)

# Apply with rate limit
Δrad = clamp(Δrad, -0.02*rad[i], +0.02*rad[i])  # 2% max per frame
rad[i] += Δrad
```

**Parameters:**
- `RADIUS_COMPLIANCE = 0.01` (soft, smooth changes)
  - Lower = stiffer (faster response)
  - Higher = softer (smoother, more damped)
- `RADIUS_RATE_LIMIT = 0.02` (2% per frame, regardless of FPS)

**Why XPBD?**
- Decouples response from frame rate
- Prevents "radius shocks" that inject overlaps
- Smooth convergence even if target degree fluctuates

---

## Additional Improvements

### XSPH Velocity Smoothing (Anti-Jitter)
**Purpose:** Kill high-frequency oscillations after PBD.

```python
# After all PBD passes, before rendering
for i in range(N):
    # Average velocity of geometric neighbors
    v_avg = mean(vel[j] for j in neighbors(i))
    
    # Blend with own velocity
    vel[i] = (1 - XSPH_EPSILON) * vel[i] + XSPH_EPSILON * v_avg
```

**Parameters:**
- `XSPH_EPSILON = 0.03` (3% blend per frame)
- Only affects velocity, not position
- Optional: Can skip if velocities aren't used (pure PBD)

---

### Dynamic Grid Cell Size
**Purpose:** Ensure grid cells match current particle radii.

```python
# Monitor max_radius each frame
if abs(current_max_radius - last_max_radius) / last_max_radius > 0.05:
    # Radii drifted >5%, rebuild grid structure
    CELL_SIZE = 2 * current_max_radius
    GRID_RES = ceil(DOMAIN_SIZE / CELL_SIZE)
    GRID_TOTAL_CELLS = GRID_RES^3
    
    # Reallocate grid fields (or use pre-allocated max size)
    last_max_radius = current_max_radius
```

**Strategy:**
- Pre-allocate for worst-case `CELL_SIZE = 2 * R_MAX` (no realloc)
- If `R_MAX` is truly hard-clamped, grid never needs resizing
- This is mostly a safeguard

---

## Implementation Order

### Step 1: Update `config.py`
Add new parameters:
```python
# Adaptive PBD
PBD_BASE_PASSES = 4
PBD_MAX_PASSES = 24
PBD_ADAPTIVE_SCALE = 20.0  # Passes added per unit depth

# Correction clamping
MAX_DISPLACEMENT_FRAC = 0.2  # 20% of radius per pass
DISPLACEMENT_MULTIPLIER = 2.0  # Allow more for multi-neighbor cases

# Deep overlap force fallback
DEEP_OVERLAP_THRESHOLD = 0.10  # Trigger rescue at 10% of R_MAX
DEEP_OVERLAP_EXIT = 0.07  # Exit rescue when below 7% (hysteresis)
FORCE_STIFFNESS_MULTIPLIER = 1.0  # Tune 0.5–2.0
FORCE_SUBSTEPS_MIN = 2
FORCE_SUBSTEPS_MAX = 4
FORCE_DAMPING = 0.9  # Velocity damping per substep
GLOBAL_DAMPING = 0.995  # Per-frame global damping (0.5% loss)

# XPBD radius adaptation
RADIUS_COMPLIANCE = 0.02  # Softness of constraint (0.02 often smoother than 0.01)
RADIUS_RATE_LIMIT = 0.02  # 2% max change per frame

# XSPH velocity smoothing
XSPH_EPSILON = 0.03  # 3% blend with neighbors
XSPH_ENABLED = True

# Dynamic grid
GRID_UPDATE_THRESHOLD = 0.05  # Rebuild if max_radius drifts >5%
```

---

### Step 2: Add `compute_max_overlap()` kernel in `dynamics.py`
```python
# Temporary field for per-particle local max depth
local_max_depth = ti.field(dtype=ti.f32, shape=N)

@ti.kernel
def compute_local_max_overlaps():
    """
    Computes per-particle maximum overlap depth.
    First pass: local max per particle (no contention).
    """
    for i in range(N):
        pi = pos[i]
        ri = rad[i]
        my_cell = cell_of_point(pi)
        
        local_max = 0.0
        
        # Check 27 neighboring cells
        for offset_x in ti.static(range(-1, 2)):
            for offset_y in ti.static(range(-1, 2)):
                for offset_z in ti.static(range(-1, 2)):
                    nc = my_cell + ti.Vector([offset_x, offset_y, offset_z])
                    nc_id = cell_index_from_coord(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    for k in range(start, start + count):
                        j = cell_indices[k]
                        if i < j:  # Check each pair once
                            pj = pos[j]
                            rj = rad[j]
                            
                            delta = periodic_delta(pi, pj, DOMAIN_SIZE)
                            dist = delta.norm()
                            
                            target_dist = ri + rj
                            if dist < target_dist:
                                depth = target_dist - dist
                                local_max = ti.max(local_max, depth)
        
        local_max_depth[i] = local_max

@ti.kernel
def reduce_max_depth() -> ti.f32:
    """
    Second pass: global reduction of per-particle maxima.
    Avoids atomic contention from all threads.
    """
    global_max = 0.0
    for i in range(N):
        global_max = ti.max(global_max, local_max_depth[i])
    return global_max

def compute_max_overlap():
    """
    Wrapper: two-pass reduction for robust max_depth.
    """
    compute_local_max_overlaps()
    return reduce_max_depth()
```

---

### Step 3: Update `project_overlaps()` in `dynamics.py`

**3a) Add total correction clamping** (already partially done, but verify):
```python
# Inside the outer particle loop, after accumulating all corrections:
# (This is the fix from earlier - ensure it's in place)

# CRITICAL: Clamp total accumulated correction to prevent tunneling
max_total_displacement = MAX_DISPLACEMENT_FRAC * rad[i] * DISPLACEMENT_MULTIPLIER
correction_magnitude = ti.sqrt(correction.dot(correction))
if correction_magnitude > max_total_displacement:
    correction = correction * (max_total_displacement / correction_magnitude)

# Apply atomically
for d in ti.static(range(3)):
    ti.atomic_add(pos[i][d], correction[d])
```

**3b) Optionally add deep-overlap detection:**
```python
# Inside the pair loop (i, j):
depth = (ri + rj) - dist

# Mark pairs as "deep" for force fallback (Option A: separate pass)
# OR apply force directly here (Option B: inline)
```

**Recommendation:** Start with Option A (separate force pass), cleaner architecture.

---

### Step 4: Add `apply_repulsive_forces()` kernel in `dynamics.py`
```python
@ti.kernel
def apply_repulsive_forces(dt_substep: ti.f32):
    """
    Applies repulsive forces for deep overlaps using velocity integration.
    Runs for 2-4 substeps when deep overlaps are detected.
    
    CRITICAL: Uses i < j to process each pair once, applies equal-and-opposite
    impulses to both particles symmetrically.
    """
    for i in range(N):
        pi = pos[i]
        ri = rad[i]
        mi = 1.0  # Assume unit mass
        
        my_cell = cell_of_point(pi)
        
        # Accumulate force on particle i
        force_i = ti.Vector([0.0, 0.0, 0.0])
        
        # Check 27 neighboring cells
        for offset_x in ti.static(range(-1, 2)):
            for offset_y in ti.static(range(-1, 2)):
                for offset_z in ti.static(range(-1, 2)):
                    nc = my_cell + ti.Vector([offset_x, offset_y, offset_z])
                    nc_id = cell_index_from_coord(nc)
                    
                    start = cell_start[nc_id]
                    count = cell_count[nc_id]
                    for k in range(start, start + count):
                        j = cell_indices[k]
                        
                        # CRITICAL: Use i < j to avoid double-applying pairs
                        if i < j:
                            pj = pos[j]
                            rj = rad[j]
                            mj = 1.0
                            
                            delta = periodic_delta(pi, pj, DOMAIN_SIZE)
                            dist = delta.norm()
                            
                            target_dist = ri + rj
                            if dist < target_dist:
                                depth = target_dist - dist
                                
                                # Only apply force if deep overlap
                                threshold = DEEP_OVERLAP_THRESHOLD * ti.min(ri, rj)
                                if depth > threshold:
                                    # Repulsive force proportional to depth
                                    if dist > EPS:
                                        direction = delta / dist
                                    else:
                                        # Random direction if coincident
                                        direction = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]).normalized()
                                    
                                    # Effective mass
                                    m_eff = (mi * mj) / (mi + mj)
                                    k = FORCE_STIFFNESS_MULTIPLIER * m_eff / (dt_substep * dt_substep)
                                    
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
def integrate_velocities(dt_substep: ti.f32):
    """
    Integrate velocities → positions for force fallback substep.
    Separate kernel for clarity.
    """
    for i in range(N):
        pos[i] += vel[i] * dt_substep
        vel[i] *= FORCE_DAMPING  # Per-substep damping
        
        # Wrap to periodic domain
        for d in ti.static(range(3)):
            if pos[i][d] < 0.0:
                pos[i][d] += DOMAIN_SIZE
            elif pos[i][d] >= DOMAIN_SIZE:
                pos[i][d] -= DOMAIN_SIZE

@ti.kernel
def apply_global_damping():
    """
    Apply gentle global damping to prevent slow energy accumulation.
    Call once per frame (not per substep).
    """
    for i in range(N):
        vel[i] *= 0.995  # 0.5% damping per frame
```

---

### Step 5: Add velocity field in `grid.py`
```python
# Add to existing fields
vel = ti.Vector.field(DIM, dtype=ti.f32, shape=N)

@ti.kernel
def init_velocities():
    """Initialize all velocities to zero at start."""
    for i in range(N):
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
```

---

### Step 6: Update `update_radii()` to use XPBD in `dynamics.py`
```python
@ti.kernel
def update_radii_xpbd(dt: ti.f32):
    """
    XPBD-compliant radius adaptation.
    Frame-rate independent, rate-limited.
    """
    for i in range(N):
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
        
        # Hard clamp to [R_MIN, R_MAX]
        rad[i] = ti.max(R_MIN, ti.min(R_MAX, r_new))
```

---

### Step 7: Add XSPH smoothing kernel in `dynamics.py`
```python
@ti.kernel
def apply_xsph_smoothing():
    """
    Smooth velocities using XSPH: blend with neighbors.
    Reduces jitter after PBD.
    """
    # Temporary buffer for smoothed velocities
    vel_smoothed = ti.Vector.field(DIM, dtype=ti.f32, shape=N)
    
    for i in range(N):
        pi = pos[i]
        my_cell = cell_of_point(pi)
        
        # Accumulate neighbor velocities
        v_sum = ti.Vector([0.0, 0.0, 0.0])
        count = 0
        
        # Check 27 neighboring cells
        for offset_x in ti.static(range(-1, 2)):
            for offset_y in ti.static(range(-1, 2)):
                for offset_z in ti.static(range(-1, 2)):
                    nc = my_cell + ti.Vector([offset_x, offset_y, offset_z])
                    nc_id = cell_index_from_coord(nc)
                    
                    start = cell_start[nc_id]
                    cnt = cell_count[nc_id]
                    for k in range(start, start + cnt):
                        j = cell_indices[k]
                        if i != j:
                            # Simple neighbor criterion: within 2*R_MAX
                            pj = pos[j]
                            delta = periodic_delta(pi, pj, DOMAIN_SIZE)
                            dist_sq = delta.dot(delta)
                            
                            if dist_sq < (2.0 * R_MAX) ** 2:
                                v_sum += vel[j]
                                count += 1
        
        # Compute average neighbor velocity
        if count > 0:
            v_avg = v_sum / count
        else:
            v_avg = vel[i]  # No neighbors, keep own velocity
        
        # Blend
        vel_smoothed[i] = (1.0 - XSPH_EPSILON) * vel[i] + XSPH_EPSILON * v_avg
    
    # Copy back
    for i in range(N):
        vel[i] = vel_smoothed[i]
```

---

### Step 8: Update main loop in `run.py`

```python
# === Initialize before main loop ===
init_velocities()  # Zero all velocities
rescue_mode = False  # State for hysteresis
rescue_frame_count = 0  # Telemetry
total_rescue_substeps = 0  # Telemetry

# Main simulation loop
while window.running:
    if not paused:
        # === 0. Rebuild grid (current radii) ===
        rebuild_grid()
        
        # === 1. Compute max overlap depth ===
        max_depth = compute_max_overlap()
        
        # === 2. Determine adaptive PBD pass count ===
        passes_needed = max(PBD_BASE_PASSES, 
                            min(PBD_MAX_PASSES, 
                                int(PBD_BASE_PASSES + PBD_ADAPTIVE_SCALE * (max_depth / (0.2 * R_MAX)))))
        
        # === 3. Deep overlap force fallback (with hysteresis) ===
        # Hysteresis: trigger at THRESHOLD, exit at EXIT (lower)
        if max_depth > DEEP_OVERLAP_THRESHOLD * R_MAX:
            rescue_mode = True
        elif max_depth < DEEP_OVERLAP_EXIT * R_MAX:
            rescue_mode = False
        
        if rescue_mode:
            rescue_frame_count += 1
            
            # Determine substeps adaptively
            substeps = FORCE_SUBSTEPS_MIN
            if max_depth > 0.2 * R_MAX:
                substeps = FORCE_SUBSTEPS_MAX
            
            total_rescue_substeps += substeps
            
            dt_substep = DT / substeps
            for substep in range(substeps):
                apply_repulsive_forces(dt_substep)
                integrate_velocities(dt_substep)
                rebuild_grid()  # Grid changes after position updates
            
            # Recompute max_depth after rescue
            max_depth_after = compute_max_overlap()
            if frame % 100 == 0:
                print(f"[Force Rescue] {substeps} substeps, depth {max_depth:.6f} → {max_depth_after:.6f}")
        
        # === 4. Adaptive PBD passes ===
        for pass_idx in range(passes_needed):
            rebuild_grid()  # Fresh neighbors each pass
            project_overlaps()
        
        # === 5. XSPH velocity smoothing ===
        if XSPH_ENABLED:
            apply_xsph_smoothing()
        
        # === 6. Global damping (prevent slow energy accumulation) ===
        apply_global_damping()
        
        # === 7. Count neighbors (geometric) ===
        rebuild_grid()
        count_neighbors()
        
        # === 8. Adapt radii (XPBD, using geometric degree for now) ===
        update_radii_xpbd(DT)
        
        # === 9. Update colors ===
        update_colors()
        
        frame += 1
        
        # === Telemetry & Logging ===
        if frame % 100 == 0:
            deg_np = deg.to_numpy()
            rad_np = rad.to_numpy()
            
            # Compute % of pairs with deep overlaps (optional, more expensive)
            # deep_pair_count = count_deep_pairs()  # Implement if needed
            
            rescue_pct = 100.0 * rescue_frame_count / frame
            avg_substeps = total_rescue_substeps / max(1, rescue_frame_count)
            
            print(f"[Frame {frame:4d}] Passes={passes_needed}, MaxDepth={max_depth:.6f} | " +
                  f"Degree: mean={deg_np.mean():.2f}, min={deg_np.min()}, max={deg_np.max()} | " +
                  f"Radius: mean={rad_np.mean():.4f}, min={rad_np.min():.4f}, max={rad_np.max():.4f} | " +
                  f"Rescue: {rescue_pct:.1f}% frames, {avg_substeps:.1f} avg substeps")
    
    # Rendering (unchanged)
    # ...
```

---

## Expected Outcomes (Success Criteria)

### Quantitative Metrics
1. **Degree stability:**
   - Mean degree: 6-12 (geometric contacts)
   - Max degree: <20 (not 4997!)
   - Should converge and stabilize within 500 frames

2. **Overlap resolution:**
   - `max_depth < 0.05 * R_MAX` in steady state
   - No visible interpenetration

3. **Radius variation:**
   - Standard deviation of radii: >10% of mean
   - At least 500+ unique radius values (out of 5000 particles)
   - Min/max ratio: >2x

4. **Performance:**
   - Base case (no rescue): 4 passes → ~current FPS
   - Rescue mode (rare): 24 passes + 4 substeps → ~50% FPS dip, temporary
   - Steady state: Should return to base passes within 100 frames after disturbance

### Visual Checks
- No uniform blob
- Clear size variation (small red particles, large blue particles)
- Particles jiggle slightly but don't oscillate wildly
- No tunneling or "teleporting"

### Debug Output
```
[Frame  100] Passes=4, MaxDepth=0.000234 | Degree: mean=7.34, min=2, max=14 | Radius: mean=0.0045, min=0.0021, max=0.0079
[Frame  200] Passes=6, MaxDepth=0.000512 | Degree: mean=8.12, min=1, max=18 | Radius: mean=0.0046, min=0.0019, max=0.0080
[Force Rescue] 4 substeps, depth 0.001234 → 0.000345
[Frame  300] Passes=4, MaxDepth=0.000198 | Degree: mean=7.89, min=0, max=15 | Radius: mean=0.0047, min=0.0023, max=0.0078
```

**Good signs:**
- Passes stay at 4 most of the time
- Rare spikes to 6-12 passes (adaptive response)
- Force rescue triggers <1% of frames
- Degrees stable, max <20

**Bad signs (failure):**
- Passes always at 24 (PBD can't converge)
- Force rescue every frame (overlaps never resolve)
- Degrees oscillate wildly or explode again
- All radii converge to R_MAX or R_MIN (no individuality)

---

## Failure Modes and Contingencies

### If PBD still collapses (degrees → 4997)
**Diagnosis:** Correction clamp too loose, or force fallback not aggressive enough.

**Fixes:**
1. Reduce `MAX_DISPLACEMENT_FRAC` from 0.2 → 0.1
2. Increase `FORCE_STIFFNESS_MULTIPLIER` from 1.0 → 2.0
3. Add more substeps (4 → 8)
4. Reduce `RADIUS_RATE_LIMIT` from 0.02 → 0.01 (slower radius changes)

### If particles oscillate or jitter
**Diagnosis:** PBD is too stiff, or XSPH not effective.

**Fixes:**
1. Increase `RADIUS_COMPLIANCE` from 0.01 → 0.05 (softer)
2. Increase `XSPH_EPSILON` from 0.03 → 0.1 (more damping)
3. Add global damping: `vel *= 0.98` each frame

### If all radii converge (no variation)
**Diagnosis:** Density problem (too sparse or too dense), or geometric contacts are wrong metric.

**Fixes:**
1. Adjust initial packing: `N=2000, DOMAIN_SIZE=0.10` for higher density
2. Increase `R_MIN` from 0.002 → 0.003 to force more contacts
3. **OR proceed to Phase B** (topological neighbors) if geometric approach is fundamentally flawed

### If performance tanks (FPS <5)
**Diagnosis:** Too many adaptive passes, or force fallback every frame.

**Fixes:**
1. Reduce `PBD_MAX_PASSES` from 24 → 12
2. Reduce `FORCE_SUBSTEPS_MAX` from 4 → 2
3. Increase `DEEP_OVERLAP_THRESHOLD` from 0.1 → 0.2 (trigger less often)

---

## Phase A Validation Checklist

Before proceeding to Phase B (topological neighbors), verify:

- [ ] Degrees stay below 50 (preferably 6-15)
- [ ] Max degree never exceeds 100 (not 4997)
- [ ] Visual: Particles have clearly different sizes
- [ ] Console: "PROOF: >500 UNIQUE radius values" (from 'S' export)
- [ ] No visible overlap (zoom in and inspect)
- [ ] FPS stays >10 for N=5000
- [ ] Passes stay at 4-6 most frames (not always 24)
- [ ] Force rescue triggers <5% of frames

**If all checkboxes pass:** Phase A is complete, ready for Phase B (witness test + topological degree).

**If any fail:** Debug using contingency plans above.

---

## Notes for Phase B (Preview)

Once Phase A is stable, we will:
1. Keep geometric contacts for physics (PBD, no changes)
2. Add `compute_topological_degree()` using witness test
3. Run topo count every K=20 frames
4. EMA smooth: `deg_topo_ema = mix(deg_topo_ema, deg_topo, 0.1)` every frame
5. Switch radius adaptation to use `deg_topo_ema` instead of `deg` (geometric)
6. Update target from ~8 (geometric) to ~14 (topological)

This decouples physics (geometric, fast) from structure (topological, infrequent).

---

## Implementation Refinements Applied

The following critical refinements were integrated based on production experience:

1. **Robust max_depth computation**: Two-pass reduction (local max → global) avoids atomic contention
2. **Symmetric force pairs**: Uses `i < j` to process each pair once, applies equal-and-opposite impulses
3. **PBD clamp placement**: Accumulate all corrections locally, clamp once per particle, then apply atomically
4. **Velocity initialization**: All velocities zeroed at start
5. **Global damping**: Added 0.5% per-frame damping (`vel *= 0.995`) to prevent energy accumulation
6. **Rescue hysteresis**: Trigger at 0.10, exit at 0.07 to prevent mode-flapping
7. **Separated integration**: `apply_repulsive_forces` → `integrate_velocities` → `rebuild_grid` in each substep
8. **Enhanced telemetry**: Logs passes, max_depth, rescue %, avg substeps per frame
9. **Conservative constants**: `RADIUS_COMPLIANCE = 0.02` (softer), `GLOBAL_DAMPING = 0.995`

---

## Time Estimate
- Implementation: ~1-2 hours
- Testing: ~30 min
- Debugging (if needed): ~1-2 hours
- **Total:** 3-5 hours for Phase A completion

---

**End of Phase A Blueprint**

