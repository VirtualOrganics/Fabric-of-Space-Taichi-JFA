# FSC-Only Foam Simulator - Implementation Blueprint

## ⚠️ SUPERSEDED - See FSC_ONLY_FINAL.md

This blueprint (v1) has been replaced by the final locked spec in **`FSC_ONLY_FINAL.md`**.

**Key changes in v2:**
- Degree removed EVERYWHERE (even GUI)
- Adaptive EMA with backpressure mechanism
- Hold between settle and next JFA (no drift)
- MAX_STEP_PCT policy clarified
- Telemetry cadence defined

---

## Overview (v1 - For Reference Only)

Transform the current hybrid system into a clean FSC-only controller with:
- Poisson disk initial distribution
- FSC-based size control (no geometric degree in control loop)
- Adaptive EMA radius smoothing
- Fixed GUI distribution tracker
- Continuous PBD during settling

---

## Phase 1: Cleanup & Removal (Delete Degree-Based Code)

### 1.1 Remove from `run.py`

**Delete these functions:**
- `smooth_degree()` calls in control path
- `decide_action()` calls in control path
- `set_radius_targets()` calls in control path
- `update_startup_phase()` (hybrid FSC startup)
- `apply_fsc_controller_hybrid()` (complex FSC controller)

**Keep for GUI only:**
- `count_neighbors()` - still needed for distribution display
- `deg` field - still needed for GUI stats

**Remove entire blocks:**
- Lines ~1070-1200: Hybrid FSC startup logic (Phase 0/1/2/3)
- Lines ~1201-1215: Degree-based fallback controller
- All `startup_phase`, `gamma_blend`, `fsc_low_auto` field usage

### 1.2 Simplify Control to Single Path

**Replace entire measurement block with:**
```python
if adjustment_timer[None] <= 0:
    # 1. Count neighbors (for GUI display only)
    count_neighbors(...)
    
    # 2. Run JFA → compute FSC
    if JFA_ENABLED:
        fsc_array, jfa_stats = jfa.run_jfa(pos, rad, active_n)
        
        # 3. Set targets based on FSC
        set_fsc_targets(
            rad, rad_target, active_n,
            float(gui_fsc_low), float(gui_fsc_high),
            float(growth_rate_rt[None]),
            float(R_MIN), float(R_MAX)
        )
        
        # 4. Print telemetry
        print_fsc_telemetry(...)
    
    # 5. Reset timer
    adjustment_timer[None] = adjustment_frames_rt[None]
else:
    # Settling: nudge + PBD
    nudge_radii_adaptive_ema(rad, rad_target, active_n, adjustment_frames_rt[None], adjustment_timer[None])
    adjustment_timer[None] -= 1
```

---

## Phase 2: New Functions to Add

### 2.1 Add to `dynamics.py`

**Function 1: Set FSC Targets**
```python
@ti.kernel
def set_fsc_targets(
    rad: ti.template(),
    rad_target: ti.template(),
    n: ti.i32,
    fsc_low: ti.i32,
    fsc_high: ti.i32,
    growth_rate: ti.f32,
    r_min: ti.f32,
    r_max: ti.f32
):
    """
    Set radius targets based on FSC band.
    
    Args:
        fsc_low: Lower band threshold (grow below this)
        fsc_high: Upper band threshold (shrink above this)
        growth_rate: Percentage change per cycle (e.g., 0.05 = 5%)
    """
    for i in range(n):
        f = jfa.fsc[i]  # Use raw FSC
        r0 = rad[i]
        
        if f < fsc_low:
            # Too few neighbors → grow
            r_target = r0 * (1.0 + growth_rate)
        elif f > fsc_high:
            # Too many neighbors → shrink
            r_target = r0 * (1.0 - growth_rate)
        else:
            # In band → hold steady
            r_target = r0
        
        # Clamp to bounds
        rad_target[i] = ti.min(r_max, ti.max(r_min, r_target))
```

**Function 2: Adaptive EMA Nudging**
```python
@ti.kernel
def nudge_radii_adaptive_ema(
    rad: ti.template(),
    rad_target: ti.template(),
    n: ti.i32,
    total_frames: ti.i32,
    frames_remaining: ti.i32
):
    """
    Nudge radii toward targets using adaptive EMA.
    
    Alpha scales with total_frames so particles reach ~95% of target
    regardless of settling period duration.
    
    Formula: alpha = 1.0 - pow(0.05, 1.0 / total_frames)
    - 10 frames → alpha ≈ 25.9% per frame
    - 30 frames → alpha ≈ 9.4% per frame
    - 60 frames → alpha ≈ 4.8% per frame
    
    Args:
        total_frames: Total adjustment period (from slider)
        frames_remaining: Current countdown value
    """
    # Compute adaptive alpha (fraction to move toward target per frame)
    # This ensures particles reach ~95% of target by end of period
    alpha = 1.0 - ti.pow(0.05, 1.0 / ti.f32(total_frames))
    
    for i in range(n):
        r_current = rad[i]
        r_target = rad_target[i]
        
        # EMA blend: move alpha% toward target
        r_new = r_current * (1.0 - alpha) + r_target * alpha
        
        rad[i] = r_new
```

**Function 3: Poisson Disk Sampling (Python)**
```python
def seed_particles_poisson(n, domain_size, r_start, min_dist_factor=2.0):
    """
    Generate Poisson disk distribution with minimum distance constraint.
    
    Args:
        n: Target number of particles
        domain_size: Cubic domain side length
        r_start: Starting radius for all particles
        min_dist_factor: Minimum distance = min_dist_factor × r_start (default 2.0)
    
    Returns:
        pos_np: (n, 3) array of positions
        rad_np: (n,) array of radii (all r_start)
    
    Algorithm: Bridson's Poisson disk sampling (dart throwing with spatial grid)
    """
    min_dist = min_dist_factor * r_start
    cell_size = min_dist / np.sqrt(3)  # Grid cell size
    grid_res = int(np.ceil(domain_size / cell_size))
    
    # Spatial hash grid for O(1) neighbor checks
    grid = {}
    active = []
    pos_list = []
    
    # Seed first point
    first = np.random.uniform(0, domain_size, 3)
    pos_list.append(first)
    active.append(0)
    grid[tuple((first / cell_size).astype(int))] = 0
    
    max_attempts = 30  # Attempts per active point
    
    while len(pos_list) < n and len(active) > 0:
        # Pick random active point
        idx = np.random.randint(len(active))
        p_idx = active[idx]
        p = pos_list[p_idx]
        
        found = False
        for _ in range(max_attempts):
            # Generate candidate in annulus [min_dist, 2×min_dist]
            r = min_dist * (1 + np.random.random())
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            
            offset = r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            candidate = (p + offset) % domain_size  # PBC wrap
            
            # Check minimum distance in grid
            cell = tuple((candidate / cell_size).astype(int))
            valid = True
            
            # Check 27 neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_cell = (
                            (cell[0] + dx) % grid_res,
                            (cell[1] + dy) % grid_res,
                            (cell[2] + dz) % grid_res
                        )
                        if neighbor_cell in grid:
                            neighbor_idx = grid[neighbor_cell]
                            neighbor_pos = pos_list[neighbor_idx]
                            
                            # PBC-aware distance
                            delta = candidate - neighbor_pos
                            delta = delta - domain_size * np.round(delta / domain_size)
                            dist = np.linalg.norm(delta)
                            
                            if dist < min_dist:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                pos_list.append(candidate)
                grid[cell] = len(pos_list) - 1
                active.append(len(pos_list) - 1)
                found = True
                break
        
        if not found:
            # No valid candidate found, remove from active list
            active.pop(idx)
    
    # Convert to arrays
    pos_np = np.array(pos_list[:n], dtype=np.float32)
    rad_np = np.full(n, r_start, dtype=np.float32)
    
    print(f"[Poisson Init] Generated {len(pos_np)} particles (target: {n})")
    print(f"               min_dist={min_dist:.6f} (factor={min_dist_factor}×r_start)")
    
    return pos_np, rad_np
```

### 2.2 Add to `run.py`

**Replace `seed_particles()` with:**
```python
def seed_particles(n):
    """
    Seed particles with Poisson disk distribution.
    
    Args:
        n: Number of particles to seed
    
    Returns:
        Actual number seeded (may be slightly less if Poisson fails to place all)
    """
    r_start = R_START_MANUAL
    
    # Generate Poisson distribution
    pos_np, rad_np = seed_particles_poisson(n, DOMAIN_SIZE, r_start, min_dist_factor=2.0)
    
    actual_n = len(pos_np)
    
    # Write to Taichi fields
    for i in range(actual_n):
        pos[i] = pos_np[i]
        rad[i] = rad_np[i]
    
    # Wrap positions (PBC-aware)
    wrap_seeded_positions(actual_n)
    
    print(f"[Init] Seeded {actual_n} particles with Poisson disk distribution")
    print(f"       min_dist={2.0 * r_start:.6f} (2×r_start)")
    print(f"       Radius: uniform at {r_start:.6f}")
    
    return actual_n
```

---

## Phase 3: Fix GUI Distribution Tracker

### 3.1 Change in `run.py` GUI Section

**Current (WRONG):**
```python
# Lines ~1338-1344 (approximately)
count_red = np.sum(deg_np < gui_deg_low)
count_green = np.sum((deg_np >= gui_deg_low) & (deg_np <= gui_deg_high))
count_blue = np.sum(deg_np > gui_deg_high)
```

**Replace with (CORRECT):**
```python
# Compute FSC distribution (not degree!)
fsc_np = jfa.fsc.to_numpy()[:active_n] if JFA_ENABLED else np.zeros(active_n)

count_below = np.sum(fsc_np < gui_fsc_low)
count_in_band = np.sum((fsc_np >= gui_fsc_low) & (fsc_np <= gui_fsc_high))
count_above = np.sum(fsc_np > gui_fsc_high)

pct_below = 100.0 * count_below / active_n
pct_in_band = 100.0 * count_in_band / active_n
pct_above = 100.0 * count_above / active_n
```

### 3.2 Update GUI Labels

**Change text from:**
```python
window.GUI.text("Distribution:")
window.GUI.text(f"  <{gui_deg_low}: {pct_red:.1f}%")
window.GUI.text(f"  {gui_deg_low}-{gui_deg_high}: {pct_green:.1f}%")
window.GUI.text(f"  >{gui_deg_high}: {pct_blue:.1f}%")
```

**To:**
```python
window.GUI.text("FSC Distribution:")
window.GUI.text(f"  <{gui_fsc_low}: {pct_below:.1f}% (growing)")
window.GUI.text(f"  {gui_fsc_low}-{gui_fsc_high}: {pct_in_band:.1f}% (stable)")
window.GUI.text(f"  >{gui_fsc_high}: {pct_above:.1f}% (shrinking)")
```

### 3.3 Add FSC Stats Section

**Add after distribution:**
```python
window.GUI.text("")
window.GUI.text("=== FSC Stats ===")
if JFA_ENABLED and fsc_np is not None:
    fsc_mean = float(fsc_np.mean())
    fsc_min = int(fsc_np.min())
    fsc_max = int(fsc_np.max())
    window.GUI.text(f"Avg:  {fsc_mean:.2f}")
    window.GUI.text(f"Min:  {fsc_min}")
    window.GUI.text(f"Max:  {fsc_max}")
else:
    window.GUI.text("(JFA disabled)")
```

### 3.4 Rename Sliders

**Change from:**
```python
gui_deg_low = window.GUI.slider_int("Min (grow below)", gui_deg_low, 1, 20)
gui_deg_high = window.GUI.slider_int("Max (shrink above)", gui_deg_high, 1, 30)
```

**To:**
```python
gui_fsc_low = window.GUI.slider_int("FSC Low (grow below)", gui_fsc_low, 1, 20)
gui_fsc_high = window.GUI.slider_int("FSC High (shrink above)", gui_fsc_high, 1, 30)
```

---

## Phase 4: Update Config

### 4.1 Simplify `config.py`

**Remove these (degree-based):**
```python
DEG_LOW = 3
DEG_HIGH = 5
GAIN_GROW = 0.05
GAIN_SHRINK = 0.05
HYSTERESIS = 0.0
STREAK_LOCK = 1
MOMENTUM = 0.0
STREAK_CAP = 4
```

**Remove these (hybrid FSC startup):**
```python
FSC_WARMUP_FRAMES = 200
FSC_RAMP_FRAMES = 20
FSC_SHRINK_FLOOR = 0.6
POWER_BETA_START = 0.7
POWER_BETA_END = 1.0
FSC_CUSHION_GROW = 0.5
FSC_CUSHION_SHRINK = 1.0
FSC_KP = 0.10
FSC_G_MAX_GROW = 0.06
FSC_G_MAX_SHRINK = 0.03
FSC_R_EMA = 0.30
```

**Keep/update these (FSC-only):**
```python
# FSC Control Band
FSC_LOW = 8             # Grow below this FSC value
FSC_HIGH = 20           # Shrink above this FSC value

# Growth Rhythm
GROWTH_RATE_DEFAULT = 0.05      # 5% per measurement cycle
ADJUSTMENT_FRAMES_DEFAULT = 30   # Frames to settle

# JFA Configuration
JFA_ENABLED = True
JFA_RUN_INTERVAL = 1        # Run every measurement (for testing)
JFA_VOXEL_SCALE = 2.8       # Voxel size = scale × r_mean
JFA_RES_MIN = 64            # Minimum JFA grid resolution
JFA_RES_MAX = 192           # Maximum JFA grid resolution
POWER_BETA = 1.0            # Power diagram weight (standard)

# Starting conditions
R_START_MANUAL = 0.0045     # Starting radius for Poisson distribution
POISSON_MIN_DIST_FACTOR = 2.0  # Min distance = factor × r_start
```

---

## Phase 5: Update Telemetry

### 5.1 Measurement Frame Telemetry

**Replace existing telemetry with:**
```python
if adjustment_timer[None] <= 0:
    # ... after JFA runs and targets set ...
    
    # Get stats
    fsc_np = jfa.fsc.to_numpy()[:active_n]
    rad_np = rad.to_numpy()[:active_n]
    target_np = rad_target.to_numpy()[:active_n]
    
    fsc_mean = float(fsc_np.mean())
    fsc_min = int(fsc_np.min())
    fsc_max = int(fsc_np.max())
    
    r_mean_before = float(rad_np.mean())
    r_target_mean = float(target_np.mean())
    
    # Count actions
    n_grow = int(np.sum(fsc_np < gui_fsc_low))
    n_shrink = int(np.sum(fsc_np > gui_fsc_high))
    n_hold = active_n - n_grow - n_shrink
    
    # Print
    print(f"\n[FSC Measurement] Frame {frame}")
    print(f"  JFA: res={jfa_res_dynamic}³, time={jfa_stats['time_ms']:.1f}ms")
    print(f"  FSC: μ={fsc_mean:.1f} [{fsc_min}, {fsc_max}]")
    print(f"  Band: [{gui_fsc_low}, {gui_fsc_high}]")
    print(f"  Actions: grow={n_grow} (<{gui_fsc_low}) | hold={n_hold} | shrink={n_shrink} (>{gui_fsc_high})")
    print(f"  Radii: r_mean={r_mean_before:.6f} → target={r_target_mean:.6f}")
```

---

## Phase 6: Remove Unused Fields

### 6.1 Delete from `run.py` Field Allocations

**Remove these (degree control):**
```python
action = ti.field(dtype=ti.i32, shape=MAX_N)
lock_pulses = ti.field(dtype=ti.i32, shape=MAX_N)
streak = ti.field(dtype=ti.i32, shape=MAX_N)
deg_smoothed = ti.field(dtype=ti.f32, shape=MAX_N)
```

**Remove these (hybrid startup):**
```python
startup_phase = ti.field(dtype=ti.i32, shape=())
stable_counter = ti.field(dtype=ti.i32, shape=())
gamma_blend = ti.field(dtype=ti.f32, shape=())
fsc_low_auto = ti.field(dtype=ti.i32, shape=())
fsc_high_auto = ti.field(dtype=ti.i32, shape=())
prev_r_mean = ti.field(dtype=ti.f32, shape=())
last_action = ti.field(dtype=ti.i8, shape=MAX_N)
curr_action = ti.field(dtype=ti.i8, shape=MAX_N)
p_in_count = ti.field(dtype=ti.i32, shape=())
flip_count = ti.field(dtype=ti.i32, shape=())
```

**Keep these (still needed):**
```python
pos, rad, rad_target, deg, color  # Core fields
vel, vel_temp                      # Motion/XSPH
v_jit, mean_radius                 # Brownian (if enabled)
adjustment_timer                   # Measurement cycle control
```

---

## Phase 7: Testing Protocol

### 7.1 Initial Test (Baseline)
1. Run with default config (FSC band [8, 20])
2. Check Poisson distribution (particles evenly spaced initially)
3. Watch FSC distribution stabilize over ~10 cycles
4. Verify no particles stuck at 0 or max radius

### 7.2 Causality Test (Critical)
**Test A: Force Growth**
- Set `gui_fsc_high = 5` (way below typical FSC ~12)
- Expected: Most particles grow, r_mean increases over 2-3 cycles

**Test B: Force Shrink**
- Set `gui_fsc_low = 18` (way above typical FSC ~12)
- Expected: Most particles shrink, r_mean decreases over 2-3 cycles

**Test C: Balanced Band**
- Set `gui_fsc_low = 10`, `gui_fsc_high = 14`
- Expected: Distribution shows ~33% each (below/in/above), r_mean stable

### 7.3 Slider Response Test
**Growth Rate:**
- Change from 5% → 10%
- Expected: Faster approach to equilibrium, rougher motion

**Adjustment Frames:**
- Change from 30 → 10 frames
- Expected: Faster settling, slightly rougher (but stable with adaptive EMA)

---

## Summary of Changes by File

### `config.py`
- ❌ Remove degree-based params (DEG_LOW/HIGH, GAIN_*, HYSTERESIS, etc.)
- ❌ Remove hybrid FSC params (WARMUP, RAMP, CUSHION, etc.)
- ✅ Keep FSC_LOW/HIGH, GROWTH_RATE, ADJUSTMENT_FRAMES
- ✅ Add POISSON_MIN_DIST_FACTOR

### `dynamics.py`
- ✅ Add `set_fsc_targets()` kernel
- ✅ Add `nudge_radii_adaptive_ema()` kernel
- ✅ Add `seed_particles_poisson()` function
- ❌ Remove/don't use `decide_action()`, `set_radius_targets()` in control

### `run.py`
- ❌ Remove hybrid FSC controller block (lines ~1070-1200)
- ❌ Remove degree fallback controller (lines ~1201-1215)
- ✅ Replace with simple FSC measurement → set targets → settle loop
- ✅ Replace `seed_particles()` with Poisson version
- ✅ Fix GUI distribution to show FSC (not degree)
- ✅ Add FSC stats display
- ✅ Rename sliders to gui_fsc_low/high
- ❌ Remove unused fields (action, lock_pulses, streak, etc.)

### `grid.py`
- ✅ Keep as-is (used for PBD only)

### `jfa.py`
- ✅ Keep as-is (already returns FSC correctly)

---

## Acceptance Criteria

✅ **Poisson distribution** - Particles start evenly spaced
✅ **FSC-only control** - No degree-based logic in control path
✅ **GUI shows FSC** - Distribution tracker shows FSC percentages
✅ **Causality works** - Moving band sliders changes r_mean within 2-3 cycles
✅ **Adaptive EMA** - Settling period adjustable, always reaches ~95% of target
✅ **No overlaps** - PBD keeps particles separated during settling
✅ **Clean code** - No unused fields, no commented-out degree paths

---

## Implementation Order (Recommended)

1. **Phase 1** - Remove degree control (clean up run.py)
2. **Phase 2** - Add new functions (dynamics.py, Poisson init)
3. **Phase 4** - Update config.py (simplify parameters)
4. **Phase 3** - Fix GUI (distribution + sliders)
5. **Phase 5** - Update telemetry (print FSC stats)
6. **Phase 6** - Remove unused fields (memory cleanup)
7. **Phase 7** - Test (causality, sliders, stability)

Each phase can be done and tested independently before moving to the next.

