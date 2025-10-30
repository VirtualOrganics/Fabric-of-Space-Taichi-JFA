# FSC-Only Foam Simulator - Final Specification

**Version:** 2.0 (Final)  
**Status:** Locked - Ready for Implementation

---

## Changelog from Blueprint v1

### Critical Changes:
1. ✅ **Degree removed EVERYWHERE** - No `deg` field, no `count_neighbors()`, GUI shows FSC-only
2. ✅ **Adaptive EMA settle** - Replaces immediate pump, continuous gentle pressure for PBD
3. ✅ **Backpressure mechanism** - Auto-slows growth when overlaps high
4. ✅ **Hold between cycles** - Radii freeze after settling until next JFA (no drift)
5. ✅ **MAX_STEP_PCT policy** - Config default with optional advanced slider
6. ✅ **Telemetry cadence** - Console on measurement frames, HUD at ~10Hz

---

## System Architecture

### Core Loop

**Measurement Frame** (every `JFA_RUN_INTERVAL` frames):
1. Run JFA at dynamic resolution → get raw `fsc[i]`
2. Set targets based on FSC band + growth slider
3. Reset settle counter: `settle_frames_left = ADJUSTMENT_FRAMES`
4. Print telemetry (FSC stats, band counts, r_mean before→target)

**Every Frame** (continuous physics):
1. Compute adaptive EMA alpha from total frames
2. Compute backpressure from current overlaps
3. Nudge radii toward targets (if `settle_frames_left > 0`)
4. Apply per-frame cap (`MAX_STEP_PCT`)
5. Run PBD (adaptive passes with fresh grid)
6. Update HUD (~10Hz: alpha, overlaps, backpressure, countdown)
7. Render

**Between Settle End and Next JFA:**
- Radii **hold steady** (no nudging, no drift toward stale targets)
- PBD continues (maintains separation)

---

## Controller Design (Definitive)

### A) Measurement Frame

**When:** `frame % JFA_RUN_INTERVAL == 0`

**Steps:**
```python
# 1. Run JFA
jfa_res_dynamic = compute_jfa_resolution(r_mean)
fsc_array, jfa_stats = jfa.run_jfa(pos, rad, active_n, jfa_res_dynamic)

# 2. Set targets based on FSC band
set_fsc_targets(
    n=active_n,
    f_low=gui_fsc_low,
    f_high=gui_fsc_high,
    growth_pct=growth_rate_rt[None]  # From slider (e.g., 0.05 = 5%)
)

# 3. Reset settle counter
settle_frames_left = adjustment_frames_rt[None]

# 4. Print telemetry
print_measurement_telemetry(fsc_array, jfa_stats, rad, rad_target, active_n)
```

### B) Every Frame (Physics Loop)

**Steps:**
```python
# 1. Compute adaptive alpha (scaled by total frames)
alpha = alpha_for_frames(adjustment_frames_rt[None])

# 2. Compute backpressure from overlaps
o_max = compute_max_overlap(pos, rad, cell_start, cell_count, cell_indices, local_max_depth, active_n)
backpressure = backpressure_from_overlap(o_max, CONTACT_TOL)

# 3. Nudge radii (only if settling)
if settle_frames_left > 0:
    nudge_radii_adaptive_ema(
        n=active_n,
        alpha=alpha,
        max_step_pct=MAX_STEP_PCT,
        r_min=R_MIN,
        r_max=R_MAX,
        backpressure=backpressure
    )
    settle_frames_left -= 1

# 4. Run PBD (adaptive passes)
rebuild_grid()
for pass in range(adaptive_pbd_passes):
    rebuild_grid()  # Fresh neighbors each pass
    project_overlaps(pos, rad, cell_start, cell_count, cell_indices, active_n)

# 5. Update HUD (every 6 frames @ 60fps = 10Hz)
if frame % 6 == 0:
    update_hud(alpha, o_max, backpressure, settle_frames_left)

# 6. Render
```

---

## Taichi Kernels (Drop-In Ready)

### Kernel 1: Set FSC Targets

**When:** Measurement frame only

```python
@ti.kernel
def set_fsc_targets(
    n: ti.i32,
    f_low: ti.i32,
    f_high: ti.i32,
    growth_pct: ti.f32
):
    """
    Set radius targets based on FSC band.
    
    Args:
        f_low: FSC lower bound (grow below this)
        f_high: FSC upper bound (shrink above this)
        growth_pct: Fractional change per cycle (e.g., 0.05 = 5%)
    
    Uses raw jfa.fsc[i] (no EMA smoothing).
    """
    for i in range(n):
        r0 = rad[i]
        f = jfa.fsc[i]  # RAW FSC
        
        if f < f_low:
            # Too few neighbors → grow
            rad_target[i] = ti.min(r0 * (1.0 + growth_pct), R_MAX)
        elif f > f_high:
            # Too many neighbors → shrink
            rad_target[i] = ti.max(r0 * (1.0 - growth_pct), R_MIN)
        else:
            # In band → hold steady
            rad_target[i] = r0
```

### Kernel 2: Adaptive EMA Nudge with Backpressure

**When:** Every frame during settling

```python
@ti.kernel
def nudge_radii_adaptive_ema(
    n: ti.i32,
    alpha: ti.f32,
    max_step_pct: ti.f32,
    r_min: ti.f32,
    r_max: ti.f32,
    backpressure: ti.f32
):
    """
    Nudge radii toward targets using adaptive EMA with backpressure.
    
    Args:
        alpha: EMA blend factor (computed from total frames)
        max_step_pct: Per-frame hard cap (e.g., 0.12 = 12% max step)
        backpressure: Scaling factor from overlaps (0.25 to 1.0)
    
    Formula:
        delta = alpha × (target - current) × backpressure
        delta = clamp(delta, -cap, +cap)
        radius = clamp(radius + delta, r_min, r_max)
    """
    for i in range(n):
        r0 = rad[i]
        rt = ti.min(r_max, ti.max(r_min, rad_target[i]))
        
        # Compute desired step
        delta = alpha * (rt - r0)
        
        # Apply backpressure (reduces step when overlaps high)
        delta *= backpressure
        
        # Per-frame cap (prevents shocks)
        cap = max_step_pct * r0
        if delta > cap:
            delta = cap
        elif delta < -cap:
            delta = -cap
        
        # Apply and clamp
        rad[i] = ti.min(r_max, ti.max(r_min, r0 + delta))
```

---

## Helper Functions (Python)

### Alpha Computation

```python
def alpha_for_frames(frames: int) -> float:
    """
    Compute EMA alpha to reach ~95% of target after `frames`.
    
    Formula: alpha = 1 - (0.05)^(1/frames)
    
    Examples:
        10 frames → alpha ≈ 0.259 (25.9% per frame)
        30 frames → alpha ≈ 0.094 (9.4% per frame)
        60 frames → alpha ≈ 0.048 (4.8% per frame)
    
    Returns:
        Alpha value in [0, 1]
    """
    return 1.0 - (0.05 ** (1.0 / max(1, frames)))
```

### Backpressure Computation

```python
def backpressure_from_overlap(o_max: float, contact_tol: float) -> float:
    """
    Compute backpressure scaling factor from current max overlap.
    
    When overlaps are light → full speed (1.0)
    When overlaps approach/exceed tolerance → slow down (0.25)
    
    Args:
        o_max: Maximum overlap fraction (from compute_max_overlap)
        contact_tol: Contact tolerance threshold (e.g., 0.035)
    
    Returns:
        Backpressure factor in [0.25, 1.0]
    """
    if contact_tol <= 0:
        return 1.0
    
    raw_factor = (contact_tol - o_max) / contact_tol
    return max(0.25, min(1.0, raw_factor))
```

---

## Configuration Parameters

### Core FSC Control

```python
# FSC Band (adjustable via GUI sliders)
FSC_LOW = 8             # Grow below this FSC value
FSC_HIGH = 20           # Shrink above this FSC value

# Growth Rate (adjustable via GUI slider)
GROWTH_RATE_DEFAULT = 0.05      # 5% per measurement cycle

# Settling Period (adjustable via GUI slider)
ADJUSTMENT_FRAMES_DEFAULT = 30   # Frames to reach ~95% of target

# Per-Frame Safety Cap (config default, optional advanced slider)
MAX_STEP_PCT = 0.12             # 12% max change per frame (prevents shocks)
```

### JFA Configuration

```python
JFA_ENABLED = True
JFA_RUN_INTERVAL = 1        # Run every measurement (for testing; can raise later)
JFA_VOXEL_SCALE = 2.8       # Voxel size = scale × r_mean
JFA_RES_MIN = 64            # Minimum JFA grid resolution
JFA_RES_MAX = 192           # Maximum JFA grid resolution
POWER_BETA = 1.0            # Power diagram weight (standard)
MIN_FACE_VOXELS = 2         # Permissive for testing
```

### Particle Setup

```python
N = 10000                       # Number of particles
DOMAIN_SIZE = 0.189             # Cubic domain side length
R_START_MANUAL = 0.0045         # Starting radius
R_MIN = 0.002                   # Minimum radius bound
R_MAX = 0.010                   # Maximum radius bound

# Poisson Disk Distribution
POISSON_MIN_DIST_FACTOR = 2.0   # Min distance = factor × r_start
```

### PBD & Physics

```python
CONTACT_TOL = 0.035             # Contact tolerance (3.5%)
CELL_SIZE = 0.031               # Grid cell size (for PBD only)
GRID_RES = 7                    # Grid resolution (for PBD only)

PBD_BASE_PASSES = 4             # Base PBD passes
PBD_MAX_PASSES = 8              # Max adaptive passes
PBD_ADAPTIVE_SCALE = 20.0       # Scaling factor for overlap depth

# Optional: XSPH, damping, jitter (unchanged)
```

---

## GUI Layout (FSC-Only)

### Main Stats Panel

```
=== FSC Stats ===
Avg:  12.3
Min:  6
Max:  18

=== FSC Distribution ===
<8:     15.2% (growing)
8-20:   68.5% (stable)
>20:    16.3% (shrinking)

Target: Balance ~33% each for equilibrium
```

### Control Sliders

```
=== FSC Control Band ===
FSC Low (grow below):    [slider: 1-30, default 8]
FSC High (shrink above): [slider: 1-30, default 20]

=== Growth Rhythm ===
Growth rate per cycle:   [slider: 1%-10%, default 5%]
Adjustment frames:       [slider: 10-60, default 30]

=== Advanced (collapsible) ===
Max step per frame:      [slider: 5%-20%, default 12%]
JFA run interval:        [slider: 1-10, default 1]
```

### Radius Bounds

```
=== Radius Limits ===
Current: [0.0023, 0.0089]
Min radius: [slider: 0.001-0.01, default 0.002]
Max radius: [slider: 0.005-0.05, default 0.010]
Starting:   [slider: 0.001-0.01, default 0.0045]
(Press 'R' to reset particles to starting size)
```

### Visualization (Size-Based Filtering Only)

```
=== Visualization ===
Color Mode:       [Size Heatmap / Size Band]
Palette:          [Viridis / Turbo / Inferno]
Band min:         [slider: 0-0.01]
Band max:         [slider: 0-0.01]
Hide out-of-band: [checkbox]
Dim level:        [slider: 0-50%]
```

**Note:** No degree-based visualization options.

---

## Telemetry Output

### Console (Measurement Frames Only)

```
[FSC Measurement] Frame 180
  JFA: res=96³, time=12.3ms
  FSC: μ=12.3 [6, 18]
  Band: [8, 20]
  Actions: grow=1520 (<8) | hold=6850 | shrink=1630 (>20)
  Radii: r_mean=0.004523 → target=0.004589 (Δ=+1.5%)
  Score: μ=0.98 σ=0.12, asym=2.1%, overflow=0.0%
```

### HUD (Updated at ~10Hz)

**Top-right overlay (small, non-intrusive):**
```
Settling: 18/30 frames
Alpha: 9.4%
Overlaps: 1.2%
Backpressure: 0.92×
```

**When not settling:**
```
Holding (next FSC in 42 frames)
```

---

## Field Allocations (Minimal)

### Keep These:

```python
# Core particle data
pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)
rad = ti.field(dtype=ti.f32, shape=MAX_N)
rad_target = ti.field(dtype=ti.f32, shape=MAX_N)
color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)

# Physics (if enabled)
vel = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)          # For force fallback
vel_temp = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)     # For XSPH
v_jit = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)        # For Brownian

# Grid (PBD only)
cell_count = ti.field(dtype=ti.i32, shape=GRID_RES**3)
cell_start = ti.field(dtype=ti.i32, shape=GRID_RES**3)
cell_write = ti.field(dtype=ti.i32, shape=GRID_RES**3)
cell_indices = ti.field(dtype=ti.i32, shape=MAX_N)

# Overlap tracking
local_max_depth = ti.field(dtype=ti.f32, shape=MAX_N)

# Rendering (filtered viz)
idx_render = ti.field(dtype=ti.i32, shape=MAX_RENDER)
render_count = ti.field(dtype=ti.i32, shape=())
pos_render = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER)
rad_render = ti.field(dtype=ti.f32, shape=MAX_RENDER)
col_render = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER)

# Rhythm control
adjustment_timer = ti.field(dtype=ti.i32, shape=())
grow_rate_rt = ti.field(dtype=ti.f32, shape=())
adjustment_frames_rt = ti.field(dtype=ti.i32, shape=())
r_start_rt = ti.field(dtype=ti.f32, shape=())

# Visualization controls
viz_mode_rt = ti.field(dtype=ti.i32, shape=())
viz_band_min_rt = ti.field(dtype=ti.f32, shape=())
viz_band_max_rt = ti.field(dtype=ti.f32, shape=())
viz_hide_out_rt = ti.field(dtype=ti.i32, shape=())
viz_palette_rt = ti.field(dtype=ti.f32, shape=())
viz_dim_alpha_rt = ti.field(dtype=ti.f32, shape=())
```

### DELETE These (Degree-Based):

```python
# REMOVE - no longer used
deg = ti.field(dtype=ti.i32, shape=MAX_N)              # ❌ Degree field
deg_smoothed = ti.field(dtype=ti.f32, shape=MAX_N)     # ❌ Smoothed degree
action = ti.field(dtype=ti.i32, shape=MAX_N)           # ❌ Degree-based actions
lock_pulses = ti.field(dtype=ti.i32, shape=MAX_N)      # ❌ Hysteresis
streak = ti.field(dtype=ti.i32, shape=MAX_N)           # ❌ Momentum
```

### DELETE These (Hybrid FSC):

```python
# REMOVE - hybrid startup logic
startup_phase = ti.field(dtype=ti.i32, shape=())       # ❌
stable_counter = ti.field(dtype=ti.i32, shape=())      # ❌
gamma_blend = ti.field(dtype=ti.f32, shape=())         # ❌
fsc_low_auto = ti.field(dtype=ti.i32, shape=())        # ❌
fsc_high_auto = ti.field(dtype=ti.i32, shape=())       # ❌
prev_r_mean = ti.field(dtype=ti.f32, shape=())         # ❌
last_action = ti.field(dtype=ti.i8, shape=MAX_N)       # ❌
curr_action = ti.field(dtype=ti.i8, shape=MAX_N)       # ❌
p_in_count = ti.field(dtype=ti.i32, shape=())          # ❌
flip_count = ti.field(dtype=ti.i32, shape=())          # ❌
```

---

## Implementation Phases (Ordered)

### Phase 1: Core Cleanup ✅
- Delete degree fields (`deg`, `deg_smoothed`, `action`, `lock_pulses`, `streak`)
- Delete hybrid FSC fields (all startup_phase, gamma, auto calibration)
- Delete `count_neighbors()` function
- Delete `decide_action()`, `smooth_degree()` calls in control path
- Delete `apply_fsc_controller_hybrid()` kernel

### Phase 2: Add New Kernels ✅
- Add `set_fsc_targets()` kernel to `dynamics.py`
- Add `nudge_radii_adaptive_ema()` kernel to `dynamics.py`
- Add `alpha_for_frames()` helper to `run.py`
- Add `backpressure_from_overlap()` helper to `run.py`

### Phase 3: Replace Control Loop ✅
- Replace measurement block with FSC-only path
- Add settle counter logic (`settle_frames_left`)
- Integrate backpressure computation
- Remove all degree-based branching

### Phase 4: Fix GUI ✅
- Delete degree stats display
- Add FSC stats display (avg, min, max)
- Fix distribution tracker to use `jfa.fsc` only
- Rename sliders to `gui_fsc_low` / `gui_fsc_high`
- Add optional advanced slider for `MAX_STEP_PCT`

### Phase 5: Update Telemetry ✅
- Console: Print on measurement frames only
- HUD: Update at ~10Hz (every 6 frames)
- Show alpha, o_max, backpressure, settle_frames_left

### Phase 6: Poisson Init ✅
- Add `seed_particles_poisson()` to `dynamics.py`
- Replace `seed_particles()` in `run.py`
- Test initial distribution (verify min distance)

### Phase 7: Config Cleanup ✅
- Remove degree-based params (DEG_LOW/HIGH, GAIN_*, HYSTERESIS, etc.)
- Remove hybrid FSC params (WARMUP, RAMP, CUSHION, etc.)
- Keep FSC-only params (FSC_LOW/HIGH, GROWTH_RATE, ADJUSTMENT_FRAMES)
- Add MAX_STEP_PCT, POISSON_MIN_DIST_FACTOR

---

## Testing Protocol

### Test 1: Poisson Distribution (Initial State)
1. Run with default config
2. Press 'F' (freeze frame) immediately
3. Check CSR integrity (sum(cell_count) == N)
4. Sample 100 particle pairs, verify min distance ≥ 2×r_start

**Expected:** Even spacing, no clusters, no isolated particles

### Test 2: FSC Causality (Critical)

**A) Force Growth:**
- Set `gui_fsc_high = 5` (way below typical ~12)
- Expected: Most particles grow, r_mean increases over 2-3 cycles
- Check: Distribution shows ~90% "growing"

**B) Force Shrink:**
- Set `gui_fsc_low = 18` (way above typical ~12)
- Expected: Most particles shrink, r_mean decreases over 2-3 cycles
- Check: Distribution shows ~90% "shrinking"

**C) Balanced Band:**
- Set `gui_fsc_low = 10`, `gui_fsc_high = 14`
- Expected: Distribution ~33% each (below/in/above), r_mean stable
- Check: After 10 cycles, distribution doesn't drift

### Test 3: Slider Response

**Growth Rate:**
- Start at 5%, run 5 cycles
- Change to 10%, run 5 cycles
- Expected: Faster approach to equilibrium, slightly rougher motion

**Adjustment Frames:**
- Start at 30 frames (smooth)
- Change to 10 frames (fast)
- Expected: Faster settling, rougher but stable (adaptive alpha prevents overshoot)

**MAX_STEP_PCT:**
- Set growth to 10%, adjustment to 10 frames
- Lower MAX_STEP_PCT from 12% to 5%
- Expected: Smoother motion, longer effective settling time

### Test 4: Backpressure Mechanism

**Scenario:** Force rapid growth in congested band
- Set `gui_fsc_high = 5` (force all to grow)
- Watch HUD: backpressure should drop from 1.0 → ~0.5 as overlaps rise
- Expected: Growth slows automatically, PBD keeps up, no explosions

### Test 5: Hold Between Cycles

**Scenario:** Long JFA interval
- Set `JFA_RUN_INTERVAL = 10`, `ADJUSTMENT_FRAMES = 5`
- Expected: Radii settle for 5 frames, then **hold steady** for 5 frames until next JFA
- Check console: r_mean doesn't drift during hold period

---

## Acceptance Criteria (Final)

✅ **No degree anywhere** - No deg field, no count_neighbors, GUI shows FSC only  
✅ **Poisson distribution** - Particles start evenly spaced (min distance enforced)  
✅ **FSC causality works** - Moving band sliders changes r_mean within 2-3 cycles  
✅ **Adaptive EMA stable** - Adjusting frames slider doesn't break convergence  
✅ **Backpressure works** - Growth auto-slows when overlaps high  
✅ **Hold works** - Radii freeze after settling until next JFA  
✅ **No overlaps** - PBD maintains separation during aggressive growth  
✅ **GUI accurate** - Distribution tracker matches actual FSC percentages  
✅ **Telemetry clean** - Console on measurement frames, HUD at 10Hz  

---

## File Structure Summary

### Modified Files:
- `config.py` - Remove 20+ params, keep FSC-only, add MAX_STEP_PCT
- `dynamics.py` - Add 2 kernels + Poisson init
- `run.py` - Replace control loop, fix GUI, update telemetry, remove deg fields
- `grid.py` - No changes (used for PBD only)
- `jfa.py` - No changes (already correct)

### Lines of Code:
- **Deleted:** ~500 lines (degree control, hybrid FSC, unused fields)
- **Added:** ~200 lines (new kernels, Poisson, backpressure)
- **Net:** ~300 lines removed (simpler, cleaner)

---

**This specification is locked and ready for implementation.**

---

# Appendix A: Optional Enhancements (Non-Core)

These refinements improve the controller physics without changing the core design. All are **opt-in** via config and can be toggled individually.

---

## Enhancement 1: Soft Damping Band

**Purpose:** Prevent "chattering" when particles oscillate at FSC band edges.

**Config Addition:**
```python
FSC_DEADBAND = 1.0  # ±FSC units around band edges for smooth ramp
```

**Helper Function (add to run.py):**
```python
@ti.func
def smoothstep(e0: ti.f32, e1: ti.f32, x: ti.f32) -> ti.f32:
    """Hermite interpolation for smooth 0→1 ramp."""
    t = ti.min(1.0, ti.max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)
```

**Updated Kernel:**
```python
@ti.kernel
def set_fsc_targets(n: ti.i32, f_low: ti.i32, f_high: ti.i32, growth_pct: ti.f32):
    """
    Set radius targets with soft damping near band edges.
    
    Gain ramps smoothly:
    - FSC = f_low - 1  →  gain = 0%   (no growth)
    - FSC = f_low      →  gain = 100% (full growth)
    - FSC in [f_low - 1, f_low] → smooth ramp via smoothstep
    """
    db = ti.cast(FSC_DEADBAND, ti.f32)
    
    for i in range(n):
        r0 = rad[i]
        f = ti.cast(jfa.fsc[i], ti.f32)
        
        if f < f_low:
            # Smooth ramp from (f_low - deadband) to f_low
            gain = smoothstep(ti.cast(f_low, ti.f32) - db, ti.cast(f_low, ti.f32), f)
            rad_target[i] = ti.min(r0 * (1.0 + growth_pct * gain), R_MAX)
        elif f > f_high:
            # Smooth ramp from f_high to (f_high + deadband)
            gain = smoothstep(ti.cast(f_high, ti.f32), ti.cast(f_high, ti.f32) + db, f)
            rad_target[i] = ti.max(r0 * (1.0 - growth_pct * gain), R_MIN)
        else:
            # In band → hold steady
            rad_target[i] = r0
```

**Why:** Prevents binary on/off flips when FSC oscillates around boundary (e.g., 7.9 ↔ 8.1).

---

## Enhancement 2: Per-Particle Backpressure

**Purpose:** Let jammed regions slow down while loose regions grow normally.

**Config Addition:**
```python
BACKPRESSURE_MODE = "local"  # "local" | "global" | "off"
```

**Updated Kernel:**
```python
@ti.kernel
def nudge_radii_adaptive_ema(
    n: ti.i32,
    alpha: ti.f32,
    max_step_pct: ti.f32,
    r_min: ti.f32,
    r_max: ti.f32,
    back_global: ti.f32,
    mode: ti.i32  # 0=off, 1=global, 2=local
):
    """
    Nudge radii with optional per-particle backpressure.
    
    mode:
        0 = off       (no backpressure, delta not scaled)
        1 = global    (use global back_global for all particles)
        2 = local     (use per-particle local_max_depth[i])
    """
    for i in range(n):
        r0 = rad[i]
        rt = ti.min(r_max, ti.max(r_min, rad_target[i]))
        d = alpha * (rt - r0)
        
        # Compute backpressure factor
        b = 1.0
        if mode == 1:
            # Global backpressure (from max overlap across all particles)
            b = back_global
        elif mode == 2:
            # Local backpressure (from this particle's overlap depth)
            b = (CONTACT_TOL - local_max_depth[i]) / CONTACT_TOL
            b = ti.min(1.0, ti.max(0.25, b))
        
        # Apply backpressure
        d *= b
        
        # Per-frame cap
        cap = max_step_pct * r0
        if d > cap:
            d = cap
        elif d < -cap:
            d = -cap
        
        # Apply and clamp
        rad[i] = ti.min(r_max, ti.max(r_min, r0 + d))
```

**Call Site (in run.py):**
```python
# Compute mode from config
if BACKPRESSURE_MODE == "local":
    bp_mode = 2
elif BACKPRESSURE_MODE == "global":
    bp_mode = 1
else:
    bp_mode = 0  # off

# Call kernel with mode
nudge_radii_adaptive_ema(
    active_n, alpha, MAX_STEP_PCT,
    R_MIN, R_MAX, back_global, bp_mode
)
```

**Why:** Localized congestion shouldn't slow down the entire foam. Uses existing `local_max_depth[i]` (zero overhead).

---

## Enhancement 3: HUD Clarity

**Change:**
```python
# Before
print(f"Backpressure: {back:.2f}×")

# After
print(f"Speed: {int(back*100)}%")
```

**Why:** "Speed: 92%" is more intuitive than "Backpressure: 0.92×" for non-technical users.

**Optional:** Add tooltip/caption:
```python
window.GUI.text(f"Speed: {int(back*100)}%")
window.GUI.text("  (growth damping)")  # Small gray caption
```

---

## Enhancement 4: Config Ergonomics

**Addition to config.py:**
```python
# Per-frame step cap (safety rail)
MAX_STEP_PCT = 0.12              # Default: 12% max change per frame
MAX_STEP_PCT_RANGE = (0.05, 0.20)  # Valid range for GUI slider (if Advanced open)
```

**GUI Code (if Advanced panel expanded):**
```python
if show_advanced:
    max_step = window.GUI.slider_float(
        "Max step per frame",
        max_step_rt[None],
        MAX_STEP_PCT_RANGE[0],
        MAX_STEP_PCT_RANGE[1]
    )
    max_step_rt[None] = max_step
```

**Why:** Self-documenting ranges prevent users from setting unrealistic values (e.g., 80% per frame).

---

## Enhancement 5: Radius Bounds Safety Test

**Purpose:** Verify radii never escape [R_MIN, R_MAX] during extreme forcing.

**Test Function:**
```python
def test_radius_bounds(rad_np, r_min, r_max, frame, active_n):
    """
    Assert radii stay within bounds.
    
    Call on every measurement frame during causality tests.
    """
    r_min_actual = rad_np[:active_n].min()
    r_max_actual = rad_np[:active_n].max()
    
    assert r_min_actual >= r_min, \
        f"Frame {frame}: Particles below R_MIN: {r_min_actual:.6g} < {r_min}"
    
    assert r_max_actual <= r_max, \
        f"Frame {frame}: Particles above R_MAX: {r_max_actual:.6g} > {r_max}"
    
    print(f"✓ Frame {frame}: Bounds OK [{r_min_actual:.6f}, {r_max_actual:.6f}] ⊂ [{r_min}, {r_max}]")
```

**Usage (in measurement block):**
```python
if adjustment_timer[None] <= 0:
    # ... after JFA and set targets ...
    
    # Run bounds check during testing
    if RUN_SAFETY_TESTS:  # Config flag
        rad_np = rad.to_numpy()[:active_n]
        test_radius_bounds(rad_np, R_MIN, R_MAX, frame, active_n)
```

**Why:** Catches clamping bugs or config errors where R_MIN/R_MAX get violated.

---

## Config Summary (All Enhancements)

**Add to config.py:**
```python
# Enhancement 1: Soft damping band
FSC_DEADBAND = 1.0              # ±FSC units for smooth ramp near edges

# Enhancement 2: Per-particle backpressure
BACKPRESSURE_MODE = "local"     # "local" | "global" | "off"

# Enhancement 4: Step cap ergonomics
MAX_STEP_PCT = 0.12             # Default per-frame cap
MAX_STEP_PCT_RANGE = (0.05, 0.20)  # GUI slider range

# Enhancement 5: Safety tests
RUN_SAFETY_TESTS = False        # Enable bounds checking (testing only)
```

---

## Implementation Note

These enhancements are **non-core** and fully **opt-in**:
- Default behavior matches original spec (with better physics)
- Can be disabled individually via config
- Zero breaking changes to core loop

**Recommended defaults:**
- `FSC_DEADBAND = 1.0` (smooth, prevents chattering)
- `BACKPRESSURE_MODE = "local"` (better physics, no overhead)
- `RUN_SAFETY_TESTS = False` (disable after initial validation)

---

**End of Appendix A**

