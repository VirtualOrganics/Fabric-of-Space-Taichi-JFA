# GitHub Issue Templates - Optimization Roadmap

Copy-paste these into GitHub Issues to track the optimization work.

---

## Issue 1: ✅ Add JFA Cadence Control (COMPLETED)

**Title:** Add JFA cadence control for multi-rate optimization

**Labels:** `enhancement`, `performance`, `completed`

**Description:**

Implement JFA decimation to run topology computation less frequently than physics/rendering, with safety rails to prevent drift.

### Implementation (Completed)

- [x] Add `JFA_CADENCE`, `JFA_WARMSTART_FRAMES`, and `JFA_WATCHDOG_INTERVAL` to `config.py`
- [x] Implement frame counter and watchdog logic in `run.py`
- [x] Add warm-start period (30 frames) for topology stabilization
- [x] Add periodic watchdog refresh (every 30 JFA runs)
- [x] Log JFA execution pattern

### Results

- **Performance Gain:** 2.4× speedup (4.2 FPS → ~10 FPS on 10k particles)
- **Topology Cost:** 77% → ~31% of frame time
- **Safety:** Warm-start + watchdog prevent drift

### Testing

```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
./venv/bin/python run.py
# Observe: JFA runs every 5 frames after frame 30
# FPS should be ~10 (was ~4.2 before)
```

### Acceptance Criteria

- [x] Configurable JFA cadence (runs every N frames)
- [x] Warm-start period where JFA runs every frame
- [x] Periodic watchdog refresh
- [x] No topology drift over 1000+ frames
- [x] FPS improvement of 2-3×

---

## Issue 2: Adaptive JFA Resolution Heuristic

**Title:** Implement adaptive JFA resolution based on mean radius

**Labels:** `enhancement`, `performance`

**Description:**

Dynamically adjust JFA grid resolution based on current particle size distribution. Smaller particles require higher resolution for accurate face detection; larger particles can use coarser grids.

### Goal

- **Expected Gain:** 1.3-1.5× additional speedup (10 FPS → 13-15 FPS)
- **Rationale:** Avoid wasting compute on unnecessarily high resolution when radii are large

### Implementation Plan

**1. Add Configuration (`config.py`):**
```python
# Adaptive JFA Resolution
JFA_ADAPTIVE_RES = True              # Enable adaptive resolution
JFA_VOXELS_PER_DIAMETER = 12.0       # Target voxels per particle diameter
```

**2. Compute Dynamic Resolution (`run.py`, before JFA call):**
```python
if JFA_ADAPTIVE_RES and jfa_should_run:
    r_mean_current = rad.to_numpy()[:active_n].mean()
    diameter = 2.0 * r_mean_current
    # Target: JFA_VOXELS_PER_DIAMETER voxels per diameter
    dynamic_res = int(DOMAIN_SIZE / (diameter / JFA_VOXELS_PER_DIAMETER))
    # Clamp to bounds
    jfa_res = max(JFA_RES_MIN, min(JFA_RES_MAX, dynamic_res))
    jfa.set_resolution(jfa_res)
```

**3. Log Resolution Changes:**
```python
if prev_jfa_res != jfa_res:
    print(f"[JFA] Adaptive res: {jfa_res}³ (r_mean={r_mean_current:.6f})")
    prev_jfa_res = jfa_res
```

### Testing

```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
./venv/bin/python run.py
# Adjust FSC band to [5, 30] → particles grow → resolution should decrease
# Adjust FSC band to [15, 20] → particles shrink → resolution should increase
# Monitor: Resolution adapts dynamically, FPS improves 1.3-1.5×
```

### Acceptance Criteria

- [ ] Resolution scales with `1 / r_mean`
- [ ] Resolution clamped to `[JFA_RES_MIN, JFA_RES_MAX]`
- [ ] Changes logged to console
- [ ] No topology quality regression (asym%, overflow% stable)
- [ ] FPS improvement of 1.3-1.5× over fixed resolution

---

## Issue 3: Fix fp64 Welford for σ(P) + Add σ(r)

**Title:** Implement numerically stable variance calculation with fp64 Welford

**Labels:** `bug`, `telemetry`, `enhancement`

**Description:**

The current pressure variance calculation (`σ(P)`) suffers from fp32 underflow when computing variance of small pressure values (r³ for small radii). This causes `σ(P)=0` even when the equilibrator is active.

### Problem

- **Current:** Two-pass naive variance, fp32 accumulators
- **Issue:** Underflows when `r³` is very small (< 1e-7)
- **Impact:** Misleading telemetry; can't tell if pressure equilibration is working

### Solution

Replace with Welford's algorithm using fp64 accumulators (✅ **Already Implemented**)

**Changes Made:**
1. ✅ Replaced two-pass method with Welford's single-pass algorithm in `dynamics.py`
2. ✅ Used `ti.f64` accumulators for `mean` and `m2` (sum of squared deviations)
3. ✅ Added σ(r) to telemetry in `run.py` for comparison
4. ✅ Updated docstrings with reference to Welford's algorithm

### New Telemetry Format

```
[Pressure] r_min={...} r_max={...} σ(r)={...} σ(P)={...}
```

- `σ(r)` = radius standard deviation (computed in fp64 on CPU)
- `σ(P)` = pressure standard deviation (r^P_exp, computed in fp64 on GPU)

**Diagnostic Value:**
- If `σ(r) > 0` but `σ(P) = 0` → still has underflow (shouldn't happen with Welford + fp64)
- If both are non-zero → pressure equilibration working correctly

### Testing

```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
./venv/bin/python run.py
# Wait for [EQ Debug] output every 30 frames
# Verify: σ(P) is non-zero (e.g., σ(P) ≈ 1e-8 to 1e-6)
# Verify: σ(r) and σ(P) both show variance
```

### Acceptance Criteria

- [x] Welford's algorithm implemented with fp64 accumulators
- [x] σ(P) non-zero when particles have different radii
- [x] σ(r) added to telemetry for comparison
- [x] No fp32 underflow artifacts

---

## Issue 4: Benchmark Harness & Performance Badge (DEFERRED)

**Title:** Add reproducible benchmark script and performance tracking

**Labels:** `tooling`, `performance`, `documentation`, `deferred`

**Status:** Deferred until Phase 2 optimization is stable. The old benchmark script was removed due to architecture mismatch (expected class-based JFA, current code is functional).

**Description:**

Create a lightweight headless benchmark that:
- Disables rendering in `run.py`
- Runs for fixed N frames
- Parses `[PERF]` log output for timing breakdown
- Computes mean FPS and per-stage averages

**Note:** Current `run.py` already emits structured `[PERF]` logs every 60 frames:
```
[PERF] Frame 180: grid=0.000s  pbd=0.047s  topo=—  render=0.006s  | FPS≈4.3
       Breakdown: grid=0%  pbd=20%  topo=77%  render=3%
```

### Implementation (TODO - Post Phase 2)

Will create `scripts/bench_headless.py` that:
1. Imports and calls `main()` from `run.py` with `RENDER=False`
2. Captures console output
3. Parses `[PERF]` lines
4. Computes aggregate statistics

### Acceptance Criteria

- [ ] Headless benchmark script
- [ ] Parses existing `[PERF]` telemetry
- [ ] Reproducible results (fixed seed)
- [ ] JSON output for CI/CD tracking
- [ ] Document usage in README.md

---

## Issue 5: Spatial Decimation (Dirty Tiles) for JFA

**Title:** Implement spatial decimation for JFA using dirty tile tracking

**Labels:** `enhancement`, `performance`, `advanced`

**Description:**

**⚠️ Advanced Optimization - Implement after Issues 2-4**

Only re-compute JFA in regions where particles have moved or changed size significantly. This is the most complex optimization but offers the largest potential gain.

### Goal

- **Expected Gain:** 2-3× additional speedup (13-15 FPS → 30-40 FPS)
- **Complexity:** High (tile tracking, halo regions, invalidation logic)

### Implementation Plan

**1. Tile Grid (`config.py`):**
```python
# Spatial Decimation
JFA_TILE_SIZE = 32                   # Tile size in voxels
JFA_QUIET_THRESHOLD = 0.01           # Max movement/radius change to stay "quiet"
JFA_HALO_WIDTH = 2                   # Tiles around dirty region to refresh
```

**2. Tile State Tracking (`jfa.py`):**
```python
# Add to JFAContext:
self.tile_dirty = ti.field(dtype=ti.i32, shape=(tiles_x, tiles_y, tiles_z))
self.tile_last_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)
self.tile_last_rad = ti.field(dtype=ti.f32, shape=MAX_N)
```

**3. Mark Dirty Tiles (before JFA):**
```python
@ti.kernel
def mark_dirty_tiles(pos, rad, last_pos, last_rad, n, threshold):
    for i in range(n):
        moved = (pos[i] - last_pos[i]).norm() > threshold
        resized = abs(rad[i] - last_rad[i]) > threshold
        if moved or resized:
            # Mark tile + halo as dirty
            tile = world_to_tile(pos[i])
            for dz in range(-HALO, HALO+1):
                for dy in range(-HALO, HALO+1):
                    for dx in range(-HALO, HALO+1):
                        mark_tile_dirty(tile + (dx, dy, dz))
```

**4. Selective JFA Rasterization:**
```python
# Only rasterize particles in/near dirty tiles
# Only run flood passes on dirty tile boundaries
```

**5. Update Last State:**
```python
@ti.kernel
def update_tile_cache(pos, rad, last_pos, last_rad, n):
    for i in range(n):
        last_pos[i] = pos[i]
        last_rad[i] = rad[i]
```

### Challenges

1. **Halo Region:** Need to refresh surrounding tiles to catch neighbors
2. **Periodic Boundaries:** Tile wrapping logic
3. **Cold Start:** First frame after reset needs full refresh
4. **Memory:** Extra fields for tracking

### Testing Strategy

1. Implement with `JFA_SPATIAL_DECIMATION = False` flag
2. Run side-by-side comparison (full vs spatial)
3. Visual validation: no artifacts at tile boundaries
4. Performance: measure speedup vs complexity cost

### Acceptance Criteria

- [ ] Tile tracking implemented
- [ ] Dirty marking logic correct (includes halo)
- [ ] PBC-aware tile wrapping
- [ ] No visual artifacts
- [ ] FPS improvement of 2-3×
- [ ] Configurable via flag (can disable)
- [ ] Documented in README

---

## Summary

| Issue | Status | Expected Gain | Complexity | Priority |
|-------|--------|---------------|------------|----------|
| 1. JFA Cadence | ✅ Done | 2.4× | Low | Complete |
| 2. Adaptive Resolution | 📋 Open | 1.3-1.5× | Low | **Next** |
| 3. fp64 Welford | ✅ Done | N/A (bugfix) | Low | Complete |
| 4. Benchmark Script | 🔄 Deferred | N/A (tooling) | Low | Post-Phase 2 |
| 5. Spatial Decimation | 📋 Open | 2-3× | High | Future |

**Cumulative Expected Speedup:** 4.2 FPS → ~40-50 FPS (10-12× total)

---

## How to Use These Issues

1. Go to your GitHub repo: `https://github.com/VirtualOrganics/Fabric-of-Space-Taichi-JFA/issues`
2. Click "New Issue"
3. Copy-paste the relevant issue content above
4. Add appropriate labels
5. Submit

Markdown formatting will be preserved!

