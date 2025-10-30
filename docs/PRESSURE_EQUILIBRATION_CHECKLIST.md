# **FSC Pressure Equilibration - Implementation Checklist**
**Version 2.1 - Production Implementation Guide (with Refinements)**

---

## **🚀 Pre-Flight Refinements Summary**

This implementation incorporates the following critical refinements:

1. **✅ True Volume Conservation:** ΣV computed as explicit `r³` (not `P_exp`-dependent) for experiment-independent tracking
2. **✅ Normalized Overlap Metric:** `max(0, (Ra+Rb-d)/(Ra+Rb))` scales cleanly as radii evolve
3. **✅ Neighbor List Verification:** Explicit check for `jfa.neighbors[i, k]` structure (critical dependency)
4. **✅ Numeric Stability:** Kernels stay `f32` (fast), telemetry can use `f64` if needed (optional)
5. **✅ Clamp Leak Visibility:** Track clamped particle count to detect pressure boundary effects
6. **✅ Philosophy Alignment:** Equilibration is primary driver; FSC provides optional long-term guidance

---

## **📋 Pre-Implementation Verification**

Before starting implementation, verify these prerequisites:

- [ ] **1.1** Current system is stable with FSC-only control working
- [ ] **1.2** `jfa.py` computes FSC values correctly (verify `jfa.fsc[i]` is populated)
- [ ] **1.3** Confirm `run.py` has access to `jfa` module and FSC values
- [ ] **1.4** Note current `FSC μ` values in telemetry (for comparison after)
- [ ] **1.5** Exit handler bug fixed (no `deg` references remaining)
- [ ] **1.6** Git commit current state: `git commit -am "Pre-pressure-equilibration checkpoint"`

---

## **🔧 Phase 1: Configuration & Field Allocation**

### **Step 1.1: Add Constants to `config.py`**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/config.py`

Add these new constants to the configuration file:

```python
# ========================================
# PRESSURE EQUILIBRATION
# ========================================
PRESSURE_EQUILIBRATION_ENABLED = True   # Master switch for pressure equilibration
PRESSURE_K = 0.03                       # Base diffusion coefficient (start conservative)
PRESSURE_EXP = 3.0                      # Volume exponent (3 for 3D, 2 for 2D)
PRESSURE_PAIR_CAP = 0.02                # Per-pair ΔV cap (fraction of min(V_i, V_j))
MAX_EQ_NEI = 10                         # Max neighbors equilibrated per site per frame
EQ_MICRO_PBD_ITERS = 1                  # Micro-PBD iterations after equilibration (if needed)
EQ_OVERLAP_THRESHOLD = 0.05             # Normalized penetration depth to trigger micro-PBD
                                        # Formula: max(0, (Ra+Rb-d)/(Ra+Rb)) > threshold
```

**Verification:**
- [ ] **1.1.1** Constants added to `config.py`
- [ ] **1.1.2** No syntax errors when importing `config.py`
- [ ] **1.1.3** Run: `python -c "import config; print(config.PRESSURE_K)"`

---

### **Step 1.2: Allocate `delta_r` Field in `dynamics.py`**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/dynamics.py`

Add the temporary storage field for Jacobi updates:

```python
# Add to imports at top of file
from config import (
    N, DOMAIN_SIZE, R_MIN, R_MAX, CONTACT_TOL,
    FSC_LOW, FSC_HIGH, GROWTH_PCT, R_EMA, 
    ADJUSTMENT_FRAMES, MAX_STEP_PCT, FSC_DEADBAND,
    BACKPRESSURE_MODE,
    PRESSURE_EQUILIBRATION_ENABLED, PRESSURE_K, PRESSURE_EXP,  # NEW
    PRESSURE_PAIR_CAP, MAX_EQ_NEI                               # NEW
)

# Add field allocation (near other field definitions)
delta_r = ti.field(dtype=ti.f32, shape=N)  # Jacobi temporary for pressure equilibration
```

**Verification:**
- [ ] **1.2.1** `delta_r` field allocated
- [ ] **1.2.2** Imports updated with new constants
- [ ] **1.2.3** No import errors when running simulation

---

### **Step 1.3: Verify JFA Neighbor Data Structure**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/jfa.py`

**Critical:** Confirm that JFA provides neighbor information. Look for:
- `jfa.neighbors[i, k]` - 2D field storing neighbor IDs
- `jfa.fsc[i]` - count of neighbors (already exists)
- Populated during face detection (after JFA step)

**Check in `jfa.py`:**
1. Search for field allocations like `ti.field(dtype=ti.i32, shape=(N, MAX_NEIGHBORS))`
2. Look for where neighbor IDs are written during face detection
3. Verify `MAX_NEIGHBORS` constant exists and is reasonable (e.g., 32)

**If neighbors field does NOT exist:**
- Add to `jfa.py`:
  ```python
  MAX_NEIGHBORS = 32
  neighbors = ti.field(dtype=ti.i32, shape=(N, MAX_NEIGHBORS))
  # Initialize to -1 (invalid)
  ```
- Modify face detection loop to store neighbor IDs:
  ```python
  if face_detected and fsc[i] < MAX_NEIGHBORS:
      neighbors[i, fsc[i]] = j  # Store before incrementing
      fsc[i] += 1
  ```

**Verification:**
- [ ] **1.3.1** Identified neighbor data structure in `jfa.py` (or added it)
- [ ] **1.3.2** Confirmed it's populated after JFA runs
- [ ] **1.3.3** Documented the access pattern: `jfa.neighbors[i, k]` for k in [0, fsc[i])
- [ ] **1.3.4** Test: Print `jfa.neighbors[0, 0:jfa.fsc[0]]` after JFA runs

**Note:** This is the most critical dependency. Without neighbor IDs, equilibration cannot run.

---

## **🧮 Phase 2: Core Kernel Implementation**

### **Step 2.1: Add Hash Function for Stochastic Neighbor Selection**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/dynamics.py`

Add this helper function before the main kernel:

```python
@ti.func
def hash_func(i: ti.i32, frame: ti.i32) -> ti.i32:
    """
    Deterministic hash for stochastic neighbor selection.
    Uses Linear Congruential Generator (LCG) for GPU determinism.
    """
    return (1103515245 * (i + 12345 * frame) + 12345) & 0x7fffffff
```

**Verification:**
- [ ] **2.1.1** Hash function added
- [ ] **2.1.2** Compiles without errors
- [ ] **2.1.3** Test: verify it produces consistent results across runs

---

### **Step 2.2: Implement `equilibrate_pressure` Kernel**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/dynamics.py`

Add the main pressure equilibration kernel:

```python
@ti.kernel
def equilibrate_pressure(
    n: ti.i32,
    frame: ti.i32,
    k: ti.f32,
    P_exp: ti.f32,
    pair_cap: ti.f32,
    max_nei: ti.i32
):
    """
    Volume-conserving pressure equilibration across FSC neighbors.
    
    Uses Jacobi iteration to avoid race conditions:
    1. Compute volume differences across neighbor pairs
    2. Apply capped volume exchange
    3. Update radii from new volumes
    
    Args:
        n: Number of active particles
        frame: Current frame number (for hash)
        k: Diffusion coefficient
        P_exp: Volume exponent (3.0 for 3D)
        pair_cap: Max ΔV as fraction of min(V_i, V_j)
        max_nei: Max neighbors to process per particle
    """
    
    # Step 1: Initialize deltas to zero
    for i in range(n):
        delta_r[i] = 0.0
    
    # Step 2: Compute volume exchanges (Jacobi style)
    for i in range(n):
        # Get current volume (P ∝ r^3)
        Vi = rad[i] ** P_exp
        
        # Get FSC neighbor count
        fsc_count = jfa.fsc[i]
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
            # NOTE: This assumes jfa.neighbors[i, nei_idx] exists
            # Adjust based on actual JFA data structure
            j = jfa.neighbors[i, nei_idx]
            
            if j < 0 or j >= n or i >= j:
                # Skip invalid, self, or already-processed pairs
                continue
            
            # Get neighbor volume
            Vj = rad[j] ** P_exp
            
            # Compute volume difference (pressure gradient proxy)
            delta_V_raw = k * (Vi - Vj)
            
            # Cap by minimum volume (stability)
            V_min = ti.min(Vi, Vj)
            delta_V = ti.max(-pair_cap * V_min, ti.min(pair_cap * V_min, delta_V_raw))
            
            # Exchange volume (local conservation)
            Vi_new = Vi - delta_V
            Vj_new = Vj + delta_V
            
            # Convert back to radius change
            delta_r[i] += (Vi_new ** (1.0 / P_exp)) - rad[i]
            delta_r[j] += (Vj_new ** (1.0 / P_exp)) - rad[j]
    
    # Step 3: Apply deltas with hard clamps
    for i in range(n):
        rad[i] = ti.max(R_MIN, ti.min(R_MAX, rad[i] + delta_r[i]))
```

**Verification:**
- [ ] **2.2.1** Kernel compiles without errors
- [ ] **2.2.2** All Ti variables properly typed
- [ ] **2.2.3** No race conditions (verified Jacobi pattern)

---

## **🔌 Phase 3: Integration into Main Loop**

### **Step 3.1: Wire Equilibration into `run.py`**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/run.py`

Find the main simulation loop, specifically after the FSC controller runs. Add:

```python
# After FSC controller (set_fsc_targets + nudge_radii_adaptive_ema)
# and BEFORE rendering

if PRESSURE_EQUILIBRATION_ENABLED and active_n > 0:
    # Run pressure equilibration
    dynamics.equilibrate_pressure(
        n=active_n,
        frame=frame_count,
        k=PRESSURE_K,
        P_exp=PRESSURE_EXP,
        pair_cap=PRESSURE_PAIR_CAP,
        max_nei=MAX_EQ_NEI
    )
```

**Verification:**
- [ ] **3.1.1** Call added in correct location (after FSC, before render)
- [ ] **3.1.2** Simulation runs without crashes
- [ ] **3.1.3** No syntax errors

---

### **Step 3.2: Add Optional Micro-PBD Safety Net**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/run.py`

After equilibration, add optional overlap check using normalized penetration depth:

```python
if PRESSURE_EQUILIBRATION_ENABLED and EQ_MICRO_PBD_ITERS > 0:
    # Compute normalized overlap: max(0, (Ra+Rb-d)/(Ra+Rb))
    # This scales cleanly as radii evolve
    max_overlap = dynamics.compute_max_normalized_overlap(active_n)
    
    if max_overlap > EQ_OVERLAP_THRESHOLD:
        # Run micro-PBD to resolve new overlaps
        grid.rebuild(pos, rad, active_n)
        
        for _ in range(EQ_MICRO_PBD_ITERS):
            dynamics.project_overlaps(active_n)
        
        # Rebuild grid after micro-PBD
        grid.rebuild(pos, rad, active_n)
```

**Note:** You'll need to add this helper in `dynamics.py`:
```python
@ti.kernel
def compute_max_normalized_overlap(n: ti.i32) -> ti.f32:
    """
    Compute maximum normalized penetration depth across all particles.
    Returns: max(0, (Ra+Rb-d)/(Ra+Rb)) for all pairs in contact.
    """
    max_pen = 0.0
    for i in range(n):
        # Use grid to find neighbors (same as PBD)
        cell_idx = grid.get_cell_index(pos[i])
        # ... iterate over neighbor cells ...
        # For each pair (i, j):
        #   d = distance(pos[i], pos[j])
        #   sum_r = rad[i] + rad[j]
        #   pen = max(0.0, (sum_r - d) / sum_r)
        #   max_pen = max(max_pen, pen)
    return max_pen
```

**Verification:**
- [ ] **3.2.1** Micro-PBD conditional added
- [ ] **3.2.2** Only triggers when normalized overlap exceeds threshold
- [ ] **3.2.3** Grid rebuild happens after micro-PBD
- [ ] **3.2.4** Overlap formula scales with radius changes

---

## **📊 Phase 4: Telemetry & Observability**

### **Step 4.1: Add Pressure Statistics Computation**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/dynamics.py`

Add helper kernel to compute pressure statistics:

```python
@ti.kernel
def compute_pressure_stats(n: ti.i32, P_exp: ti.f32) -> ti.types.vector(5, ti.f32):
    """
    Compute pressure statistics for telemetry.
    
    Uses explicit r³ for volume conservation tracking (independent of P_exp).
    Counts particles hitting radius clamps to detect pressure boundaries.
    
    Returns: [P_mean, P_std, sum_V, clamp_count, spill_count]
    """
    P_sum = 0.0
    P_sq_sum = 0.0
    V_sum = 0.0  # True volume: always r³ in 3D
    clamp_count = 0
    
    for i in range(n):
        # Pressure (can vary with P_exp for experiments)
        P = rad[i] ** P_exp
        P_sum += P
        P_sq_sum += P * P
        
        # Volume: always r³ for conservation tracking
        V_sum += rad[i] ** 3.0
        
        # Count clamped particles (potential pressure sinks/sources)
        if rad[i] <= R_MIN + 1e-6 or rad[i] >= R_MAX - 1e-6:
            clamp_count += 1
    
    P_mean = P_sum / ti.cast(n, ti.f32)
    P_variance = (P_sq_sum / ti.cast(n, ti.f32)) - (P_mean * P_mean)
    P_std = ti.sqrt(ti.max(0.0, P_variance))
    
    # Spill count currently unused (reserved for future clamp-aware redistribution)
    spill_count = 0.0
    
    return ti.Vector([P_mean, P_std, V_sum, ti.cast(clamp_count, ti.f32), spill_count])
```

**Verification:**
- [ ] **4.1.1** Kernel compiles
- [ ] **4.1.2** Returns correct vector type (5 components)
- [ ] **4.1.3** Handles edge cases (n=0, etc.)
- [ ] **4.1.4** ΣV uses `r³` explicitly (not P_exp)

---

### **Step 4.2: Add Telemetry Output to `run.py`**
**Location:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/run.py`

In the measurement frame telemetry block, add:

```python
# After existing FSC telemetry, add:
if PRESSURE_EQUILIBRATION_ENABLED:
    stats = dynamics.compute_pressure_stats(active_n, PRESSURE_EXP)
    P_mean, P_std, sum_V, clamp_count, spill_count = stats
    clamp_pct = 100.0 * clamp_count / active_n if active_n > 0 else 0.0
    
    print(f"[Pressure Equilibration]")
    print(f"  σ(P): {P_std:.6f} | ΣV: {sum_V:.4f} | Clamped: {clamp_pct:.1f}%")
    
    # Optional: track volume drift over time
    # if frame == 10: initial_V = sum_V
    # if frame > 10: print(f"  ΔV: {100*(sum_V/initial_V - 1):.2f}%")
```

**Verification:**
- [ ] **4.2.1** Telemetry prints correctly
- [ ] **4.2.2** Values are reasonable
- [ ] **4.2.3** No performance impact from telemetry
- [ ] **4.2.4** ΣV remains roughly constant (within 5% after startup)

---

## **🧪 Phase 5: Testing & Validation**

### **Step 5.1: Conservative Startup Test**
**Goal:** Verify system runs without crashes

**Test Protocol:**
1. Set conservative parameters:
   ```python
   PRESSURE_K = 0.01
   MAX_EQ_NEI = 6
   ```
2. Start simulation
3. Let run for 50+ frames
4. Observe:
   - [ ] No crashes
   - [ ] `σ(P)` is computed
   - [ ] `ΣV` stays roughly constant
   - [ ] No runaway growth/shrink

---

### **Step 5.2: Equilibration Effect Test**
**Goal:** Verify pressure variance decreases

**Test Protocol:**
1. Note initial `σ(P)` value at frame 10
2. Increase `PRESSURE_K` to `0.03`
3. Run for 100 frames
4. Compare `σ(P)` at frame 110
5. Verify:
   - [ ] `σ(P)` decreased by at least 10%
   - [ ] `ΣV` changed by less than 5% (volume conservation)
   - [ ] Visual: particles look more uniform in size

---

### **Step 5.3: Causal Response Test**
**Goal:** Verify equilibration responds to FSC changes

**Test Protocol:**
1. Let system stabilize (100+ frames)
2. Move FSC sliders to create pressure imbalance:
   - Set `FSC_LOW = 15` (force shrinking)
3. Observe next 30 frames:
   - [ ] `σ(P)` increases initially (due to FSC forcing changes)
   - [ ] `σ(P)` then decreases (equilibration working)
   - [ ] System reaches new stable state
4. Reset FSC band to normal
5. Verify:
   - [ ] System recovers
   - [ ] No oscillations

---

### **Step 5.4: Neighbor Budget Test**
**Goal:** Verify budgeting doesn't break equilibration

**Test Protocol:**
1. Set `MAX_EQ_NEI = 3` (very restrictive)
2. Run for 100 frames
3. Verify:
   - [ ] Still sees `σ(P)` decrease (slower is OK)
   - [ ] No crashes or stuck states
4. Increase to `MAX_EQ_NEI = 15`
5. Verify:
   - [ ] Faster convergence
   - [ ] No instabilities

---

### **Step 5.5: Stability Stress Test**
**Goal:** Verify no blow-ups at high diffusion rates

**Test Protocol:**
1. Gradually increase `PRESSURE_K`:
   - Start: 0.03
   - Increase by 0.01 every 50 frames
   - Stop at 0.10 or when instability appears
2. Monitor:
   - [ ] Radius stays within bounds
   - [ ] No NaN or Inf values
   - [ ] `ΣV` variance stays < 10%
3. Document maximum stable `PRESSURE_K` value

---

## **📈 Phase 6: Tuning & Optimization**

### **Step 6.1: Find Optimal `PRESSURE_K`**
**Goal:** Balance speed vs stability

**Tuning Protocol:**
1. Start at `PRESSURE_K = 0.03`
2. Measure time to reduce `σ(P)` by 50% from initial value
3. Increase `K` by 0.01 increments
4. Record:
   - Convergence time
   - Stability (oscillations?)
   - Visual quality
5. Choose `K` where:
   - [ ] Convergence is fast enough (subjective)
   - [ ] No visible oscillations
   - [ ] `ΣV` variance < 5%

**Recommended:** `K = 0.03 - 0.05`

---

### **Step 6.2: Optimize Neighbor Budget**
**Goal:** Minimize computation while maintaining quality

**Tuning Protocol:**
1. Start at `MAX_EQ_NEI = 10`
2. Measure `σ(P)` convergence rate
3. Reduce to 8, then 6, then 4
4. For each value, compare:
   - Convergence time
   - Final `σ(P)` value
   - Frame time (from PERF output)
5. Choose smallest `MAX_EQ_NEI` where:
   - [ ] Convergence time < 2× baseline
   - [ ] Final `σ(P)` within 10% of full neighbor version

**Recommended:** `MAX_EQ_NEI = 8 - 10`

---

### **Step 6.3: Adjust Pair Cap if Needed**
**Goal:** Prevent hot spots from dominating

**Tuning Protocol:**
1. Watch for individual particles with rapid size changes
2. If particles flicker or oscillate:
   - Reduce `PRESSURE_PAIR_CAP` from 0.02 to 0.01
3. If equilibration is too slow:
   - Increase to 0.03
4. Choose value where:
   - [ ] No visible flicker
   - [ ] Smooth size changes
   - [ ] Reasonable convergence speed

**Recommended:** `PRESSURE_PAIR_CAP = 0.02`

---

## **✅ Phase 7: Final Validation**

### **Step 7.1: Full System Integration Check**

Run complete test with all systems active:
- [ ] FSC controller running
- [ ] Pressure equilibration running
- [ ] PBD running
- [ ] No crashes for 300+ frames
- [ ] Telemetry shows:
  - `σ(P)` decreasing
  - `FSC μ` in band
  - `ΣV` stable
  - Clamp % < 5%

---

### **Step 7.2: Visual Quality Assessment**

Subjective checks:
- [ ] Particles look uniformly sized (within FSC-driven variation)
- [ ] No flickering or oscillations
- [ ] Smooth, organic motion
- [ ] Visually convincing foam behavior

---

### **Step 7.3: Performance Benchmark**

Compare before/after equilibration:
- [ ] Measure FPS before: ________
- [ ] Measure FPS after: ________
- [ ] Calculate overhead: _______%
- [ ] Verify overhead < 15%

---

## **📝 Phase 8: Documentation & Cleanup**

### **Step 8.1: Document Final Configuration**

Record optimal parameters found:
```python
PRESSURE_K = _______
MAX_EQ_NEI = _______
PRESSURE_PAIR_CAP = _______
```

---

### **Step 8.2: Add Comments to Code**

Ensure all new code has:
- [ ] Docstrings on kernels
- [ ] Inline comments explaining key steps
- [ ] References to this blueprint where relevant

---

### **Step 8.3: Git Commit**

Commit the complete implementation:
```bash
git add config.py dynamics.py run.py jfa.py
git commit -m "feat: Add FSC pressure equilibration system

- Implements volume-conserving pressure diffusion
- Adds Jacobi-based race-free updates
- Includes budgeted neighbor processing
- Adds pressure telemetry (σ(P), ΣV, clamps)
- Optimal params: K=X, MAX_NEI=Y, PAIR_CAP=Z"
```

- [ ] Committed to git
- [ ] Pushed to remote (if applicable)

---

## **🎯 Success Criteria**

The implementation is complete and successful when:

✅ **Functional:**
- System runs stably for 500+ frames
- `σ(P)` decreases over time
- `ΣV` variance < 5%

✅ **Observable:**
- Telemetry shows clear pressure equilibration
- Visual appearance improves (more uniform)

✅ **Performant:**
- Overhead < 15%
- No frame drops or stuttering

✅ **Documented:**
- Code commented
- Optimal parameters recorded
- Git committed

---

## **🐛 Troubleshooting Guide**

### Issue: `σ(P)` not decreasing
**Possible causes:**
- `PRESSURE_K` too low → increase to 0.05
- Neighbor structure not populated → verify JFA integration
- FSC changes overwhelming equilibration → reduce `GROWTH_PCT`

### Issue: Oscillations or instability
**Possible causes:**
- `PRESSURE_K` too high → reduce to 0.01
- `PRESSURE_PAIR_CAP` too large → reduce to 0.01
- Race condition (unlikely if Jacobi) → verify `delta_r` pattern

### Issue: `ΣV` not conserved
**Possible causes:**
- Clamping at bounds → check clamp % (should be < 5%)
- Volume calculation error → verify `P_exp = 3.0`
- Pair processing asymmetry → check `i >= j` condition

### Issue: Performance degradation
**Possible causes:**
- `MAX_EQ_NEI` too high → reduce to 6
- Micro-PBD running too often → raise `EQ_OVERLAP_THRESHOLD`
- Telemetry overhead → reduce print frequency

---

## **📚 Reference**

### Key Equations
1. **Volume:** `V = r³` (in 3D)
2. **Pressure proxy:** `P = r³`
3. **Volume exchange:** `ΔV = k(Vi - Vj)`
4. **Capped exchange:** `ΔV = clamp(ΔV, -c·min(Vi,Vj), +c·min(Vi,Vj))`
5. **Radius update:** `r_new = (V_new)^(1/3)`

### Time Complexity
- Per-frame cost: `O(N · MAX_EQ_NEI)`
- With `MAX_EQ_NEI = 10`: effectively `O(N)` with small constant

---

## **🎨 Design Philosophy & Architecture Notes**

### **Two-Channel Control System**

The complete system operates on two complementary timescales:

| Channel | Timescale | Goal | Method |
|---------|-----------|------|--------|
| **FSC Controller** | Slow (EMA, ~30 frames) | Target topology (connectivity) | Global FSC band enforcement |
| **Pressure Equilibrator** | Fast (per-frame) | Local mechanical consistency | Pressure diffusion across neighbors |

**Key Insight:** The pressure equilibrator is the **primary driver** of foam dynamics. The FSC controller can operate with:
- Wide band (neutral, permissive)
- Light gain (gentle guidance)
- Or even disabled temporarily

This means:
1. **Pressure equilibration works independently** - it will balance local pressures even without FSC changes
2. **FSC provides long-term topology goals** - optional guidance toward a target connectivity
3. **No forced target face number** - the system finds its own equilibrium based on local pressure balance

### **Combustion-Engine Analogy**

If you later want dynamic, cycling behavior ("breathing foam"):
- **Modulate `PRESSURE_K`** temporally (e.g., sine wave)
- **Modulate FSC band width** (expand/contract target)
- **Keep mechanics unchanged** - the equilibration + FSC kernels remain the same

Example:
```python
# Breathing foam (optional, future)
cycle_phase = (frame % 120) / 120.0  # 0..1
PRESSURE_K_LIVE = PRESSURE_K * (0.5 + 0.5 * sin(2π * cycle_phase))
```

### **Numeric Stability**

**Kernel precision:** `f32` throughout (fast, GPU-friendly)
- Volume exchange is stable with capping
- Jacobi pattern prevents race conditions
- Clamping at bounds provides natural reservoirs

**Telemetry precision:** 
- Use `f32` for real-time display
- If running very long simulations (1000+ frames) and noticing ΣV drift:
  - Consider `f64` accumulation for telemetry stats only
  - Do NOT change kernel precision (performance hit)
  - Use Python-side accumulation: `sum_V_f64 = sum(rad_np**3)`

**Conservation expectations:**
- ΣV should remain within **±5%** after startup (frames 50+)
- Clamped particles act as pressure sinks/sources (this is intentional)
- If clamp% > 10%, consider widening `R_MIN`/`R_MAX` bounds

---

---

## **📇 Quick Reference Card**

### **Critical Path to First Run**
1. ✅ Verify `jfa.neighbors[i, k]` exists (Phase 1.3) ⭐ **BLOCKER**
2. Add constants to `config.py` (Phase 1.1)
3. Add `delta_r` field to `dynamics.py` (Phase 1.2)
4. Implement `equilibrate_pressure()` kernel (Phase 2.2)
5. Wire into `run.py` main loop (Phase 3.1)
6. Add telemetry (Phase 4.2)
7. Run conservative startup test (Phase 5.1)

### **Default Parameters (Start Here)**
```python
PRESSURE_K = 0.03              # Conservative diffusion rate
MAX_EQ_NEI = 10                # Neighbor budget
PRESSURE_PAIR_CAP = 0.02       # Per-pair safety cap
EQ_OVERLAP_THRESHOLD = 0.05    # Trigger micro-PBD
```

### **Success Metrics**
- ✅ System runs 500+ frames without crash
- ✅ `σ(P)` decreases over time (pressure variance drops)
- ✅ `ΣV` stable within ±5% (volume conserved)
- ✅ Clamp% < 5% (most particles within bounds)
- ✅ Visual: more uniform particle sizes

---

**Document Version:** 2.1  
**Last Updated:** 2025-10-29  
**Status:** ✅ Ready for Implementation (with refinements integrated)

