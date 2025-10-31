# REBUILD BLUEPRINT - FSC Manual Pump (Clean Room Design)

**Date:** 2025-10-28  
**Purpose:** Clean implementation of FSC-driven size control based on 50+ prompts of debugging lessons

---

## Executive Summary

We've debugged the system extensively and proven:
- ✅ Grid neighbor counting CAN be accurate (0% miss rate achieved)
- ✅ Degree-based control WORKS (slow but functional)
- ✅ JFA runs and computes Voronoi diagrams
- ❌ FSC control NEVER actually ran (blocked by trust gates)
- ❌ "Manual mode" was an illusion (sliders moved, control didn't execute)

**Root cause:** Trust gates, phase logic, and EMA smoothing blocked direct FSC→radius control, even in "manual" mode.

**Solution:** Three-phase rebuild, each phase fully verified before moving to the next.

---

## Lessons Learned (The Hard Way)

### 1. **Trust Gates = Silent Kill Switch**
```python
# What we had (WRONG):
if asymmetry_pct < 10% and score_mu > 0.5 and pairs > 0.5*N:
    apply_fsc_control()  # This NEVER executed
else:
    # FSC sliders move, GUI updates, but nothing happens
```

**Lesson:** "Manual mode" must mean MANUAL - no hidden validation preventing control.

### 2. **Adjustment Cadence Hides Changes**
```python
# Tiny nudges over 30 frames = invisible
for frame in range(30):
    rad[i] += (target - rad[i]) / 30  # 3.3% change per frame
```

**Lesson:** When testing control, apply changes IMMEDIATELY (1 frame) so causality is obvious.

### 3. **Telemetry Can Lie**
- GUI percentages were calculated from **slider positions**, not actual particle states
- Console showed "action: grow=X shrink=Y" from **degree controller**, not FSC
- Mean degree looked wrong because **CONTACT_TOL was too tight** (0.015 vs needed 0.035)

**Lesson:** Trust code paths, not derived metrics. Trace execution to verify control actually runs.

### 4. **Two Controllers = Confusion**
- Degree control (grid-based, working)
- FSC control (JFA-based, gated off)
- Never clear which one was active

**Lesson:** One controller at a time. Prove each works independently before combining.

### 5. **Grid Works When Tuned Right**
After fixing:
- `CELL_SIZE = r_cut_typical` (not too fine)
- `reach = ceil(r_cut / actual_cell_size)` (no hard cap)
- `CONTACT_TOL = 0.035` (covers measured miss margin of 2.7%)
- Squared distance predicate everywhere
- `eps_hash = 1e-7` for boundary stability

Result: **0.0% miss rate on freeze-frame validation**

**Lesson:** Grid is debugged and trustworthy (when configured properly).

### 6. **JFA Runs But FSC Unreliable**
- JFA executes, generates Voronoi diagrams
- But: `FSC μ=0.0` consistently (no neighbors detected)
- Asymmetry high, score normalization wrong
- Likely issues: `MIN_FACE_VOXELS` threshold, `EXPECTED_FACE_SCALE`, resolution too coarse

**Lesson:** JFA needs calibration before FSC can be trusted for control.

---

## Three-Phase Rebuild Plan

### **Phase 1: Degree-Only Baseline (VERIFY CURRENT STATE)**

**Goal:** Confirm we have a stable, working degree-based system.

**What to keep:**
- Grid neighbor counting (with all fixes: CONTACT_TOL=0.035, squared predicates, eps_hash)
- Degree-based control (DEG_LOW, DEG_HIGH thresholds)
- PBD separation
- Rendering

**What to REMOVE temporarily:**
- All JFA/FSC code and imports
- All phase management (startup_phase, gamma_blend, etc.)
- All FSC sliders and telemetry
- Hybrid controller logic

**Configuration for stability:**
```python
# config.py
N = 10000
R_START_MANUAL = 0.0045  # Ensures initial contact
R_MIN_MANUAL = 0.002     # Prevents collapse
DEG_LOW = 4              # Narrow band for visible action
DEG_HIGH = 6
GAIN_GROW = 0.02         # Reduced for stability (was 0.05)
GAIN_SHRINK = 0.02
ADJUSTMENT_FRAMES = 30   # Keep existing cadence for now
CONTACT_TOL = 0.035      # Proven to work
```

**Success criteria:**
1. ✅ Grid validation shows 0% miss rate
2. ✅ Mean degree oscillates gently around 5 (not wild swings 0.07 → 7.29)
3. ✅ Radius mean changes visibly over 100 frames (print every 10 frames)
4. ✅ Export shows increasing # of unique radii
5. ✅ No indentation errors, runs for 500+ frames without crash

**Deliverable:** Clean `run_degree_only.py` (or cleaned `run.py` with JFA removed)

---

### **Phase 2: FSC Manual Pump (MINIMAL, NO GATES)**

**Goal:** Add ONLY direct FSC→radius control. Prove sliders drive the simulation.

**What to add:**
- JFA computation (but keep existing proven JFA code from `jfa.py`)
- FSC sliders (gui_fsc_low, gui_fsc_high)
- Direct FSC pump kernel (NO gates, NO phases, NO EMA on application)

**What NOT to add yet:**
- Trust gates (asymmetry/score/pairs checks)
- Phase management
- Auto-calibration
- Hybrid blending
- Any "if JFA looks bad, skip control" logic

**New kernel (minimal FSC pump):**
```python
@ti.kernel
def apply_fsc_direct(
    n: ti.i32,
    fsc_low: ti.f32,
    fsc_high: ti.f32,
    r_min: ti.f32,
    r_max: ti.f32
):
    """
    Direct FSC control: no gates, no smoothing, immediate application.
    
    FSC < fsc_low  → grow by 5%
    FSC > fsc_high → shrink by 5%
    Otherwise      → hold
    """
    for i in range(n):
        fsc_val = jfa.fsc_ema[i]  # Or raw fsc[i] if EMA not needed
        r_old = rad[i]
        
        # Simple thresholding
        if fsc_val < fsc_low:
            r_new = r_old * 1.05
        elif fsc_val > fsc_high:
            r_new = r_old * 0.95
        else:
            r_new = r_old
        
        # Clamp and write
        rad[i] = ti.max(r_min, ti.min(r_max, r_new))
```

**Control loop (simplified):**
```python
# Every measurement frame (e.g., every 30 frames):
if frame % MEASUREMENT_CADENCE == 0:
    # 1. Run JFA
    jfa.compute_power_diagram(pos, rad, active_n)
    jfa.accumulate_faces(...)
    jfa.compute_fsc(active_n)
    
    # 2. Apply FSC control IMMEDIATELY (no adjustment phase, no nudging)
    apply_fsc_direct(
        active_n,
        float(gui_fsc_low),   # Read directly from sliders
        float(gui_fsc_high),
        float(R_MIN),
        float(R_MAX)
    )
    
    # 3. Print telemetry
    fsc_np = jfa.fsc_ema.to_numpy()[:active_n]
    fsc_mean = fsc_np.mean()
    in_band = np.sum((fsc_np >= gui_fsc_low) & (fsc_np <= gui_fsc_high))
    print(f"[FSC] μ={fsc_mean:.1f} | band=[{gui_fsc_low},{gui_fsc_high}] "
          f"in-band={100*in_band/active_n:.1f}%")
    
    # 4. Export radii to prove change
    if frame % 60 == 0:
        r_np = rad.to_numpy()[:active_n]
        print(f"[PROOF] r: μ={r_np.mean():.6f} unique={len(np.unique(r_np))}")
```

**JFA configuration (start conservative):**
```python
# config.py
JFA_ENABLED = True
JFA_RES_MIN = 96          # Proven to run fast
JFA_RES_MAX = 128
JFA_RUN_INTERVAL = 1      # Run every measurement frame (no skipping)
MIN_FACE_VOXELS = 3       # Lower threshold (was 6, may be too strict)
EXPECTED_FACE_SCALE = 300 # From previous tuning
```

**Success criteria:**
1. ✅ Move FSC_LOW slider → console shows "in-band %" change within 1-2 measurement cycles
2. ✅ Set FSC_HIGH below current mean → radii shrink (r_mean decreases)
3. ✅ Set FSC_LOW above current mean → radii grow (r_mean increases)
4. ✅ Export shows r_mean changing by 5-10% over 5 measurement cycles
5. ✅ JFA reports FSC > 0 (not 0.0) - if still 0, JFA calibration needed first

**If FSC still reports 0.0:**
- This is a JFA calibration issue, NOT a control issue
- Options:
  - Lower `MIN_FACE_VOXELS` to 1-2 (very permissive)
  - Increase `JFA_RES_MIN` to 128 (more voxels = more faces)
  - Check `POWER_BETA` (try 0.8 or 1.2)
  - Add diagnostic: print raw face counts before thresholding
- **Do NOT add trust gates** - fix JFA or use degree as fallback

**Deliverable:** `run_fsc_manual.py` with working direct FSC control

---

### **Phase 3: Refinement (ONLY AFTER PHASE 2 WORKS)**

**Goal:** Add stability features without breaking direct control.

**What to add (one at a time, verify after each):**

1. **Smoothing on application** (optional)
   ```python
   # Instead of r*=1.05 instantly, nudge over N frames
   rad_target[i] = r_old * 1.05
   # Then use existing nudge_radii_to_targets() over ADJUSTMENT_FRAMES
   ```
   - Only add if instant jumps cause PBD problems

2. **Asymmetric gains** (optional)
   ```python
   if fsc_val < fsc_low:
       r_new = r_old * 1.06  # Grow faster
   elif fsc_val > fsc_high:
       r_new = r_old * 0.97  # Shrink slower
   ```
   - Only add if system collapses too fast

3. **Trust gates** (optional, last resort)
   ```python
   # Only apply FSC if JFA looks healthy
   if asymmetry_pct < 5% and mean_score > 0.8:
       apply_fsc_direct(...)
   else:
       # Fallback to degree control, but PRINT WARNING
       print("[WARN] JFA unhealthy, using degree fallback")
   ```
   - Only add if bad FSC data causes instability
   - Must print warning so user knows control switched

4. **Dual control mode** (optional)
   ```python
   # Use both degree AND FSC for different purposes
   # E.g., degree for coarse adjustment, FSC for fine-tuning
   ```
   - Only add if single controller insufficient

**Success criteria:**
- Each addition verified independently
- No silent behavior changes
- Sliders always have visible effect (even if dampened)

---

## Implementation Checklist

### Phase 1: Degree Baseline
- [ ] Remove all JFA imports and calls from `run.py`
- [ ] Remove FSC sliders from GUI
- [ ] Set `DEG_LOW=4, DEG_HIGH=6, GAIN=0.02`
- [ ] Run for 500 frames, verify stable oscillation around deg=5
- [ ] Verify grid validation shows 0% miss
- [ ] Export radii every 100 frames, confirm mean is changing

### Phase 2: FSC Direct
- [ ] Add back JFA import and initialization
- [ ] Create `apply_fsc_direct()` kernel (no gates)
- [ ] Add FSC sliders to GUI (separate from degree sliders)
- [ ] Wire sliders directly to control (read gui values every measurement)
- [ ] Remove any "if asymmetry > X, skip FSC" logic
- [ ] Print `[FSC]` line every measurement with mean, band, in-band%
- [ ] Test: move sliders, watch console for immediate response
- [ ] Export radii, verify r_mean responds to slider changes

### Phase 3: Refinement
- [ ] (Only proceed if Phase 2 works perfectly)
- [ ] Add ONE feature at a time
- [ ] Verify after each addition
- [ ] Keep diagnostic prints for all gating decisions

---

## Key Design Principles (Learned the Hard Way)

1. **Explicit over implicit**
   - No hidden phase transitions
   - No silent fallbacks
   - Print what the system is actually doing

2. **One thing at a time**
   - Prove degree works before adding FSC
   - Prove FSC works before adding gates
   - Never debug two things simultaneously

3. **Immediate feedback**
   - When testing, use 1-frame application
   - Add smoothing AFTER causality is proven

4. **Trust code, not metrics**
   - Trace execution paths
   - Add `print("FSC control executing")` in kernel
   - Don't assume telemetry is accurate

5. **Fail loudly**
   - If JFA fails, print warning (don't silently skip)
   - If gates block control, print warning
   - User should always know why nothing is happening

---

## Expected Outcomes

### Phase 1 (Degree only)
```
[Measure] frame=100 | deg: μ=5.2 [2,9] | r_mean: 0.00450 → 0.00455
[Measure] frame=200 | deg: μ=4.8 [1,8] | r_mean: 0.00455 → 0.00452
[Measure] frame=300 | deg: μ=5.1 [2,9] | r_mean: 0.00452 → 0.00458
```
Stable, gentle oscillation. No wild swings.

### Phase 2 (FSC direct)
```
[Measure] frame=100
  [Degree] μ=5.2 (not used, just monitoring)
  [FSC] μ=12.3 | band=[10,15] in-band=67%
  [PROOF] r: μ=0.004550 unique=89

User moves FSC_HIGH slider from 15 → 11

[Measure] frame=130
  [FSC] μ=12.3 | band=[10,11] in-band=22%  ← most now "high"
  [PROOF] r: μ=0.004321 unique=112  ← SHRUNK by 5%
```
Sliders → immediate effect on mean radius within 1-2 cycles.

### Phase 3 (Refined)
Similar to Phase 2, but smoother (if smoothing added).

---

## What This Fixes

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Sliders don't work | Trust gates block FSC | Remove gates in Phase 2 |
| Changes invisible | 30-frame smoothing | Test with immediate application first |
| GUI percentages fake | Calculated from sliders | Fixed (now uses actual particle states) |
| Two controllers fighting | Both active, unclear priority | Phase 1: degree only. Phase 2: FSC only |
| Wild oscillations | Gains too high (0.05) | Reduce to 0.02 in Phase 1 |
| JFA returns FSC=0 | Thresholds/resolution wrong | Calibrate in Phase 2 (or use degree fallback) |
| Grid miss rate | Predicate mismatches, reach cap | Fixed (proven 0% on freeze-frame) |

---

## Verification Tests

### Test 1: Degree Stability
```
1. Run Phase 1 for 500 frames
2. Every 100 frames, check: 3 < deg_mean < 7
3. r_mean should change by <10% over 500 frames (stable)
4. No crashes, no wild swings
```

### Test 2: FSC Causality
```
1. Run Phase 2, let stabilize
2. Note current r_mean (e.g., 0.00450)
3. Set FSC_HIGH = current_fsc_mean - 2
4. Wait 2 measurement cycles (~60 frames)
5. Check r_mean decreased by ~5-10%
6. Repeat with FSC_LOW to test growth
```

### Test 3: Slider Range
```
1. Set FSC band very wide [5, 25] → most in-band → r_mean stable
2. Set FSC band very narrow [12, 13] → most out-of-band → r_mean changes fast
3. Verify in-band % tracks with slider positions
```

---

## Diagnostic Checklist (When Things Go Wrong)

If **FSC still reports 0.0:**
- [ ] Check `MIN_FACE_VOXELS` (try 1-2)
- [ ] Check JFA resolution (try 128³)
- [ ] Print raw face counts before thresholding
- [ ] Check `POWER_BETA` (try 0.8, 1.0, 1.2)

If **r_mean doesn't change:**
- [ ] Print "FSC CONTROL EXECUTING" inside kernel
- [ ] Check if gates are blocking (search for asymmetry/score checks)
- [ ] Verify sliders are being read (print gui_fsc_low in loop)
- [ ] Check if radii are clamped at R_MIN or R_MAX

If **wild oscillations:**
- [ ] Reduce gains (0.02 → 0.01)
- [ ] Increase measurement cadence (30 → 60 frames between pulses)
- [ ] Check if band is too narrow (widen it)

If **grid miss rate returns:**
- [ ] Verify `CONTACT_TOL = 0.035`
- [ ] Check `CELL_SIZE` (should be ~r_cut_typical)
- [ ] Run freeze-frame diagnostic (press 'F')
- [ ] Check reach calculation (should be >= r_cut/cell_size)

---

## Code Skeleton for Phase 2

```python
# run_fsc_manual.py (simplified structure)

import taichi as ti
import numpy as np
from config import *
from grid import *
from dynamics import *
import jfa

ti.init(arch=ti.gpu)
jfa.init_jfa()

# ... allocate fields ...

# GUI
window = ti.ui.Window("FSC Manual Pump", (1920, 1080))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# FSC sliders (separate from degree sliders if keeping both)
gui_fsc_low = 8.0
gui_fsc_high = 16.0

MEASUREMENT_CADENCE = 30  # Run FSC every 30 frames

while window.running:
    frame += 1
    
    # === 1. PBD (every frame) ===
    rebuild_grid()
    project_overlaps(...)
    
    # === 2. FSC Control (every MEASUREMENT_CADENCE frames) ===
    if frame % MEASUREMENT_CADENCE == 0:
        # 2A. Run JFA
        jfa.compute_power_diagram(pos, rad, active_n)
        jfa.accumulate_faces(...)
        jfa.compute_fsc(active_n)
        
        # 2B. Apply FSC directly (NO GATES)
        apply_fsc_direct(
            active_n,
            float(gui_fsc_low),
            float(gui_fsc_high),
            float(R_MIN),
            float(R_MAX)
        )
        
        # 2C. Telemetry
        fsc_np = jfa.fsc_ema.to_numpy()[:active_n]
        fsc_mean = fsc_np.mean()
        in_band = np.sum((fsc_np >= gui_fsc_low) & (fsc_np <= gui_fsc_high))
        print(f"[FSC] frame={frame} μ={fsc_mean:.1f} "
              f"band=[{gui_fsc_low:.0f},{gui_fsc_high:.0f}] "
              f"in-band={100*in_band/active_n:.1f}%")
        
        if frame % 60 == 0:
            r_np = rad.to_numpy()[:active_n]
            print(f"[PROOF] r: μ={r_np.mean():.6f} σ={r_np.std():.6f} "
                  f"unique={len(np.unique(r_np))}")
    
    # === 3. Render ===
    # ... existing rendering code ...
    
    # === 4. GUI ===
    with window.GUI.sub_window("FSC Control", 0.01, 0.01, 0.3, 0.4):
        window.GUI.text(f"Frame: {frame}")
        
        # FSC sliders
        gui_fsc_low = window.GUI.slider_float("FSC Min (grow)", 
                                               gui_fsc_low, 4, 20)
        gui_fsc_high = window.GUI.slider_float("FSC Max (shrink)", 
                                                gui_fsc_high, 8, 30)
        
        # Stats (actual FSC, not slider-derived)
        fsc_np = jfa.fsc_ema.to_numpy()[:active_n]
        fsc_mean = fsc_np.mean()
        window.GUI.text(f"FSC mean: {fsc_mean:.1f}")
        
        # Radius stats
        r_np = rad.to_numpy()[:active_n]
        window.GUI.text(f"Radius: μ={r_np.mean():.4f} σ={r_np.std():.4f}")
    
    window.show()
```

---

## Final Notes

**Why this will work:**
1. We've proven each piece independently (grid, degree control, JFA execution)
2. We know the exact failure mode (trust gates blocking FSC)
3. We have a phased approach with clear verification at each step

**Why it failed before:**
- Tried to do everything at once
- Hidden complexity (phases, gates, blending)
- No verification between changes
- Telemetry was misleading

**The new rule:**
> If you can't verify it works in isolation, don't combine it.

---

**Next Session Plan:**
1. Review this blueprint together
2. Decide: start from Phase 1 (verify baseline) or jump to Phase 2 (if confident in current degree system)
3. Implement ONE phase completely before moving to next
4. Verify with sliders, not just console output

**Key success metric:**
> User moves slider → particles visibly respond within 30 frames. No guessing, no "maybe it's working slowly."

---

*This blueprint represents 50+ prompts of hard-won debugging knowledge. The mistakes are documented so we don't repeat them.*

