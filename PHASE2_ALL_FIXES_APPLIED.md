# Phase 2: All 10 Surgical Fixes Applied

## Problem Observed
Particles were **blowing up** on first run → pump IS working, but needs tuning and has interference from other systems.

---

## All Fixes Applied

### ✅ **Fix 1: Kill ALL JFA Gates in Manual Mode**
**Location:** `run.py` line ~974

**Before:**
```python
jfa_should_run = (JFA_ENABLED and adjustment_timer[None] <= 0 and
                 ((FSC_MANUAL_MODE and True) or ...))
```

**After:**
```python
if JFA_ENABLED and FSC_MANUAL_MODE:
    jfa_should_run = True  # Always run, no conditions, no gates
elif JFA_ENABLED and adjustment_timer[None] <= 0:
    jfa_should_run = (jfa_measurement_counter % JFA_RUN_INTERVAL == 0)
else:
    jfa_should_run = False
```

**Why:** Ensures JFA runs EVERY measurement frame in manual mode with zero gates (no `last_fsc_mean >= 2.0` checks, no cadence limits).

---

### ✅ **Fix 2: Degree Controller is Already Guarded**
**Location:** `run.py` line ~1075, ~1121

The structure is:
```python
if JFA_ENABLED and FSC_MANUAL_MODE:
    # Manual pump (runs when manual mode on)
elif JFA_ENABLED:
    # Hybrid controller (only runs when manual mode OFF)
else:
    # Fallback degree controller
```

**Status:** Already correctly structured - degree controller doesn't run in manual mode.

---

### ✅ **Fix 3: Use Raw FSC (Not EMA) for Instant Response**
**Location:** `run.py` line ~510 (pump kernel), line ~1091 (telemetry)

**Before:**
```python
f = jfa.fsc_ema[i]  # Smoothed over dozens of frames
```

**After:**
```python
f = jfa.fsc[i]  # Raw FSC for instant response
```

**Why:** EMA smears changes over 10-30 frames. Raw FSC shows immediate effect of slider changes.

---

### ✅ **Fix 4: Removed Trust Gates Around Manual Mode**
**Status:** No `fsc_trusted` gates found in manual mode execution path. Manual pump runs unconditionally when `FSC_MANUAL_MODE=True`.

---

### ✅ **Fix 5: Adjustment Timer Stays at 0 in Manual Mode**
**Location:** `run.py` line ~1253

```python
if not (JFA_ENABLED and FSC_MANUAL_MODE):
    adjustment_timer[None] = adjustment_frames_rt[None]  # Normal mode: start adjustment
else:
    adjustment_timer[None] = 0  # Manual mode: stay at 0 (measure every frame)
```

**Why:** Prevents 30-frame nudging cycle from hiding immediate changes.

---

### ✅ **Fix 6: Reduced Pump Rate + Widened Bounds**

#### Pump Rate (run.py line ~519):
**Before:** ±5% per measurement
**After:** ±2% per measurement

```python
if f < fsc_low:
    r_new = r0 * 1.02  # Grow by 2% (was 1.05)
elif f > fsc_high:
    r_new = r0 * 0.98  # Shrink by 2% (was 0.95)
```

#### Bounds (config.py line ~50):
**Before:**
```python
R_MIN_MANUAL = 0.002
R_MAX_MANUAL = 0.0500
```

**After:**
```python
R_MIN_MANUAL = 0.0001  # Very permissive for testing
R_MAX_MANUAL = 0.1000  # Very wide for testing
```

**Why:** 
- Smaller steps prevent explosion
- Wide bounds prevent clipping (a 2% step won't get clamped to 0%)
- No shrink floor applied in manual pump kernel

---

### ✅ **Fix 7: Three-Point Telemetry (Catch Late Overwrites)**
**Location:** `run.py` lines ~1085, ~1117, ~1259

```python
# TELEMETRY A: Before JFA
rad_before_jfa = rad.to_numpy()[:active_n]
r_mean_before_jfa = float(rad_before_jfa.mean())
print(f"[TELEMETRY A] r_mean BEFORE JFA: {r_mean_before_jfa:.6f}")

# ... JFA runs ...

# TELEMETRY B: Right after pump write
rad_after_pump = rad.to_numpy()[:active_n]
r_mean_after_pump = float(rad_after_pump.mean())
print(f"[TELEMETRY B] r_mean RIGHT AFTER PUMP: {r_mean_after_pump:.6f}")

# ... rest of measurement block ...

# TELEMETRY C: End of measurement
rad_end_of_frame = rad.to_numpy()[:active_n]
r_mean_end_of_frame = float(rad_end_of_frame.mean())
print(f"[TELEMETRY C] r_mean END OF MEASUREMENT: {r_mean_end_of_frame:.6f}")

# If B and C differ, something overwrote rad[]!
if abs(r_mean_end_of_frame - r_mean_after_pump) > 1e-6:
    print(f"⚠️ WARNING: Something overwrote rad[] after the pump!")
```

**Why:** 
- **A→B:** Confirms FSC exists and pump executes
- **B→C:** Catches late-frame overwrites (PBD, nudger, etc.)
- If **B changes** but **C reverts**, we've found the culprit

---

### ✅ **Fix 8: Set Power Beta Every Measurement in Manual Mode**
**Location:** `run.py` line ~1088

```python
# FIX 8: Set power beta to 1.0 for standard power diagram EVERY measurement
jfa.power_beta_current[None] = 1.0
```

**Why:** Prevents beta drift from hybrid controller's ramp logic. Standard power diagram (β=1.0) gives predictable FSC.

---

### ✅ **Fix 9: GUI Shows Degree, Pump Uses FSC**
**Status:** This is expected behavior - documented only, no code change needed.

- GUI degree histogram won't move immediately when FSC bands change
- **Watch `r_mean` and FSC telemetry** to validate pump causality
- Degree histogram will eventually adjust as radii stabilize

---

### ✅ **Fix 10: No XPBD Radius Updates Found**
**Status:** Verified no `update_radii_xpbd()` or `update_radii_with_actions()` calls in the measurement path.

The only place radii change in manual mode is the pump kernel itself.

---

## Expected Telemetry Output (Every Measurement)

```
[TELEMETRY A] r_mean BEFORE JFA: 0.004500
[JFA Step 4] FSC μ=12.3 [6,18] | EMA μ=12.1
...
[TELEMETRY B] r_mean RIGHT AFTER PUMP: 0.004590  ← Changed! (grew 2%)
[FSC MANUAL] Applying ±2% now (no gates, no smoothing)
             band=[8,20] FSC(raw) μ=12.3 [6,18]
             grow=500 shrink=200 hold=9300
             r_mean=0.004590 range=[0.004100,0.005200]
[TELEMETRY C] r_mean END OF MEASUREMENT: 0.004590  ← Same as B (no overwrites!)
```

---

## How to Test (Updated Protocol)

### 1. **Start the app:**
```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
./venv/bin/python run.py
```

### 2. **Watch telemetry** (should print every ~30 frames):
- **TELEMETRY A → B:** Should see r_mean change by ~±2%
- **TELEMETRY B → C:** Should stay the same (no overwrites)
- **FSC(raw) μ:** Should be in reasonable range (8-20)

### 3. **Test Causality:**

#### Test A: Force Growth
- Edit `config.py`: Set `FSC_HIGH = 6` (way below typical FSC ~12)
- Restart app
- **Expected:** 
  - Most particles in "grow" zone
  - r_mean increases ~2% per measurement
  - Visible growth on screen

#### Test B: Force Shrink
- Edit `config.py`: Set `FSC_LOW = 18` (way above typical FSC ~12)
- Restart app
- **Expected:**
  - Most particles in "shrink" zone
  - r_mean decreases ~2% per measurement
  - Visible shrink on screen

#### Test C: Balanced Band
- Edit `config.py`: Set `FSC_LOW = 10`, `FSC_HIGH = 14`
- Restart app
- **Expected:**
  - Roughly balanced grow/shrink/hold counts
  - r_mean stabilizes near current value

---

## If Particles Still Blow Up

If growth is still too aggressive:

1. **Reduce pump rate further** (run.py line ~519):
   ```python
   r_new = r0 * 1.01  # 1% instead of 2%
   r_new = r0 * 0.99  # 1% shrink
   ```

2. **Check TELEMETRY B → C:**
   - If they differ → late-frame overwrite (we'll debug which line)
   - If they're same → pump is working, just tune the rate

3. **Check FSC values:**
   - If `FSC(raw) μ = 0.0` → JFA not detecting faces (increase `JFA_RES_MIN`)
   - If `FSC(raw) μ > 100` → power beta issue or face detection bug

---

## If Particles Freeze / Don't Move

If r_mean doesn't change:

1. **Check TELEMETRY A → B:**
   - If different → pump wrote successfully
   - If same → pump didn't execute or was clamped

2. **Check TELEMETRY B → C:**
   - If B changed but C reverts → **LATE OVERWRITE** (smoking gun!)
   - Print which line is guilty (add more prints between B and C)

3. **Check action counts:**
   - If `grow=0 shrink=0 hold=10000` → all particles in-band (move sliders wider apart)
   - If `grow=5000` but r_mean stays same → clipping or overwrite

---

## What This Achieves

✅ **Zero gates** - JFA runs every measurement, no `last_fsc_mean` checks  
✅ **Zero degree interference** - degree controller doesn't run in manual mode  
✅ **Instant feedback** - raw FSC (not EMA) for 1-frame response  
✅ **No adjustment smoothing** - timer stays at 0, pump applies immediately  
✅ **Wide bounds** - 0.0001 to 0.1000 prevents clipping  
✅ **Smaller steps** - ±2% instead of ±5% prevents explosion  
✅ **Three-point telemetry** - catches overwrites if they exist  
✅ **Standard power beta** - set to 1.0 every measurement  

This is the absolute cleanest FSC manual pump possible. If it doesn't work now, the three-point telemetry will tell us exactly where and why.

