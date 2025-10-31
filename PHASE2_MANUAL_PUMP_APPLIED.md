# Phase 2: Manual FSC Pump - Implementation Summary

## ⚠️ SUPERSEDED - See PHASE2_ALL_FIXES_APPLIED.md

This document describes the initial implementation. After first run revealed particles blowing up, we applied 10 additional surgical fixes documented in **`PHASE2_ALL_FIXES_APPLIED.md`**.

---

## Changes Applied (Fresh Start for FSC Testing)

All surgical fixes from the feedback have been successfully implemented. This gives us a **clean, minimal FSC manual pump** with zero hidden logic.

---

## What Was Changed

### 1. Config (config.py)
✅ **Enabled Manual Mode:**
- `FSC_MANUAL_MODE = True` (was False)
- `JFA_RUN_INTERVAL = 1` (run every measurement frame)
- `MIN_FACE_VOXELS = 2` (very permissive for debugging)

**Why:** Makes JFA permissive enough to guarantee non-zero FSC values during testing.

---

### 2. Run Script (run.py)

✅ **Added Minimal FSC Pump Kernel** (line ~485):
```python
@ti.kernel
def apply_fsc_pump_simple(n_active, fsc_low, fsc_high, r_min, r_max):
    """
    Simple ±5% pump based on FSC band:
    - FSC < low  → grow 5%
    - FSC > high → shrink 5%
    - In band    → hold
    
    No gates, no smoothing, immediate effect.
    """
    for i in range(n_active):
        f = jfa.fsc_ema[i]
        r0 = rad[i]
        
        if f < fsc_low:
            r_new = r0 * 1.05    # Grow
        elif f > fsc_high:
            r_new = r0 * 0.95    # Shrink
        else:
            r_new = r0           # Hold
        
        rad[i] = clamp(r_new, r_min, r_max)
```

✅ **Force JFA to Run Every Measurement Frame in Manual Mode** (line ~972):
```python
jfa_should_run = (JFA_ENABLED and adjustment_timer[None] <= 0 and
                 ((FSC_MANUAL_MODE and True)  # always run in manual mode
                  or (not FSC_MANUAL_MODE and (jfa_measurement_counter % JFA_RUN_INTERVAL == 0))))
```

**Why:** Ensures FSC is computed every measurement cycle for immediate feedback.

✅ **Replaced Hybrid Controller with Simple Pump** (line ~1070):
```python
if JFA_ENABLED and FSC_MANUAL_MODE:
    # Set power beta to standard value
    jfa.power_beta_current[None] = 1.0
    
    # Apply simple pump
    apply_fsc_pump_simple(active_n, FSC_LOW, FSC_HIGH, R_MIN, R_MAX)
    
    # Print telemetry
    print(f"[FSC MANUAL] Applying ±5% now (no gates, no smoothing)")
    print(f"             band=[{FSC_LOW},{FSC_HIGH}] FSC μ={fsc_mean:.1f}")
    print(f"             grow={n_grow} shrink={n_shrink} hold={n_hold}")
    print(f"             r_mean={r_mean_after:.6f}")
```

**Why:** Removes all trust gates, phases, and hidden logic. Pure cause→effect testing.

✅ **Skip Adjustment Cycle in Manual Mode** (line ~1235):
```python
if not (JFA_ENABLED and FSC_MANUAL_MODE):
    adjustment_timer[None] = adjustment_frames_rt[None]  # Normal mode: start adjustment
else:
    adjustment_timer[None] = 0  # Manual mode: stay at 0 (measure every frame)
```

**Why:** No adjustment smoothing - changes apply immediately (1-frame response).

---

## How to Test (2-Minute Protocol)

### 1. **Start the app:**
```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
./venv/bin/python run.py
```

### 2. **Let it run for 2-3 measurement cycles.**
You should see:
```
[JFA Step 4] FSC μ=... ...
[FSC MANUAL] Applying ±5% now (no gates, no smoothing)
             band=[8,20] FSC μ=12.3 [6,18]
             grow=500 shrink=200 hold=9300
             r_mean=0.004523
```

### 3. **Test Causality: Move FSC_HIGH Below Mean**
- In `config.py`, set `FSC_HIGH = 10` (below reported mean ~12)
- Restart app
- **Expected:** Within 1-2 measurement prints:
  - `r_mean` should **decrease**
  - Large `shrink` count
  - Visible radius reduction on screen

### 4. **Test Causality: Move FSC_LOW Above Mean**
- In `config.py`, set `FSC_LOW = 14` (above reported mean ~12)
- Restart app
- **Expected:** Within 1-2 measurement prints:
  - `r_mean` should **increase**
  - Large `grow` count
  - Visible radius growth on screen

### 5. **Freeze-Frame Test (Optional):**
- Press `F` to verify grid miss rate is still ~0%
- This confirms we didn't break the grid system

---

## If FSC μ is Stuck at 0.0

If you see `FSC μ=0.0` after a few cycles, try:

1. **Increase resolution:**
   - `JFA_RES_MIN = 128` (in config.py)

2. **Make even more permissive:**
   - `MIN_FACE_VOXELS = 1` (in config.py)

3. **Check power beta:**
   - Should see `jfa.power_beta_current[None] = 1.0` in telemetry
   - Verify it's being set in manual mode section

---

## What This Achieves

✅ **Phase 2 Goal:** Prove FSC can drive radii with direct cause→effect
✅ **No Hidden Logic:** Every change is visible in console
✅ **Immediate Feedback:** 1-frame response (no 30-frame smoothing)
✅ **Verifiable:** Move slider → see r_mean change within 2 cycles
✅ **Grid Intact:** 0% miss rate preserved (unchanged grid system)

---

## Next Steps (After Phase 2 Works)

Once you confirm the sliders control `r_mean` reliably:

1. **Add smoothing back** (if desired) - one parameter at a time
2. **Add trust gates** (if desired) - with loud print statements
3. **Tune FSC band** - find optimal [low, high] for degree ~5

But **only after** confirming basic causality works in this minimal setup.

---

## Design Principles Applied

✅ **Explicit over implicit** - no hidden behavior
✅ **One thing at a time** - prove pump works before adding features
✅ **Immediate feedback** - 1-frame response for testing
✅ **Fail loudly** - clear print statements show what's happening

This is the cleanest possible FSC test - if it doesn't work now, we know exactly what to debug.

