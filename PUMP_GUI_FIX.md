# Pump GUI Fix - The Actual Problem

## What Was Wrong

**Pump was reading hardcoded config values, NOT the GUI sliders.**

- **GUI sliders showed:** [7, 16]
- **Pump was using:** [3, 5] (from config.py)
- **Degree avg:** 4.01 (in the 3-5 "hold" band → no action)
- **Result:** Nothing happened when you moved sliders

## The Fix (3 lines changed in run.py)

### 1. Pump Call (line ~1106)
**Before:**
```python
apply_fsc_pump_simple(
    int(active_n),
    float(FSC_LOW), float(FSC_HIGH),  # Config values [3,5]
    float(R_MIN), float(R_MAX)
)
```

**After:**
```python
apply_fsc_pump_simple(
    int(active_n),
    float(gui_deg_low), float(gui_deg_high),  # GUI sliders!
    float(R_MIN), float(R_MAX)
)
```

### 2. Action Counts (line ~1101)
**Before:**
```python
n_grow = int(np.sum(deg_np < FSC_LOW))    # Config [3,5]
n_shrink = int(np.sum(deg_np > FSC_HIGH))
```

**After:**
```python
n_grow = int(np.sum(deg_np < gui_deg_low))    # GUI sliders
n_shrink = int(np.sum(deg_np > gui_deg_high))
```

### 3. Telemetry Print (line ~1122)
**Before:**
```python
print(f"band=[{FSC_LOW},{FSC_HIGH}] ...")  # Showed [3,5]
```

**After:**
```python
print(f"band=[{gui_deg_low},{gui_deg_high}] ...")  # Shows actual GUI values
```

---

## Key Code Locations

### **run.py** (1626 lines total)

**Pump kernel (line ~507):**
```python
@ti.kernel
def apply_fsc_pump_simple(
    n_active: ti.i32,
    fsc_low: ti.f32, fsc_high: ti.f32,
    r_min: ti.f32, r_max: ti.f32
):
    for i in range(n_active):
        f = ti.f32(deg[i])  # Uses DEGREE
        
        r0 = rad[i]
        r_new = r0
        
        if f < fsc_low:
            r_new = r0 * 1.02  # Grow 2%
        elif f > fsc_high:
            r_new = r0 * 0.98  # Shrink 2%
        
        r_new = ti.min(r_max, ti.max(r_min, r_new))
        rad[i] = r_new
```

**Pump call (line ~1105):**
```python
if JFA_ENABLED and FSC_MANUAL_MODE:
    # Get degree stats
    deg_np = deg.to_numpy()[:active_n]
    signal_mean = float(deg_np.mean())
    
    # Count actions based on GUI sliders
    n_grow = int(np.sum(deg_np < gui_deg_low))
    n_shrink = int(np.sum(deg_np > gui_deg_high))
    
    # Apply pump with GUI slider values
    apply_fsc_pump_simple(
        int(active_n),
        float(gui_deg_low), float(gui_deg_high),
        float(R_MIN), float(R_MAX)
    )
```

---

## Expected Behavior Now

With GUI sliders at [7, 16] and degree avg = 4.01:

**Degree = 4.01 < 7 → GROW**
- `n_grow` should be ~10000 (all particles)
- `n_shrink` should be 0
- r_mean should **increase by ~2%** per measurement
- Telemetry shows: `band=[7,16] DEGREE μ=4.0`

---

## Test Now

```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
./venv/bin/python run.py
```

**Watch console:**
- Band should show [7, 16] (matching GUI)
- With deg avg ~4, should see `grow=10000`
- r_mean should increase each cycle

**Adjust GUI sliders:**
- Move "Min (grow below)" to 2 → should stop growing (deg > 2)
- Move "Max (shrink above)" to 3 → should start shrinking (deg > 3)

---

## If You Want the Full Files

**Main files:**
1. `run.py` (1626 lines) - Pump kernel + control logic
2. `config.py` (468 lines) - Configuration

**Key sections in run.py:**
- Line ~507: Pump kernel definition
- Line ~974: JFA run logic
- Line ~1075: Manual mode block (where pump is called)
- Line ~1105: Pump call with GUI sliders

You can send these files to get them checked.

---

## What I Screwed Up

I was reading config values instead of GUI state. The sliders were updating `gui_deg_low` and `gui_deg_high` variables, but the pump was hardcoded to use `FSC_LOW` and `FSC_HIGH` from config.py.

Classic disconnected-UI bug. My bad.

