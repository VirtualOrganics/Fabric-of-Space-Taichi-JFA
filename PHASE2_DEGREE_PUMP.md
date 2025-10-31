# Phase 2: Switched to Degree-Based Pump

## Problem Identified

**UI/Control Mismatch:**
- GUI shows **degree** stats [avg=627, band=[3,5]]
- Pump was using **FSC** band [8,20]
- User watching degree sliders but pump ignoring them → confusing causality test

**Symptoms:**
- Degree avg > 5 (should shrink based on GUI)
- But particles growing (because FSC < 8)
- Can't verify causality when watching wrong metrics

---

## Solution: Option C - Drive Pump by Degree

**Decision Rationale:**
1. **GUI-visible testing** - User can see degree in real-time
2. **Direct causality** - Move degree slider → see immediate effect
3. **Simpler Phase 2** - Test pump mechanism first, FSC signal later
4. **One-line change** - Easy to swap back to FSC after verification

---

## Changes Made

### 1. Pump Signal (run.py line ~511)
**Before:**
```python
f = jfa.fsc[i]  # Use raw FSC
```

**After:**
```python
f = ti.f32(deg[i])  # Use degree (GUI-visible)
```

### 2. Telemetry (run.py line ~1095)
**Before:**
```python
fsc_raw_np = jfa.fsc.to_numpy()[:active_n]
fsc_mean = float(fsc_raw_np.mean())
n_grow = int(np.sum(fsc_raw_np < FSC_LOW))
```

**After:**
```python
deg_np = deg.to_numpy()[:active_n]
signal_mean = float(deg_np.mean())
n_grow = int(np.sum(deg_np < FSC_LOW))  # FSC_LOW now means degree low
```

### 3. Config Thresholds (config.py line ~93)
**Before:**
```python
FSC_LOW  = 8   # Target FSC lower bound
FSC_HIGH = 20  # Target FSC upper bound
```

**After:**
```python
FSC_LOW  = 3   # Degree lower bound (grow below this)
FSC_HIGH = 5   # Degree upper bound (shrink above this)
```

### 4. Print Messages (run.py line ~1121)
Now shows:
```
[PUMP MANUAL] Applying ±2% now (no gates, no smoothing)
              band=[3,5] DEGREE μ=627.6 [479,817]
              grow=0 shrink=10000 hold=0
              r_mean=0.004590
              NOTE: Pump uses DEGREE (GUI-visible), not FSC
```

---

## Expected Behavior Now

### **Degree > 5 → Shrink**
- If degree avg = 627 (way > 5)
- Pump should shrink ALL particles by 2% per measurement
- r_mean should **decrease** visibly

### **Degree < 3 → Grow**
- If degree avg drops below 3
- Pump should grow ALL particles by 2% per measurement
- r_mean should **increase** visibly

### **3 ≤ Degree ≤ 5 → Hold**
- If degree in band
- Pump does nothing
- r_mean stays constant

---

## Test Protocol

### 1. **Run the app:**
```bash
cd /Users/chimel/Desktop/Cursor_FoS-Custom-Grid
./venv/bin/python run.py
```

### 2. **Watch console telemetry:**
```
[TELEMETRY A] r_mean BEFORE JFA: 0.004500
[TELEMETRY B] r_mean RIGHT AFTER PUMP: 0.004410  ← Should shrink (deg > 5)
[PUMP MANUAL] band=[3,5] DEGREE μ=627.6 [479,817]
              grow=0 shrink=10000 hold=0
              NOTE: Pump uses DEGREE (GUI-visible), not FSC
[TELEMETRY C] r_mean END OF MEASUREMENT: 0.004410  ← Should match B
```

### 3. **Expected Changes:**
- **A → B:** r_mean should **shrink by ~2%** (degree way > 5)
- **B → C:** r_mean should **stay same** (no overwrites)
- **shrink count:** Should be ~10000 (all particles)

### 4. **Visual Check:**
- Particles should visibly shrink on screen
- Degree histogram should stay > 5 initially (takes time to adjust)

---

## Troubleshooting

### **If particles still grow (deg > 5):**
1. Check TELEMETRY A → B: does r_mean decrease?
   - If YES → pump works, something else growing them
   - If NO → pump not executing or clamped

2. Check action counts in console:
   - Should see `shrink=10000 grow=0` (all shrinking)
   - If not, band thresholds are wrong

3. Check for overwrites:
   - If B < A but C > B → something overwrote rad[]
   - Add prints between B and C to find culprit

### **If degree avg = 627 (seems wrong):**
This is suspiciously high - degrees should be ~3-6 for a foam. Possible causes:
1. Degree counter is broken (counting all stencil pairs, not just contacts)
2. Reach is too large (counting particles far away)
3. CONTACT_TOL is way too high

**For now:** Ignore the actual degree values, just test if pump responds to the *changes* in degree when you adjust sliders in GUI.

---

## Next Steps (After Verification)

### **Once degree→radius causality proven:**
1. Swap back to FSC signal (1-line change):
   ```python
   f = jfa.fsc[i]  # Back to FSC
   ```

2. Set FSC thresholds appropriately:
   ```python
   FSC_LOW = 8
   FSC_HIGH = 16
   ```

3. Update telemetry to show FSC stats again

### **This two-step approach:**
1. **Step 1 (now):** Prove pump mechanism works using degree (GUI-visible)
2. **Step 2 (next):** Prove FSC signal works using console-visible metrics

Separates "does pump work?" from "is FSC computed correctly?"

---

## Note on Degree = 627

If degree really is averaging 627, that's a **separate bug** in the neighbor counting system. Normal foam degree should be ~3-6. Possible fixes:
1. Check `CONTACT_TOL` - maybe too loose
2. Check `reach` calculation - maybe stencil too large
3. Check `count_neighbors` kernel - maybe counting wrong

But fixing that is **separate from proving pump works**. For now, we just need to see if r_mean responds to degree changes (even if degree values are wrong).

