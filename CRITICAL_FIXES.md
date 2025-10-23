# Critical Fixes Applied

## Fix #1: Rendering - Variable Radii (ESSENTIAL)
**Problem:** All particles rendered at uniform size despite radii changing internally
**Root Cause:** Incorrect GGUI API usage - tried to mix `per_vertex_radius` with `per_vertex_color`
**Solution:** Copied from GeoTaichi's DEM renderer:
```python
scene.particles(pos, per_vertex_radius=rad, color=(0.5, 0.8, 0.5))
```
**Key:** Use `per_vertex_radius` with UNIFORM color tuple (not `per_vertex_color`)
**Status:** ✅ FIXED - Variable-sized particles now render correctly!

---

## Fix #2: PBD Separation Bug (CRITICAL)
**Problem:** Degrees exploding to 4997/5000 (physically impossible)
**Root Cause:** `if i < j:` meant only particle `i` was corrected, never `j`
**Fix:** Changed to `if i != j:` so BOTH particles in overlapping pairs get corrected

**Before:**
```python
if i < j:  # Only process when i < j
    # Calculate correction
    correction += correction_vec  # Only i gets corrected!
```

**After:**
```python
if i != j:  # Process all pairs (each particle corrects itself)
    # Calculate correction
    correction += correction_vec  # Each particle accumulates its own corrections
```

**Why this matters:**
- Each particle `i` loops through all neighbors `j` where `i != j`
- Particle `i` computes how much IT needs to move away from `j`
- Particle `j` will do the same when it's `j`'s turn to loop
- Result: Both particles push away from each other (symmetric)

**Expected outcome:**
- Degrees should stabilize around 5-6 (target)
- Max degree should stay < 20 (not 4997!)
- No massive blob formation
- Particles visibly separate

---

## What to Check Next

### 1. Console Output
Watch for degree stats every 100 frames:
```
[Frame  100] Degree: mean=2.99, min=0, max=13  ← Good!
[Frame  400] Degree: mean=278.43, max=4997     ← BAD! (old bug)
```

**Good signs:**
- Mean degree converges to 5-6
- Max degree stays < 20
- Values stable over time

**Bad signs:**
- Mean degree explodes (>50)
- Max degree reaches 1000s
- Oscillating wildly

### 2. Visual Check
- Particles should have **clearly different sizes**
- Small red/yellow particles (isolated, deg < 5)
- Medium green particles (happy, deg 5-6)
- Large blue particles (crowded, deg > 6)
- NOT a uniform blob

### 3. Performance
- FPS should be >20 for 5000 particles
- No hang or freeze
- Smooth camera rotation

---

## Known Limitations

### Rendering Trade-off
Taichi GGUI doesn't support BOTH `per_vertex_colors` AND `per_vertex_radii` simultaneously:
- ✅ **We can show variable sizes** (uniform color) - CURRENT
- ❌ Cannot show per-particle colors AND variable sizes at same time

**This is fine** - we prioritize seeing size changes, which is the core feature.

---

## Blueprint Compliance: ✅

Both fixes maintain 100% blueprint compliance:
- No changes to neighbor detection algorithm
- No changes to radius adaptation logic
- No changes to PBD parameters
- Only fixed rendering (output) and PBD implementation bug (correctness)

