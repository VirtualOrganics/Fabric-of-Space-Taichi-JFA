# Visualization Problem - Need Expert Help

## THE CORE ISSUE

**Simulation works correctly** (console shows `min=0.0051, max=0.0060` - clear size variation)
**Visualization shows uniform particles** (screenshot shows all particles same size)

---

## WHAT WE'VE TRIED (ALL FAILED)

### Attempt 1: `radius=mean_rad` with `per_vertex_color`
```python
scene.particles(pos, radius=0.004, per_vertex_color=color)
```
**Result:** All uniform size (expected - no per_vertex_radius specified)

### Attempt 2: `per_vertex_radius` with `per_vertex_color`
```python
scene.particles(pos, per_vertex_radius=rad, per_vertex_color=color)
```
**Result:** `TypeError: missing 1 required positional argument: 'radius'`

### Attempt 3: Both `radius` and `per_vertex_radius`
```python
scene.particles(pos, radius=0.004, per_vertex_radius=rad, color=(0.5, 0.8, 0.5))
```
**Result:** NO ERROR, but visual shows uniform size (current state)

### Attempt 4: NumPy array for radius
```python
scene.particles(pos, radius=rad_np, color=(0.5, 0.8, 0.5))
```
**Result:** `TypeError: incompatible function arguments` (expects scalar, not array)

### Attempt 5: Individual particle rendering
```python
for i in range(N):
    scene.particles(pos[i:i+1], radius=rad_np[i], color=...)
```
**Result:** `AttributeError: 'Vector' object has no attribute 'shape'`

---

## GEOTAICHI COMPARISON

**GeoTaichi DOES show variable-sized particles** (see paper figures)

Their rendering code:
```python
# From GeoTaichi/src/dem/DEMBase.py:155
ui_scene.particles(scene.particle.x, per_vertex_radius=scene.particle.rad, color=self.sims.particle_color)
```

**Key differences:**
1. **Scene creation:** GeoTaichi uses `window.get_scene()`, we were using `ti.ui.Scene()` (deprecated)
   - ✅ **FIXED**: Changed to `window.get_scene()` - TESTING NOW
2. **Taichi version:** GeoTaichi was tested with Taichi 1.6.0, we're using 1.7.4
3. **Field type:** GeoTaichi's `scene.particle.rad` is a Taichi Struct field, ours is a plain `ti.field(ti.f32)`
4. **Color parameter:** GeoTaichi passes `self.sims.particle_color` (tuple), we pass `(0.5, 0.8, 0.5)`

---

## ENVIRONMENT

- **Taichi:** 1.7.4, llvm 15.0.7
- **Backend:** Metal (Apple Silicon)
- **Python:** 3.13.5
- **OS:** macOS

---

## VERIFICATION THAT SIMULATION IS CORRECT

Console output proves radii ARE changing:
```
[Frame  100] Degree: mean=3.13, min=0, max=15 | Radius: mean=0.0060, min=0.0051, max=0.0060
[Frame  200] Degree: mean=5.11, min=0, max=15 | Radius: mean=0.0057, min=0.0044, max=0.0060
```

- Min radius: 0.0044
- Max radius: 0.0060
- Ratio: 1.36x difference
- This SHOULD be visually obvious if rendering correctly

---

## QUESTIONS FOR TAICHI EXPERTS

1. **Does `per_vertex_radius` actually work in Taichi 1.7.4?**
   - If yes, what are we doing wrong?
   - If no, what's the correct way to render variable-sized particles?

2. **Is the deprecated `ti.ui.Scene()` the problem?**
   - Should we use `window.get_scene()` instead?
   - Does this affect `per_vertex_radius` functionality?

3. **Does `per_vertex_radius` require a specific field structure?**
   - Plain `ti.field(ti.f32)` vs Taichi Struct fields?
   - Does data layout matter?

4. **Is there a minimum size difference threshold for visibility?**
   - Our 1.36x ratio should be obvious, right?
   - Or do we need larger differences?

5. **Could this be a Metal backend issue?**
   - Does `per_vertex_radius` work on CUDA but not Metal?

---

## FILES TO INSPECT

- **Our rendering code:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/run.py` (line 223)
- **GeoTaichi's working code:** `/Users/chimel/Desktop/GeoTaichi/src/dem/DEMBase.py` (line 155)
- **Our field definitions:** `/Users/chimel/Desktop/Cursor_FoS-Custom-Grid/grid.py` (lines 10-13)

---

## WHAT WE NEED

Either:
1. **Fix for `per_vertex_radius`** to make it actually work
2. **Alternative method** to render variable-sized particles in Taichi GGUI
3. **Confirmation** that it's impossible, so we can switch to matplotlib/Blender export

---

## CURRENT WORKAROUND (UNACCEPTABLE)

Uniform rendering - simulation is correct but we can't see it:
```python
scene.particles(pos, radius=0.004, color=(0.5, 0.8, 0.5))
```

This defeats the entire purpose of building a custom grid system for variable-sized particles.

