# 🧩 Fabric of Space - Custom Taichi Grid

**Dynamic Voronoi Foam — Phase B (Growth Rhythm + Lévy Diffusion)**

---

## 🌌 Overview

This simulator models a dynamic 3D foam composed of thousands of interacting spherical "cells."
Each cell has a radius *r*ᵢ and a position *p*ᵢ, and together they form a smoothed-particle approximation of a Voronoi / power diagram.

The foam evolves toward a balanced topology where each cell maintains a target degree band (number of neighbors) while staying overlap-free.
**Phase B** introduces **temporal separation** and **topological relaxation** for stable, lifelike evolution at real-time frame rates.

---

## 🚀 Key Features (Phase B)

### 1️⃣ Discrete Growth Rhythm

Instead of continuous chaotic updates, growth now happens in **pulses**:

| Phase | What happens | Duration |
|-------|-------------|----------|
| **Pulse Frame** | Count neighbors → apply one ±Δr step | 1 frame |
| **Relax Window** | No counting or growth; only motion + Lévy | `RELAX_INTERVAL` frames |
| **Idle** | PBD only, until next pulse | `GROWTH_INTERVAL` frames |

This gives every process time to settle before the next measurement.

---

### 2️⃣ Runtime GUI Sliders

Interactive control panel (left panel) with **no snap-back**:

| Slider | Range | Effect |
|--------|-------|--------|
| **Growth rate per pulse** | 0.01 – 0.10 | Percentage radius change (±) on each pulse |
| **Frames between pulses** | 5 – 120 | Controls pulse frequency |
| **Relax frames after pulse** | 0 – 60 | Time for foam to settle before next measurement |

Sliders modify live Taichi 0-D fields (`grow_rate_rt`, `grow_interval_rt`, `relax_interval_rt`) that the kernels read every frame.

---

### 3️⃣ Lévy Positional Diffusion (Topological Regularization)

During the **relax window only**, particles undergo a smooth positional diffusion:

```
Δpᵢ = α · Σⱼ clamp((dⱼ - dᵢ) / span, -1, 1) · (pⱼ - pᵢ)
```

- Runs **only during relax frames** (not during growth)
- Uses smoothed degree values (`deg_smoothed`)
- Step clamped to `LEVY_STEP_FRAC × mean_radius`
- Periodic-boundary-safe (PBC-aware shifts)

**Typical parameters** (in `config.py`):

```python
LEVY_ENABLED = True
LEVY_ALPHA = 0.04
LEVY_DEG_SPAN = 10.0
LEVY_STEP_FRAC = 0.15
LEVY_USE_TOPO_DEG = False  # set True once Gabriel topology is restored
```

---

### 4️⃣ Automatic Rate-Limit Enforcement

The radius-update kernel now receives a **runtime rate-limit parameter**:

```python
rate_limit_rt = max(RADIUS_RATE_LIMIT, grow_rate_rt[None])
update_radii_xpbd(grow_rate_rt[None], rate_limit_rt, …)
```

Ensures visible growth even when GUI rate > static limit.

---

### 5️⃣ Detailed Per-Pulse Telemetry

Each pulse prints a structured report:

```
[Pulse] frame=180 | rate=0.060 gap=18 relax=12
        deg: μ=5.89 [1,14]
        r_mean: 0.004743 → 0.004921 (Δ=+0.000178, +3.75%)
        clipped: 28/5000 (0.6%) [min=8 max=20]
        max_depth=0.000156, passes=4
```

**Meaning:**
- **rate/gap/relax**: active GUI settings
- **deg μ[min,max]**: neighbor count stats
- **r_mean**: average radius before/after pulse
- **clipped**: fraction hitting min/max bounds
- **max_depth/passes**: PBD load after pulse

---

## 🧠 Temporal Architecture (Loop Order)

```python
if pulse_timer <= 0:
    rebuild_grid()
    count_neighbors()
    update_radii_xpbd(grow_rate_rt, rate_limit_rt)
    relax_timer = relax_interval_rt
    pulse_timer = grow_interval_rt
else:
    pulse_timer -= 1

project_overlaps()  # PBD every frame

if relax_timer > 0:
    relax_timer -= 1
    if LEVY_ENABLED:
        compute_mean_radius()
        smooth_degree()
        levy_position_diffusion()
```

---

## ⚙️ Default Parameters (`config.py`)

```python
DEG_LOW, DEG_HIGH = 3, 5          # target neighbor band
RADIUS_RATE_LIMIT = 0.015         # absolute per-frame cap
GROWTH_RATE_DEFAULT = 0.04        # 4% per pulse
GROWTH_INTERVAL_DEFAULT = 20      # frames between pulses
RELAX_INTERVAL_DEFAULT = 10       # relax frames
LEVY_ALPHA = 0.04                 # diffusion strength
LEVY_STEP_FRAC = 0.15             # max step size
```

---

## 📊 Tuning Guide

| Goal | rate | gap | relax | Notes |
|------|------|-----|-------|-------|
| **Stable, alive** | 0.04 | 20 | 10 | Default balance |
| **Faster compression** | 0.06 – 0.08 | 15 – 18 | 12 – 14 | Increases motion pressure |
| **Smoother** | 0.03 | 25 | 15 | Softer convergence |

**Monitor:**
- **deg μ** → should drift toward 3–5 (or your custom band)
- **r_mean** → oscillates gently around steady state
- **max_depth** → stays < 0.001
- **FPS** ≈ 100–120 for N ≈ 5000

---

## 🖥️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VirtualOrganics/Fabric-of-Space-Taichi.git
cd Fabric-of-Space-Taichi

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### Run Simulation

**Option 1: Using the launch script (recommended)**
```bash
./run.sh
```

**Option 2: Manual activation**
```bash
source venv/bin/activate
python3 run.py
deactivate  # when done
```

### Controls

- **Right-click + drag**: Rotate camera
- **Mouse wheel**: Zoom in/out
- **WASD**: Move camera
- **SPACE**: Pause/Resume simulation
- **S**: Export particle data (CSV)
- **ESC**: Exit

### GUI Panel

- **Particle Count**: Adjust N and restart simulation
- **Degree Stats**: View distribution and adjust band thresholds
- **Radius Limits**: Set min/max bounds
- **Growth Rhythm**: Control rate, gap, and relax (live, no snap-back!)
- **Visualization**: Toggle center-point vs. full-sphere rendering

---

## 🧪 Quick Start for Developers

### Project Structure

```
Cursor_FoS-Custom-Grid/
├── config.py          # All simulation parameters
├── grid.py            # Spatial hashing and neighbor detection
├── dynamics.py        # Radius adaptation, PBD, Lévy diffusion
├── topology.py        # Gabriel graph topology analysis (Phase B)
├── run.py             # Main loop, GUI, and visualization
├── run.sh             # Launch script
├── requirements.txt   # Python dependencies
└── venv/              # Virtual environment (created during setup)
```

### Dependencies

- **Python**: 3.8+ (tested on 3.13)
- **Taichi**: 1.7.0+ (GPU-accelerated, Metal/CUDA/Vulkan)
- **NumPy**: 1.24.0+ (array operations)

### Configuration Quick Reference

Edit `config.py` to customize:

**Particle Properties:**
```python
N = 5000                    # Number of particles
DOMAIN_SIZE = 0.15          # Cubic domain size
R_MIN, R_MAX = 0.0020, 0.0080  # Radius bounds
```

**Degree Adaptation:**
```python
DEG_LOW, DEG_HIGH = 3, 5    # Target neighbor band
GAIN_GROW, GAIN_SHRINK = 0.03, 0.03  # Growth/shrink rate
```

**PBD Separation:**
```python
PBD_BASE_PASSES = 4         # Minimum PBD passes per frame
PBD_MAX_PASSES = 8          # Maximum (adaptive scaling)
GAP_FRACTION = 0.015        # Target separation cushion
```

**Growth Rhythm:**
```python
GROWTH_RATE_DEFAULT = 0.04     # 4% per pulse
GROWTH_INTERVAL_DEFAULT = 20   # Frames between pulses
RELAX_INTERVAL_DEFAULT = 10    # Relax window duration
```

**Lévy Diffusion:**
```python
LEVY_ENABLED = True         # Toggle positional diffusion
LEVY_ALPHA = 0.04           # Diffusion strength
LEVY_STEP_FRAC = 0.15       # Max step size (fraction of mean radius)
```

**Periodic Boundaries:**
```python
PBC_ENABLED = True          # Toggle periodic boundaries
```

### Run Options

**Standard run:**
```bash
python3 run.py
```

**With performance profiling (built-in):**
- Console logs every 60 frames: grid, PBD, topology, render timings
- Pulse logs every pulse: degree, radius, clipped%, depth, passes

**Export data:**
- Press **S** during simulation to export CSV snapshot
- File: `particle_data_frame_N.csv` (ID, X, Y, Z, Radius, Degree)

### Development Tips

1. **Fast iteration**: Lower `N` to 1000 for quick testing
2. **Debug PBD**: Watch `max_depth` and `passes` in console logs
3. **Tune rhythm**: Use GUI sliders for live experimentation
4. **Profile GPU**: Taichi's `ti.profiler` tools available (see Taichi docs)
5. **Test PBC**: Set `PBC_ENABLED = False` to compare bounded vs. periodic

---

## 🔬 Architecture Details

### Spatial Hashing Grid

- **Cell size**: `2 × R_MAX` (conservative, ensures all pairs detected)
- **Grid resolution**: `GRID_RES³` (e.g., 13³ = 2197 cells for default domain)
- **Neighbor search**: 27-stencil (self + 26 neighbors)
- **Periodic wrap**: Minimum-image convention for cross-boundary pairs

### XPBD Radius Adaptation

Frame-rate independent radius changes using compliance-based constraints:

```python
desired_change = gain × r_old
delta_r = desired_change / (1 + α / dt²)
delta_r = clamp(delta_r, -rate_limit × r_old, +rate_limit × r_old)
r_new = clamp(r_old + delta_r, r_min, r_max)
```

### PBD Overlap Resolution

Adaptive multi-pass Position-Based Dynamics:
- **Base passes**: 4 (normal case)
- **Max passes**: 8 (scales with `max_depth`)
- **Displacement cap**: 20% of radius per pass (anti-tunneling)
- **Gap target**: 1.5% breathing room between particles

### Brownian Motion (Optional)

Ornstein-Uhlenbeck jitter for visual interest:
- **RMS drift**: 10% of mean radius per second
- **Time scale**: 1.0s (smooth meander)
- **Clamp**: 20% of target gap (PBD-stable)

---

## 📈 Performance

| Configuration | FPS | Notes |
|--------------|-----|-------|
| N=5,000, PBC ON | ~100–120 | M1/M2 Mac, Metal backend |
| N=5,000, Topology ON | ~20–40 | Gabriel graph is expensive |
| N=10,000, PBC ON | ~40–60 | Grid rebuild dominates |

**Bottlenecks:**
1. **Grid rebuild**: ~40% of frame time (necessary for dynamic radii)
2. **PBD passes**: ~30% (scales with overlap depth)
3. **Topology**: ~20% (only when `TOPO_ENABLED=True`)

---

## 🧩 Phase B Changelog

- ✅ Added growth/relax rhythm system (discrete pulses + temporal separation)
- ✅ Implemented runtime Taichi controls (no snap-back sliders)
- ✅ Integrated Lévy positional diffusion → runs only during relax frames
- ✅ Added automatic rate-limit enforcement
- ✅ Enhanced per-pulse telemetry and console diagnostics
- ✅ Cleaned old `LEVY_CADENCE` logic
- ✅ Verified stability ≈ 100 FPS @ N=5000

---

## 🔮 Future Work (Phase C Preview)

- Re-enable **Gabriel graph topology** → set `LEVY_USE_TOPO_DEG=True`
- Add **automatic relax-length adaptation** based on overlap depth
- Extend **HUD** for live pulse metrics and FPS tracking
- Optional: integrate Lévy diffusion strength slider
- Explore **topology-aware growth** (use Gabriel neighbors instead of geometric)

---

## 🧪 Acceptance Tests

1. **Grid accuracy**: Neighbor counts match brute-force reference
2. **Radius individuality**: Isolated particles grow, crowded particles shrink independently
3. **PBD stability**: No tunneling or explosions after 1000+ frames
4. **Periodic wrap**: Particles near boundaries detect cross-boundary neighbors
5. **Slider persistence**: GUI values don't snap back (runtime Taichi fields)
6. **Pulse timing**: Growth happens only at pulse frames, Lévy only during relax
7. **Rate-limit enforcement**: `rate_limit_rt = max(RADIUS_RATE_LIMIT, grow_rate_rt)`

---

## 🧠 Summary

**Phase B** transforms the simulator from a continuously thrashing system into a **self-paced, rhythm-driven foam** where growth, motion, and topology are decoupled in time.

The result is a **stable yet dynamic fabric** that visually breathes and evolves toward equilibrium — the foundation for the upcoming topological **Phase C**.

---

## 📄 License

MIT

---

## 👤 Author

Built for the **Fabric of Space** project — exploring emergent foam-like behavior in variable-radius particle systems.

**Repository**: [github.com/VirtualOrganics/Fabric-of-Space-Taichi](https://github.com/VirtualOrganics/Fabric-of-Space-Taichi)

---

## 🙏 Acknowledgments

- **Taichi Graphics**: GPU-accelerated Python framework
- **PBD/XPBD**: Müller et al., Macklin et al.
- **Lévy diffusion**: Inspired by Voronoi centroidal relaxation
- **Gabriel graphs**: Computational geometry for proximity analysis

---

**Happy Simulating!** 🌌✨
