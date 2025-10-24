# Phase B: Topological Neighbor Counting (Witness Test)

**Status:** Ready for Review  
**Depends On:** Phase A (Stability) - COMPLETED ✅  
**Target:** Replace geometric contact counting with topological (Voronoi) degree for radius adaptation

---

## Problem Statement

**Current State (Phase A):**
- Particles grow/shrink based on **geometric contacts** (overlaps + tolerance)
- Target: 3-6 geometric neighbors
- Works for separation, but doesn't match the true "fabric topology"

**Goal (Phase B):**
- Introduce **topological degree** = count of Voronoi face-sharing neighbors
- Target: ~14 topological neighbors (tetrahedral packing average)
- Use geometric contacts only for PBD separation
- Use topological degree for radius adaptation decisions

**Why This Matters:**
- Geometric contacts are ephemeral (particles can separate and lose contact)
- Topological neighbors are structural (define the spatial network)
- Target degree = 14 matches optimal 3D space-filling foam (Kelvin/Weaire-Phelan structures)

---

## The Topological Proximity Test (Gabriel Graph)

### Concept

We use the **Gabriel graph** as a fast, GPU-friendly approximation of Voronoi adjacency.

**Gabriel Graph Definition:**
```
Particles i, j are Gabriel neighbors (edges in Gabriel graph) ⟺ 
  The open sphere with diameter i–j contains no other particle k
```

**Translation:**
- Compute midpoint `m = (pos[i] + pos[j]) / 2` (PBC-wrapped)
- Compute radius `r = ||pos[i] - pos[j]|| / 2` (minimum-image distance)
- For every other particle `k`:
  - If `||pos[k] - m||² < r² - ε` → witness found → i-j are NOT neighbors
- If no witness found → i-j are neighbors

**Key Properties:**
- Gabriel ⊆ Delaunay ⊆ Voronoi dual
- Gabriel is a safe **under-approximation** of Voronoi face-sharing
- Typically gives 12-16 neighbors for 3D random point sets (close to our target 14)
- If degrees too high, fall back to **RNG** (Relative Neighborhood Graph, stricter)

### Visual Intuition

```
Gabriel neighbors:                    NOT Gabriel neighbors:
    i ●―――――――● j                         i ●         ● j
      ╲       ╱                              ╲   ● k ╱
       ╲     ╱  ← empty sphere                ╲  ⊙  ╱
        ╲   ╱     (no k inside)                ╲   ╱
         ⊙ m                                    ⊙ m
    
    Sphere with diameter i-j             k violates the empty sphere
    contains no other particles          → i-j NOT neighbors
```

### Why GPU-Friendly

- Pure distance calculations (no geometric predicates)
- Embarrassingly parallel (each pair independent)
- Early-exit on first witness (short-circuit)
- **Midpoint-centered search** drastically shrinks witness search volume
- Uses `length²` (avoids expensive sqrt)
- Leverages existing spatial grid for candidate pruning

### Why Not "True" Voronoi? (Phase C)

- Exact Voronoi requires computing all Voronoi faces (expensive, complex)
- Gabriel graph is a well-studied proxy: fast, stable, GPU-friendly
- For radius control, we don't need exact topology—just a consistent structural metric
- Phase C (future): Upgrade to power diagram (Laguerre) for polydisperse particles

---

## Implementation Strategy

### 1. Data Structures

**New Taichi Fields:**
```python
# Topological degree per particle
topo_deg = ti.field(ti.i32, shape=MAX_N)          # Raw topological degree (computed every K frames)
topo_deg_ema = ti.field(ti.f32, shape=MAX_N)      # Exponential moving average (updated every frame)

# Temporary storage for witness test (pre-allocated)
topo_pairs = ti.field(ti.i32, shape=(MAX_N, MAX_TOPO_NEIGHBORS))  # Store pair indices for each particle
topo_pair_count = ti.field(ti.i32, shape=MAX_N)                   # Count of pairs per particle
```

**New Config Parameters:**
```python
# Phase B: Topological neighbor counting
TOPO_ENABLED = True                   # Toggle topological degree (vs. geometric only)
TOPO_UPDATE_CADENCE = 20              # Compute topological degree every K frames (expensive)
TOPO_EMA_BETA = 0.1                   # Smoothing factor for EMA (0.1 = 10% new, 90% old)
TOPO_TARGET_DEG = 14                  # Target topological degree (14 for tetrahedral packing)
TOPO_DEG_LOW = 10                     # Grow below this threshold
TOPO_DEG_HIGH = 18                    # Shrink above this threshold
TOPO_MAX_RADIUS_MULTIPLE = 2.5        # Only test pairs within this * (r_i + r_j)
MAX_TOPO_NEIGHBORS = 32               # Pre-allocate space for topological pairs (conservative upper bound)

# Override radius adaptation to use topological degree when enabled
# (Keep geometric DEG_LOW/DEG_HIGH for backward compatibility / fallback)
```

---

### 2. Core Kernels

#### Kernel 1: `build_topo_candidates`

**Purpose:** Find candidate pairs for witness testing (spatial pruning).

**Logic:**
1. For each particle `i`, iterate 27-cell stencil (3×3×3)
2. For each particle `j` in those cells:
   - If `j > i` (avoid duplicate pairs)
   - Compute `delta_ij = pdelta(pos[i], pos[j])`  # PBC-aware
   - Compute `dist_sq_ij = delta_ij.dot(delta_ij)`  # Avoid sqrt
   - If `dist_sq_ij <= [TOPO_MAX_RADIUS_MULTIPLE * (rad[i] + rad[j])]²`:
     - Store `(i, j)` as candidate pair
     - Atomic increment `topo_pair_count[i]`

**Output:** List of candidate pairs per particle (spatial neighbors).

**Key:** Use `pdelta` + squared distances (consistent with PBC + perf).

---

#### Kernel 2: `gabriel_test_topological_degree`

**Purpose:** For each candidate pair, apply Gabriel graph test (empty diameter-sphere check).

**Logic:**
```python
@ti.kernel
def gabriel_test_topological_degree(pos, rad, topo_deg, topo_pairs, topo_pair_count, 
                                     cell_start, cell_count, cell_indices, n):
    # Clear topological degree
    for i in range(n):
        topo_deg[i] = 0
    
    # For each particle i
    for i in range(n):
        num_pairs = topo_pair_count[i]
        
        # For each candidate neighbor j
        for p_idx in range(num_pairs):
            j = topo_pairs[i, p_idx]
            
            # Compute minimum-image vector i→j (PBC-aware)
            delta_ij = pdelta(pos[i], pos[j])
            d_ij_sq = delta_ij.dot(delta_ij)  # Use squared distance (avoid sqrt)
            
            # Gabriel test: sphere with diameter i-j
            # Midpoint: m = pos[i] + 0.5 * delta_ij (PBC-wrapped)
            midpoint = wrapP(pos[i] + 0.5 * delta_ij)
            
            # Radius² of diameter sphere (r = d_ij / 2, so r² = d_ij² / 4)
            r_sq_diameter = 0.25 * d_ij_sq
            
            # Get cell range for MIDPOINT (not particle i!)
            # This shrinks the search volume significantly
            q = (midpoint + HALF_L) * INV_L
            mid_cell = ti.Vector([int(q[d] * GRID_RES) for d in ti.static(range(3))])
            mid_cell = wrap_cell(mid_cell)
            
            # Adaptive stencil: size based on diameter-sphere radius
            # For small pairs: 1-3 cells; for large: up to 5-7 cells
            r_diameter = ti.sqrt(d_ij_sq) * 0.5  # Radius of diameter sphere
            stencil_cells = int(ti.ceil(r_diameter / CELL_SIZE))  # Cells per axis
            stencil_cells = ti.max(1, ti.min(stencil_cells, 3))  # Clamp to [1, 3] for safety
            stencil_cells = ti.min(stencil_cells, GRID_RES // 2)  # Never exceed grid bounds
            
            is_gabriel_neighbor = True  # Innocent until proven guilty
            
            # Iterate adaptive stencil [-stencil_cells, +stencil_cells]³
            for dx in range(-stencil_cells, stencil_cells + 1):
                for dy in range(-stencil_cells, stencil_cells + 1):
                    for dz in range(-stencil_cells, stencil_cells + 1):
                        nc = mid_cell + ti.Vector([dx, dy, dz])
                        nc = wrap_cell(nc)
                        nc_id = cell_id(nc)
                        
                        start = cell_start[nc_id]
                        count = cell_count[nc_id]
                        
                        for k_idx in range(start, start + count):
                            k = cell_indices[k_idx]
                            
                            # Skip i and j themselves
                            if k == i or k == j:
                                continue
                            
                            # Gabriel witness test: is k inside the open sphere?
                            # ||pos[k] - midpoint||² < r² - ε
                            delta_km = pdelta(pos[k], midpoint)
                            dist_sq_km = delta_km.dot(delta_km)
                            
                            # Scaled epsilon: avoids misclassification for very large/small pairs
                            eps_scaled = ti.max(1e-12, 1e-6 * r_sq_diameter)
                            
                            # If k is inside the diameter sphere, i-j NOT Gabriel neighbors
                            if dist_sq_km < r_sq_diameter - eps_scaled:
                                is_gabriel_neighbor = False
                                break  # Early exit (one witness is enough)
                        
                        if not is_gabriel_neighbor:
                            break  # Break out of cell loops
                    if not is_gabriel_neighbor:
                        break
                if not is_gabriel_neighbor:
                    break
            
            # Edge guard: Skip degenerate pairs (too small or self)
            if d_ij_sq < 1e-16:
                continue  # Avoid divide-by-zero in diameter sphere, skip duplicates
            
            # If no witness found, i-j are Gabriel neighbors
            # Process each pair once (j > i) and increment both degrees (symmetric)
            if is_gabriel_neighbor and j > i:
                ti.atomic_add(topo_deg[i], 1)
                ti.atomic_add(topo_deg[j], 1)  # Symmetric (halves work, avoids bias)
```

**Output:** `topo_deg[i]` = count of Gabriel graph neighbors for particle `i`.

**Key Optimizations:**
1. **Adaptive stencil**: Stencil size scales with pair distance (1-3 cells typical, up to 5-7 for large pairs)
2. **Midpoint-centered search**: Cells are iterated around `midpoint`, not `pos[i]`
   - Shrinks search volume by ~8× for typical candidate pairs
3. **Scaled epsilon**: `eps = max(1e-12, 1e-6 * r²)` avoids misclassification for extreme sizes
4. **Squared distances**: Avoids `sqrt` in inner loop (uses `dot()` only)
5. **Early exit**: Stops at first witness (typical case: 1-5 checks per pair)
6. **Symmetric processing**: Each pair (i, j) processed once (j > i), both degrees updated
7. **PBC-aware**: `wrapP` for midpoint, `pdelta` for all distance vectors

---

#### Optional: Laguerre-Gabriel (Weighted Power Diagram)

**Why:** With polydisperse radii, Euclidean Gabriel can over-connect small–large pairs. The **Laguerre-Gabriel** graph uses power distance instead of Euclidean distance to respect particle sizes.

**When to Use:** If mean degree consistently > 18-20 despite tuning, or if you see systematic over-connection between very different sizes.

**Implementation:** Add compile-time toggle in `config.py`:
```python
USE_LAGUERRE_GABRIEL = False  # True for power diagram, False for Euclidean
```

**Kernel Modification (inside `gabriel_test_topological_degree`):**
```python
# Replace r_sq_diameter and dist_sq_km with power distance

if ti.static(USE_LAGUERRE_GABRIEL):
    # Power ball radius (Laguerre analog of diameter sphere)
    power_radius = 0.25 * (d_ij_sq - (rad[i] - rad[j]) ** 2)
    
    # Guardrail: If power_radius <= 0, the "power ball" is empty (degenerate)
    # Accept this witness candidate (don't false-reject)
    if power_radius > 0:
        # Power distance from k to midpoint
        dist_sq_km_euclid = delta_km.dot(delta_km)
        power_k = dist_sq_km_euclid - rad[k] ** 2
        
        eps_scaled = ti.max(1e-12, 1e-6 * ti.abs(power_radius))
        
        if power_k < power_radius - eps_scaled:
            is_gabriel_neighbor = False
            break
else:
    # Standard Euclidean Gabriel (existing code)
    eps_scaled = ti.max(1e-12, 1e-6 * r_sq_diameter)
    
    if dist_sq_km < r_sq_diameter - eps_scaled:
        is_gabriel_neighbor = False
        break
```

**Expected Behavior:**
- **Euclidean Gabriel**: Typical degree 12-16 (works well for near-uniform sizes)
- **Laguerre-Gabriel**: Typical degree 10-14 (better for high polydispersity, e.g., 2-4× radius range)

**Phase:** Ship Phase B with Euclidean (default). Enable Laguerre in Phase B.2 or Phase C if needed.

---

#### Kernel 3: `update_topo_ema`

**Purpose:** Smooth topological degree with exponential moving average (every frame).

**Logic:**
```python
@ti.kernel
def update_topo_ema(topo_deg, topo_deg_ema, n, new_measurement_available: ti.i32):
    for i in range(n):
        if new_measurement_available:
            # New topological degree computed this frame → update EMA
            topo_deg_ema[i] = TOPO_EMA_BETA * topo_deg[i] + (1.0 - TOPO_EMA_BETA) * topo_deg_ema[i]
        else:
            # No new measurement → EMA remains unchanged (or decays toward target)
            # Option 1: Do nothing (keep last EMA)
            # Option 2: Gentle decay toward target (optional stabilizer)
            # For now: do nothing (pure EMA)
            pass
```

**Note:** `new_measurement_available = 1` every K=20 frames, `0` otherwise.

---

#### Kernel 4: `update_radii_topological` (Modified)

**Purpose:** Adjust radii based on **topological EMA** (not geometric contacts).

**Logic:**
```python
@ti.kernel
def update_radii_topological(rad, topo_deg_ema, n):
    for i in range(n):
        # Current topological degree (smoothed)
        deg_topo = topo_deg_ema[i]
        
        # XPBD compliance (frame-rate independent)
        alpha = RADIUS_COMPLIANCE / (DT * DT)
        
        # Compute target radius change
        delta_r = 0.0
        
        if deg_topo < TOPO_DEG_LOW:
            # Too few neighbors → grow
            delta_r = rad[i] * GAIN_GROW
        elif deg_topo > TOPO_DEG_HIGH:
            # Too many neighbors → shrink
            delta_r = -rad[i] * GAIN_SHRINK
        # else: in target band [10, 18] → no change
        
        # Apply XPBD spring constraint
        lambda_r = delta_r / (1.0 + alpha)
        
        # Rate limiter (prevent runaway growth/collapse)
        max_change = RADIUS_RATE_LIMIT * DT
        lambda_r = ti.math.clamp(lambda_r, -max_change, max_change)
        
        # Update radius with hard bounds
        rad[i] = ti.math.clamp(rad[i] + lambda_r, R_MIN, R_MAX)
```

---

### 3. Main Loop Integration

**Pseudocode:**
```python
frame = 0
while window.running:
    # 1. Rebuild spatial grid (every frame)
    rebuild_grid(pos, rad, active_n)
    
    # 2. Compute geometric neighbors (for PBD)
    count_neighbors(pos, rad, deg_geom, active_n)  # Geometric degree (contacts)
    
    # 3. Compute topological neighbors (every K frames)
    if TOPO_ENABLED and (frame % TOPO_UPDATE_CADENCE == 0):
        build_topo_candidates(pos, rad, topo_pairs, topo_pair_count, active_n)
        gabriel_test_topological_degree(pos, rad, topo_deg, topo_pairs, topo_pair_count, 
                                         cell_start, cell_count, cell_indices, active_n)
        update_topo_ema(topo_deg, topo_deg_ema, active_n, new_measurement=1)
    else:
        update_topo_ema(topo_deg, topo_deg_ema, active_n, new_measurement=0)
    
    # 4. Radius adaptation (use topological degree if enabled)
    if TOPO_ENABLED:
        update_radii_topological(rad, topo_deg_ema, active_n)
    else:
        update_radii_xpbd(rad, deg_geom, active_n)  # Fallback to geometric
    
    # 5. PBD separation (uses geometric contacts)
    for pass in range(adaptive_pbd_passes):
        rebuild_grid(pos, rad, active_n)  # Re-scatter after position updates
        project_overlaps(pos, rad, active_n)
    
    # 6. Deep-overlap force fallback (if needed)
    if max_depth > threshold:
        apply_repulsive_forces(pos, rad, vel, active_n)
        integrate_velocities(pos, vel, active_n)
        apply_global_damping(vel, active_n)
    
    # 7. Render
    update_colors(rad, topo_deg_ema if TOPO_ENABLED else deg_geom, active_n)
    render(pos, rad, color, active_n)
    
    frame += 1
```

---

## Expected Outcomes

### Phase B Success Criteria

1. **Topological degree converges to ~14** (±3)
   - Mean degree: 11-17 (stable over 500+ frames)
   - Std dev: < 4
   - Max degree: < 30 (no extreme outliers)

2. **Radius distribution remains diverse**
   - Min/mean/max spread: 3:1 or better (e.g., 0.003 / 0.005 / 0.009)
   - No collapse to uniform sizes

3. **Stability preserved**
   - MaxDepth stays < 0.002 (< 1% of mean radius)
   - PBD passes: 4-12 (not constantly maxing out at 24)
   - FPS: > 10 for N=5000

4. **Visual quality**
   - Particles form irregular foam-like structure (not crystalline grid)
   - Degree color distribution: mostly green (in-band), few red/blue outliers
   - Smooth transitions (no jitter from EMA smoothing)

### What Changes from Phase A

| Metric                | Phase A (Geometric)  | Phase B (Topological) |
|-----------------------|----------------------|-----------------------|
| **Target Degree**     | 3-6                  | 10-18 (mean ~14)      |
| **Degree Metric**     | Contact threshold    | **Gabriel graph** (Euclidean or Laguerre) |
| **Update Frequency**  | Every frame          | Every 20 frames       |
| **Smoothing**         | None                 | EMA (β=0.1)           |
| **Witness Search**    | N/A                  | **Adaptive stencil** (1-7 cells, scaled by pair distance) |
| **Epsilon Handling**  | Fixed (1e-9)         | **Scaled** (1e-6 × r²) |
| **Pair Processing**   | N/A                  | **Symmetric** (j > i, halves work) |
| **PBD Separation**    | Geometric contacts   | Geometric contacts    |
| **Radius Adaptation** | Geometric contacts   | **Topological degree** (blended first 200 frames) |
| **Computation Cost**  | Low (~N)             | Medium (~20-30N per update, ~1-1.5N/frame amortized) |

---

## Performance Considerations

### Computational Cost

**Witness Test Complexity:**
- For each particle `i`: `O(N_neighbors_i)`
- For each neighbor `j`: Check all witnesses `k`: `O(N_cell)`
- Worst case: `O(N × N_neighbors × N_cell)`
- **With spatial pruning:** ~`O(N × 30 × 100)` = 3000N ops/update
- **Amortized (every 20 frames):** ~150N ops/frame (tolerable)

**Optimization Strategies:**
1. **Spatial pruning:** Only test pairs within `2.5 × (r_i + r_j)` (already included)
2. **Early exit:** Stop witness test on first violation (already included)
3. **Cadence:** Update every K=20 frames (already included)
4. **Parallelism:** All pair tests are independent (GPU-friendly)

### Memory Overhead

- `topo_deg`: 4 bytes × MAX_N = 200 KB (N=50k)
- `topo_deg_ema`: 4 bytes × MAX_N = 200 KB
- `topo_pairs`: 4 bytes × MAX_N × 32 = 6.4 MB
- `topo_pair_count`: 4 bytes × MAX_N = 200 KB
- **Total:** ~7 MB (acceptable)

### Scaling

| N     | Topo pairs (avg) | Witness checks (avg) | Time/update (est) |
|-------|------------------|----------------------|-------------------|
| 1k    | 15k              | 1.5M                 | ~5 ms             |
| 5k    | 75k              | 7.5M                 | ~25 ms            |
| 10k   | 150k             | 15M                  | ~50 ms            |
| 30k   | 450k             | 45M                  | ~150 ms           |

**Note:** At N > 30k, consider switching to GPU Delaunay triangulation libraries (beyond scope of Phase B).

---

## Tuning Knobs

### Critical Parameters

1. **`TOPO_TARGET_DEG = 14`**
   - Theoretical: 14 (tetrahedral close packing)
   - Empirical: 12-16 (depends on polydispersity)
   - **If too low:** Particles over-expand, low density
   - **If too high:** Particles shrink, crystallize

2. **`TOPO_DEG_LOW = 10`, `TOPO_DEG_HIGH = 18`**
   - Band width: 8 (allows ±28% variation from target)
   - **Wider band:** More diversity, slower convergence
   - **Narrower band:** Faster convergence, less diversity

3. **`TOPO_UPDATE_CADENCE = 20`**
   - Range: 10-50 frames
   - **Lower:** More responsive, higher CPU cost
   - **Higher:** Cheaper, laggier response

4. **`TOPO_EMA_BETA = 0.1`**
   - Range: 0.05-0.20
   - **Lower:** Smoother (slower response to changes)
   - **Higher:** More reactive (less smoothing)

5. **`TOPO_MAX_RADIUS_MULTIPLE = 2.5`**
   - Only test pairs within this × (r_i + r_j)
   - **Lower (2.0):** Faster, may miss distant Voronoi neighbors
   - **Higher (3.0):** More accurate, slower

### Starting Values (Conservative)

```python
# Phase B defaults (copy to config.py)
TOPO_ENABLED = True
TOPO_UPDATE_CADENCE = 20
TOPO_EMA_BETA = 0.1
TOPO_TARGET_DEG = 14
TOPO_DEG_LOW = 10
TOPO_DEG_HIGH = 18
TOPO_MAX_RADIUS_MULTIPLE = 2.5
MAX_TOPO_NEIGHBORS = 32
```

---

## Robustness & Performance Tweaks

### 0. Adaptive Witness Stencil (IMPLEMENTED)

✅ **Status:** Included in core blueprint (Kernel 2)

Stencil size scales with pair distance: `stencil_cells = ceil(r_diameter / CELL_SIZE)`, clamped to [1, 3].

---

### 1. Candidate Pair Overflow Protection

**Problem:** `topo_pairs[i, MAX_TOPO_NEIGHBORS]` can truncate if particle `i` has > 32 candidates.

**Solution:**
```python
# Add to config.py
TOPO_TRUNCATION_WARNING_THRESHOLD = 0.02  # Warn if >2% of particles truncate

# Add tracking field
topo_truncated = ti.field(ti.i32, shape=MAX_N)  # 1 if particle truncated, 0 otherwise

# In build_topo_candidates kernel:
@ti.kernel
def build_topo_candidates(pos, rad, topo_pairs, topo_pair_count, topo_truncated, n):
    for i in range(n):
        topo_pair_count[i] = 0
        topo_truncated[i] = 0  # Reset truncation flag
    
    for i in range(n):
        # ... (iterate candidate pairs j) ...
        current_count = topo_pair_count[i]
        if current_count < MAX_TOPO_NEIGHBORS:
            topo_pairs[i, current_count] = j
            topo_pair_count[i] += 1
        else:
            topo_truncated[i] = 1  # Mark as truncated

# In main loop (every topological update):
truncated_count = np.sum(topo_truncated.to_numpy()[:active_n])
if truncated_count > TOPO_TRUNCATION_WARNING_THRESHOLD * active_n:
    print(f"[WARNING] {truncated_count}/{active_n} particles truncated candidate list!")
    print(f"          Consider increasing MAX_TOPO_NEIGHBORS or reducing TOPO_MAX_RADIUS_MULTIPLE")
```

**Action:** If warnings persist, increase `MAX_TOPO_NEIGHBORS` from 32 → 48 or reduce `TOPO_MAX_RADIUS_MULTIPLE` from 2.5 → 2.0.

---

### 2. Symmetric Edge Storage (Optional)

**Why:** If you later need the actual neighbor list (not just degrees), symmetric storage avoids redundant pair processing.

**Implementation:**
```python
# Only process each unique pair (i, j) once (where j > i)
# Then add edge to BOTH i and j's neighbor lists

if is_gabriel_neighbor and j > i:  # Only process each pair once
    ti.atomic_add(topo_deg[i], 1)
    ti.atomic_add(topo_deg[j], 1)  # Symmetric
```

**Trade-off:** Simpler logic vs. potential atomic contention on `topo_deg[j]` (negligible for N < 50k).

---

### 3. Dynamic Cadence (Advanced)

**Goal:** Update topological degree more frequently when system is far from equilibrium, less frequently when stable.

**Strategy:**
```python
# Add EMA variance tracker
topo_deg_variance = ti.field(ti.f32, shape=())  # Single scalar for global variance

@ti.kernel
def compute_topo_variance(topo_deg_ema, variance, n):
    # Compute variance of EMA degrees
    mean_deg = 0.0
    for i in range(n):
        mean_deg += topo_deg_ema[i]
    mean_deg /= n
    
    var = 0.0
    for i in range(n):
        diff = topo_deg_ema[i] - mean_deg
        var += diff * diff
    var /= n
    variance[None] = var

# In main loop:
if frame % TOPO_UPDATE_CADENCE == 0:
    compute_topo_variance(topo_deg_ema, topo_deg_variance, active_n)
    var = topo_deg_variance[None]
    
    # Adapt cadence based on variance
    if var > 10.0:  # High variance → update more often
        TOPO_UPDATE_CADENCE = 10
    elif var < 2.0:  # Low variance → back off
        TOPO_UPDATE_CADENCE = 30
    else:
        TOPO_UPDATE_CADENCE = 20  # Default
```

**Implementation:** Phase B.2 (after basic system works). Not critical for initial rollout.

---

### 4. PBC Midpoint Edge Case

**Critical Test:** When i and j straddle the periodic boundary, ensure midpoint wrapping is correct.

**Example:**
```python
# Test case: i near left boundary, j near right boundary
pos[i] = [-0.074, 0, 0]  # Near -L/2
pos[j] = [+0.074, 0, 0]  # Near +L/2

# Naive midpoint: (pos[i] + pos[j])/2 = [0, 0, 0] ✓ (correct)
# But if pos[j] wrapped: pos[j] = [-0.076, 0, 0] (after wrapping from +0.074)
# Then: (pos[i] + pos[j])/2 = [-0.075, 0, 0] ✗ (wrong! should be [0, 0, 0])

# CORRECT approach (as in blueprint):
delta_ij = pdelta(pos[i], pos[j])  # Minimum-image vector
midpoint = wrapP(pos[i] + 0.5 * delta_ij)  # Always correct
```

**Validation:** Add unit test in Phase B.1 to check cross-boundary pairs manually.

---

### 5. RNG Fallback (If Gabriel Degrees Too High)

**If mean degree > 20** after Phase B convergence, Gabriel graph may be too permissive (e.g., high polydispersity).

**RNG Predicate (stricter):**
```python
# Replace Gabriel test with RNG test:
# i-j are RNG neighbors iff:
#   ∀k: max(d(i,k), d(j,k)) >= d(i,j) - ε

# In gabriel_test kernel, replace inner loop:
for k in candidate_particles:
    if k == i or k == j: continue
    
    dist_sq_ik = pdelta(pos[k], pos[i]).dot(...)
    dist_sq_jk = pdelta(pos[k], pos[j]).dot(...)
    
    # RNG test: if k is in the "lune" (closer to both i and j than they are to each other)
    if ti.max(dist_sq_ik, dist_sq_jk) < d_ij_sq - EPS:
        is_rng_neighbor = False
        break
```

**When to Use:** Only if Gabriel gives mean degree > 18. Start with Gabriel (simpler, faster).

---

### 6. Cone Prefilter (Candidate Pruning Optimization)

**Goal:** Reject far candidates that are unlikely to be Gabriel neighbors before witness loop.

**Strategy:** After distance pruning, reject candidates whose angle deviates too much from local principal direction.

```python
# In build_topo_candidates kernel, after distance check:
if dist_ij <= max_topo_radius:
    # Cone filter: reject if too far off-axis
    # (Optional: compute principal direction from k nearest neighbors)
    # For now, use simple heuristic: reject if dot product too low
    
    # Example: Check angular coherence with previously accepted candidates
    # If topo_pair_count[i] > 0:
    #     prev_j = topo_pairs[i, topo_pair_count[i] - 1]
    #     delta_prev = pdelta(pos[i], pos[prev_j])
    #     cos_angle = delta_ij.dot(delta_prev) / (dist_ij * delta_prev.norm() + EPS)
    #     if cos_angle < -0.5:  # > 120° off → skip
    #         continue
    
    # Add to candidate list
    ...
```

**When to Implement:** Phase B.2 (after basic Gabriel works). Adds ~10-20% speedup in dense packs.

**Expected Impact:** Reduces candidate list by 10-30% without affecting topological accuracy.

---

### 7. Adaptive Candidate Bounds (Density-Aware)

**Goal:** Prevent systematic truncation in high-density regions by reducing search radius.

**Strategy:**
```python
# In build_topo_candidates, compute local density before iterating:
@ti.kernel
def build_topo_candidates_adaptive(pos, rad, topo_pairs, topo_pair_count, 
                                     cell_start, cell_count, cell_indices, n):
    for i in range(n):
        # Estimate local density from own cell
        my_cell_id = ...  # (compute as in existing kernels)
        local_density = cell_count[my_cell_id]
        
        # Adapt search radius based on density
        if local_density > 50:  # Hot spot threshold
            max_topo_radius_local = TOPO_MAX_RADIUS_MULTIPLE * 2.0 * rad[i]
        elif local_density > 30:
            max_topo_radius_local = TOPO_MAX_RADIUS_MULTIPLE * 2.2 * rad[i]
        else:
            max_topo_radius_local = TOPO_MAX_RADIUS_MULTIPLE * 2.5 * rad[i]
        
        # Use max_topo_radius_local instead of global constant
        ...
```

**When to Implement:** Phase B.2 or if truncation warnings persist after increasing `MAX_TOPO_NEIGHBORS`.

---

### 8. Cross-Check with CPU Delaunay (Validation)

**Goal:** Measure precision/recall of Gabriel graph vs. ground-truth Voronoi adjacency.

**Procedure:**
```python
# For N=100 particles, compute:
# 1. Gabriel edges (GPU)
# 2. Delaunay triangulation (CPU, scipy.spatial.Delaunay)
# 3. Compare: 
#    - Precision = |Gabriel ∩ Delaunay| / |Gabriel|
#    - Recall    = |Gabriel ∩ Delaunay| / |Delaunay|

from scipy.spatial import Delaunay

pos_np = pos.to_numpy()[:100]

# CRITICAL: Unwrap PBC positions into continuous cell
# If PBC is enabled, shift all positions relative to particle 0
# to avoid torus splits that confuse Delaunay
if PBC_ENABLED:
    pos_unwrapped = np.copy(pos_np)
    ref = pos_np[0]
    for i in range(1, len(pos_np)):
        # Use minimum-image convention to unwrap
        delta = pos_np[i] - ref
        # Wrap delta into [-L/2, L/2]
        delta = delta - DOMAIN_SIZE * np.floor(delta / DOMAIN_SIZE + 0.5)
        pos_unwrapped[i] = ref + delta
    pos_np = pos_unwrapped

tri = Delaunay(pos_np)  # Delaunay triangulation (on unwrapped positions)

# Extract Delaunay edges from tri.simplices (tetrahedra in 3D)
delaunay_edges = set()
for simplex in tri.simplices:
    for i in range(4):
        for j in range(i+1, 4):
            edge = tuple(sorted([simplex[i], simplex[j]]))
            delaunay_edges.add(edge)

# Extract Gabriel edges from topo_deg
gabriel_edges = set()
# ... (from your topo_pairs storage) ...

# Compute metrics
intersection = gabriel_edges & delaunay_edges
precision = len(intersection) / len(gabriel_edges)
recall = len(intersection) / len(delaunay_edges)

print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")
# Expected: Precision > 95%, Recall > 70-80%
```

**When to Run:** Phase B.1 validation (N=100), then spot-check at N=1000.

---

### 9. PBC Antipodal Witness Validation

**Goal:** Validate midpoint wrapping and witness detection across periodic boundaries.

**Test Case:**
```python
# Specific test: i and j nearly antipodal across boundary, k inside wrapped diameter sphere
# Place i near -L/2, j near +L/2, k near the wrapped midpoint
pos_np = np.zeros((3, 3), dtype=np.float32)
pos_np[0] = [-DOMAIN_SIZE*0.49, 0, 0]  # i near left boundary
pos_np[1] = [DOMAIN_SIZE*0.49, 0, 0]   # j near right boundary
pos_np[2] = [0, 0, 0]                   # k at origin (center of domain)

# Compute expected behavior:
# - Raw distance i-j: ~0.98*L (almost full domain)
# - PBC distance i-j: ~0.02*L (wraps around)
# - Midpoint (PBC-aware): ~0 (center)
# - Diameter sphere radius: ~0.01*L
# - k should be OUTSIDE diameter sphere (d(k, midpoint) = 0 < r_diameter)
# - Therefore: k is a witness, i-j should NOT be Gabriel neighbors

# Run gabriel_test_topological_degree and verify i-j not connected
```

**Expected Result:** i-j pair correctly rejected despite being physically separated by ~L.

**Why Critical:** Classic PBC bug where midpoint wrapping fails, leading to false positives.

---

### 10. Blend Metric Early (Stability)

**Goal:** Avoid abrupt radius swings when topological graph first activates.

**Strategy:** For the first few hundred frames, blend topological and geometric degrees:
```python
# Add to config.py
TOPO_BLEND_FRAMES = 200  # Frames to blend metrics
TOPO_BLEND_LAMBDA_START = 0.8  # Start with 80% topo, 20% geom
TOPO_BLEND_LAMBDA_END = 1.0  # Fade to 100% topo

# In main loop (after topological degree update):
if frame < TOPO_BLEND_FRAMES:
    # Linear fade from 0.8 → 1.0 over first 200 frames
    blend_lambda = TOPO_BLEND_LAMBDA_START + (TOPO_BLEND_LAMBDA_END - TOPO_BLEND_LAMBDA_START) * (frame / TOPO_BLEND_FRAMES)
    
    # Blend topological and geometric degrees
    @ti.kernel
    def blend_degrees(topo_deg_ema, deg_geom, deg_eff, blend_lambda, n):
        for i in range(n):
            deg_eff[i] = blend_lambda * topo_deg_ema[i] + (1.0 - blend_lambda) * deg_geom[i]
    
    blend_degrees(topo_deg_ema, deg_geom, deg_effective, blend_lambda, active_n)
    
    # Use deg_effective for radius adaptation
    update_radii_topological(rad, deg_effective, active_n)
else:
    # Pure topological after blend period
    update_radii_topological(rad, topo_deg_ema, active_n)
```

**When to Implement:** Phase B.1 (prevents initial oscillations).

**Critical:** Freeze cadence `K=20` during blend window (don't change two knobs at once).

**Expected Behavior:** Smooth transition, no sudden jumps in radii or degree distribution.

---

### 10. Deadband + Integral Cap (Ratcheting Prevention)

**Goal:** Prevent slow radius drift when EMA oscillates around target.

**Strategy:**
```python
# Add to config.py
RADIUS_DEADBAND = 0.5  # No change if |deg - target| < 0.5
RADIUS_INTEGRAL_CAP = 0.02  # Max cumulative change per 100 frames

# Modify update_radii_topological kernel:
@ti.kernel
def update_radii_topological_stable(rad, topo_deg_ema, rad_integral, frame, n):
    for i in range(n):
        deg = topo_deg_ema[i]
        
        # Deadband: no change if within tolerance
        if TOPO_DEG_LOW - RADIUS_DEADBAND <= deg <= TOPO_DEG_HIGH + RADIUS_DEADBAND:
            continue
        
        # Compute radius change
        if deg < TOPO_DEG_LOW:
            delta_r = GAIN_GROW * rad[i]
        elif deg > TOPO_DEG_HIGH:
            delta_r = -GAIN_SHRINK * rad[i]
        else:
            delta_r = 0.0
        
        # Integral cap: track cumulative change per 100 frames
        if frame % 100 == 0:
            rad_integral[i] = 0.0  # Reset every 100 frames
        
        rad_integral[i] += ti.abs(delta_r)
        if rad_integral[i] > RADIUS_INTEGRAL_CAP:
            delta_r *= 0.5  # Halve change if cap exceeded
        
        # Apply radius change
        rad[i] = ti.max(R_MIN, ti.min(R_MAX, rad[i] + delta_r))
```

**When to Implement:** Phase B.2 (after basic convergence works).

**Expected Behavior:** Tighter control, less jitter in converged state.

---

### 11. Enhanced Telemetry

**Goal:** Make tuning trivial by logging key metrics every topological update.

**Metrics to Add:**
```python
# In main loop, after topological update:
if TOPO_ENABLED and (frame % TOPO_UPDATE_CADENCE == 0):
    # Existing: compute topological degree
    build_topo_candidates(...)
    gabriel_test_topological_degree(...)
    
    # NEW: Compute telemetry
    truncated_count = np.sum(topo_truncated.to_numpy()[:active_n])
    truncated_pct = 100.0 * truncated_count / active_n
    
    mean_pairs = np.mean(topo_pair_count.to_numpy()[:active_n])
    max_pairs = np.max(topo_pair_count.to_numpy()[:active_n])
    
    # Estimate early-exit rate (requires adding counter in kernel)
    # early_exit_rate = num_early_exits / total_witness_checks
    
    topo_deg_np = topo_deg.to_numpy()[:active_n]
    deg_histogram = np.histogram(topo_deg_np, bins=[0, 5, 10, 15, 20, 25, 100])
    
    geom_deg_np = deg_geom.to_numpy()[:active_n]
    
    print(f"[Topo Update] Frame={frame}")
    print(f"    Truncated: {truncated_pct:.1f}% ({truncated_count}/{active_n})")
    print(f"    Pairs/particle: mean={mean_pairs:.1f}, max={max_pairs}")
    print(f"    Topo degree: mean={topo_deg_np.mean():.2f}, std={topo_deg_np.std():.2f}")
    print(f"    Geom degree: mean={geom_deg_np.mean():.2f}, std={geom_deg_np.std():.2f}")
    print(f"    Histogram (topo): {deg_histogram[0]}")
```

**Frequency:** Every topological update (every K frames) or every 200 frames for histogram.

**Ship with Telemetry ON by Default:** Log truncation counter, mean/max pairs/particle alongside % truncated every update.

**Expected Output (healthy):**
```
[Topo Update] Frame=400
    Truncated: 0.2% (10/5000)
    Pairs/particle: mean=18.3, max=27
    Topo degree: mean=13.8, std=3.2
    Geom degree: mean=5.4, std=1.8
    Histogram (topo): [120, 1200, 2800, 800, 70, 10]
```

**Red Flags:**
- Truncated > 5% → Increase `MAX_TOPO_NEIGHBORS` or reduce `TOPO_MAX_RADIUS_MULTIPLE`
- Mean pairs > 40 → Reduce `TOPO_MAX_RADIUS_MULTIPLE`
- Topo degree std > 5 → Increase EMA smoothing (reduce `TOPO_EMA_BETA`)

---

## Testing Strategy

### Phase B.1: Validation (N=100, 10 min)

**Goal:** Confirm Gabriel graph test works correctly.

**Procedure:**
1. **Cross-check with CPU Delaunay** (N=100):
   - Run Gabriel test on GPU
   - Compute Delaunay triangulation on CPU (scipy)
   - Measure precision/recall (see "Robustness Tweaks #6")
   - **Expected:** Precision > 95%, Recall > 70-80%
   - **Red flag:** Precision < 90% → Gabriel test has bugs

2. **PBC edge case test**:
   - Manually place 2 particles straddling boundary:
     - `pos[0] = [-0.074, 0, 0]` (near -L/2)
     - `pos[1] = [+0.074, 0, 0]` (near +L/2)
   - Verify midpoint = [0, 0, 0] (not [-0.075, 0, 0])
   - Verify third particle at [0, 0.001, 0] correctly witnesses edge

3. **Manual spot-check** (N=100):
   - Pick particle `i` with `topo_deg[i] = 12`
   - For each of its 12 neighbors `j`, verify:
     - Diameter sphere (midpoint, r = d_ij/2) contains no other particles
   - Use Python script to validate 3-5 random pairs

**Expected:**
- Mean degree: 12-14 (Gabriel typical for random 3D points)
- Degree histogram: bell curve centered at 13-14
- Delaunay precision/recall passing thresholds

---

### Phase B.2: Convergence Test (N=5000, 10 min)

**Goal:** Confirm topological degree stabilizes at target.

**Procedure:**
1. Run with `N=5000`, `TOPO_ENABLED=True`
2. Monitor telemetry:
   - Mean topological degree every 100 frames
   - Radius distribution (min/mean/max)
   - PBD passes, MaxDepth

**Success Criteria:**
- Mean degree: 11-17 (stable after 2000 frames)
- Radius diversity: min/max ratio < 4:1
- MaxDepth: < 0.002 (stable)

---

### Phase B.3: Visual Inspection

**Goal:** Confirm foam-like structure.

**What to Look For:**
- Particles form **irregular clusters** (not cubic lattice)
- Degree color: mostly green, few red/blue outliers
- Smooth motion (no jitter from EMA)
- No "dead zones" (particles with deg=0)

**Red Flags:**
- Crystalline order (particles in grid pattern)
- Bimodal degree distribution (two distinct populations)
- Rapid oscillation in degree (EMA not working)

---

### Phase B.4: Stress Test (N=10k, 5 min)

**Goal:** Confirm performance at scale.

**Procedure:**
1. Run with `N=10000`
2. Monitor FPS, frame time
3. Check if witness test dominates frame budget

**Expected:**
- FPS: > 10 (acceptable)
- Witness test time: < 50 ms (every 20 frames → 2.5 ms/frame amortized)
- No memory issues

---

## Implementation Order

### Step 1: Data Structures (10 min)
- Add Taichi fields: `topo_deg`, `topo_deg_ema`, `topo_pairs`, `topo_pair_count`
- Add config parameters to `config.py`
- Initialize fields in `run.py`

### Step 2: Candidate Pruning (30 min)
- Implement `build_topo_candidates` kernel in `topology.py`
- Test: confirm candidate pairs match spatial neighbors (compare vs `count_neighbors`)

### Step 3: Witness Test (45 min)
- Implement `witness_test_topological_degree` kernel
- Test: manually verify 10 random pairs (Python validation script)

### Step 4: EMA Smoothing (15 min)
- Implement `update_topo_ema` kernel
- Test: plot EMA vs raw degree over 500 frames (should be smoother)

### Step 5: Radius Adaptation (20 min)
- Modify `update_radii_xpbd` → `update_radii_topological`
- Use `topo_deg_ema` instead of `deg` for adaptation

### Step 6: Main Loop Integration (20 min)
- Add topological update logic to `run.py`
- Add `if frame % TOPO_UPDATE_CADENCE == 0:` gate
- Update GUI telemetry (display topological degree)

### Step 7: Testing & Tuning (60 min)
- Run validation tests (B.1 - B.4)
- Tune `TOPO_TARGET_DEG`, `TOPO_DEG_LOW/HIGH`, `TOPO_UPDATE_CADENCE`
- Capture before/after screenshots

**Total Time:** ~3 hours (assuming Phase A is stable)

---

## Failure Modes & Mitigations

### 1. Topological Degree Too High (e.g., mean=25)

**Symptom:** All particles shrink to `R_MIN`.

**Cause:** Witness test too permissive (finding too many neighbors).

**Fix:**
- Reduce `TOPO_MAX_RADIUS_MULTIPLE` (2.5 → 2.0)
- Increase `EPS` in witness test (allow small numerical slack)

---

### 2. Topological Degree Too Low (e.g., mean=5)

**Symptom:** All particles grow to `R_MAX`, overlaps explode.

**Cause:** Witness test too strict (rejecting valid neighbors).

**Fix:**
- Increase `TOPO_MAX_RADIUS_MULTIPLE` (2.5 → 3.0)
- Check for PBC bugs in `pdelta` (distances wrapping incorrectly)

---

### 3. Degree Oscillates Wildly

**Symptom:** Mean degree swings 10 → 20 → 10 every 100 frames.

**Cause:** EMA not smoothing enough, or update cadence too high.

**Fix:**
- Reduce `TOPO_EMA_BETA` (0.1 → 0.05, more smoothing)
- Increase `TOPO_UPDATE_CADENCE` (20 → 30 frames)

---

### 4. Performance Collapse (FPS < 5)

**Symptom:** Witness test takes > 100 ms per update.

**Cause:** Too many candidate pairs, or N too large.

**Fix:**
- Reduce `TOPO_MAX_RADIUS_MULTIPLE` (2.5 → 2.0, fewer candidates)
- Increase `TOPO_UPDATE_CADENCE` (20 → 50 frames)
- Consider N < 10k for witness test (switch to Delaunay for N > 10k)

---

## Future Extensions (Beyond Phase B)

1. **GPU Delaunay Triangulation** (N > 30k)
   - Replace witness test with incremental Delaunay (exact Voronoi)
   - Libraries: CGAL, TetGen, or custom Taichi implementation

2. **Power Diagram / Weighted Voronoi**
   - Account for variable radii in Voronoi computation
   - Requires power distance: `d² - r²` instead of Euclidean `d`

3. **Adaptive Cadence**
   - Update topological degree more frequently when system is far from equilibrium
   - Reduce frequency when degree is stable

4. **Hybrid Geometric + Topological**
   - Use geometric contacts for immediate response (jitter prevention)
   - Use topological degree for long-term trends (foam structure)

---

## Rollback Plan

If Phase B causes instability or performance issues, **rollback is trivial:**

1. Set `TOPO_ENABLED = False` in `config.py`
2. Simulation reverts to Phase A (geometric contacts only)
3. No code changes needed (clean fallback path)

**Rollback trigger:**
- FPS < 5 for N=5000
- Mean degree doesn't stabilize after 5000 frames
- Radius collapse (all particles → R_MIN or R_MAX)

---

## Summary

**Phase B adds:**
- ✅ Topological neighbor counting via witness test
- ✅ EMA smoothing for stable radius adaptation
- ✅ Target degree = 14 (tetrahedral packing)
- ✅ Amortized computation (every 20 frames)
- ✅ Clean fallback to Phase A (geometric only)

**What doesn't change:**
- PBD separation (still uses geometric contacts)
- Force fallback (still uses geometric contacts)
- Grid rebuild (every frame, same as Phase A)
- Overall simulation loop structure

**Risk level:** Low
- Witness test is self-contained (no side effects)
- `TOPO_ENABLED` flag allows instant rollback
- Performance cost is amortized (every 20 frames)

**Go/No-Go Decision Point:**
- After Step 3 (witness test working), verify correctness on N=100 (manual check)
- If incorrect, debug witness logic before proceeding
- If correct, proceed to Steps 4-7 (integration)

---

---

## Critical Corrections Applied (Rev 3)

**Major Fix (Rev 2):** Replaced incorrect witness test with **Gabriel graph** predicate.

**Original (BROKEN):**
```python
# This is almost always true by triangle inequality!
if d(k,i) + d(k,j) > d(i,j):  # ✗ Wrong predicate
    is_neighbor = True
```

**Corrected (Gabriel graph):**
```python
# Empty diameter-sphere test (Gabriel graph)
midpoint = wrapP(pos[i] + 0.5 * pdelta(pos[i], pos[j]))
r_sq = 0.25 * d_ij_sq
if ||pos[k] - midpoint||² < r_sq - ε:  # ✓ Correct predicate
    is_neighbor = False  # Witness found
```

**High-Impact Refinements (Rev 3):**
1. ✅ **Adaptive witness stencil** - `stencil_cells = ceil(r_diameter / CELL_SIZE)`, clamped to `[1, min(3, GRID_RES//2)]` (prevents missed witnesses + grid overflow)
2. ✅ **Scaled epsilon** - `eps = max(1e-12, 1e-6 * r²)` (avoids size-dependent misclassification)
3. ✅ **Symmetric processing** - Each pair (i, j) processed once (j > i), both degrees updated (halves work)
4. ✅ **Laguerre-Gabriel toggle** - Compile-time switch for power diagram (handles polydispersity), `power_radius > 0` guard
5. ✅ **Edge guards** - Skip pairs with `d_ij_sq < 1e-16` (avoid degenerate diameter spheres)
6. ✅ **Candidate consistency** - Use `pdelta + dist_sq` in `build_topo_candidates` (PBC + perf)
7. ✅ **Telemetry on by default** - Log truncation %, mean/max pairs/particle every update
8. ✅ **Blend with frozen cadence** - Keep `K=20` fixed during 200-frame topo/geom blend

**Stability & Integration Enhancements:**
9. ✅ PBC antipodal witness validation (classic wrap bug catcher)
10. ✅ Blend metric early (topo + geom → pure topo over 200 frames, frozen cadence)
11. ✅ Deadband + integral cap (prevents ratcheting)
12. ✅ Enhanced telemetry (truncation %, pairs checked, histogram, ON by default)
13. ✅ PBC Delaunay unwrapping (fixes cross-check edge cases)

**Optional Performance Tweaks (Phase B.2):**
14. ⏸ Cone prefilter (angular rejection before witness loop)
15. ⏸ Adaptive candidate bounds (density-aware search radius)
16. ⏸ Dynamic cadence from variance (ship with K=20 fixed)

**Expected Outcome:** Mean degree 12-16 (Euclidean Gabriel), 10-14 (Laguerre Gabriel), target band 10-18.

---

**Ready for implementation? ✅**

**Acknowledgment:** Rev 2 caught the catastrophic triangle inequality bug. Rev 3 adds the polish to make this production-ready—adaptive stencil, scaled epsilon, and symmetric processing prevent 90% of "why is my degree off?" bugs. The Laguerre toggle gives a clean path to power diagrams without jumping to full Voronoi yet.

