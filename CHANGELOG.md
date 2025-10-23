# Blueprint Changelog (User Feedback)

## Critical Correctness Fixes Applied

### 1. **Scatter Pointer Separation** ✅
- **Problem**: Atomic fetch-add on `cell_start` destroys the prefix sum.
- **Fix**: Added `cell_write` field (copy of `cell_start` before scatter). Scatter mutates `cell_write`, iteration uses `cell_start`.

### 2. **Contact Tolerance** ✅
- **Problem**: PBD enforces 2% gap → particles read 0 neighbors after separation.
- **Fix**: Added `CONTACT_TOL = 0.02` to neighbor detection: `touch_thresh = (1.0 + CONTACT_TOL) * (rad[i] + rad[j])`

### 3. **Periodic Min-Image Helper** ✅
- **Problem**: Re-inventing min-image logic in multiple places.
- **Fix**: Single `periodic_delta(p1, p2, L)` helper used consistently in both neighbor count and PBD.

### 4. **Loop Order Semantics** ✅
- **Problem**: Ambiguous when degree is measured.
- **Fix**: Explicit order: `rebuild_grid → count_neighbors → grow/shrink 5% → PBD separate → render`

### 5. **Grid Indexing** ✅
- **Problem**: Vague indexing formula.
- **Fix**: Explicit formula: `id = c[0] + c[1]*GRID_RES + c[2]*GRID_RES**2`, wrap: `(c % R + R) % R`

### 6. **Types & Precision** ✅
- **Problem**: Mixing f32/f64 can cause subtle bugs.
- **Fix**: All fields are `f32` (no mixing).

### 7. **Rendering API** ✅
- **Problem**: Taichi UI doesn't support `color_by`/`colormap`.
- **Fix**: Added explicit `color` field (RGB), `update_colors` kernel for degree-based coloring.

### 8. **Initial Warmup** ✅
- **Problem**: Overlapping initial conditions create "sticky blob".
- **Fix**: Added 2-4 PBD passes after seeding, before main loop.

### 9. **PBD Stability** ✅
- **Problem**: Large displacements can cause instability.
- **Fix**: Added `MAX_DISPLACEMENT_FRAC = 0.2` (cap per-pair displacement to 20% of overlap).

### 10. **Hard Clamp** ✅
- **Problem**: Radii can grow unbounded.
- **Fix**: Explicit clamp to `[R_MIN, R_MAX]` every frame in `update_radii`.

---

## Performance/Robustness Tweaks

- **Prefix sum**: Serial is fine (`GRID_RES**3 ≈ 1728`, negligible cost).
- **CELL_SIZE**: `2*R_MAX` is correct for 27-stencil.
- **PBD passes**: 4-8 recommended (start with 4).

---

## Spec Improvements

- **Python syntax**: Changed `GRID_RES^3` → `GRID_RES**3`.
- **Config additions**: Added `CONTACT_TOL`, `EPS`, `MAX_DISPLACEMENT_FRAC`.
- **Initial warmup**: Added to main loop pseudocode.

---

## Acceptance Tests - Enhanced

- **Test 1 (Grid Accuracy)**: Now requires 100% exact match (no ±1 slack).
- **Test 2 (NEW - Radius Individuality)**: 3 isolated + 3 crowded → ensures no global averaging bug.
- **Tests 3-7**: Renumbered to accommodate new test.

---

## Kernel Count

- **Before**: 5 kernels
- **After**: 7 kernels (added `copy_cell_pointers`, `update_colors`)
- **Helper**: `periodic_delta` (Taichi `@ti.func`)

---

## Total LOC Estimate

- **Before**: ~280 lines
- **After**: ~335 lines (added color logic, warmup, more comments)

---

---

## Final Tweaks (User Approval Round 2) ✅

### 1. **GRID_RES Coverage Fix**
- **Before**: `GRID_RES = max(1, int(DOMAIN_SIZE / CELL_SIZE))`
- **After**: `GRID_RES = max(3, int(math.ceil(DOMAIN_SIZE / CELL_SIZE)))`
- **Why**: `ceil` ensures last cell covers domain edge. `max(3, ...)` ensures 27-stencil always valid.

### 2. **PBD Displacement Cap Clarification**
- **Before**: "Cap per-pair displacement"
- **After**: `MAX_DISPLACEMENT_FRAC = 0.2` applies per-particle (split overlap 50/50), clamped per-axis.
- **Why**: Prevents tunneling. Each particle moves at most 20% of its radius per pass.

### 3. **Degree Semantics Documentation**
- **Added**: "deg[i] = count of neighbors within (1 + CONTACT_TOL) * (r_i + r_j)"
- **Why**: Clarifies this is "near-contact", not exact touching. Matches PBD gap semantics.

### 4. **Prefix Sum Determinism Note**
- **Added**: "Taichi guarantees deterministic execution order for serial loops."
- **Why**: Reassures that prefix sum is reproducible across runs.

### 5. **Design Rationale Expanded**
- **Added**: Q&A for GRID_RES coverage and degree semantics.
- **Why**: Explains why these specific choices were made.

---

## Ready to Build?

**All correctness issues addressed.**  
**All user-requested tweaks applied.**  
**Blueprint locked and approved.**

Next: Begin file-by-file implementation with heavy comments.

