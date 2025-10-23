# Final Pre-Build Checklist

## User-Requested Tweaks (All Applied ✅)

### 1. GRID_RES Coverage ✅
- **Issue**: `int(DOMAIN_SIZE / CELL_SIZE)` can leave last cell short of domain edge.
- **Fix**: `GRID_RES = max(3, int(math.ceil(DOMAIN_SIZE / CELL_SIZE)))`
- **Why**: `ceil` guarantees last cell covers edge. No particles fall outside grid.

### 2. Min Grid Size ✅
- **Issue**: If `CELL_SIZE` is very large (big radii), `GRID_RES` could be 1 or 2.
- **Fix**: `max(3, ...)` forces minimum 3x3x3 grid.
- **Why**: 27-stencil always valid. No edge cases with 1x1x1 or 2x2x2 grids.

### 3. PBD Displacement Cap ✅
- **Issue**: Need to clarify per-particle vs. per-pair capping.
- **Fix**: `MAX_DISPLACEMENT_FRAC = 0.2` applies per-particle (split overlap 50/50), clamped per-axis.
- **Why**: Prevents tunneling. Each particle moves at most 20% of its radius per pass.

### 4. Degree Semantics ✅
- **Issue**: "Degree" could mean exact touching or near-contact.
- **Fix**: Documented: `deg[i]` = neighbors within `(1 + CONTACT_TOL) * (r_i + r_j)`.
- **Why**: Matches PBD gap. Particles with `deg=0` will grow (not stuck).

### 5. Prefix Sum Determinism ✅
- **Issue**: Serial scan order must be deterministic.
- **Fix**: Added note: "Taichi guarantees deterministic execution order for serial loops."
- **Why**: Reassures that prefix sum is reproducible across runs.

---

## Blueprint Status

**All correctness issues addressed.**  
**All user-requested tweaks applied.**  
**Ready for implementation.**

---

## File Checklist (To Build)

1. **`config.py`** (25 lines)
   - Constants with `math.ceil`, `max(3, ...)`, updated comments
   
2. **`grid.py`** (150 lines)
   - 7 kernels: `clear_grid`, `count_particles_per_cell`, `prefix_sum`, `copy_cell_pointers`, `scatter_particles`, `count_neighbors`, `update_colors`
   - Helper: `periodic_delta`
   
3. **`dynamics.py`** (70 lines)
   - `update_radii`: Simple 5% rule with hard clamp
   - `project_overlaps`: PBD with per-particle displacement cap, min-image, per-axis clamp
   
4. **`run.py`** (90 lines)
   - Initialization + warmup (4 PBD passes)
   - Main loop: rebuild → count → adapt → PBD → color → render
   - GUI setup + camera

5. **`tests/`** (4 test files)
   - `test_grid_accuracy.py`: Brute-force vs grid (100% exact match)
   - `test_radius_individuality.py`: 3 isolated + 3 crowded (per-particle check)
   - `test_pbd_separation.py`: Overlap reduction
   - `test_periodicity.py`: Boundary wrapping

6. **`README.md`** (quick start)
7. **`requirements.txt`** (taichi, numpy)

---

## Estimated Build Time

- **config.py**: 5 min
- **grid.py**: 40 min (7 kernels + comments)
- **dynamics.py**: 25 min (2 kernels + PBD logic)
- **run.py**: 30 min (main loop + GUI)
- **tests/**: 30 min (4 test scripts)
- **docs**: 10 min (README, requirements.txt)

**Total: ~2.5 hours** (with heavy commenting)

---

## Next Action

**User approval** → Begin file-by-file implementation with full comments.

