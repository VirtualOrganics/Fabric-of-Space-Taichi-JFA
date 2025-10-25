"""
Main entry point for Fabric of Space - Custom Taichi Grid.

This script:
1. Initializes Taichi and allocates fields
2. Seeds particles with random positions and radii
3. Performs warmup PBD passes to prevent initial stickiness
4. Runs main loop: rebuild → count → adapt → PBD → color → render

Controls:
  - Right-click drag: Rotate camera
  - Mouse wheel: Zoom
  - SPACE: Pause/Resume
  - S: Export particle data (PROOF radii are changing!)
  - ESC: Exit
"""

import taichi as ti
import numpy as np
import time  # For performance profiling

# Import configuration and kernels
from config import (
    N, DOMAIN_SIZE, R_MIN, R_MAX, GRID_RES, VIS_SCALE, FPS_TARGET,
    PBD_BASE_PASSES, PBD_MAX_PASSES, PBD_ADAPTIVE_SCALE, PBD_SUBSTEPS,
    DEEP_OVERLAP_THRESHOLD, DEEP_OVERLAP_EXIT, FORCE_SUBSTEPS_MIN, FORCE_SUBSTEPS_MAX,
    RESCUE_ENABLED, RESCUE_STRENGTH,
    XSPH_ENABLED, DT, DEG_LOW, DEG_HIGH,
    PBC_ENABLED,
    GAIN_GROW, GAIN_SHRINK,
    # Phase B: Topological neighbor counting (optional, off by default)
    TOPO_ENABLED, TOPO_UPDATE_CADENCE, TOPO_EMA_ALPHA, TOPO_DEG_LOW, TOPO_DEG_HIGH,
    TOPO_BLEND_FRAMES, TOPO_BLEND_LAMBDA_START, TOPO_BLEND_LAMBDA_END,
    TOPO_TRUNCATION_WARNING_THRESHOLD,
    TOPO_BATCHES, TOPO_PAIR_SUBSAMPLE_Q, TOPO_WRITE_TO_EMA,
    USE_KNN_TOPO, KNN_TOPO_K,
    # Lévy Positional Diffusion (Track 2: topological regularization)
    LEVY_ENABLED, LEVY_ALPHA, LEVY_DEG_SPAN, LEVY_STEP_FRAC, LEVY_USE_TOPO_DEG,
    # Growth/Relax Rhythm (defaults only; runtime uses Taichi fields)
    GROWTH_RATE_DEFAULT, GROWTH_INTERVAL_DEFAULT, RELAX_INTERVAL_DEFAULT
)
from grid import (
    clear_grid, clear_all_particles, count_particles_per_cell, prefix_sum, copy_cell_pointers,
    scatter_particles, count_neighbors, update_colors,
    wrapP
)
from dynamics import (
    project_overlaps, init_velocities, compute_max_overlap,
    apply_repulsive_forces, integrate_velocities, apply_global_damping,
    update_radii_xpbd, apply_xsph_smoothing,
    init_jitter_velocities, apply_brownian, integrate_jitter,
    compute_mean_radius, smooth_degree, levy_position_diffusion,
    update_colors_by_size, filter_write_indices, gather_filtered_to_render
)

# Phase B: Topological neighbor counting
import topology

# ==============================================================================
# Initialize Taichi
# ==============================================================================

ti.init(arch=ti.gpu)  # Use GPU (Metal on Mac, CUDA on NVIDIA, Vulkan otherwise)

print(f"[Taichi] Initialized with backend: {ti.cfg.arch}")
print(f"[Config] N={N}, Domain={DOMAIN_SIZE}, R=[{R_MIN}, {R_MAX}]")
print(f"[Config] Grid: {GRID_RES}³ cells, PBD: {PBD_BASE_PASSES}-{PBD_MAX_PASSES} passes (adaptive)")

# ==============================================================================
# Allocate Taichi fields (with MAX capacity for runtime particle count changes)
# ==============================================================================

MAX_N = 50000  # Maximum number of particles (allocate for this, use active_n at runtime)

# Particle data
pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)    # Positions
rad = ti.field(dtype=ti.f32, shape=MAX_N)              # Radii
deg = ti.field(dtype=ti.i32, shape=MAX_N)              # Degree (neighbor count)
deg_smoothed = ti.field(dtype=ti.f32, shape=MAX_N)     # Smoothed degree (EMA, for Lévy diffusion)
color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)  # RGB color for rendering
vel = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)    # Velocities (for force fallback)
vel_temp = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)  # Temporary buffer for XSPH

# Rendering buffers (for filtered visualization in band mode)
MAX_RENDER = MAX_N  # Maximum render buffer size
idx_render = ti.field(dtype=ti.i32, shape=MAX_RENDER)  # Filtered particle indices
render_count = ti.field(dtype=ti.i32, shape=())        # Number of particles to render

# Taichi render buffers (Metal requires Taichi fields for per-vertex attributes)
pos_render = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER)  # Filtered positions
rad_render = ti.field(dtype=ti.f32, shape=MAX_RENDER)            # Filtered radii
col_render = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER)  # Filtered colors

# Brownian motion (OU jitter velocities)
v_jit = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)   # Jitter velocities (OU process)
mean_radius = ti.field(dtype=ti.f32, shape=())           # Scalar: mean radius (for jitter scaling)

# Runtime growth rhythm controls (0D Taichi fields - GUI edits these directly)
grow_rate_rt = ti.field(dtype=ti.f32, shape=())          # Growth rate per pulse (runtime)
grow_interval_rt = ti.field(dtype=ti.i32, shape=())      # Frames between pulses (runtime)
relax_interval_rt = ti.field(dtype=ti.i32, shape=())     # Relax frames after pulse (runtime)
pulse_timer = ti.field(dtype=ti.i32, shape=())           # Frames until next pulse
relax_timer = ti.field(dtype=ti.i32, shape=())           # Frames remaining in relax window

# Visualization runtime controls (0D Taichi fields - GUI edits these directly)
viz_mode_rt = ti.field(dtype=ti.i32, shape=())           # 0=Degree, 1=Size Heatmap, 2=Size Band
viz_band_min_rt = ti.field(dtype=ti.f32, shape=())       # Band min radius (for mode 2)
viz_band_max_rt = ti.field(dtype=ti.f32, shape=())       # Band max radius (for mode 2)
viz_hide_out_rt = ti.field(dtype=ti.i32, shape=())       # 0=dim, 1=hide out-of-band
viz_palette_rt = ti.field(dtype=ti.i32, shape=())        # 0=Viridis, 1=Turbo, 2=Inferno
viz_dim_alpha_rt = ti.field(dtype=ti.f32, shape=())      # Dim factor for out-of-band particles

# Debug mode for filter testing (0=Normal, 1=ALL, 2=EVERY_OTHER, 3=MIDDLE_THIRD)
VIZ_FILTER_FORCE_MODE = ti.field(dtype=ti.i32, shape=())

# Grid data
cell_count = ti.field(dtype=ti.i32, shape=GRID_RES**3)  # Particles per cell
cell_start = ti.field(dtype=ti.i32, shape=GRID_RES**3)  # Prefix sum (read-only)
cell_write = ti.field(dtype=ti.i32, shape=GRID_RES**3)  # Write pointer (scatter)
cell_indices = ti.field(dtype=ti.i32, shape=MAX_N)      # Sorted particle IDs

# Phase A: Additional fields for stability
local_max_depth = ti.field(dtype=ti.f32, shape=MAX_N)   # Per-particle max overlap (for reduction)

# Phase B: Topological neighbor counting fields (optional, only if enabled)
if TOPO_ENABLED:
    topology.allocate_fields(MAX_N)
    print(f"[Memory] Allocated topological fields (Phase B)")

# Runtime variable for active number of particles
active_n = N  # Start with config default

print(f"[Memory] Allocated {MAX_N} particles (max capacity), {GRID_RES**3} cells")
print(f"[Memory] Active particles: {active_n}")

# ==============================================================================
# Initialize particle positions and radii
# ==============================================================================

def init_rhythm_runtime():
    """Initialize growth/relax rhythm runtime fields from config defaults."""
    grow_rate_rt[None] = GROWTH_RATE_DEFAULT
    grow_interval_rt[None] = GROWTH_INTERVAL_DEFAULT
    relax_interval_rt[None] = RELAX_INTERVAL_DEFAULT
    pulse_timer[None] = GROWTH_INTERVAL_DEFAULT  # First pulse after N frames
    relax_timer[None] = 0

def init_visual_runtime():
    """Initialize visualization runtime fields to sensible defaults."""
    viz_mode_rt[None] = 0          # Start with degree-based colors
    # Default band: FULL radius range initially (user adjusts after seeing actual range)
    # Don't use config R_MIN/R_MAX because particles evolve to different ranges!
    viz_band_min_rt[None] = 0.0    # Start at zero (user will adjust)
    viz_band_max_rt[None] = 0.010  # Start at slider max (user will adjust)
    viz_hide_out_rt[None] = 0      # Dim by default (not hide)
    viz_palette_rt[None] = 1       # Turbo (punchy, good contrast)
    viz_dim_alpha_rt[None] = 0.08  # Subtle dim for out-of-band
    VIZ_FILTER_FORCE_MODE[None] = 0  # Start with normal filter

@ti.kernel
def wrap_seeded_positions(n: ti.i32):
    """Wrap seeded positions into PBC primary cell (always-wrapped invariant)."""
    for i in range(n):
        pos[i] = wrapP(pos[i])

def seed_particles(n):
    """
    Seed particles with random positions and radii.
    
    Args:
        n: Number of particles to seed
    
    Positions: Uniform random in [-L/2, L/2)³ if PBC, else [0, DOMAIN_SIZE)³
    Radii: Uniform random in [R_MIN, R_MAX]
    
    After seeding, positions are wrapped to maintain always-wrapped invariant.
    """
    if PBC_ENABLED:
        # Seed in centered domain [-L/2, L/2)³ for symmetric initial condition
        half_L = DOMAIN_SIZE * 0.5
        pos_np = np.random.uniform(-half_L, half_L, (n, 3)).astype(np.float32)
    else:
        # Seed in bounded domain [0, DOMAIN_SIZE)³
        pos_np = np.random.uniform(0, DOMAIN_SIZE, (n, 3)).astype(np.float32)
    
    rad_np = np.random.uniform(R_MIN, R_MAX, n).astype(np.float32)
    
    # Only write to the first n elements
    for i in range(n):
        pos[i] = pos_np[i]
        rad[i] = rad_np[i]
    
    # Wrap positions (PBC-aware, maintains always-wrapped invariant)
    wrap_seeded_positions(n)
    
    print(f"[Init] Seeded {n} particles")
    if PBC_ENABLED:
        print(f"       Position range: [{-half_L}, {half_L}] (PBC centered)")
    else:
        print(f"       Position range: [0, {DOMAIN_SIZE}]")
    print(f"       Radius range: [{rad_np.min():.6f}, {rad_np.max():.6f}]")

def warmup_pbd(n):
    """
    Run initial PBD passes to separate overlapping particles.
    
    Args:
        n: Number of active particles
    
    Without this, the initial random configuration often has many overlaps,
    leading to a sticky blob that's hard to separate later.
    
    We do 4 passes before the main loop starts.
    """
    print(f"[Warmup] Running {PBD_BASE_PASSES} PBD passes to separate initial overlaps...")
    
    for _ in range(PBD_BASE_PASSES):
        # Rebuild grid
        clear_grid(cell_count)
        count_particles_per_cell(pos, cell_count, n)
        prefix_sum(cell_count, cell_start)
        copy_cell_pointers(cell_start, cell_write)
        scatter_particles(pos, cell_write, cell_indices, n)
        
        # Project overlaps
        project_overlaps(pos, rad, cell_start, cell_count, cell_indices, n)
    
    print(f"[Warmup] Complete. Particles separated.")

def initialize_simulation(n):
    """
    Initialize or restart the simulation with n particles.
    Clears ALL particle data first to avoid showing stale particles.
    
    Args:
        n: Number of particles to initialize
    
    Returns:
        n (validated and clamped to MAX_N)
    """
    global active_n
    
    # Validate and clamp n
    n = max(100, min(n, MAX_N))  # Clamp to [100, MAX_N]
    active_n = n
    
    # Clear ALL particle data (move inactive particles out of view)
    # This prevents stale data from previous runs with more particles
    clear_all_particles(pos, rad, vel, deg, color, MAX_N)
    
    # Seed only the active particles
    seed_particles(n)
    
    # Initialize velocities for active particles
    init_velocities(vel, n)
    init_jitter_velocities(v_jit, n)
    print("[Init] Velocities initialized to zero")
    
    # Initialize growth/relax rhythm (load defaults into runtime fields)
    init_rhythm_runtime()
    
    # Initialize visualization settings
    init_visual_runtime()
    
    # Warmup PBD
    warmup_pbd(n)
    
    # Phase 0: PBC Self-Check (deterministic validation)
    if PBC_ENABLED:
        pos_np = pos.to_numpy()[:n]
        half_L = DOMAIN_SIZE * 0.5
        
        # Check 1: All positions in [-L/2, L/2)
        assert np.all(-half_L <= pos_np) and np.all(pos_np < half_L), \
            f"PBC check failed: positions outside [-L/2, L/2). Range: [{pos_np.min()}, {pos_np.max()}]"
        
        # Check 2: Cross-boundary distance
        # Find two particles near opposite sides (if any)
        near_left = np.where(pos_np[:, 0] < -half_L + 0.01)[0]
        near_right = np.where(pos_np[:, 0] > half_L - 0.01)[0]
        
        if len(near_left) > 0 and len(near_right) > 0:
            i = near_left[0]
            j = near_right[0]
            raw_dist = np.linalg.norm(pos_np[i] - pos_np[j])
            # True PBC distance should be much smaller
            pbc_dist_approx = min(raw_dist, DOMAIN_SIZE - raw_dist)
            print(f"[PBC Check] Cross-boundary pair: raw_dist={raw_dist:.4f}, pbc_dist≈{pbc_dist_approx:.4f}")
        
        print("[PBC Check] ✓ Startup self-check passed")
    
    return n

# Initial setup
active_n = initialize_simulation(active_n)

# ==============================================================================
# Main simulation loop
# ==============================================================================

def rebuild_grid():
    """
    Rebuild spatial grid for current particle positions.
    
    Must be called every frame after particles move or change size.
    Uses active_n for the current number of particles.
    """
    clear_grid(cell_count)
    count_particles_per_cell(pos, cell_count, active_n)
    prefix_sum(cell_count, cell_start)
    copy_cell_pointers(cell_start, cell_write)
    scatter_particles(pos, cell_write, cell_indices, active_n)

# Initialize GUI
window = ti.ui.Window("Fabric of Space - Custom Grid", (1024, 768), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()  # FIX: Use window.get_scene() not ti.ui.Scene()
camera = ti.ui.Camera()

# Camera setup (pivot around center of domain)
domain_center = DOMAIN_SIZE / 2.0  # Center of cubic volume
camera.position(domain_center, domain_center, domain_center + 0.2)  # Start slightly above center
camera.lookat(domain_center, domain_center, domain_center)  # Always look at center
camera.up(0, 1, 0)

print("\n" + "="*70)
print("FABRIC OF SPACE - CUSTOM TAICHI GRID")
print("="*70)
print(f"Controls:")
print(f"  - Right-click + drag: Rotate camera")
print(f"  - Mouse wheel: Zoom in/out")
print(f"  - WASD: Move camera")
print(f"  - SPACE: Pause/Resume")
print(f"  - S: Export particle data")
print(f"  - ESC: Exit")
print("="*70 + "\n")

# Phase A: Telemetry state
rescue_mode = False  # State for hysteresis
rescue_frame_count = 0  # Telemetry
total_rescue_substeps = 0  # Telemetry

# GUI: Adjustable degree band thresholds
gui_deg_low = DEG_LOW    # Start with config defaults
gui_deg_high = DEG_HIGH
show_centers_only = False  # Toggle for center-point visualization

# GUI: Adjustable radius limits
gui_r_min = R_MIN  # Minimum radius (meters)
gui_r_max = R_MAX  # Maximum radius (meters)

# GUI: Adjustable particle count
gui_n_particles = N  # Number of particles (can be changed and restarted)
restart_requested = False  # Flag to trigger restart

# Radius range tracking (for size-based visualization)
r_obs_min = R_MIN  # Observed minimum radius (updated periodically)
r_obs_max = R_MAX  # Observed maximum radius (updated periodically)

# Main loop
paused = False
frame = 0

while window.running:
    # Handle keyboard input
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.SPACE:
            paused = not paused
            print(f"[Control] {'Paused' if paused else 'Resumed'}")
        elif window.event.key == 's' or window.event.key == 'S':
            # EXPORT DATA TO PROVE RADII ARE CHANGING
            pos_np = pos.to_numpy()
            rad_np = rad.to_numpy()
            deg_np = deg.to_numpy()
            
            import csv
            export_file = f"particle_data_frame_{frame}.csv"
            with open(export_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'X', 'Y', 'Z', 'Radius', 'Degree'])
                for i in range(active_n):
                    writer.writerow([i, pos_np[i][0], pos_np[i][1], pos_np[i][2], rad_np[i], deg_np[i]])
            
            print(f"\n{'='*70}")
            print(f"[EXPORT] Saved {export_file}")
            print(f"         Frame: {frame}, N={active_n}")
            print(f"         Radius stats: min={rad_np[:active_n].min():.6f}, mean={rad_np[:active_n].mean():.6f}, max={rad_np[:active_n].max():.6f}, std={rad_np[:active_n].std():.6f}")
            print(f"         First 20 radii: {rad_np[:20]}")
            print(f"         Last 20 radii:  {rad_np[active_n-20:active_n] if active_n >= 20 else rad_np[:active_n]}")
            print(f"         PROOF: {len(np.unique(np.round(rad_np[:active_n], 6)))} UNIQUE radius values out of {active_n}")
            print(f"{'='*70}\n")
        elif window.event.key == ti.ui.ESCAPE:
            print("[Control] Exiting...")
            break
    
    # Check for restart request
    if restart_requested:
        print(f"\n[Restart] Reinitializing with {gui_n_particles} particles...")
        active_n = initialize_simulation(gui_n_particles)
        gui_n_particles = active_n  # Update GUI to reflect clamped value
        restart_requested = False
        frame = 0  # Reset frame counter
        rescue_frame_count = 0
        rescue_mode = False
        print(f"[Restart] Complete. Active particles: {active_n}\n")
    
    if not paused:
        # === PERFORMANCE PROFILING ===
        t_start = time.perf_counter()
        ti.sync()
        
        # === 0. Rebuild grid (current radii) ===
        rebuild_grid()
        ti.sync()
        t_grid = time.perf_counter()
        
        # === 1. Compute max overlap depth ===
        max_depth = compute_max_overlap(pos, rad, cell_start, cell_count, cell_indices, local_max_depth, active_n)
        
        # === 2. Determine adaptive PBD pass count ===
        passes_needed = max(PBD_BASE_PASSES, 
                            min(PBD_MAX_PASSES, 
                                int(PBD_BASE_PASSES + PBD_ADAPTIVE_SCALE * (max_depth / (0.2 * R_MAX)))))
        
        # === 3. Deep overlap force fallback (Track 1: gentle rescue) ===
        # Hysteresis logic
        if max_depth > DEEP_OVERLAP_THRESHOLD * R_MAX:
            rescue_mode = True
        elif max_depth < DEEP_OVERLAP_EXIT * R_MAX:
            rescue_mode = False
        
        # Apply rescue forces if enabled and rescue mode is active
        if RESCUE_ENABLED and rescue_mode:
            rescue_frame_count += 1
            # Apply soft repulsive forces (scaled by RESCUE_STRENGTH)
            dt_rescue = DT * RESCUE_STRENGTH
            apply_repulsive_forces(pos, rad, vel, cell_start, cell_count, cell_indices, dt_rescue, active_n)
            integrate_velocities(pos, vel, active_n, dt_rescue)
        
        # === 4. Adaptive PBD passes ===
        for pass_idx in range(passes_needed):
            rebuild_grid()  # Fresh neighbors each pass
            project_overlaps(pos, rad, cell_start, cell_count, cell_indices, active_n)
        
        # === 5. XSPH velocity smoothing ===
        if XSPH_ENABLED:
            apply_xsph_smoothing(pos, vel, vel_temp, rad, cell_start, cell_count, cell_indices, active_n)
        
        # === 6. Global damping (prevent slow energy accumulation) ===
        apply_global_damping(vel, active_n)
        
        # === 6B. Brownian motion (OU jitter) - Track 1 Option B ===
        # Compute mean radius for jitter scaling
        rad_np = rad.to_numpy()[:active_n]
        mean_radius[None] = rad_np.mean()
        
        # Apply smooth Brownian drift (OU process)
        apply_brownian(v_jit, rad, mean_radius, active_n, DT)
        
        # Integrate jitter into positions (PBC-safe)
        integrate_jitter(pos, v_jit, active_n, DT)
        
        # === 6C. Lévy Positional Diffusion - Track 2 (Topological Regularization) ===
        # NOTE: Lévy now runs ONLY during relax frames (see step 8 below).
        # This section is intentionally left empty (Lévy moved to relax window).
        
        # === 7. Keep the grid current for this frame's geometry ===
        rebuild_grid()
        ti.sync()
        t_pbd = time.perf_counter()
        
        # === 8. PULSE gate: measure + ONE discrete growth step, then schedule relax ===
        if pulse_timer[None] <= 0:
            # 8A. Measure with current geometry
            count_neighbors(pos, rad, deg, cell_start, cell_count, cell_indices, active_n)
            
            # Store radius stats BEFORE pulse (for telemetry)
            rad_np_before = rad.to_numpy()[:active_n]
            deg_np_before = deg.to_numpy()[:active_n]
            r_mean_before = rad_np_before.mean()
            deg_mean_before = deg_np_before.mean()
            
            # 8B. One discrete growth/shrink step (±grow_rate_rt)
            deg_low_target = TOPO_DEG_LOW if TOPO_ENABLED else gui_deg_low
            deg_high_target = TOPO_DEG_HIGH if TOPO_ENABLED else gui_deg_high
            
            # Auto-enforce: rate limit >= growth rate (so pulse is visible)
            from config import RADIUS_RATE_LIMIT
            rate_limit_rt = max(RADIUS_RATE_LIMIT, float(grow_rate_rt[None]))
            
            # Apply pulse with runtime rate limit
            update_radii_xpbd(rad, deg, active_n, DT, gui_r_min, gui_r_max, 
                             deg_low_target, deg_high_target, 
                             grow_rate_rt[None], grow_rate_rt[None], rate_limit_rt)
            
            # Store radius stats AFTER pulse (for telemetry)
            rad_np_after = rad.to_numpy()[:active_n]
            r_mean_after = rad_np_after.mean()
            
            # Count how many radii hit bounds (clipped)
            clipped_min = np.sum(rad_np_after <= gui_r_min + 1e-6)
            clipped_max = np.sum(rad_np_after >= gui_r_max - 1e-6)
            clipped_total = clipped_min + clipped_max
            clipped_pct = 100.0 * clipped_total / active_n if active_n > 0 else 0.0
            
            # 8C. Schedule relax window and next pulse
            relax_timer[None] = relax_interval_rt[None]
            pulse_timer[None] = grow_interval_rt[None]
            
            # 8D. Enhanced telemetry (every pulse, with before/after stats)
            delta_r_mean = r_mean_after - r_mean_before
            delta_r_pct = 100.0 * delta_r_mean / r_mean_before if r_mean_before > 0 else 0.0
            
            print(f"[Pulse] frame={frame:4d} | rate={grow_rate_rt[None]:.3f} gap={grow_interval_rt[None]} relax={relax_interval_rt[None]}")
            print(f"        deg: μ={deg_mean_before:.2f} [{deg_np_before.min()},{deg_np_before.max()}]")
            print(f"        r_mean: {r_mean_before:.6f} → {r_mean_after:.6f} (Δ={delta_r_mean:+.6f}, {delta_r_pct:+.2f}%)")
            print(f"        clipped: {clipped_total}/{active_n} ({clipped_pct:.1f}%) [min={clipped_min} max={clipped_max}]")
            print(f"        max_depth={max_depth:.6f}, passes={passes_needed}")
        else:
            pulse_timer[None] -= 1
        
        # === 9. MOTION happens every frame (PBD already done above) ===
        
        # === 10. LÉVY: runs ONLY during relax frames ===
        if relax_timer[None] > 0:
            relax_timer[None] -= 1
            if LEVY_ENABLED:
                # Compute mean radius for step size clamping
                compute_mean_radius(rad, active_n, mean_radius)
                
                # Smooth degree values (EMA with α=0.25 for stable diffusion)
                smooth_degree(deg, deg_smoothed, active_n, 0.25)
                
                # Apply Lévy diffusion (particles shift toward better-connected neighbors)
                levy_position_diffusion(pos, deg_smoothed, cell_start, cell_count, cell_indices,
                                       mean_radius, active_n, LEVY_ALPHA, LEVY_DEG_SPAN, LEVY_STEP_FRAC)
        
        # === 11. Topological neighbor counting (Phase B - OPTIONAL, OFF BY DEFAULT) ===
        # Note: This runs independently of growth rhythm (for topology analysis only)
        topo_did_run = False
        if TOPO_ENABLED:
            if frame % TOPO_UPDATE_CADENCE == 0:
                topo_did_run = True
                # Build candidate pairs (spatial pruning)
                topology.build_topo_candidates(pos, rad, cell_start, cell_count, cell_indices, active_n)
                
                # Gabriel graph test (witness-based, with hash-based subsampling and early exit)
                topology.gabriel_test_topological_degree(pos, rad, cell_start, cell_count, cell_indices, active_n, frame)
            
            # Update EMA (every frame, even if topo not recomputed)
            topology.update_topo_ema(TOPO_EMA_ALPHA, active_n)
            
            # === Blend topological + geometric degrees (first 200 frames) ===
            if frame < TOPO_BLEND_FRAMES:
                # Linear blend from 80% topo → 100% topo
                blend_lambda = TOPO_BLEND_LAMBDA_START + \
                               (TOPO_BLEND_LAMBDA_END - TOPO_BLEND_LAMBDA_START) * (frame / TOPO_BLEND_FRAMES)
                
                # Compute blended degree
                deg_topo_np = topology.topo_deg_ema.to_numpy()[:active_n]
                deg_geom_np = deg.to_numpy()[:active_n]
                deg_blended_np = blend_lambda * deg_topo_np + (1.0 - blend_lambda) * deg_geom_np
                
                # Write back to deg field (used by radius adaptation)
                for i in range(active_n):
                    deg[i] = int(deg_blended_np[i])
            else:
                # Pure topological (after blend period)
                deg_topo_np = topology.topo_deg_ema.to_numpy()[:active_n]
                for i in range(active_n):
                    deg[i] = int(deg_topo_np[i])
        
        # === 12. Update observed radius range (every 15 frames for GUI context) ===
        if frame % 15 == 0:
            rad_np_sample = rad.to_numpy()[:active_n]
            r_obs_min = float(rad_np_sample.min()) if active_n > 0 else R_MIN
            r_obs_max = float(rad_np_sample.max()) if active_n > 0 else R_MAX
        
        # === 13. Update colors (size-based or degree-based, runtime-switchable) ===
        update_colors_by_size(rad, deg, color, active_n, r_obs_min, r_obs_max,
                             gui_deg_low, gui_deg_high,
                             viz_mode_rt[None], viz_band_min_rt[None], viz_band_max_rt[None],
                             viz_hide_out_rt[None], viz_palette_rt[None], viz_dim_alpha_rt[None])
        
        ti.sync()
        t_topo = time.perf_counter()
        
        frame += 1
        
        # === Telemetry & Logging ===
        if frame % 100 == 0:
            deg_np = deg.to_numpy()[:active_n]
            rad_np = rad.to_numpy()[:active_n]
            
            rescue_pct = 100.0 * rescue_frame_count / frame if frame > 0 else 0.0
            rescue_status = f"{rescue_pct:.1f}%" if RESCUE_ENABLED else f"{rescue_pct:.1f}% (disabled)"
            
            # Phase B telemetry (topological degree)
            if TOPO_ENABLED:
                topo_deg_np = topology.topo_deg.to_numpy()[:active_n]
                topo_deg_ema_np = topology.topo_deg_ema.to_numpy()[:active_n]
                topo_pair_count_np = topology.topo_pair_count.to_numpy()[:active_n]
                topo_truncated_np = topology.topo_truncated.to_numpy()[:active_n]
                
                truncated_pct = 100.0 * np.sum(topo_truncated_np) / active_n if active_n > 0 else 0.0
                
                print(f"[Frame {frame:4d}] N={active_n}, Passes={passes_needed}, MaxDepth={max_depth:.6f}")
                print(f"    Geom Deg: mean={deg_np.mean():.2f}, min={deg_np.min()}, max={deg_np.max()}")
                print(f"    Topo Deg (raw): mean={topo_deg_np.mean():.2f}, min={topo_deg_np.min()}, max={topo_deg_np.max()}")
                print(f"    Topo Deg (EMA): mean={topo_deg_ema_np.mean():.2f}, min={topo_deg_ema_np.min():.2f}, max={topo_deg_ema_np.max():.2f}")
                print(f"    Topo Pairs: mean={topo_pair_count_np.mean():.1f}, max={topo_pair_count_np.max()}")
                print(f"    Topo Truncated: {truncated_pct:.2f}% ({np.sum(topo_truncated_np)}/{active_n})")
                print(f"    Radius: mean={rad_np.mean():.4f}, min={rad_np.min():.4f}, max={rad_np.max():.4f}")
                print(f"    Rescue: {rescue_status}")
                
                # Warning if truncation is too high
                if truncated_pct > TOPO_TRUNCATION_WARNING_THRESHOLD * 100:
                    print(f"    [WARNING] Truncation >{TOPO_TRUNCATION_WARNING_THRESHOLD*100:.1f}%! " +
                          f"Consider increasing MAX_TOPO_NEIGHBORS or reducing TOPO_MAX_RADIUS_MULTIPLE")
            else:
                # Track 1 (Fast Path): Geometric PCC only, simple telemetry
                print(f"[Frame {frame:4d}] N={active_n}, Passes={passes_needed}, MaxDepth={max_depth:.6f} | " +
                      f"Degree: mean={deg_np.mean():.2f}, min={deg_np.min()}, max={deg_np.max()} | " +
                      f"Radius: mean={rad_np.mean():.4f}, min={rad_np.min():.4f}, max={rad_np.max():.4f} | " +
                      f"Rescue: {rescue_status}")
    
    # === 6. Render ===
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
    # Keep camera pivoting around domain center
    camera.lookat(domain_center, domain_center, domain_center)
    
    # Decide which particles to render based on visualization mode
    use_filtered = (viz_mode_rt[None] == 2 and viz_hide_out_rt[None] == 1)
    
    if use_filtered:
        # === FILTERED PATH: Band mode with hide ===
        
        # 1. Build index list on GPU
        r_np_all = rad.to_numpy()[:active_n]
        rmin_obs = float(r_np_all.min()) if active_n > 0 else 0.0
        rmax_obs = float(r_np_all.max()) if active_n > 0 else 0.0
        mode = int(VIZ_FILTER_FORCE_MODE[None])
        band_min = float(viz_band_min_rt[None])
        band_max = float(viz_band_max_rt[None])
        
        filter_write_indices(rad, idx_render, render_count, MAX_RENDER,
                            mode, band_min, band_max, active_n, rmin_obs, rmax_obs)
        n_render = int(render_count[None])
        
        # Diagnostic output every 60 frames
        if frame % 60 == 0:
            print(f"[BandDbg] mode={mode} band=[{band_min:.6f},{band_max:.6f}] " +
                  f"obs=[{rmin_obs:.6f},{rmax_obs:.6f}] -> kept {n_render}/{active_n} " +
                  f"({100.0*n_render/max(1,active_n):.1f}%)")
        
        if n_render > 0:
            # 2. GPU gather: copy filtered particles to render buffers (on-GPU, fast)
            # This avoids .from_numpy() shape mismatch and keeps everything as Taichi fields
            gather_filtered_to_render(
                pos, rad, color,
                idx_render,
                pos_render, rad_render, col_render,
                n_render, active_n
            )
            
            # 3. Set camera & lights
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.0, 0.5), color=(1, 1, 1))
            
            # 4. Draw with Taichi fields (Metal requires Taichi fields for per-vertex attrs)
            if show_centers_only:
                scene.particles(pos_render, radius=0.0005, per_vertex_color=col_render)
            else:
                scene.particles(pos_render, radius=0.001, per_vertex_radius=rad_render, per_vertex_color=col_render)
            
            canvas.scene(scene)
        else:
            # Nothing to draw
            canvas.scene(scene)
        
        # CRITICAL: Skip unfiltered path (prevents double-draw)
        # Continue to GUI section below, don't execute else branch
        
    else:
        # === UNFILTERED PATH: Normal rendering ===
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.0, 0.5), color=(1, 1, 1))
        
        if show_centers_only:
            scene.particles(pos, radius=0.0005, per_vertex_color=color)
        else:
            scene.particles(pos, radius=0.001, per_vertex_radius=rad, per_vertex_color=color)
        
        canvas.scene(scene)
    
    # === 7. GUI Control Panel ===
    # Compute statistics for distribution (only active particles)
    deg_np = deg.to_numpy()[:active_n]
    rad_np = rad.to_numpy()[:active_n]
    
    count_red = np.sum(deg_np < gui_deg_low)
    count_green = np.sum((deg_np >= gui_deg_low) & (deg_np <= gui_deg_high))
    count_blue = np.sum(deg_np > gui_deg_high)
    
    pct_red = 100.0 * count_red / active_n
    pct_green = 100.0 * count_green / active_n
    pct_blue = 100.0 * count_blue / active_n
    
    # IMGUI window for controls and stats (expanded height for particle count control)
    window.GUI.begin("Control Panel", 0.01, 0.01, 0.32, 0.70)
    
    # Particle count section
    window.GUI.text(f"=== Particle Count ===")
    window.GUI.text(f"Active: {active_n} / {MAX_N}")
    gui_n_particles = window.GUI.slider_int("New N", gui_n_particles, 100, MAX_N)
    if window.GUI.button("RESTART"):
        restart_requested = True
    window.GUI.text("")
    
    # Degree statistics section
    window.GUI.text(f"=== Degree Stats ===")
    window.GUI.text(f"Avg:  {deg_np.mean():.2f}")
    window.GUI.text(f"Min:  {deg_np.min()}")
    window.GUI.text(f"Max:  {deg_np.max()}")
    window.GUI.text("")
    window.GUI.text("Distribution:")
    window.GUI.text(f"  <{gui_deg_low}: {pct_red:.1f}%")
    window.GUI.text(f"  {gui_deg_low}-{gui_deg_high}: {pct_green:.1f}%")
    window.GUI.text(f"  >{gui_deg_high}: {pct_blue:.1f}%")
    window.GUI.text("")
    window.GUI.text("Band Thresholds:")
    gui_deg_low = window.GUI.slider_int("Min (grow below)", gui_deg_low, 1, 20)
    gui_deg_high = window.GUI.slider_int("Max (shrink above)", gui_deg_high, 1, 30)
    window.GUI.text("")
    
    # Radius limits section
    window.GUI.text(f"=== Radius Limits ===")
    window.GUI.text(f"Current: {rad_np.min():.5f} - {rad_np.max():.5f}")
    gui_r_min = window.GUI.slider_float("Min radius", gui_r_min, 0.0001, 0.0100)
    gui_r_max = window.GUI.slider_float("Max radius", gui_r_max, 0.0001, 0.0200)
    window.GUI.text("")
    
    # Growth Rhythm section (runtime controls - no snap-back!)
    window.GUI.text(f"=== Growth Rhythm ===")
    rate = window.GUI.slider_float("Growth rate per pulse", grow_rate_rt[None], 0.01, 0.10)
    gap = window.GUI.slider_int("Frames between pulses", grow_interval_rt[None], 5, 120)
    relx = window.GUI.slider_int("Relax frames after pulse", relax_interval_rt[None], 0, 60)
    
    # Commit slider values to runtime fields (no snap-back!)
    grow_rate_rt[None] = rate
    grow_interval_rt[None] = gap
    relax_interval_rt[None] = relx
    
    # Status line showing current timer state
    if relax_timer[None] > 0:
        window.GUI.text(f"Relaxing... {relax_timer[None]} frames left")
    else:
        window.GUI.text(f"Next pulse in {pulse_timer[None]} frames")
    window.GUI.text("")
    
    # System configuration section
    window.GUI.text(f"=== System Config ===")
    pbc_status = "ON" if PBC_ENABLED else "OFF"
    window.GUI.text(f"PBC: {pbc_status} (restart to change)")
    topo_status = "ON" if TOPO_ENABLED else "OFF"
    window.GUI.text(f"Topology (Gabriel): {topo_status} (slow!)")
    window.GUI.text("")
    
    # Visualization section
    window.GUI.text(f"=== Visualization ===")
    
    # Color mode selector
    mode_labels = ["Degree", "Size Heatmap", "Size Band"]
    viz_mode_rt[None] = window.GUI.slider_int("Color Mode", viz_mode_rt[None], 0, 2)
    window.GUI.text(f"  Mode: {mode_labels[viz_mode_rt[None]]}")
    
    # Debug filter mode (only show in band mode with hide enabled)
    if viz_mode_rt[None] == 2 and viz_hide_out_rt[None] == 1:
        debug_mode_labels = ["Normal", "ALL", "EVERY_OTHER", "MIDDLE_THIRD"]
        debug = window.GUI.slider_int("Filter debug", int(VIZ_FILTER_FORCE_MODE[None]), 0, 3)
        VIZ_FILTER_FORCE_MODE[None] = int(debug)
        window.GUI.text(f"  Debug: {debug_mode_labels[debug]}")
    
    # Palette selector (for size modes)
    if viz_mode_rt[None] >= 1:
        palette_labels = ["Viridis", "Turbo", "Inferno"]
        viz_palette_rt[None] = window.GUI.slider_int("Palette", viz_palette_rt[None], 0, 2)
        window.GUI.text(f"  {palette_labels[viz_palette_rt[None]]}")
    
    # Size band controls (only for mode 2)
    if viz_mode_rt[None] == 2:
        window.GUI.text(f"  Observed: [{r_obs_min:.5f}, {r_obs_max:.5f}]")
        
        # Quick-set buttons
        if window.GUI.button("Use Full Range"):
            viz_band_min_rt[None] = r_obs_min
            viz_band_max_rt[None] = r_obs_max
        
        if window.GUI.button("Use Middle Third"):
            span = r_obs_max - r_obs_min
            viz_band_min_rt[None] = r_obs_min + span * 0.33
            viz_band_max_rt[None] = r_obs_min + span * 0.67
        
        band_min = viz_band_min_rt[None]
        band_max = viz_band_max_rt[None]
        
        band_min = window.GUI.slider_float("Band min", band_min, 0.0, 0.010)
        band_max = window.GUI.slider_float("Band max", band_max, 0.0, 0.010)
        
        # Ensure min <= max
        if band_max < band_min:
            band_max = band_min
        
        viz_band_min_rt[None] = band_min
        viz_band_max_rt[None] = band_max
        
        # Hide/dim toggle
        hide_bool = viz_hide_out_rt[None] == 1
        hide_bool = window.GUI.checkbox("Hide out-of-band", hide_bool)
        viz_hide_out_rt[None] = 1 if hide_bool else 0
        
        # Show count of in-band particles if hiding
        if viz_hide_out_rt[None] == 1:
            # Count particles in band (quick Python check for display)
            rad_np_band = rad.to_numpy()[:active_n]
            in_band_count = np.sum((rad_np_band >= band_min) & (rad_np_band <= band_max))
            in_band_pct = 100.0 * in_band_count / active_n if active_n > 0 else 0.0
            window.GUI.text(f"  Showing: {in_band_count}/{active_n} ({in_band_pct:.1f}%)")
        
        # Dim slider (only if not hiding)
        if viz_hide_out_rt[None] == 0:
            dim = viz_dim_alpha_rt[None]
            dim = window.GUI.slider_float("Dim level", dim, 0.0, 0.5)
            viz_dim_alpha_rt[None] = dim
    
    window.GUI.text("")
    show_centers_only = window.GUI.checkbox("Show centers only", show_centers_only)
    window.GUI.end()
    
    window.show()
    ti.sync()
    t_render = time.perf_counter()
    
    # === PERFORMANCE PROFILING OUTPUT ===
    if not paused and frame % 60 == 0:  # Every 60 frames
        dt_grid = t_grid - t_start
        dt_pbd = t_pbd - t_grid
        dt_topo = t_topo - t_pbd
        dt_render = t_render - t_topo
        dt_total = t_render - t_start
        
        topo_status = f"{dt_topo:.3f}s" if topo_did_run else "—"
        fps_estimate = 1.0 / dt_total if dt_total > 0 else 0.0
        
        print(f"\n[PERF] Frame {frame}: grid={dt_grid:.3f}s  pbd={dt_pbd:.3f}s  topo={topo_status}  render={dt_render:.3f}s  | FPS≈{fps_estimate:.1f}")
        print(f"       Breakdown: grid={100*dt_grid/dt_total:.0f}%  pbd={100*dt_pbd/dt_total:.0f}%  topo={100*dt_topo/dt_total:.0f}%  render={100*dt_render/dt_total:.0f}%")

print("\n[Exit] Simulation ended.")
print(f"       Total frames: {frame}")
print(f"       Active particles: {active_n}")
print(f"       Final mean degree: {deg.to_numpy()[:active_n].mean():.2f}")
print(f"       Final mean radius: {rad.to_numpy()[:active_n].mean():.4f}")

