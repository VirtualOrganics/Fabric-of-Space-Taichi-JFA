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
  - R: Reset all particles to current slider value
  - S: Export particle data (PROOF radii are changing!)
  - F: Freeze-frame diagnostic test (Phase A1 - temporal skew test)
  - ESC: Exit
"""

import taichi as ti
import numpy as np
import time  # For performance profiling
import math  # For box scaling calculations

# Import configuration and kernels
from config import (
    N, DOMAIN_SIZE, R_MIN, R_MAX, R_START_MANUAL, CELL_SIZE, GRID_RES, VIS_SCALE, FPS_TARGET,
    PBD_BASE_PASSES, PBD_MAX_PASSES, PBD_ADAPTIVE_SCALE, PBD_SUBSTEPS,
    DEEP_OVERLAP_THRESHOLD, DEEP_OVERLAP_EXIT, FORCE_SUBSTEPS_MIN, FORCE_SUBSTEPS_MAX,
    RESCUE_ENABLED, RESCUE_STRENGTH,
    # Geometric Repacking (push-pull during adjustment phase)
    REPACK_ENABLED, REPACK_DEADZONE_TAU, REPACK_BETA_PUSH, REPACK_BETA_PULL,
    REPACK_EXTRA_PASSES, REPACK_STEP_CAP_FRAC,
    XSPH_ENABLED, DT, DEG_LOW, DEG_HIGH,
    PBC_ENABLED,
    GAIN_GROW, GAIN_SHRINK,
    CONTACT_TOL,  # Contact tolerance for neighbor detection
    # Phase B: Topological neighbor counting (optional, off by default)
    TOPO_ENABLED, TOPO_UPDATE_CADENCE, TOPO_EMA_ALPHA, TOPO_DEG_LOW, TOPO_DEG_HIGH,
    TOPO_BLEND_FRAMES, TOPO_BLEND_LAMBDA_START, TOPO_BLEND_LAMBDA_END,
    TOPO_TRUNCATION_WARNING_THRESHOLD,
    TOPO_BATCHES, TOPO_PAIR_SUBSAMPLE_Q, TOPO_WRITE_TO_EMA,
    USE_KNN_TOPO, KNN_TOPO_K,
    # Lévy Positional Diffusion (Track 2: topological regularization)
    LEVY_ENABLED, LEVY_ALPHA, LEVY_DEG_SPAN, LEVY_STEP_FRAC, LEVY_USE_TOPO_DEG,
    # Growth/Relax Rhythm (defaults only; runtime uses Taichi fields)
    GROWTH_RATE_DEFAULT, ADJUSTMENT_FRAMES_DEFAULT,
    # Decision Stability (hysteresis + streaks)
    HYSTERESIS, STREAK_LOCK, MOMENTUM, STREAK_CAP,
    # Auto-scaling (startup)
    AUTO_SCALE_RADII, PHI_TARGET,
    # JFA Power Diagram (experimental)
    JFA_ENABLED, JFA_RUN_INTERVAL, JFA_EMA_ALPHA,
    # FSC-Only Controller (Phase 2)
    FSC_MANUAL_MODE, FSC_LOW, FSC_HIGH, GROWTH_PCT, ADJUSTMENT_FRAMES,
    MAX_STEP_PCT, MAX_STEP_PCT_RANGE,
    FSC_DEADBAND, BACKPRESSURE_MODE, RUN_SAFETY_TESTS,
    # Pressure Equilibration
    PRESSURE_EQUILIBRATION_ENABLED, PRESSURE_K, PRESSURE_EXP,
    PRESSURE_PAIR_CAP, MAX_EQ_NEI,
    # Brownian Motion
    BROWNIAN_ENABLED, BROWNIAN_STRENGTH, BROWNIAN_DAMPING,
    # Warm-start & Debug
    WARMSTART_FRAMES, FSC_ZERO_RATE_THRESH, EQ_DEBUG_EVERY
)

# Import R_REF if auto-scaling is enabled
try:
    from config import R_REF
except ImportError:
    R_REF = None  # Not available when AUTO_SCALE_RADII = False
from grid import (
    clear_grid, clear_all_particles, count_particles_per_cell, prefix_sum, copy_cell_pointers,
    scatter_particles, update_colors,
    wrapP
)

# Import JFA module (if enabled)
if JFA_ENABLED:
    import jfa
from dynamics import (
    project_overlaps, project_with_pull, init_velocities, compute_max_overlap,
    apply_repulsive_forces, integrate_velocities, apply_global_damping,
    update_radii_xpbd, apply_xsph_smoothing,
    init_jitter_velocities, apply_brownian, integrate_jitter,
    compute_mean_radius, levy_position_diffusion,
    update_colors_by_size, filter_write_indices, gather_filtered_to_render,
    # FSC-Only Controller (Phase 2)
    set_fsc_targets, nudge_radii_adaptive_ema,
    # Pressure Equilibration
    equilibrate_pressure, compute_pressure_stats,
    # Brownian Motion
    apply_brownian_motion
)

# Phase B: Topological neighbor counting
import topology

# ==============================================================================
# Initialize Taichi
# ==============================================================================

ti.init(arch=ti.gpu)  # Use GPU (Metal on Mac, CUDA on NVIDIA, Vulkan otherwise)

# Initialize JFA fields (if enabled)
if JFA_ENABLED:
    jfa.init_jfa()
    print("[JFA] Power diagram enabled")
    jfa.print_jfa_config()

print(f"[Taichi] Initialized with backend: {ti.cfg.arch}")
print(f"[Config] N={N}, Domain={DOMAIN_SIZE}, R=[{R_MIN}, {R_MAX}]")
print(f"[Config] Grid: {GRID_RES}³ cells, PBD: {PBD_BASE_PASSES}-{PBD_MAX_PASSES} passes (adaptive)")

# ==============================================================================
# Allocate Taichi fields (with MAX capacity for runtime particle count changes)
# ==============================================================================

MAX_N = 50000  # Maximum number of particles (allocate for this, use active_n at runtime)

# Particle data
pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_N)    # Positions
rad = ti.field(dtype=ti.f32, shape=MAX_N)              # Current radii
rad_target = ti.field(dtype=ti.f32, shape=MAX_N)       # Target radii (for smooth adjustment)
delta_r = ti.field(dtype=ti.f32, shape=MAX_N)          # Jacobi temporary for pressure equilibration
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

# Repacking telemetry (0D fields for push/pull counts)
repack_push_count = ti.field(dtype=ti.i32, shape=())     # Number of push corrections per frame
repack_pull_count = ti.field(dtype=ti.i32, shape=())     # Number of pull corrections per frame

# Runtime growth rhythm controls (0D Taichi fields - GUI edits these directly)
grow_rate_rt = ti.field(dtype=ti.f32, shape=())          # Growth/shrink rate per cycle (runtime)
adjustment_frames_rt = ti.field(dtype=ti.i32, shape=())  # Frames to adjust size (runtime)
adjustment_timer = ti.field(dtype=ti.i32, shape=())      # Countdown: 0 = measure, >0 = adjusting

# Runtime particle sizing control
r_start_rt = ti.field(dtype=ti.f32, shape=())            # Starting radius (for reset/GUI)

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
    """Initialize growth rhythm runtime fields from config defaults."""
    grow_rate_rt[None] = GROWTH_RATE_DEFAULT
    adjustment_frames_rt[None] = ADJUSTMENT_FRAMES_DEFAULT
    adjustment_timer[None] = 0  # Start by measuring immediately
    r_start_rt[None] = R_START_MANUAL  # Starting radius for particles

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
    Radii: Log-normal around R_REF (if auto-scaling), else uniform in [R_MIN, R_MAX]
    
    After seeding, positions are wrapped to maintain always-wrapped invariant.
    """
    if PBC_ENABLED:
        # Seed in centered domain [-L/2, L/2)³ for symmetric initial condition
        half_L = DOMAIN_SIZE * 0.5
        pos_np = np.random.uniform(-half_L, half_L, (n, 3)).astype(np.float32)
    else:
        # Seed in bounded domain [0, DOMAIN_SIZE)³
        pos_np = np.random.uniform(0, DOMAIN_SIZE, (n, 3)).astype(np.float32)
    
    # Seed all particles with starting radius (optionally with variance)
    # Growth/shrink system will naturally create size distribution over time
    
    # === ADD RADIUS VARIANCE TO CREATE INITIAL PRESSURE GRADIENTS ===
    # Without variance, system starts at perfect equilibrium (σ(P)=0)
    INITIAL_RADIUS_VARIANCE = 0.30  # ±30% variance (set to 0.0 for uniform)
    
    if INITIAL_RADIUS_VARIANCE > 0.0:
        # Random radii centered on R_START with ±variance
        rad_np = R_START_MANUAL * (1.0 + (np.random.rand(n).astype(np.float32) - 0.5) * 2.0 * INITIAL_RADIUS_VARIANCE)
        # Clamp to valid range
        rad_np = np.clip(rad_np, R_MIN, R_MAX)
        print(f"[Init] Seeded radii: μ={rad_np.mean():.6f} σ={rad_np.std():.6f} (±{INITIAL_RADIUS_VARIANCE*100:.0f}% variance)")
    else:
        rad_np = np.full(n, R_START_MANUAL, dtype=np.float32)
        print(f"[Init] Seeded radii: uniform at R_START={R_START_MANUAL:.6f}")
    
    print(f"[Init] Bounds: R_MIN={R_MIN:.6f}, R_MAX={R_MAX:.6f} (will evolve naturally)")
    
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
    clear_all_particles(pos, rad, vel, color, MAX_N)
    
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
# Phase 3: FSC-Driven Controller (Device-Side)
# ==============================================================================
# These functions implement the FSC-based growth controller that replaces
# the degree-based growth logic. All computation happens on GPU.

@ti.func
def smoothstep01(x: ti.f32) -> ti.f32:
    """
    Smooth interpolation function (Hermite interpolation).
    
    Args:
        x: Input value (automatically clamped to [0, 1])
    
    Returns:
        Smoothed value in [0, 1] with zero derivatives at endpoints
    
    This creates gentle transitions for the FSC controller ramps.
    """
    t = ti.min(1.0, ti.max(0.0, x))
    return t * t * (3.0 - 2.0 * t)

@ti.func
def lerp_fsc(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32:
    """
    Linear interpolation (used for EMA smoothing of radius changes).
    
    Args:
        a: Start value
        b: End value
        t: Blend factor [0, 1]
    
    Returns:
        Interpolated value: a + t * (b - a)
    """
    return a + t * (b - a)

@ti.func
def smoothstep(e0: ti.f32, e1: ti.f32, x: ti.f32) -> ti.f32:
    """
    Smoothstep function for damping band near FSC boundaries.
    
    Args:
        e0: Lower edge
        e1: Upper edge
        x: Value to interpolate
    
    Returns:
        Smoothed value in [0, 1]
    """
    t = ti.min(1.0, ti.max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)

def alpha_for_frames(frames: int) -> float:
    """
    Compute adaptive EMA alpha to reach ~95% of target in N frames.
    
    Formula: α = 1 - (0.05)^(1/N)
    
    Args:
        frames: Number of frames to reach 95% convergence
    
    Returns:
        EMA alpha value
    """
    return 1.0 - (0.05 ** (1.0 / max(1, frames)))

def backpressure_from_overlap(o_max: float, contact_tol: float) -> float:
    """
    Compute global backpressure factor from maximum overlap.
    
    Reduces growth/shrink rate when overlaps are high, giving PBD time to resolve.
    
    Args:
        o_max: Maximum overlap depth across all particles
        contact_tol: Contact tolerance threshold
    
    Returns:
        Backpressure factor in [0.25, 1.0]
    """
    if contact_tol <= 0:
        return 1.0
    raw = (contact_tol - o_max) / contact_tol
    return max(0.25, min(1.0, raw))

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
    
    # === PHASE A3: CSR INTEGRITY CHECK ===
    # Verify sum(cell_count) == active_N (every particle assigned to exactly one cell)
    # This is a dev-time sanity check; can be disabled in production builds
    global _csr_check_counter
    if not hasattr(rebuild_grid, '_csr_check_counter'):
        rebuild_grid._csr_check_counter = 0
    
    rebuild_grid._csr_check_counter += 1
    if rebuild_grid._csr_check_counter % 300 == 0:  # Check every 300 frames (minimal overhead)
        cell_count_np = cell_count.to_numpy()
        total_assigned = int(cell_count_np.sum())
        if total_assigned != active_n:
            print(f"\n[CSR INTEGRITY FAIL] Frame ~{rebuild_grid._csr_check_counter}: sum(cell_count)={total_assigned} ≠ active_N={active_n}")
            print(f"                     Particle assignment corrupted! (Overflow, race, or index bug)")
        # No print if pass - silent success (avoid spam)
    
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
print(f"  - R: Reset particles to current slider value")
print(f"  - S: Export particle data")
print(f"  - F: Freeze-frame diagnostic (test temporal skew hypothesis)")
print(f"  - ESC: Exit")
print("="*70)
if PRESSURE_EQUILIBRATION_ENABLED:
    print(f"\nPRESSURE EQUILIBRATION:")
    print(f"  k={PRESSURE_K:.3f} | P_exp={PRESSURE_EXP:.1f} | pair_cap={PRESSURE_PAIR_CAP:.2f}")
    print(f"  max_nei={MAX_EQ_NEI} | EQ_DEBUG_EVERY={EQ_DEBUG_EVERY}")
print(f"\nWARM-START:")
print(f"  WARMSTART_FRAMES={WARMSTART_FRAMES} | FSC_ZERO_RATE_THRESH={FSC_ZERO_RATE_THRESH*100:.0f}%")
print("="*70 + "\n")

# Phase A: Telemetry state
rescue_mode = False  # State for hysteresis
rescue_frame_count = 0  # Telemetry
total_rescue_substeps = 0  # Telemetry

# GUI: Visualization toggles
show_centers_only = False  # Toggle for center-point visualization

# GUI: Adjustable radius limits
gui_r_min = R_MIN  # Minimum radius (meters)
gui_r_max = R_MAX  # Maximum radius (meters)

# GUI: Adjustable FSC band (live control)
gui_fsc_low = FSC_LOW
gui_fsc_high = FSC_HIGH

# GUI: Adjustable particle count
gui_n_particles = N  # Number of particles (can be changed and restarted)
restart_requested = False  # Flag to trigger restart

# Radius range tracking (for size-based visualization)
r_obs_min = R_MIN  # Observed minimum radius (updated periodically)
r_obs_max = R_MAX  # Observed maximum radius (updated periodically)

# Main loop
paused = False
frame = 0
freeze_frame_test = False  # Phase A1: diagnostic flag to test temporal skew hypothesis
jfa_measurement_counter = 0  # Track measurement frames for JFA cadence control
settle_frames_left = 0  # FSC-Only Controller: frames remaining in settling phase

while window.running:
    # Handle keyboard input
    if window.get_event(ti.ui.PRESS):
        # Debug: Confirm key presses are being received
        print(f"[DEBUG] Key pressed: {window.event.key}")
        
        if window.event.key == ti.ui.SPACE:
            paused = not paused
            print(f"[Control] {'Paused' if paused else 'Resumed'}")
        elif window.event.key == 's' or window.event.key == 'S':
            # EXPORT DATA TO PROVE RADII ARE CHANGING
            pos_np = pos.to_numpy()
            rad_np = rad.to_numpy()
            
            import csv
            export_file = f"particle_data_frame_{frame}.csv"
            with open(export_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'X', 'Y', 'Z', 'Radius', 'FSC'])
                fsc_np = jfa.fsc.to_numpy()
                for i in range(active_n):
                    writer.writerow([i, pos_np[i][0], pos_np[i][1], pos_np[i][2], rad_np[i], fsc_np[i]])
            
            print(f"\n{'='*70}")
            print(f"[EXPORT] Saved {export_file}")
            print(f"         Frame: {frame}, N={active_n}")
            print(f"         Radius stats: min={rad_np[:active_n].min():.6f}, mean={rad_np[:active_n].mean():.6f}, max={rad_np[:active_n].max():.6f}, std={rad_np[:active_n].std():.6f}")
            print(f"         First 20 radii: {rad_np[:20]}")
            print(f"         Last 20 radii:  {rad_np[active_n-20:active_n] if active_n >= 20 else rad_np[:active_n]}")
            print(f"         PROOF: {len(np.unique(np.round(rad_np[:active_n], 6)))} UNIQUE radius values out of {active_n}")
            print(f"{'='*70}\n")
        elif window.event.key == 'r' or window.event.key == 'R':
            # RESET all particles to current slider value
            r_start_value = r_start_rt[None]
            for i in range(active_n):
                rad[i] = r_start_value
            print(f"\n[Reset] All {active_n} particles reset to radius={r_start_value:.6f}")
            print(f"        (Particles will grow/shrink from uniform size)")
        elif window.event.key == 'f' or window.event.key == 'F':
            # Phase A1: Trigger freeze-frame test to diagnose temporal skew
            freeze_frame_test = True
            print(f"\n{'='*70}")
            print(f"[FREEZE-FRAME TEST] Phase A1 Diagnostic")
            print(f"         Frame {frame}: Triggering frozen frame to test temporal skew hypothesis")
            print(f"         Skipping PBD, jitter, growth → rebuild grid → count → validate")
            print(f"         Expected: miss rate <1% if temporal skew is the cause")
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
        # === FREEZE-FRAME VALIDATOR ===
        # Press 'F' to validate PBC and grid integrity (no motion, just checks)
        if freeze_frame_test:
            print(f"\n{'='*70}")
            print(f"[FREEZE-FRAME] Frame {frame} - PBC & Grid Integrity Check")
            print(f"{'='*70}")
            print(f"               Skipping motion updates for this frame...")
            
            # Rebuild grid on frozen snapshot (skip forward/backward motion)
            rebuild_grid()
            
            # Check CSR integrity: sum(cell_count) == active_n
            cell_count_np = cell_count.to_numpy()[:GRID_RES**3]
            total_assigned = int(cell_count_np.sum())
            print(f"               CSR Check: sum(cell_count) = {total_assigned}, active_n = {active_n}")
            if total_assigned == active_n:
                print(f"               ✅ PASS - Every particle assigned to exactly one cell")
            else:
                print(f"               ❌ FAIL - CSR integrity broken (mismatch: {abs(total_assigned - active_n)} particles)")
            
            # Check PBC: all particles should be within primary cell [-L/2, L/2)³ (centered wrapping)
            pos_np = pos.to_numpy()[:active_n]
            half_L = DOMAIN_SIZE * 0.5
            out_of_bounds = np.sum((pos_np < -half_L) | (pos_np >= half_L))
            print(f"               PBC Check: particles out of bounds = {out_of_bounds}")
            if out_of_bounds == 0:
                print(f"               ✅ PASS - All particles within primary cell [-L/2, L/2)³")
            else:
                print(f"               ❌ FAIL - {out_of_bounds} coordinate(s) out of [-L/2, L/2)³")
            
            print(f"{'='*70}\n")
            
            # Reset flag and skip rest of this frame's updates
            freeze_frame_test = False
            frame += 1
            continue
        
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
        
        # === 3B. FSC-Only Controller: Every-frame settling + PBD backpressure ===
        if JFA_ENABLED and FSC_MANUAL_MODE:
            # Compute adaptive EMA alpha
            alpha = alpha_for_frames(ADJUSTMENT_FRAMES)
            
            # Compute max overlap for backpressure
            o_max = compute_max_overlap(pos, rad, local_max_depth, cell_start, cell_count, cell_indices, active_n)
            back_global = backpressure_from_overlap(o_max, CONTACT_TOL)
            
            # Map mode string → enum
            bp_mode = 2 if BACKPRESSURE_MODE == "local" else (1 if BACKPRESSURE_MODE == "global" else 0)
            
            # Nudge radii if still settling
            if settle_frames_left > 0:
                nudge_radii_adaptive_ema(
                    rad, rad_target, local_max_depth,
                    int(active_n), float(alpha), float(MAX_STEP_PCT),
                    float(R_MIN), float(R_MAX), float(back_global), int(bp_mode)
                )
                settle_frames_left -= 1
            
            # === PRESSURE EQUILIBRATION (runs every frame) ===
            # Volume-conserving pressure diffusion across FSC neighbors
            # Complements FSC controller: FSC drives topology, pressure balances mechanics
            if PRESSURE_EQUILIBRATION_ENABLED and jfa.fsc[0] >= 0:  # Only if JFA has run at least once
                # Call equilibrator and capture stats (returns max_abs_dr, changed_count)
                max_abs_dr, changed = equilibrate_pressure(
                    n=int(active_n),
                    frame=frame,
                    pos=pos,
                    rad=rad,
                    delta_r=delta_r,
                    jfa_face_ids=jfa.face_ids,
                    jfa_fsc=jfa.fsc,
                    k=float(PRESSURE_K),
                    P_exp=float(PRESSURE_EXP),
                    pair_cap=float(PRESSURE_PAIR_CAP),
                    max_nei=int(MAX_EQ_NEI),
                    r_min=float(R_MIN),
                    r_max=float(R_MAX)
                )
                
                # Diagnostic telemetry every EQ_DEBUG_EVERY frames
                if (frame % EQ_DEBUG_EVERY) == 0:
                    rmin, rmax, sigma_p = compute_pressure_stats(int(active_n), rad, float(PRESSURE_EXP), float(R_MIN), float(R_MAX))
                    print(f"[EQ Debug] max|Δr|={max_abs_dr:.8f} | changed={changed}/{active_n}")
                    print(f"[Pressure] r_min={rmin:.6f} r_max={rmax:.6f} σ(P)={sigma_p:.6f}")
            
            # HUD update (≈10Hz)
            if frame % 6 == 0:
                speed_pct = int(back_global * 100)
                hud_text = f"Settling: {max(0, settle_frames_left)}/{ADJUSTMENT_FRAMES} | Alpha: {alpha*100:.1f}% | Speed: {speed_pct}%"
                print(f"[HUD] {hud_text}")
        
        # === BROWNIAN MOTION (thermal jitter to prevent frozen equilibrium) ===
        # Apply continuous micro-perturbation to keep foam "breathing"
        if BROWNIAN_ENABLED:
            apply_brownian_motion(
                pos=pos,
                n=int(active_n),
                strength=float(BROWNIAN_STRENGTH),
                domain_size=float(DOMAIN_SIZE)
            )
        
        # === 4. Adaptive PBD passes (with optional repacking during adjustment) ===
        # During adjustment phase: use push-pull repacking to propagate pressure
        # Otherwise: use standard PBD (push only)
        in_adjustment_phase = (adjustment_timer[None] > 0)
        use_repacking = (REPACK_ENABLED and in_adjustment_phase)
        
        # Compute mean radius ONCE before passes (avoid expensive CPU sync in loop)
        if use_repacking:
            rad_np = rad.to_numpy()[:active_n]
            mean_r = float(rad_np.mean())
        
        # Extra passes during repacking to propagate expansion pressure into shrink gaps
        total_passes = passes_needed + (REPACK_EXTRA_PASSES if use_repacking else 0)
        
        for pass_idx in range(total_passes):
            rebuild_grid()  # Fresh neighbors each pass
            
            if use_repacking:
                # Push-pull repacking: expanding particles push, shrinking create pull pressure
                # Reset telemetry counters
                repack_push_count[None] = 0
                repack_pull_count[None] = 0
                
                project_with_pull(
                    pos, rad, cell_start, cell_count, cell_indices, active_n,
                    mean_r, REPACK_BETA_PUSH, REPACK_BETA_PULL,
                    REPACK_DEADZONE_TAU, REPACK_STEP_CAP_FRAC,
                    repack_push_count, repack_pull_count
                )
            else:
                # Standard PBD (push only)
                project_overlaps(pos, rad, cell_start, cell_count, cell_indices, active_n)
        
        # === 5. XSPH velocity smoothing ===
        if XSPH_ENABLED:
            apply_xsph_smoothing(pos, vel, vel_temp, rad, cell_start, cell_count, cell_indices, active_n)
        
        # === 6. Global damping (prevent slow energy accumulation) ===
        apply_global_damping(vel, active_n)
        
        # === 6B. Brownian motion (OU jitter) - Track 1 Option B ===
        # DISABLE Brownian during adjustment to maintain contacts while size changes
        if adjustment_timer[None] == 0:
            # Only jitter between adjustment cycles
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
        
        # === 8. SMOOTH ADJUSTMENT CYCLE (continuous per-particle growth/shrink) ===
        # When timer==0: measure degree → set targets → reset timer
        # While timer>0: nudge radii toward targets + PBD → decrement timer
        
        # Track r_mean for telemetry (scoped for entire measurement block)
        r_mean_after_pump_global = 0.0
        
        if adjustment_timer[None] <= 0:
            
            # === JFA POWER DIAGRAM (PHASE 2: REFINEMENT) ===
            # Run JFA ONLY on measurement frames AFTER relaxation (timing guard)
            # This prevents frame-0 noise and ensures particles have settled
            #
            # Step 1: Cadence control - only run every N measurement frames
            # This reduces average overhead by N× while maintaining stable signal via EMA
            if JFA_ENABLED and adjustment_timer[None] <= 0:
                jfa_measurement_counter += 1
            
            # FIX 1: Force JFA to run UNCONDITIONALLY in manual mode (no gates, no cadence)
            # In manual mode, we need FSC every measurement for immediate feedback
            if JFA_ENABLED and FSC_MANUAL_MODE:
                jfa_should_run = True  # Always run, no conditions
            elif JFA_ENABLED and adjustment_timer[None] <= 0:
                jfa_should_run = (jfa_measurement_counter % JFA_RUN_INTERVAL == 0)
            else:
                jfa_should_run = False
            
            if jfa_should_run:
                # Step 2: Dynamic resolution based on mean particle radius
                # Compute mean radius and set JFA_RES dynamically
                r_mean_np = rad.to_numpy()[:active_n]
                r_mean = float(r_mean_np.mean())
                
                # Voxel size ≈ 2.5-3.0 × mean radius (ensures particles span multiple voxels)
                from config import JFA_VOXEL_SCALE, JFA_RES_MIN, JFA_RES_MAX
                voxel_size = JFA_VOXEL_SCALE * r_mean
                jfa_res_dynamic = int(round(DOMAIN_SIZE / voxel_size))
                jfa_res_dynamic = max(JFA_RES_MIN, min(jfa_res_dynamic, JFA_RES_MAX))
                
                # Update JFA resolution dynamically
                jfa.set_resolution(jfa_res_dynamic)
                
                # Start timing JFA
                t_jfa_start = time.perf_counter()
                
                # Run full JFA pipeline: init → propagate → extract FSC
                fsc_array, jfa_stats = jfa.run_jfa(pos, rad, active_n)
                
                # Validate JFA results (symmetry, overflow, self-loops)
                jfa_validation = jfa.validate_jfa(active_n)
                
                # End timing
                t_jfa = time.perf_counter() - t_jfa_start
                
                # ================================================================
                # STEP 4 - DEVICE-SIDE TELEMETRY (GPU METRICS)
                # ================================================================
                # All metrics now computed on GPU - minimal CPU-GPU transfer
                if True:  # Print every JFA frame (cadence = 3)
                    # Extract GPU-computed metrics from stats dict
                    mean_fsc = jfa_stats.get("mean_fsc", 0.0)
                    min_fsc = jfa_stats.get("min_fsc", 0)
                    max_fsc = jfa_stats.get("max_fsc", 0)
                    mean_fsc_ema = jfa_stats.get("mean_fsc_ema", 0.0)
                    mean_score = jfa_stats.get("mean_score", 0.0)
                    std_score = jfa_stats.get("std_score", 0.0)
                    overflow_pct = jfa_stats.get("overflow_pct", 0.0)
                    p_dir = jfa_stats.get("p_dir", 0)
                    p_und = jfa_stats.get("p_und_accepted", 0)
                    
                    # Asymmetry from validation (still uses old kernel for now)
                    asym_pct = 100.0 * jfa_validation["asymmetric_pairs"] / max(1, active_n)
                    
                    # Updated telemetry output
                    print(f"[JFA Step 4] FSC μ={mean_fsc:.1f} [{min_fsc},{max_fsc}] | "
                          f"EMA μ={mean_fsc_ema:.1f}")
                    print(f"      score: μ={mean_score:.2f} σ={std_score:.2f} | "
                          f"pairs: {p_und} (P_dir={p_dir})")
                    print(f"      asym={asym_pct:.1f}% overflow={overflow_pct:.1f}% | "
                          f"time={t_jfa*1000:.1f}ms | res={jfa_res_dynamic}³ | "
                          f"passes={jfa_stats['num_passes']} "
                          f"(early_exit={jfa_stats['early_exit']})")
                    
                    # Validation status with updated acceptance criteria
                    validation_passed = jfa_validation["passed"]
                    step4_criteria_met = (
                        asym_pct < 1.0 and  # Asymmetry < 1%
                        overflow_pct == 0.0 and  # No overflow
                        mean_score >= 0.8 and mean_score <= 1.2 and  # Score calibration
                        mean_fsc >= 10.0  # FSC in reasonable range (relaxed for early frames)
                    )
                    
                    if validation_passed and step4_criteria_met:
                        print(f"      ✅ Step 4 SUCCESS - All metrics in target range")
                    elif validation_passed:
                        print(f"      ✓ Validation passed (still converging)")
                    else:
                        print(f"      ⚠️  Validation FAILED")
            
            # ================================================================
            # WARM-START GATE (prevents FSC=0 runaway during startup)
            # ================================================================
            # Estimate fraction of zero-face sites on this measurement
            if JFA_ENABLED and FSC_MANUAL_MODE and jfa_should_run:
                fsc_np_warmstart = jfa.fsc.to_numpy()[:active_n]
                zero_faces = int(np.sum(fsc_np_warmstart == 0))
                f0 = float(zero_faces) / float(active_n) if active_n > 0 else 1.0
                warmstart = (frame < WARMSTART_FRAMES) or (f0 > FSC_ZERO_RATE_THRESH)
                
                if warmstart:
                    # Disable growth and let geometry settle so JFA can detect faces
                    print(f"[Warm-start] f0={f0*100:.1f}% have FSC=0 → growth DISABLED (frame {frame}/{WARMSTART_FRAMES})")
                    GROWTH_ENABLED = False
                else:
                    GROWTH_ENABLED = True
            else:
                GROWTH_ENABLED = True  # Default: growth enabled
            
            # ================================================================
            # FSC-ONLY CONTROLLER (PHASE 2)
            # ================================================================
            # Measurement frame: JFA → set targets from FSC band → reset settle counter
            
            if JFA_ENABLED and FSC_MANUAL_MODE and GROWTH_ENABLED:
                # Telemetry: r_mean BEFORE targets set
                rad_before = rad.to_numpy()[:active_n]
                r_mean_before = float(rad_before.mean())
                
                # Set power beta to 1.0 for standard power diagram
                jfa.power_beta_current[None] = 1.0
                
                # Set radius targets from FSC band (with hysteresis + EMA lag)
                # Use GUI values (live control) and EMA (smoothed FSC) for decisions
                set_fsc_targets(
                    rad, rad_target, jfa.fsc, jfa.fsc_ema,
                    int(active_n),
                    int(gui_fsc_low), int(gui_fsc_high),
                    float(GROWTH_PCT),
                    float(R_MIN), float(R_MAX)
                )
                
                # Reset settle counter
                settle_frames_left = int(ADJUSTMENT_FRAMES)
                
                # Safety test (optional, during bring-up)
                if RUN_SAFETY_TESTS:
                    r_np = rad.to_numpy()[:active_n]
                    rmin_act, rmax_act = float(r_np.min()), float(r_np.max())
                    assert rmin_act >= R_MIN and rmax_act <= R_MAX, \
                        f"Radius bounds violated: [{rmin_act:.6f}, {rmax_act:.6f}] vs [{R_MIN:.6f}, {R_MAX:.6f}]"
                
                # Telemetry: FSC stats, band counts, r_mean → target_mean
                fsc_np = jfa.fsc.to_numpy()[:active_n]
                fsc_mean = float(fsc_np.mean())
                fsc_min = int(fsc_np.min())
                fsc_max = int(fsc_np.max())
                
                target_np = rad_target.to_numpy()[:active_n]
                r_target_mean = float(target_np.mean())
                
                n_grow = int(np.sum(fsc_np < gui_fsc_low))
                n_shrink = int(np.sum(fsc_np > gui_fsc_high))
                n_hold = active_n - n_grow - n_shrink
                
                print(f"\n[JFA Measurement] frame={frame}")
                print(f"  FSC: μ={fsc_mean:.1f} [{fsc_min},{fsc_max}] | band=[{gui_fsc_low},{gui_fsc_high}]")
                print(f"  Distribution: grow={n_grow} ({100.0*n_grow/active_n:.1f}%) | "
                      f"in-band={n_hold} ({100.0*n_hold/active_n:.1f}%) | "
                      f"shrink={n_shrink} ({100.0*n_shrink/active_n:.1f}%)")
                print(f"  r_mean: {r_mean_before:.6f} → target={r_target_mean:.6f}")
                print(f"  Settling for {ADJUSTMENT_FRAMES} frames...")
            
            # End of measurement block
        
        # === 9. MOTION happens every frame (PBD already done above) ===
        # PBD + XSPH + Brownian run continuously to maintain tight packing
        # and prevent jitter. Radius adjustments happen smoothly in parallel.
        
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
            
            # === Topology tracking (FSC-only system, no degree field) ===
            # Topological degree is computed but not used for control
            pass
        
        # === 12. Update observed radius range (every 15 frames for GUI context) ===
        if frame % 15 == 0:
            rad_np_sample = rad.to_numpy()[:active_n]
            r_obs_min = float(rad_np_sample.min()) if active_n > 0 else R_MIN
            r_obs_max = float(rad_np_sample.max()) if active_n > 0 else R_MAX
        
        # === 13. Update colors (size-based visualization only, FSC-only system) ===
        update_colors_by_size(rad, color, active_n, r_obs_min, r_obs_max,
                             viz_mode_rt[None], viz_band_min_rt[None], viz_band_max_rt[None],
                             viz_hide_out_rt[None], viz_palette_rt[None], viz_dim_alpha_rt[None])
        
        ti.sync()
        t_topo = time.perf_counter()
        
        frame += 1
        
        # === Telemetry & Logging ===
        if frame % 100 == 0:
            rad_np = rad.to_numpy()[:active_n]
            
            rescue_pct = 100.0 * rescue_frame_count / frame if frame > 0 else 0.0
            rescue_status = f"{rescue_pct:.1f}%" if RESCUE_ENABLED else f"{rescue_pct:.1f}% (disabled)"
            
            # FSC-Only telemetry (no degree)
            fsc_np = jfa.fsc.to_numpy()[:active_n] if JFA_ENABLED else np.zeros(active_n)
            fsc_mean = float(fsc_np.mean()) if active_n > 0 else 0.0
            fsc_min = int(fsc_np.min()) if active_n > 0 else 0
            fsc_max = int(fsc_np.max()) if active_n > 0 else 0
            
            print(f"[Frame {frame:4d}] N={active_n}, Passes={passes_needed}, MaxDepth={max_depth:.6f}")
            print(f"    FSC: mean={fsc_mean:.1f}, min={fsc_min}, max={fsc_max} | band=[{FSC_LOW},{FSC_HIGH}]")
            print(f"    Radius: mean={rad_np.mean():.4f}, min={rad_np.min():.4f}, max={rad_np.max():.4f}")
            print(f"    Rescue: {rescue_status}")
        
        # === Grid Validation (MOVED TO MEASUREMENT BLOCK - Phase A2) ===
        # Validation now happens inside the measurement block (line ~631)
        # immediately after count_neighbors, using the same snapshot.
        # This eliminates temporal skew and should drop miss rate to <1%.
    
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
    # Compute FSC distribution statistics (only active particles)
    if JFA_ENABLED:
        fsc_np = jfa.fsc.to_numpy()[:active_n]
    else:
        fsc_np = np.zeros(active_n, dtype=np.int32)
    rad_np = rad.to_numpy()[:active_n]
    
    # FSC distribution relative to band [gui_fsc_low, gui_fsc_high]
    count_red = np.sum(fsc_np < gui_fsc_low)      # Below band (grow)
    count_green = np.sum((fsc_np >= gui_fsc_low) & (fsc_np <= gui_fsc_high))  # In-band (hold)
    count_blue = np.sum(fsc_np > gui_fsc_high)    # Above band (shrink)
    
    pct_red = 100.0 * count_red / active_n if active_n > 0 else 0.0
    pct_green = 100.0 * count_green / active_n if active_n > 0 else 0.0
    pct_blue = 100.0 * count_blue / active_n if active_n > 0 else 0.0
    
    # IMGUI window for controls and stats (expanded height for particle count control)
    window.GUI.begin("Control Panel", 0.01, 0.01, 0.32, 0.70)
    
    # Particle count section
    window.GUI.text(f"=== Particle Count ===")
    window.GUI.text(f"Active: {active_n} / {MAX_N}")
    gui_n_particles = window.GUI.slider_int("New N", gui_n_particles, 100, MAX_N)
    if window.GUI.button("RESTART"):
        restart_requested = True
    window.GUI.text("")
    
    # FSC statistics section (FSC-Only Controller)
    window.GUI.text(f"=== FSC Stats ===")
    if JFA_ENABLED and active_n > 0:
        window.GUI.text(f"Avg:  {fsc_np.mean():.1f}")
        window.GUI.text(f"Min:  {int(fsc_np.min())}")
        window.GUI.text(f"Max:  {int(fsc_np.max())}")
    else:
        window.GUI.text(f"Avg:  N/A (JFA disabled)")
        window.GUI.text(f"Min:  N/A")
        window.GUI.text(f"Max:  N/A")
    window.GUI.text("")
    window.GUI.text("Distribution:")
    window.GUI.text(f"  <{gui_fsc_low}: {pct_red:.1f}% (grow)")
    window.GUI.text(f"  {gui_fsc_low}-{gui_fsc_high}: {pct_green:.1f}% (hold)")
    window.GUI.text(f"  >{gui_fsc_high}: {pct_blue:.1f}% (shrink)")
    window.GUI.text("")
    
    # FSC band sliders (live control)
    gui_fsc_low = window.GUI.slider_int("FSC Low", gui_fsc_low, 1, 30)
    gui_fsc_high = window.GUI.slider_int("FSC High", gui_fsc_high, 1, 40)
    # Ensure low < high
    if gui_fsc_low >= gui_fsc_high:
        gui_fsc_high = gui_fsc_low + 1
    window.GUI.text("")
    
    # Radius limits section
    window.GUI.text(f"=== Radius Limits ===")
    window.GUI.text(f"Current: {rad_np.min():.5f} - {rad_np.max():.5f}")
    
    # Starting radius slider (runtime only - edit config.py to change default)
    r_start = window.GUI.slider_float("Starting radius", r_start_rt[None], 0.0005, 0.0100)
    r_start_rt[None] = r_start
    window.GUI.text(f"(Press 'R' to reset particles to this size)")
    
    # Min/max bounds sliders
    gui_r_min = window.GUI.slider_float("Min radius", gui_r_min, 0.0001, 0.0100)
    gui_r_max = window.GUI.slider_float("Max radius", gui_r_max, 0.0001, 0.0200)
    window.GUI.text("")
    
    # Growth Rhythm section (runtime controls - no snap-back!)
    window.GUI.text(f"=== Growth Rhythm ===")
    rate = window.GUI.slider_float("Growth/shrink rate per cycle", grow_rate_rt[None], 0.01, 0.10)
    adj_frames = window.GUI.slider_int("Adjustment frames (smoothness)", adjustment_frames_rt[None], 1, 120)
    
    # Commit slider values to runtime fields (no snap-back!)
    grow_rate_rt[None] = rate
    adjustment_frames_rt[None] = adj_frames
    
    # Status line showing current cycle phase
    if adjustment_timer[None] > 0:
        window.GUI.text(f"Adjusting... {adjustment_timer[None]} frames left")
    else:
        window.GUI.text(f"Measuring next frame...")
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
    
    # Color mode selector (FSC-Only: no degree mode)
    mode_labels = ["Size Heatmap", "Size Heatmap", "Size Band"]
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
print(f"       Final mean radius: {rad.to_numpy()[:active_n].mean():.4f}")
if JFA_ENABLED:
    fsc_np = jfa.fsc.to_numpy()[:active_n]
    print(f"       Final mean FSC: {fsc_np.mean():.2f}")

