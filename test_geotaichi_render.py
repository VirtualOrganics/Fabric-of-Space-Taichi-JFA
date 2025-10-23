#!/usr/bin/env python3
"""
Minimal test to exactly replicate GeoTaichi's rendering approach.
Based on GeoTaichi's src/dem/DEMBase.py:155
"""
import taichi as ti
import numpy as np

# Initialize Taichi (same as GeoTaichi)
ti.init(arch=ti.metal, default_fp=ti.f64)

# Create particle data
N = 100
pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
rad = ti.field(dtype=ti.f32, shape=N)

# Seed with varying radii
np.random.seed(42)
pos_np = np.random.uniform(0, 0.15, (N, 3)).astype(np.float32)
rad_np = np.random.uniform(0.002, 0.008, N).astype(np.float32)

pos.from_numpy(pos_np)
rad.from_numpy(rad_np)

print(f"[Test] Created {N} particles")
print(f"       Radius range: [{rad_np.min():.6f}, {rad_np.max():.6f}]")
print(f"       Radius spread: {rad_np.max() / rad_np.min():.2f}x")

# Setup GUI (GeoTaichi method)
window = ti.ui.Window("GeoTaichi Render Test", (800, 800), vsync=True)
canvas = window.get_canvas()
ui_scene = window.get_scene()  # <-- GeoTaichi calls it "ui_scene"
camera = ti.ui.Camera()

# Camera setup
camera.position(0.075, 0.075, 0.3)
camera.lookat(0.075, 0.075, 0.075)
camera.up(0.0, 1.0, 0.0)

print("\n[Test] Rendering with GeoTaichi's exact method:")
print("       ui_scene.particles(pos, per_vertex_radius=rad, color=(0.5, 0.8, 0.5))")
print("\nControls:")
print("  - Right-click drag: Rotate")
print("  - Mouse wheel: Zoom")
print("  - ESC: Exit")
print("="*70)

frame = 0
while window.running:
    # Handle ESC
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.ESCAPE:
            break
    
    # Camera controls
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
    
    # Render (GeoTaichi's exact line from DEMBase.py:155)
    ui_scene.set_camera(camera)
    ui_scene.ambient_light((0.8, 0.8, 0.8))
    ui_scene.point_light(pos=(0.1, 0.2, 0.1), color=(1, 1, 1))
    
    # TEST: SceneV2 requires radius argument - try with BOTH radius and per_vertex_radius
    # Use a tiny fallback radius to see if per_vertex_radius overrides it
    ui_scene.particles(pos, radius=0.001, per_vertex_radius=rad, color=(0.5, 0.8, 0.5))
    
    canvas.scene(ui_scene)
    window.show()
    
    frame += 1
    if frame % 100 == 0:
        print(f"[Frame {frame:4d}] Still rendering...")

print(f"\n[Exit] Test complete. Rendered {frame} frames.")

