# Fabric of Space - Custom Taichi Grid

A custom spatial grid implementation in Taichi for simulating variable-radius particles with dynamic neighbor detection and radius adaptation.

## Features

- **Custom spatial hashing grid**: Efficient neighbor search with periodic boundaries
- **Dynamic radius adaptation**: Particles grow/shrink based on neighbor count (target: 5-6 neighbors)
- **Position-Based Dynamics (PBD)**: Overlap resolution for stable particle separation
- **Real-time 3D visualization**: Interactive camera controls and color-coded particles

## Quick Start

### Installation

```bash
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
- **SPACE**: Pause/Resume simulation
- **ESC**: Exit

## Configuration

Edit `config.py` to adjust simulation parameters:

- `N`: Number of particles (default: 5000)
- `DOMAIN_SIZE`: Cubic domain size (default: 0.15)
- `R_MIN`, `R_MAX`: Radius bounds (default: 0.0015, 0.0060)
- `DEG_LOW`, `DEG_HIGH`: Target degree thresholds (default: 5, 6)
- `GAIN_GROW`, `GAIN_SHRINK`: Radius change rate (default: 5% per step)
- `PBD_PASSES`: Overlap resolution iterations (default: 4)
- `CONTACT_TOL`: Contact detection tolerance (default: 2%)

## Architecture

- `config.py`: Global constants and parameters
- `grid.py`: Spatial hashing and neighbor detection kernels
- `dynamics.py`: Radius adaptation and PBD overlap resolution
- `run.py`: Main simulation loop and visualization

## Design Principles

- **No external dependencies**: Pure Taichi + NumPy, no SPH/DEM frameworks
- **Frame-by-frame radius changes**: Grid rebuilds every loop for accurate detection
- **Conservative cell size**: `CELL_SIZE = 2 * R_MAX` ensures 27-stencil captures all neighbors
- **Periodic boundaries**: Toroidal domain with minimum-image convention

## Performance

- **5K particles**: ~20 FPS on Apple Silicon M1
- **Grid rebuild cost**: ~1-2ms per frame (acceptable for real-time use)

## Acceptance Tests

1. **Grid accuracy**: Neighbor counts match brute-force reference
2. **Radius individuality**: Isolated particles grow, crowded particles shrink independently
3. **PBD stability**: No tunneling or explosions after 1000+ frames
4. **Periodic wrap**: Particles near boundaries detect cross-domain neighbors

## License

MIT

## Author

Built for the Fabric of Space project - exploring emergent foam-like behavior in variable-radius particle systems.
