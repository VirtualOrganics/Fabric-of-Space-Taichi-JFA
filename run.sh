#!/bin/bash
# Launch script for Fabric of Space - Custom Grid

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run simulation
python3 run.py

# Deactivate on exit
deactivate

