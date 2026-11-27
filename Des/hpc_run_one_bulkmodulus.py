"""
Script to run a vcrelax given an cif file for parallel job array hpc calculations

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 12/11/2026
"""

import os
from pathlib import Path
import argparse
from datetime import datetime
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from utils import *
from calculations import run_calculation, VC_RELAX_SETTINGS_HIGH, VC_RELAX_SETTINGS_LOW
from crystal_builder import read_cif
from dataclasses import replace

# Parse input cif and output folder
parser = argparse.ArgumentParser(
    prog="run_one",
    description="Run one vc-relax calculation of a pressure series for a given cif file.",
)
parser.add_argument("--cif", required=True, help="Input CIF file")
parser.add_argument("--output_dir", required=True, help="Base output folder")
parser.add_argument("--p_max", type=float, required=True,
                    help="Maximum |pressure| (kbar); range is -p_max .. +p_max")
parser.add_argument("--n_points", type=int, default=8,
                    help="Number of pressure points, zero excluded (must be even, default 8)")
parser.add_argument("--index", type=int, required=True, help="Pressure point index (0-based)")
parser.add_argument("--bulk_modulus_guess", type=float, help="Initial guess for the bulk modulus (GPa)")
parser.add_argument("--settings", choices=["low", "high"], default="low",
                    help="Calculation settings: 'low' (default) or 'high'")
args = parser.parse_args()


# Pressure for this array task
if args.n_points < 2 or args.n_points % 2 != 0:
    sys.exit(f"n_points must be an even number >= 2 (got {args.n_points})")
if not 0 <= args.index < args.n_points:
    sys.exit(f"Index {args.index} out of range for {args.n_points} points")
# Symmetric equidistant grid with N+1 points (0 in the middle), then drop the 0 point
grid = np.linspace(-args.p_max, args.p_max, args.n_points + 1)
pressures = np.delete(grid, args.n_points // 2)
pressure = float(pressures[args.index])

# Load and rescale the structure if needed
structure = read_cif(args.cif)
if args.bulk_modulus_guess is not None:
    pressure_gpa = pressure * 0.1  # kbar -> GPa
    volume_ratio = 1.0 - pressure_gpa / args.bulk_modulus_guess
    structure.set_cell(structure.cell * volume_ratio ** (1 / 3), scale_atoms=True)

# Change the target pressure
settings = VC_RELAX_SETTINGS_LOW if args.settings == "low" else VC_RELAX_SETTINGS_HIGH
settings_targetpress = replace(settings, target_press=pressure)

# Per-pressure output folder
calc_dir = Path(args.output_dir) / f"P_{pressure:+.1f}kbar"
os.makedirs(calc_dir, exist_ok=True)

print(f"[{datetime.now():%H:%M:%S}] Starting vc-relax for {args.cif} at {pressure:.2f} kbar "
      f"(point {args.index + 1}/{args.n_points})")
run_calculation(structure, settings_targetpress, calc_dir=str(calc_dir))
print(f"[{datetime.now():%H:%M:%S}] Finished relaxing {args.cif} at {pressure:.2f} kbar")