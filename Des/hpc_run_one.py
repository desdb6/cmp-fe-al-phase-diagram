"""
Script to run a single relaxation given an cif file for parallel job array hpc calculations

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 12/11/2026
"""

import argparse
from datetime import datetime
import sys
sys.stdout.reconfigure(line_buffering=True)

from utils import *
from calculations import run_calculation, VC_RELAX_SETTINGS_HIGH, VC_RELAX_SETTINGS_LOW
from crystal_builder import read_cif

# Parse input cif and output folder
parser = argparse.ArgumentParser(
    prog='run_one',
    description='Run a single vc-relax calculation given a cif file.'
    )
parser.add_argument("--cif", required=True, help="Input CIF file")
parser.add_argument("--output_dir", required=True, help="Output folder")
parser.add_argument(
    "--settings",
    choices=["low", "high"],
    default="low",
    help="Calculation settings: 'low' (default) or 'high'",
)

args = parser.parse_args()

# Run the calculation
structure = read_cif(args.cif)
settings = VC_RELAX_SETTINGS_LOW if args.settings == "low" else VC_RELAX_SETTINGS_HIGH

print(f"[{datetime.now():%H:%M:%S}] Starting vc-relax for {args.cif} structure")
run_calculation(structure, settings, calc_dir=args.output_dir)
print(f"[{datetime.now():%H:%M:%S}] Finished relaxing {args.cif} structure")