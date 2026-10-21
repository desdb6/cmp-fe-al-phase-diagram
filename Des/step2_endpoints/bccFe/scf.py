"""
Script to run single-scf calculations for bcc Fe

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 21/10/2026
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import csv
import numpy as np
from time import perf_counter

from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile

from qe_utils import *

def single_scf(volume: float, settings: QESettings, output_folder: str, rm_calc_dir: bool = False):
    """
    Run a single self-consistent field equation for a given volume.

    volume      : Unit cell volume is Ang**3
    settings    : Settings file with ecutrho, ecutwfc, kmesh settings 
    """
    tag = f"bccFe_V{volume:.4f}"
    calc_dir = output_folder + "/" + tag
    os.mkdir(calc_dir)

    lattice_parameter = volume ** (1/3)
    atoms = atoms_bcc_Fe(lattice_parameter)

    profile = EspressoProfile(command="mpirun -np 1 /home/des/q-e/bin/pw.x",
        pseudo_dir=PSEUDO_DIR)

    atoms.calc = Espresso(
        profile=profile,
        directory=calc_dir,
        pseudopotentials={s: PSEUDOS[s] for s in set(atoms.get_chemical_symbols())},
        input_data={
            "control": {"calculation": "scf", "tstress": True, "tprnfor": True},
            "system": {"ecutwfc": settings.ecutwfc, "ecutrho": settings.ecutrho,
                        "occupations": "smearing", "smearing": "mv",
                        "degauss": 0.02, "nspin": 2}, # nspin = 2 for magnetic calculation
            "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep":200},
        },
        kpts=settings.kmesh,
    )

    tick = perf_counter() # Timer
    energy = atoms.get_potential_energy() * EV_TO_RY_CONV  # Ry
    stress = atoms.get_stress(voigt=False) * EV_TO_RY_CONV * RY_PER_ANG3_TO_KBAR_CONV # kbar
    tock = perf_counter()

    pressure_xx = -stress[0, 0]  # kbar
    pressure_yy = -stress[1, 1]  # kbar
    pressure_zz = -stress[2, 2]  # kbar

    avg_pressure = -np.trace(stress) / 3
    moment = read_total_magnetization(calc_dir)           # μB per cell

    csv_path = output_folder + "/results.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["k1", "k2", "k3", "ecutwfc", "ecutrho", "volume (Ang**3)", "lattice parameter (Ang)",
                        "energy (Ry)", "pressure xx (kbar)", "pressure yy (kbar)", "pressure zz (kbar)", "avg pressure (kbar)", "moment (muB)", "Wall time (s)"])
        w.writerow([*settings.kmesh, settings.ecutwfc, settings.ecutrho, volume, lattice_parameter,
                    energy, pressure_xx, pressure_yy, pressure_zz, avg_pressure, moment, tock-tick])
                
    if rm_calc_dir:
        remove_dir(calc_dir, allowed_root="/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints", dry_run=False)
    
    return