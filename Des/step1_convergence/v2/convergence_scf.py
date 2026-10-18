import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os, csv
import random
from time import time
import numpy as np
import shutil

from ase.io import read
from ase.units import GPa
from ase.calculators.espresso import Espresso, EspressoProfile

from utils import *

PSEUDO_DIR = "/home/des/cmp-fe-al-phase-diagram/Pseudopotentials"
PSEUDOS = {"Fe": "Fe.paw.pbe.z_16.ld1.psl.v0.2.1",
           "Al": "Al.us.pbe.z_3.ld1.psl.v1.0.0-low"}

def single_sfc(ecutwfc, ecutrho, kmesh, output_dir):
    ECUTWFC, ECUTRHO = ecutwfc, ecutrho
    KMESH = kmesh
    OUT =  output_dir + "/" + "results.csv"

    profile = EspressoProfile(command="mpirun -np 1 /home/des/q-e/bin/pw.x",
                            pseudo_dir=PSEUDO_DIR)

    tag = f"{"x".join(str(k) for k in KMESH)}_{ECUTWFC}_{ECUTRHO}"
    calc_dir = output_dir + "/" + tag
    os.mkdir(calc_dir)

    atoms = atoms_fe7al(lattice_parameter=2.83)
    atoms.set_initial_magnetic_moments(
        [3.0 if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()])
    # write_cif(atoms, calc_dir, 'Fe7Al')

    atoms.calc = Espresso(
        profile=profile,
        directory=calc_dir,
        pseudopotentials={s: PSEUDOS[s] for s in set(atoms.get_chemical_symbols())},
        input_data={
            "control": {"calculation": "scf", "tstress": True, "tprnfor": True},
            "system": {"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                        "occupations": "smearing", "smearing": "mv",
                        "degauss": 0.02, "nspin": 2},
            "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep":200},
        },
        kpts=KMESH,
    )

    tick = time() # Timer
    energy = atoms.get_potential_energy()                 # eV
    stress = atoms.get_stress(voigt=False)                # eV/Å^3, 3x3
    tock = time()

    pressure_kbar = -np.trace(stress) / 3 / GPa * 10      # 1 GPa = 10 kbar
    moment = read_total_magnetization(calc_dir)           # μB per cell

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["k1", "k2", "k3", "ecutwfc", "ecutrho",
                        "energy (eV)", "pressure (kbar)", "moment (muB)", "Wall time"])
        w.writerow([*KMESH, ECUTWFC, ECUTRHO,
                    energy, pressure_kbar, moment, tock-tick])
    print(f"mesh {tag:>8}  P = {pressure_kbar:8.2f} kbar  m = {moment:.3f} μB",
        flush=True)
    
if __name__ ==  "__main__":
    kmesh_sizes = [(4, 4, 4), (5, 5, 5), (6, 6, 6)]

    while True:
        kmesh = random.choice(kmesh_sizes)
        ecutwfc = random.randint(70, 120)
        ecutrho = random.randint(600, 1200)
        tag = f"{"x".join(str(k) for k in kmesh)}_{ecutwfc}_{ecutrho}"

        single_sfc(ecutwfc, ecutrho, kmesh, "/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/final_settings")
        shutil.rmtree("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/final_settings/" + tag)
