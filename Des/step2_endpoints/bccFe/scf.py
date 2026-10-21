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

import csv
import os
from time import perf_counter

import numpy as np
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read


def single_relax(
    volume: float,
    settings: QESettings,
    output_folder: str,
    calculation: str = "vc-relax",
    rm_calc_dir: bool = False,
    ):
        """
        Relax a bcc Fe cell starting from a given volume.

        volume      : Initial unit cell volume in Ang**3
        settings    : ecutwfc, ecutrho, kmesh
        calculation : "vc-relax" (cell + positions) or "relax" (positions only)
        """
        if calculation not in ("relax", "vc-relax"):
            raise ValueError("calculation must be 'relax' or 'vc-relax'")

        tag = f"bccFe_{calculation}_V{volume:.4f}"
        calc_dir = output_folder + "/" + tag
        os.makedirs(calc_dir, exist_ok=True)

        lattice_parameter = volume ** (1 / 3)
        atoms = atoms_bcc_Fe(lattice_parameter)

        profile = EspressoProfile(
            command="mpirun -np 1 /home/des/q-e/bin/pw.x",
            pseudo_dir=PSEUDO_DIR,
        )

        input_data = {
            "control": {
                "calculation": calculation,
                "tstress": True,
                "tprnfor": True,
                "etot_conv_thr": 1e-6,   # Ry
                "forc_conv_thr": 1e-4,   # Ry/Bohr
                "nstep": 100,
            },
            "system": {
                "ecutwfc": settings.ecutwfc, "ecutrho": settings.ecutrho,
                "occupations": "smearing", "smearing": "mv",
                "degauss": 0.02, "nspin": 2,
            },
            "electrons": {"conv_thr": 1e-8, "mixing_beta": 0.3, "electron_maxstep": 200},
            "ions": {"ion_dynamics": "bfgs"},
        }
        if calculation == "vc-relax":
            input_data["cell"] = {
                "cell_dynamics": "bfgs",
                "press": 0.0,             # target pressure (kbar)
                "press_conv_thr": 0.1,    # kbar
                "cell_dofree": "all",
            }

        atoms.calc = Espresso(
            profile=profile,
            directory=calc_dir,
            pseudopotentials={s: PSEUDOS[s] for s in set(atoms.get_chemical_symbols())},
            input_data=input_data,
            kpts=settings.kmesh,
        )

        tick = perf_counter()
        atoms.get_potential_energy()   # triggers pw.x
        tock = perf_counter()

        # The input `atoms` keeps the starting cell, so read the final structure
        # (last ionic step) from the output file.
        final = read(os.path.join(calc_dir, "espresso.pwo"), format="espresso-out", index=-1)

        energy = final.get_potential_energy() * EV_TO_RY_CONV                     # Ry
        stress = final.get_stress(voigt=False) * EV_TO_RY_CONV * RY_PER_ANG3_TO_KBAR_CONV  # kbar
        avg_pressure = -np.trace(stress) / 3
        final_volume = final.get_volume()
        final_a = final_volume ** (1 / 3)   # valid for the cubic 2-atom cell
        moment = read_total_magnetization(calc_dir)   # μB per cell

        csv_path = output_folder + "/results_relax.csv"
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["k1", "k2", "k3", "ecutwfc", "ecutrho", "calculation",
                            "initial volume (Ang**3)", "final volume (Ang**3)",
                            "final lattice parameter (Ang)", "energy (Ry)",
                            "avg pressure (kbar)", "moment (muB)", "Wall time (s)"])
            w.writerow([*settings.kmesh, settings.ecutwfc, settings.ecutrho, calculation,
                        volume, final_volume, final_a, energy,
                        avg_pressure, moment, tock - tick])

        if rm_calc_dir:
            remove_dir(calc_dir,
                    allowed_root="/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints",
                    dry_run=False)
        return final_volume, energy