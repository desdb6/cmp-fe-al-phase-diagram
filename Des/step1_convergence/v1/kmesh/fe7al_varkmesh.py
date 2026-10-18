import os, csv
from time import time
import numpy as np
from ase.io import read
from ase.units import GPa
from ase.calculators.espresso import Espresso, EspressoProfile

CIF = "CIF Files/Fe7Al-structure.cif"
PSEUDO_DIR = "/home/des/cmp-fe-al-phase-diagram/Pseudopotentials"
PSEUDOS = {"Fe": "Fe.paw.pbe.z_16.ld1.psl.v0.2.1",
           "Al": "Al.us.pbe.z_3.ld1.psl.v1.0.0-low"}
ECUTWFC, ECUTRHO = 64, 782 # Recommended values from Fe pseudopotential

# k-meshes to run, coarse -> fine. Write them as (k1, k2, k3).
KMESHES = [
    (1, 1, 1),
    (3, 3, 3),
    (5, 5, 5),
    (7, 7, 7),
    (9, 9, 9),
    (11, 11, 11),
]
OUT = "/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/kmesh/kmesh_conv.csv"

def read_total_magnetization(directory): # Needed because ase cannot read magnetic moment directly, needs to read the output line by line
    """Last 'total magnetization' line in espresso.pwo = converged value (μB/cell)."""
    val = float("nan")
    with open(os.path.join(directory, "espresso.pwo")) as f:
        for line in f:
            if "total magnetization" in line and "Bohr mag/cell" in line: # If a new value is found, overwrite. The final value is at the bottom.
                val = float(line.split("=")[1].split()[0])
    return val


def done_meshes():
    """Return what meshes have already been calculated"""
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {(int(r["k1"]), int(r["k2"]), int(r["k3"]))
                for r in csv.DictReader(f)}


profile = EspressoProfile(command="mpirun -np 1 /home/des/q-e/bin/pw.x",
                          pseudo_dir=PSEUDO_DIR)
base = read(CIF)

for kpts in KMESHES:
    kpts = tuple(int(k) for k in kpts)
    if kpts in done_meshes():
        continue

    tag = "x".join(str(k) for k in kpts)
    calc_dir = f"Des/step1_convergence/kmesh/kconv_{tag}"

    atoms = base.copy()
    atoms.set_initial_magnetic_moments(
        [3.0 if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()])

    atoms.calc = Espresso(
        profile=profile,
        directory=calc_dir,
        pseudopotentials={s: PSEUDOS[s] for s in set(atoms.get_chemical_symbols())},
        input_data={
            "control": {"calculation": "scf", "tstress": True, "tprnfor": True},
            "system": {"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                       "occupations": "smearing", "smearing": "mv",
                       "degauss": 0.02, "nspin": 2},
            "electrons": {"conv_thr": 1e-9, "mixing_beta": 0.3},
        },
        kpts=kpts,
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
            w.writerow(["k1", "k2", "k3",
                        "energy_eV", "pressure_kbar", "moment_muB", "Wall time"])
        w.writerow([*kpts,
                    energy, pressure_kbar, moment, tock-tick])
    print(f"mesh {tag:>8}  P = {pressure_kbar:8.2f} kbar  m = {moment:.3f} μB",
          flush=True)