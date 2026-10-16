# kconv.py
import os, csv
import numpy as np
from ase.io import read
from ase.units import GPa
from ase.calculators.espresso import Espresso, EspressoProfile

CIF = "CIF Files/Fe7Al-structure.cif"          # adjust to the provided file name
PSEUDO_DIR = "/Pseudopotentials"
PSEUDOS = {"Fe": "Fe.paw.pbe.z_16.ld1.psl.v0.2.1.upf",
           "Al": "Al.us.pbe.z_3.ld1.psl.v1.0.0-low.upf"}
ECUTWFC, ECUTRHO = 50, 400            # keep fixed here; converge these separately
SPACINGS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10]   # 1/Å, coarse -> fine
OUT = "kconv.csv"

def kmesh(atoms, spacing):
    """Mesh with reciprocal-space spacing <= `spacing` (in 1/Å, incl. 2π)."""
    b = 2 * np.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return tuple(int(max(1, np.ceil(x / spacing))) for x in b)

def done_spacings():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {float(r["spacing"]) for r in csv.DictReader(f)}

profile = EspressoProfile(command="mpirun -np 4 pw.x", pseudo_dir=PSEUDO_DIR)
base = read(CIF)

for spacing in SPACINGS:
    if spacing in done_spacings():
        continue

    atoms = base.copy()
    atoms.set_initial_magnetic_moments(
        [3.0 if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()])
    kpts = kmesh(atoms, spacing)

    atoms.calc = Espresso(
        profile=profile,
        directory=f"calc/kconv_{spacing:.2f}",
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

    energy = atoms.get_potential_energy()                 # eV
    stress = atoms.get_stress(voigt=False)                # eV/Å^3, 3x3
    pressure_kbar = -np.trace(stress) / 3 / GPa * 10      # 1 GPa = 10 kbar
    moment = atoms.get_magnetic_moment()                  # μB per cell

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["spacing", "k1", "k2", "k3", "nk_total",
                        "energy_eV", "pressure_kbar", "moment_muB"])
        w.writerow([spacing, *kpts, int(np.prod(kpts)),
                    energy, pressure_kbar, moment])
    print(f"spacing {spacing:.2f}  mesh {kpts}  "
          f"P = {pressure_kbar:8.2f} kbar  m = {moment:.3f} μB", flush=True)