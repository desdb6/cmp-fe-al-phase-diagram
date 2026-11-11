"""
Parsers file to read quantum espresso ooutput files.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 04/11/2026
"""
import os
from ase.io import read
from ase import Atoms
import spglib

from utils import *

def read_total_magnetization(calc_dir):
    """
    ASE is not able to read the magnetic momentum using the native function.
    This function reads the last 'total magnetization' line in espresso.pwo in the given directory, which is the converged value (μB/cell).
    """
    val = float("nan")
    with open(os.path.join(calc_dir, "espresso.pwo")) as f:
        for line in f:
            if "total magnetization" in line and "Bohr mag/cell" in line: # If a new value is found, overwrite. The final value is at the bottom.
                val = float(line.split("=")[1].split()[0])
    return val

def read_energy(calc_dir):
    """Wrapper for energy reading."""
    output = calc_dir / "espresso.pwo"
    final = read(output, format="espresso-out")
    return final.get_potential_energy() * EV_TO_RY_CONV

def read_volume(calc_dir):
    """Wrapper for volume reading."""
    output = calc_dir / "espresso.pwo"
    final = read(output, format="espresso-out")
    return final.get_volume()

def read_pressure(calc_dir):
    """Wrapper for pressure reading."""
    output = calc_dir / "espresso.pwo"
    final = read(output, format="espresso-out")
    stress = final.get_stress(voigt=False)
    return -np.trace(stress) / 3 * 10 / GPa

def read_time(calc_dir):
    """Return the wall time in seconds from the final PWSCF line."""
    text = (Path(calc_dir) / "espresso.pwo").read_text()
    matches = re.findall(r"PWSCF\s*:\s*.+?\s+CPU\s+(.+?)\s+WALL", text)
    return to_seconds(matches[-1]) if matches else None


def read_vcrelax(calc_dir: str):
    """Return all energies and volumes from a vcrelax calculation"""
    images = read(calc_dir / "espresso.pwo", format="espresso-out", index=":")
    images = images[:-1] # Drop duplicate scf calculation

    volumes   = [a.get_volume() for a in images]               # Ang**3
    energies  = [a.get_potential_energy() / Ry for a in images]  # Ry
    pressures = [-np.trace(a.get_stress(voigt=False)) / 3 / GPa * 10 for a in images]  # kbar
    return volumes, energies, pressures


def space_group(atoms: Atoms, symprec: float = 1e-5) -> str:
    """Return the international space group symbol of an Atoms object."""
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    sym = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if sym is None:
        raise ValueError("spglib could not determine the symmetry (try a larger symprec)")
    return sym.international


def chemical_formula(atoms: Atoms) -> str:
    """Return the reduced chemical formula, e.g. 'Fe7Al'."""
    return atoms.get_chemical_formula(mode="metal")

