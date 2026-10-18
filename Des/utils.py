"""
Utils file for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 17/10/2026
"""

import os
import numpy as np

from ase.build import bulk
from ase.io import write
from ase.visualize import view


def read_total_magnetization(directory):
    """
    ASE is not able to read the magnetic momentum using the native function.
    This function reads the last 'total magnetization' line in espresso.pwo in the given directory, which is the converged value (μB/cell).
    """
    val = float("nan")
    with open(os.path.join(directory, "espresso.pwo")) as f:
        for line in f:
            if "total magnetization" in line and "Bohr mag/cell" in line: # If a new value is found, overwrite. The final value is at the bottom.
                val = float(line.split("=")[1].split()[0])
    return val


def atoms_fe7al(lattice_parameter: float) -> ase.Atoms:
    """Generate an Atoms object for an Fe7Al structure"""
    atoms = bulk('Fe', 'bcc', a=lattice_parameter) * (2, 2, 2)
    atoms[0].symbol = 'Al'
    return atoms


def write_cif(atoms: ase.Atoms, path: str, filename: str) -> None:
    """Generate a cif file from an Atoms object"""
    write(path + '/' + filename + '.cif', atoms)


def vis_atoms(atoms: ase.Atoms) -> None:
    """Visualize Atoms object"""
    view(atoms)


if __name__ == "__main__":
    atoms = atoms_fe7al(2.83)
    vis_atoms(atoms)