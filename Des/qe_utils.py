"""
Utils file for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 21/10/2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from dataclasses import dataclass, replace

from ase.build import bulk
from ase.io import write
from ase.visualize import view
from ase.eos import EquationOfState
from ase.units import Ry, GPa

PSEUDO_DIR = "/home/des/cmp-fe-al-phase-diagram/Pseudopotentials"
PSEUDOS = {"Fe": "Fe.paw.pbe.z_16.ld1.psl.v0.2.1",
           "Al": "Al.us.pbe.z_3.ld1.psl.v1.0.0-low"}

EV_TO_RY_CONV = 1 / Ry
RY_PER_ANG3_TO_KBAR_CONV = 10 * Ry / GPa


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


def atoms_bcc_Fe(lattice_parameter: float, magmom: float = 0.5) -> ase.Atoms:
    """Generate an Atoms object for a Fe bcc structure"""
    atoms = bulk('Fe', 'bcc', a=lattice_parameter, cubic=True)
    atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    return atoms


def atoms_d03_fe3al(lattice_parameter: float, magmom: float = 0.5) -> ase.Atoms:
    """Generate an Atoms object for an D03-Fe3Al structure"""
    atoms = bulk('Fe', 'bcc', a=lattice_parameter, cubic=True) * (2, 2, 2)
    atom_positions = list(atoms.get_positions() / lattice_parameter)
    Al_positions = [
        [0.5, 0.5, 0.5],
        [1.5, 0.5, 1.5],
        [1.5, 1.5, 0.5],
        [0.5, 1.5, 1.5],
    ]
    for pos in Al_positions:
        index = np.where(np.all(np.isclose(atom_positions, pos, atol=1e-6), axis=1))[0][0]
        atoms[index].symbol = "Al"
    return atoms


def write_cif(atoms: ase.Atoms, path: str, filename: str) -> None:
    """Generate a cif file from an Atoms object"""
    write(path + '/' + filename + '.cif', atoms)


def vis_atoms(atoms: ase.Atoms) -> None:
    """Visualize Atoms object"""
    view(atoms)


def remove_dir(path, *, allowed_root, dry_run=True, min_depth=3):
    """Delete a directory tree only if it passes sanity checks.

    path         : directory to delete
    allowed_root : the target must be strictly inside this directory
    dry_run      : if True (default), only print what would be deleted
    min_depth    : minimum number of path components (blocks '/', '/home', ...)
    """
    target = Path(path).expanduser()

    # Never follow a symlink: resolving it would point at a different tree
    if target.is_symlink():
        raise ValueError(f"Refusing to delete symlink: {target}")

    target = target.resolve()
    root = Path(allowed_root).expanduser().resolve()

    if not target.exists():
        raise FileNotFoundError(f"No such path: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")
    if len(target.parts) < min_depth:
        raise ValueError(f"Path too shallow: {target}")
    if target in (Path.home().resolve(), Path.cwd().resolve(), root):
        raise ValueError(f"Refusing to delete protected path: {target}")
    if root not in target.parents:
        raise ValueError(f"{target} is not inside allowed root {root}")

    n_files = sum(1 for p in target.rglob("*") if p.is_file())
    if dry_run:
        print(f"[dry run] would delete {target} ({n_files} files)")
        return False

    shutil.rmtree(target)
    return True


@dataclass(frozen=True)
class QESettings:
    """General settings for Quantum Espresso calculations."""
    ecutwfc: float                 # Ry
    ecutrho: float                 # Ry
    kmesh: tuple[int, int, int]

    def to_input_data(self) -> dict:
        return {
            "system": {"ecutwfc": self.ecutwfc, "ecutrho": self.ecutrho},
        }


class EoSfit:
    """Fit an equation of state to volume (A^3) and energy (Ry) data."""

    def __init__(
        self,
        csv_path,
        energy_label="energy (Ry)",
        volume_label="volume (Ang**3)",
        **read_kwargs,
    ):
        self.csv_path = Path(csv_path)
        raw = pd.read_csv(self.csv_path, **read_kwargs)  # e.g. sep="\t"
        df = (raw[[volume_label, energy_label]]
              .dropna()
              .drop_duplicates(subset=volume_label)
              .sort_values(volume_label))
        self.V = df[volume_label].to_numpy()   # A^3
        self.E = df[energy_label].to_numpy()   # Ry
        self.fit()
        self.write_results()

    def fit(self, eos=("birchmurnaghan", "murnaghan", "vinet")):
        """Return a DataFrame with V0, E0, B0 for each EOS form."""
        rows = []
        for name in eos:
            # Inputs in A^3 and Ry -> outputs in A^3, Ry, Ry/A^3
            v0, e0, B = EquationOfState(self.V, self.E, eos=name).fit()
            rows.append({
                "eos": name,
                "V0 (A^3)": v0,
                "E0 (Ry)": e0,
                "B0 (kbar)": B * RY_PER_ANG3_TO_KBAR_CONV,
            })
        self.results = pd.DataFrame(rows)
        return self.results

    def write_results(self):
        """Write a csv file with the results of the fit."""
        output_path = self.csv_path.parent / "eos_fit_results.csv"
        self.results.to_csv(output_path, index=False)
        return output_path
        
    def plot(self, filename=None, eos="birchmurnaghan"):
        """Plot data and fit with correct axis labels (A^3, Ry)."""
        e = EquationOfState(self.V, self.E, eos=eos)
        e.fit()
        _, e0, v0, B, x, y, v, en = e.getplotdata()  # B is in Ry/A^3 here

        fig, ax = plt.subplots()
        ax.plot(v, en, "o", label="data")
        ax.plot(x, y, "-", label=eos)
        ax.set_xlabel(r"Volume (Å$^3$)")
        ax.set_ylabel("Energy (Ry)")
        ax.legend()
        ax.grid(alpha=0.5)
        ax.set_title(
            rf"$V_0$ = {v0:.2f} Å$^3$,  $E_0$ = {e0:.2f} Ry,  "
            rf"$B_0$ = {B * RY_PER_ANG3_TO_KBAR_CONV:.2f} kbar",
            fontsize=8,
        )
        if filename:
            fig.savefig(filename, dpi=200, bbox_inches="tight")
        return fig, ax


if __name__ == "__main__":
    atoms = atoms_d03_fe3al(10)
    vis_atoms(atoms)