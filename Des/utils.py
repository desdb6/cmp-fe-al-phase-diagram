"""
Utils file for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 04/11/2026
"""

import shutil
import re
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ase.units import Ry, GPa

# Unit conventions
UNIT_DICT = {"energy": "Ry", "pressure": "kbar", "magnetization": "muB/cell"}

# Conversion constants
EV_TO_RY_CONV = 1 / Ry
RY_PER_ANG3_TO_KBAR_CONV = 10 * Ry / GPa

# Experimental bcc and fcc volumes
V_PER_ATOM_FE = 11.65
V_PER_ATOM_AL = 16.6

# Formation energy paths
FORMATION_E_PATH_FE = "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/bcc_fe/formation_energy/vc-relax/results.csv"
FORMATION_E_PATH_AL = "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/fcc_al/formation_energy/vc-relax/results.csv"

def to_seconds(t: str) -> float:
    """Convert QE time strings like '1h 2m 3.45s' or '12.3s' to seconds."""
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(float(v) * units[u] for v, u in re.findall(r"([\d.]+)\s*([dhms])", t))

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


def find_convergence(x, y, tol):
    """
    Return the first x from which y stays within tol of the final value.
    Returns None if only the last point satisfies this (not converged).
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    ok = np.abs(y - y[-1]) <= tol
    bad = np.where(~ok)[0]                  # points outside tolerance
    idx = bad[-1] + 1 if bad.size else 0    # first point after the last bad one

    if idx >= len(y) - 1:                   # only the last point (trivially) passes
        return None
    return x[idx]

def get_formation_energies():
    """
    Simple function to get last calculated formation energies in Ry from results file.
    """
    fe_df = pd.read_csv(FORMATION_E_PATH_FE)
    al_df = pd.read_csv(FORMATION_E_PATH_AL)

    return {"Fe": float(fe_df["energy"].iloc[-1]),
            "Al": float(al_df["energy"].iloc[-1])}

def get_equilibrium_volumes():
    """
    Simple function to get last calculated formation energies in Ry from results file.
    """
    fe_df = pd.read_csv(FORMATION_E_PATH_FE)
    al_df = pd.read_csv(FORMATION_E_PATH_AL)

    return {"Fe": float(fe_df["volume"].iloc[-1]),
            "Al": float(al_df["volume"].iloc[-1])}

def save_nested_dict_csv(results, output_path, index_label="x"):
    """Save a nested results dict to a csv file. (Claude) """
    first = next(iter(results.values()))
    columns = [k for k in first if k != "atoms"] # Skip the atoms column

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([index_label] + columns)
        for x, res in results.items():
            writer.writerow([x] + [res[k] for k in columns])

if __name__ == "__main__":
    print(get_formation_energies())