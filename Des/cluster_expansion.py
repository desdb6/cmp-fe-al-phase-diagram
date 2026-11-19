"""
File for cluster expansion to predict optimal configurations.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 19/11/2026
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

from ase.visualize import view
from ase import Atoms
from ase.build import bulk
from ase.io import write, read
from ase.units import eV, Ry
import spglib

from utils import *
from crystal_builder import *

CSV_PATH = Path("phase_diagram_data.csv")
CIF_DIR = Path("CIF Files")

def find_cif(index: int, chemical_formula: str, root: Path = CIF_DIR) -> Path:
    """Find the cif file for a given chemical formula and index. (Claude)

    Files are named like Fe23Al4_17_C2_m_m324.cif, where the index may be
    zero-padded (17, 017, 005) depending on the number of saved cif files.
    """
    pattern = re.compile(
        rf"^{re.escape(chemical_formula)}_0*{index}_.*\.cif$"
    )
    matches = [p for p in root.rglob("*.cif") if pattern.match(p.name)]

    if not matches:
        raise FileNotFoundError(
            f"No cif found for {chemical_formula} with index {index} in {root}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple cifs found for {chemical_formula} with index {index}:\n"
            + "\n".join(str(m) for m in matches)
        )
    return matches[0]


def get_al_positions(structure: Atoms, scaled: bool = True) -> np.ndarray:
    """Return the positions of all Al atoms in the structure."""
    is_al = np.array(structure.get_chemical_symbols()) == "Al"
    if not is_al.any():
        raise ValueError("No Al atom found in structure")

    positions = structure.get_scaled_positions() if scaled else structure.get_positions()
    return positions[is_al]


def preprocess():
    """
    Prepare data for cluster expansion.

    Input: chemical formula
    Output: Dataframe containing atoms object, converged energy, and al positions.
    """
    full_df = pd.read_csv(CSV_PATH)
    df = full_df[full_df["Include in cluster expansion"]].copy()
    

    df["Cif path"] = [
        find_cif(int(idx), formula)
        for idx, formula in zip(df["Index"], df["Chemical formula"])
    ]
    df["Structure"] = [
        read_cif(structure_cif_path) for structure_cif_path in df["Cif path"]
    ]
    df["Al positions"] = [
        get_al_positions(structure) for structure in df["Structure"]
    ]

    df = df.drop(columns=["Al fraction", "Index", "Space group", "Spin polarized", "Additional label", "Date", "Include in plot"])
    df.to_csv("processed_data_cluster_expansion.csv")

if __name__ == "__main__":
    print(preprocess())