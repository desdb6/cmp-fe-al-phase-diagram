"""
File for cluster expansion to predict optimal configurations.

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 19/11/2026
"""

import time, multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import itertools

from ase.visualize import view
from ase import Atoms
from ase.build import bulk
from ase.io import write, read
from ase.units import eV, Ry
from ase.cell import Cell
import spglib
import icet
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from icet.tools.structure_mapping import map_structure_to_reference
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from Des.utils import *
except ModuleNotFoundError:
    from utils import *

try:
    from Des.crystal_builder import *
except ModuleNotFoundError:
    from crystal_builder import *

CSV_PATH = Path("phase_diagram_data.csv")
CIF_DIR = Path("CIF Files")

PREDICTION_PATHS = [
    r"CIF Files\conventional_supercells\2x2x2\2Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\2x2x2\3Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\2x2x2\4Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\3x3x3\2Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\3x3x3\3Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\3x3x3\4Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\3x3x3\5Al\cluster_expansion_results.csv",
    r"CIF Files\conventional_supercells\3x3x3\6Al\cluster_expansion_results.csv",
    r"CIF Files\primitive_supercells\2x2x2\2Al\cluster_expansion_results.csv",
    r"CIF Files\primitive_supercells\3x3x3\2Al\cluster_expansion_results.csv",
    r"CIF Files\primitive_supercells\3x3x3\3Al\cluster_expansion_results.csv",
    r"CIF Files\primitive_supercells\3x3x3\4Al\cluster_expansion_results.csv",
    r"CIF Files\primitive_supercells\3x3x3\5Al\cluster_expansion_results.csv",
    r"CIF Files\primitive_supercells\3x3x3\6Al\cluster_expansion_results.csv"
]

def _map(structure, ref, q, kwargs):
    t = time.time()
    map_structure_to_reference(structure, ref, **kwargs)
    q.put(time.time() - t)

def timed_map(structure, ref, timeout=20, **kwargs):
    q = mp.Queue()
    p = mp.Process(target=_map, args=(structure, ref, q, kwargs))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None  # timed out
    return q.get()


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


def check_equivalent(structure_1, structure_2):
    """Check if two structures are actually equivalent."""
    struct1 = AseAtomsAdaptor.get_structure(structure_1)
    struct2 = AseAtomsAdaptor.get_structure(structure_2)

    matcher = StructureMatcher(attempt_supercell=True)
    return matcher.fit(struct1, struct2)


def rescale_structure(atoms: Atoms):
    """Rescale atoms object so icet does not complain that the volumes do not match."""
    old_volume = atoms.get_volume()
    new_volume = len(atoms) * V_PER_ATOM_FE
    scale = (new_volume / old_volume) ** (1 / 3)
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    return atoms


def align_to_reference(atoms: Atoms, ref: Atoms) -> Atoms:
    """Rotate an n x n x n primitive supercell into the orientation of the primitive reference."""
    n = round(len(atoms) ** (1 / 3))
    if n ** 3 != len(atoms):
        return atoms
    target = ref.cell.array * n
    if not np.allclose(atoms.cell.cellpar(), Cell(target).cellpar(), atol=1e-3):
        return atoms
    new = atoms.copy()
    new.set_cell(target, scale_atoms=True)
    return new


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
    ref = bcc_fe_primitive(V_PER_ATOM_FE)
    df["Structure"] = [
        align_to_reference(rescale_structure(read_cif(p, align=False)), ref)
        for p in df["Cif path"]
    ]

    to_drop = set()
    for _, group in df.groupby("Al fraction"):
        for i_1, i_2 in combinations(group.index, 2):
            if i_1 in to_drop or i_2 in to_drop:
                continue
            if check_equivalent(df.loc[i_1, "Structure"], df.loc[i_2, "Structure"]):
                dE = df.loc[i_1, "Formation energy (Ry)"] - df.loc[i_2, "Formation energy (Ry)"]
                print("\nEquivalent structures found:")
                print(df.loc[i_1, "Cif path"])
                print(df.loc[i_2, "Cif path"])
                print(f"Energy difference: {dE:.2e} Ry")
                print(f"Dropping {df.loc[i_2, 'Cif path']}")
                to_drop.add(i_2)
    df = df.drop(index=list(to_drop))

    df["Al positions"] = [
        get_al_positions(structure) for structure in df["Structure"]
    ]

    df = df.drop(columns=["Al fraction", "Index", "Space group", "Spin polarized", "Additional label", "Date", "Include in plot"])
    df.to_csv(r"Des\cluster_expansion\processed_data_cluster_expansion.csv")
    return df


def cluster_expansion(df: pd.DataFrame):
    """Perform a cluster expansion on preprocessed data."""
    primitive_structure = bcc_fe_primitive(V_PER_ATOM_FE)
    cs = ClusterSpace(structure=primitive_structure,
                  cutoffs=[10.0, 6.0, 5.0],
                  chemical_symbols=['Al', 'Fe'])
    sc = StructureContainer(cluster_space=cs)
    for structure, energy, tag in zip(df["Structure"], df["Formation energy (Ry)"], df["Cif path"]):
        ideal, _ = map_structure_to_reference(structure, cs.primitive_structure)
        print(f"added {Path(tag).name}")
        sc.add_structure(structure=ideal, user_tag=str(tag),
                        properties={"mixing_energy": energy})
    opt = CrossValidationEstimator(
        fit_data=sc.get_fit_data(key='mixing_energy'), fit_method='ardr')
    opt.validate()
    opt.train()
    print(opt)
    print(len(cs), "parameters,", len(df), "structures")
    print("non-zero parameters:", opt.n_nonzero_parameters)
    print("train RMSE:", opt.rmse_train)
    print("CV RMSE:   ", opt.rmse_validation)

    ce = ClusterExpansion(
        cluster_space=cs, parameters=opt.parameters, metadata=opt.summary)
    print(ce)
    ce.write(r'Des\cluster_expansion\fitted_parameters.ce')


def check_prediction(df):
    """Plot predicted energies vs actual energies"""
    ce = ClusterExpansion.read(r'Des\cluster_expansion\fitted_parameters.ce')
    primitive_structure = bcc_fe_primitive(V_PER_ATOM_FE)

    predicted_energy = []
    energies = []
    for structure, energy in tqdm(zip(df["Structure"], df["Formation energy (Ry)"]), desc="Predicting energies..."):
        ideal, _ = map_structure_to_reference(structure, primitive_structure)
        energies.append(energy)
        predicted_energy.append(ce.predict(ideal))
    indices = range(len(energies))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel(r'Index')
    ax.set_ylabel(r'Formation energy (Ry)')
    ax.scatter(indices, energies,
            marker='o', label='reference')
    ax.scatter(indices, predicted_energy,
            marker='x', label='CE prediction')
    ax.grid(alpha=0.5)
    ax.legend()
    plt.savefig(r'Des\cluster_expansion\mixing_energy_comparison.png', bbox_inches='tight', dpi=360)


def fit_energies(cif_dir: str, window: float = 1.5e-3) -> pd.DataFrame:
    """Predict the formation of every cif file in a directory."""
    ce = ClusterExpansion.read(r'Des\cluster_expansion\fitted_parameters.ce')
    ref = bcc_fe_primitive(V_PER_ATOM_FE)

    cif_paths = sorted(Path(cif_dir).rglob("*.cif"))

    rows = []
    for path in tqdm(cif_paths, desc="Predicting energies..."):
        row = {"Cif path": str(path), "File": path.name}
        structure = align_to_reference(
            rescale_structure(read_cif(str(path), align=False)), ref
        )
        ideal, _ = map_structure_to_reference(structure, ref)
        symbols = structure.get_chemical_symbols()
        row["Chemical formula"] = structure.get_chemical_formula()
        row["Atoms"] = len(structure)
        row["Al fraction"] = symbols.count("Al") / len(symbols)
        row["Predicted formation energy (Ry)"] = float(ce.predict(ideal))
        rows.append(row)

    results = pd.DataFrame(rows)

    e_min = np.min(results["Predicted formation energy (Ry)"])
    results["Energy above composition minimum (Ry)"] = (
        results["Predicted formation energy (Ry)"] - e_min
    )
    results["Worth relaxing"] = results["Energy above composition minimum (Ry)"] <= window

    results = results.sort_values("Predicted formation energy (Ry)").reset_index(drop=True)
    results.to_csv(Path(cif_dir) / "cluster_expansion_results.csv", index=False)

    n_flagged = int(results["Worth relaxing"].sum())
    print(f"{n_flagged} of {len(results)} structures within {window:.1e} Ry of their composition minimum")
    return results


def list_all_energies():
    """List all predicted energies and fractions for plotting."""
    fractions = []
    predicted_energies = []
    tags = []
    for ce_result_path in PREDICTION_PATHS:
        df = pd.read_csv(ce_result_path)
        fractions.append(df["Al fraction"] * 100)
        predicted_energies.append(df["Predicted formation energy (Ry)"])
        tags.append(df["File"])
    fractions = list(itertools.chain.from_iterable(fractions))
    predicted_energies = list(itertools.chain.from_iterable(predicted_energies))
    tags = list(itertools.chain.from_iterable(tags))
    return fractions, predicted_energies, tags



def plot_predicted_energies():
    """Make a plot of all predicted energies with convex hull."""
    fractions, predicted_energies, tags = list_all_energies()
    fig, ax = plt.subplots(figsize=(16, 10))

    # Spin polarized data
    data = ax.scatter(
        x=fractions,
        y=predicted_energies,
        s=20, color="#0084FF", zorder=2, clip_on=False
    )

        # Hover tooltip
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, -45), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->"), zorder=10,
    )
    annot.set_visible(False)

    def on_hover(event):
        visible = annot.get_visible()
        if event.inaxes == ax:
            hit, info = data.contains(event)
            if hit:
                i = info["ind"][0]
                annot.xy = (fractions[i], predicted_energies[i])
                annot.set_text(
                    f"{tags[i]}\n"
                    f"{fractions[i]:.1f}% Al\n"
                    f"{predicted_energies[i]:.5f} Ry"
                )
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        if visible:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # Convex hull (pure elements have E_form = 0 by definition)
    ref_x = [0.0, 100.0]
    ref_y = [0.0, 0.0]
    hull_x = np.concatenate([fractions, ref_x])
    hull_y = np.concatenate([predicted_energies, ref_y])

    # x is in %, y is in Ry: rescale x to a fraction so Qhull's tolerances behave
    points = np.column_stack([hull_x / 100, hull_y])
    ch = ConvexHull(points)

    # Keep only the lower facets: outward normal has a negative y-component
    lower_simplices = ch.simplices[ch.equations[:, 1] < -1e-12]
    hull_idx = np.unique(lower_simplices.ravel())
    hull = np.column_stack([hull_x[hull_idx], hull_y[hull_idx]])
    hull = hull[np.argsort(hull[:, 0])]

    ax.scatter(hull[:, 0], hull[:, 1], s=140, facecolors="none",
            edgecolors="black", lw=1.2, zorder=4, clip_on=False)

    hull_line, = ax.plot(
        hull[:, 0], hull[:, 1],
        color="black", lw=1.2, ls="--", zorder=1,
    )

    # Set limits
    ax.set_xlim(0, 25)
    ax.margins(y=0.5)
    ax.set_ylim(top=0)

    # Move x axis ticks to top
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(left=True, labelleft=True, right=False, labelright=False)

    # Labels
    ax.set_xlabel("Predicted energies % Al", fontsize=16)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$E_{\mathrm{form}}$ (Ry/atom)", fontsize=16)
    ax.grid(alpha=0.6)

    plt.savefig("Predicted_energies.png", dpi=360)
    fig.savefig("Predicted_energies.pdf", bbox_inches="tight")   # vector, good for the report

    plt.show()


def main():
    df = preprocess()
    cluster_expansion(df)
    check_prediction(df)
    

if __name__ == "__main__":
    main()
    plot_predicted_energies()