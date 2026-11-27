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
from pathlib import Path
from tqdm import tqdm
import re
import itertools
from itertools import combinations

from ase import Atoms
from ase.cell import Cell
from ase.units import Ry
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from icet.tools.structure_mapping import map_structure_to_reference
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

RY_TO_MEV_CONV = Ry * 1000

RED = "#cc0101"
YELLOW = "#ffec46"
BLUE = "#0084FF"
GRAY = "#4D4D4D"

try:
    from Des.utils import *
except ModuleNotFoundError:
    from utils import *

try:
    from Des.crystal_builder import *
except ModuleNotFoundError:
    from crystal_builder import *

ROOT = Path(__file__).resolve().parent.parent   # project root (this file lives in Des/)
CSV_PATH = ROOT / "phase_diagram_data.csv"
CIF_DIR = ROOT / "CIF Files"
CE_DIR = ROOT / "Des" / "cluster_expansion"
CE_DIR.mkdir(parents=True, exist_ok=True)

# (cell type, supercell size, list of Al counts)
_PREDICTION_SETS = [
    ("conventional_supercells", "2x2x2", [2, 3, 4]),
    ("conventional_supercells", "3x3x3", [2, 3, 4, 5, 6, 7]),
    ("primitive_supercells",    "2x2x2", [2]),
    ("primitive_supercells",    "3x3x3", [2, 3, 4, 5, 6]),
]
PREDICTION_PATHS = [
    CIF_DIR / cell / size / f"{n}Al" / "cluster_expansion_results.csv"
    for cell, size, counts in _PREDICTION_SETS
    for n in counts
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
    matches = [
        p for p in root.rglob("*.cif")
        if pattern.match(p.name)
        and "hull_structures" not in p.relative_to(root).parts
    ]

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
    return df


def cluster_expansion(df: pd.DataFrame):
    """Perform a cluster expansion on preprocessed data."""
    primitive_structure = bcc_fe_primitive(V_PER_ATOM_FE)
    cs = ClusterSpace(structure=primitive_structure,
                  cutoffs=[6.5, 5.5, 5.0],
                  chemical_symbols=['Al', 'Fe'])
    sc = StructureContainer(cluster_space=cs)
    for structure, energy, tag in zip(df["Structure"], df["Formation energy (Ry)"], df["Cif path"]):
        ideal, _ = map_structure_to_reference(structure, cs.primitive_structure)
        print(f"added {Path(tag).name}")
        sc.add_structure(structure=ideal, user_tag=str(tag),
                        properties={"mixing_energy": energy})
    opt = CrossValidationEstimator(
        fit_data=sc.get_fit_data(key='mixing_energy'), fit_method='ardr', threshold_lambda=1e4
        )
    opt.validate()
    opt.train()
    print(opt)
    print(len(cs), "parameters,", len(df), "structures")
    print("non-zero parameters:", opt.n_nonzero_parameters)
    print("train RMSE:", opt.rmse_train)
    print("CV RMSE:   ", opt.rmse_validation)

    A, y = sc.get_fit_data(key="mixing_energy")
    print("fit matrix:", A.shape, " rank:", np.linalg.matrix_rank(A, tol=1e-8))

    # Group identical columns (orbits the training set cannot tell apart)
    _, first, counts = np.unique(np.round(A, 8), axis=1, return_index=True, return_counts=True)
    print("unique columns:", len(first), "of", A.shape[1])

    ce = ClusterExpansion(
        cluster_space=cs, parameters=opt.parameters, metadata=opt.summary)
    print(ce)
    ce.write(r'Des/cluster_expansion/fitted_parameters.ce')


def check_prediction(df):
    """Plot predicted energies vs actual energies"""
    ce = ClusterExpansion.read(r'Des/cluster_expansion/fitted_parameters.ce')
    primitive_structure = bcc_fe_primitive(V_PER_ATOM_FE)

    predicted_energy = []
    energies = []
    for structure, energy in tqdm(zip(df["Structure"], df["Formation energy (Ry)"]), desc="Predicting energies..."):
        ideal, _ = map_structure_to_reference(structure, primitive_structure)
        energies.append(energy)
        predicted_energy.append(float(ce.predict(ideal)))

    labels = [Path(p).stem for p in df["Cif path"]]   # filename without ".cif"
    indices = np.arange(len(energies))

    # Widen the figure with the number of structures so labels don't overlap
    fig, ax = plt.subplots(figsize=(max(5, 0.25 * len(indices)), 4))
    ax.set_ylabel(r'Formation energy (Ry)')
    ax.scatter(indices, energies, marker='o', label='reference')
    ax.scatter(indices, predicted_energy, marker='x', label='CE prediction')

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_xlim(-1, len(indices))

    ax.grid(alpha=0.5)
    ax.legend()
    plt.savefig(r'Des/cluster_expansion/mixing_energy_comparison.png', bbox_inches='tight', dpi=360)


def fit_energies(cif_dir: str, window: float = 6e-4) -> pd.DataFrame:
    """Predict the formation of every cif file in a directory."""
    ce = ClusterExpansion.read(r'Des/cluster_expansion/fitted_parameters.ce')
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


def repredict_all_energies(window: float = 6e-4, refit: bool = False) -> pd.DataFrame:
    """Re-predict the formation energies of all cif files in every prediction directory.

    Args:
        window: Energy window (Ry) above the composition minimum for 'Worth relaxing'.
        refit: If True, preprocess the data and refit the cluster expansion first,
               so the predictions use freshly fitted parameters.

    Returns:
        A dataframe with the combined results of all directories.
    """
    if refit:
        df = preprocess()
        cluster_expansion(df)

    all_results = []
    failed = []
    for results_path in PREDICTION_PATHS:
        cif_dir = results_path.parent
        if not cif_dir.exists():
            print(f"Skipping {cif_dir}: directory does not exist")
            failed.append((cif_dir, "missing directory"))
            continue

        print(f"\n=== {cif_dir.relative_to(CIF_DIR)} ===")
        try:
            all_results.append(fit_energies(str(cif_dir), window=window))
        except Exception as e:
            print(f"Failed for {cif_dir}: {e}")
            failed.append((cif_dir, str(e)))

    if failed:
        print("\nThe following directories failed:")
        for path, reason in failed:
            print(f"  {path}: {reason}")

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nRepredicted {len(combined)} structures across {len(all_results)} directories")
    return combined

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

def plot_eci_results():
    """Plot the effective cluster interactions against cluster radius."""
    ce = ClusterExpansion.read(r'Des/cluster_expansion/fitted_parameters.ce')
    d = ce.to_dataframe()

    # Order 0 is the constant term, order 1 the singlet; plot only pairs and triplets
    d = d[d["order"] >= 2].copy()
    d["eci_mev"] = d["eci"] * RY_TO_MEV_CONV
    d["weighted_mev"] = d["eci_mev"] * d["multiplicity"]

    nonzero = d[d["eci"] != 0]
    print(f"{len(nonzero)} of {len(d)} pair/triplet orbits have non-zero ECIs")
    print(nonzero[["order", "radius", "multiplicity", "eci_mev"]]
          .sort_values("radius").to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    for order, color, label, zorder in [(2, RED, "Pairs", 4), (3, BLUE, "Triplets", 3), (4, YELLOW, "Quadruplets", 2)]:
        sub = d[d["order"] == order]
        ax.scatter(sub["radius"], sub["eci_mev"], color=color, label=label, zorder=zorder, alpha=0.7)
    ax.axhline(0, color=GRAY, lw=2, linestyle="--", zorder=1)
    ax.set_xlabel("Cluster radius (Å)")
    ax.set_ylabel("ECI (meV)")
    ax.grid(alpha=0.5)
    ax.legend()

    plt.savefig(r'Des/cluster_expansion/eci_vs_radius.png', bbox_inches="tight", dpi=360)
    plt.show()

def cutoff_sweep(cutoff_sets=None, seeds=(0, 1, 2)):
    """Compare train/CV error and the 1NN pair ECI for different cutoffs."""
    if cutoff_sets is None:
        cutoff_sets = ([6.0, 5.0, 4.0], [7.0, 5.5, 4.5], [8.0, 6.0, 5.0],
                       [10.0, 7.0, 5.0])

    df = preprocess()
    primitive_structure = bcc_fe_primitive(V_PER_ATOM_FE)

    # The mapping does not depend on the cutoffs, so do it once
    ideals = [map_structure_to_reference(s, primitive_structure)[0]
              for s in df["Structure"]]

    def get_1nn_eci(ce):
        """Return the 1NN pair ECI in meV."""
        d = ce.to_dataframe()
        pairs = d[d["order"] == 2].sort_values("radius")
        return pairs.iloc[0]["eci"] * RY_TO_MEV_CONV

    def test_cutoff(cutoffs):
        cs = ClusterSpace(structure=primitive_structure,
                          cutoffs=cutoffs,
                          chemical_symbols=['Al', 'Fe'])
        sc = StructureContainer(cluster_space=cs)
        for ideal, energy, tag in zip(ideals, df["Formation energy (Ry)"], df["Cif path"]):
            sc.add_structure(structure=ideal, user_tag=str(tag),
                             properties={"mixing_energy": energy})

        cv_rmses = []
        for seed in seeds:
            opt = CrossValidationEstimator(
                fit_data=sc.get_fit_data(key='mixing_energy'),
                fit_method='ardr', seed=seed)
            opt.validate()
            cv_rmses.append(opt.rmse_validation * RY_TO_MEV_CONV)

        opt.train()  # final fit on all data (last seed) for the ECIs
        ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters,
                              metadata=opt.summary)
        return {
            "cutoffs": str(cutoffs),
            "n_params": len(cs),
            "n_nonzero": opt.n_nonzero_parameters,
            "train RMSE (meV)": opt.rmse_train * RY_TO_MEV_CONV,
            "CV RMSE mean (meV)": np.mean(cv_rmses),
            "CV RMSE std (meV)": np.std(cv_rmses),
            "1NN ECI (meV)": get_1nn_eci(ce),
        }

    results = pd.DataFrame([test_cutoff(c) for c in cutoff_sets])
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return results

def cluster_pruning_sweep(lambda_sets=None, seeds=(0, 1, 2)):
    """Compare train/CV error and the 1NN pair ECI for different cutoffs."""
    if lambda_sets is None:
        lambda_sets = [10 ** n for n in [3, 4, 5, 6, 7]]

    df = preprocess()
    primitive_structure = bcc_fe_primitive(V_PER_ATOM_FE)

    # The mapping does not depend on the cutoffs, so do it once
    ideals = [map_structure_to_reference(s, primitive_structure)[0]
              for s in df["Structure"]]

    def get_1nn_eci(ce):
        """Return the 1NN pair ECI in meV."""
        d = ce.to_dataframe()
        pairs = d[d["order"] == 2].sort_values("radius")
        return pairs.iloc[0]["eci"] * RY_TO_MEV_CONV

    def test_lambda(l):
        cs = ClusterSpace(structure=primitive_structure,
                          cutoffs=[8.0, 5.5, 5.0],
                          chemical_symbols=['Al', 'Fe'])
        sc = StructureContainer(cluster_space=cs)
        for ideal, energy, tag in zip(ideals, df["Formation energy (Ry)"], df["Cif path"]):
            sc.add_structure(structure=ideal, user_tag=str(tag),
                             properties={"mixing_energy": energy})

        cv_rmses = []
        for seed in seeds:
            opt = CrossValidationEstimator(
                fit_data=sc.get_fit_data(key='mixing_energy'),
                fit_method='ardr', seed=seed, threshold_lambda=l)
            opt.validate()
            cv_rmses.append(opt.rmse_validation * RY_TO_MEV_CONV)

        opt.train()  # final fit on all data (last seed) for the ECIs
        ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters,
                              metadata=opt.summary)
        return {
            "lambda": str(l),
            "n_params": len(cs),
            "n_nonzero": opt.n_nonzero_parameters,
            "train RMSE (meV)": opt.rmse_train * RY_TO_MEV_CONV,
            "CV RMSE mean (meV)": np.mean(cv_rmses),
            "CV RMSE std (meV)": np.std(cv_rmses),
            "1NN ECI (meV)": get_1nn_eci(ce),
        }

    results = pd.DataFrame([test_lambda(l) for l in lambda_sets])
    print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return results

def main():
    df = preprocess()
    cluster_expansion(df)
    repredict_all_energies()
    

if __name__ == "__main__":
    main()
    plot_eci_results()
