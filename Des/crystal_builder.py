"""
Crystal structure generation for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 04/11/2026
"""

import os
import numpy as np

from ase.visualize import view
from ase import Atoms
from ase.build import bulk
from ase.io import write, read
import spglib
from itertools import combinations

from utils import *

def write_cif(atoms: Atoms, path: str) -> None:
    """Generate a cif file from an Atoms object."""
    write(path, atoms)


def read_cif(path: str, magmom: float = 0.2) -> Atoms:
    """Generate an Atoms object from a cif file."""
    atoms = read(path)
    atoms.set_initial_magnetic_moments(
            [magmom if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()]
        )
    return atoms


def vis_atoms(atoms: Atoms) -> None:
    """Visualize Atoms object."""
    view(atoms)


def set_volume(atoms, target_volume):
    """Return a copy of atoms with the cell isotropically scaled to target_volume (A^3). (Claude)"""
    new = atoms.copy()
    factor = (target_volume / new.get_volume()) ** (1 / 3)
    new.set_cell(new.cell.array * factor, scale_atoms=True)
    return new


def fe7al(lattice_parameter: float) -> Atoms:
    """Generate an Atoms object for an Fe7Al structure"""
    atoms = bulk('Fe', 'bcc', a=lattice_parameter) * (2, 2, 2)
    atoms[0].symbol = 'Al'
    return atoms


def bcc_fe_conventional(vol: float, magmom: float = 0.2) -> Atoms:
    """Generate a conventional BCC iron atoms object"""
    lattice_parameter = vol ** (1/3)
    atoms = bulk('Fe', 'bcc', a=lattice_parameter, cubic=True)
    atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    return atoms

def bcc_fe_primitive(vol: float, magmom: float = 0.2) -> Atoms:
    """Generate a primitive BCC iron atoms object"""
    lattice_parameter = (2 * vol) ** (1/3)
    atoms = bulk('Fe', 'bcc', a=lattice_parameter)
    atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    return atoms

def fcc_al_conventional(vol: float) -> Atoms:
    """Generate a conventional FCC alimunium atoms object."""
    lattice_parameter = vol ** (1/3)
    atoms = bulk('Al', 'fcc', a=lattice_parameter, cubic=True)
    return atoms

def fcc_al_primitive(vol: float) -> Atoms:
    """Generate a primitive FCC alimunium atoms object."""
    lattice_parameter = (4 * vol) ** (1/3)
    atoms = bulk('Al', 'fcc', a=lattice_parameter)
    return atoms

def d03_fe3al(lattice_parameter: float, magmom: float = 0.2) -> Atoms:
    """
    Generate an Atoms object for an D03-Fe3Al structure
    WARNING: the lattice parameter input is the length of 1 bcc subsell, not the entire supercell
    """
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
    atoms.set_initial_magnetic_moments(
        [magmom if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()]
    )
    return atoms

def supercell_primitive_one_al(n: int, vol: float, magmom: float = 0.2) -> Atoms:
    """
    Generate an nxnxn supercell of primitive bcc Fe unit cells,
    where one atom has been changed for Al.
    """
    lattice_parameter_single_cell = (2 * vol) ** (1/3) / n # Volume times two because this cell is not cubic, check the volume formula
    atoms = bulk("Fe", crystalstructure="bcc", a=lattice_parameter_single_cell) * (n, n, n)
    atoms[0].symbol = "Al"
    atoms.set_initial_magnetic_moments(
        [magmom if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()]
    )
    return atoms

def supercell_conventional_one_al(n: int, vol: float, magmom: float = 0.2) -> Atoms:
    """
    Generate an nxnxn supercell of conventional bcc Fe unit cells,
    where one atom has been changed for Al.
    """
    lattice_parameter_single_cell = (vol) ** (1/3) / n
    atoms = bulk("Fe", crystalstructure="bcc", cubic=True, a=lattice_parameter_single_cell) * (n, n, n)
    atoms[0].symbol = "Al"
    atoms.set_initial_magnetic_moments(
        [magmom if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()]
    )
    return atoms

def gen_sym_ineq_supercells(n_cells: int, n_subs: int, vol_per_atom: float = None, magmom: float = 0.2, cell_type: str = "conventional") -> dict:
    """
    Generate a dictionary of all symmetry-inequivalent configurations of a n_cells x n_cells x n_cells supercell with n_subs substitutions.

    Algorithm:
    1. Get all symmetry operations of the supercell (spglib).
    2. Convert each operation into a permutation of atom indices.
    3. For every pair (i, j) of candidate atoms, compute the pair’s image under every permutation and take the smallest as a canonical label. Pairs with the same label are equivalent.
    4. Build one structure per unique label, make the substitution, and run spglib to get its space group.
    """
    if cell_type == "conventional":
        n_atoms = 2 * n_cells ** 3
    elif cell_type == "primitive":
        n_atoms = n_cells ** 3
    else:
        raise ValueError(f"cell_type must be 'conventional' or 'primitive', got {cell_type!r}")

    if vol_per_atom is None:
        vol_per_atom = (n_atoms * V_PER_ATOM_FE  + n_subs * (V_PER_ATOM_AL - V_PER_ATOM_FE)) / n_atoms

    formula = f"Fe{n_atoms-n_subs}Al{n_subs}"

    print("-" * 50)
    print("Starting inequivalent configuration calculation")
    print(f"{n_cells}x{n_cells}x{n_cells} {cell_type} supercell, {n_subs} substitutions")
    print(f"Formula: {formula}")
    print("-" * 50)

    # Stage 0: make the supercell without substitutions
    if cell_type == "conventional":
        lattice_parameter_single_cell = (2 * vol_per_atom) ** (1/3) # Times two because the conventional cell contains two atoms
        atoms = bulk("Fe", crystalstructure="bcc", cubic=True, a=lattice_parameter_single_cell) * (n_cells, n_cells, n_cells)
        print(f"Conventional supercell: {n_atoms} atoms, a = {lattice_parameter_single_cell:.4f} Å")
    elif cell_type == "primitive":
        lattice_parameter_single_cell = (2 * vol_per_atom) ** (1/3) # Times two because the primitive cell is not cubic
        atoms = bulk("Fe", crystalstructure="bcc", a=lattice_parameter_single_cell) * (n_cells, n_cells, n_cells)
        print(f"Primitive supercell: {n_atoms} atoms, a = {lattice_parameter_single_cell:.4f} Å")

    # Stage 1
    symprec = 1e-4
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    rots, trans = sym["rotations"], sym["translations"]
    print("Number of symmetry operations:", len(rots))

    # Stage 2
    def wrap(x):
        return np.round(x, 8) % 1.0

    def calculate_permutation(rot, trans, positions, tol=1e-3):
        """Calculate atom permutation from array of position vectors and symmetry operation."""
        new_positions = wrap(positions @ rot.T + trans)
        d = new_positions[:, None, :] - positions[None, :, :]   # (N, N, 3), compare the distances in x, y, z between all pairs of non-permuted and permuted configuration
        d -= np.round(d)                                        # periodic minimum image
        dist2 = (d ** 2).sum(-1)                                # (N, N), distances between pairs
        permutation = dist2.argmin(axis=1)                      # nearest original atom for each new one, (in each column there is a 0)
        assert np.all(dist2[np.arange(len(positions)), permutation] < tol**2), \
            "operation did not map atoms onto atoms"
        return permutation

    positions = wrap(atoms.get_scaled_positions())
    perms = np.array([calculate_permutation(R, t, positions) for R, t in zip(rots, trans)])

    print(f"Generated symmetry-equivalent permutations:")
    print(perms[0:5])
    print("...")
    print(f"Number of permutations: {len(perms)}")

    # Stage 3
    def encode_base_n_atoms(sub_indices_array):
        """
        Encode sorted atom-index subsets/lists as single integers (counting in base n_atoms).
        Accepts shape (k,) or (m, k); returns a scalar or an array of m keys.
        Needed for vectorisation of visit check.
        """
        arr = np.asarray(sub_indices_array, dtype=np.int64)
        key = np.zeros(arr.shape[:-1], dtype=np.int64)
        for i in range(arr.shape[-1]):
            key = key * n_atoms + arr[..., i]      # first element = most significant digit
        return key

    substitution_indices = np.array(list(combinations(range(n_atoms), n_subs)), dtype=np.int64) # Make a list of all possible substitutions
    all_keys = encode_base_n_atoms(substitution_indices).tolist()   # Encode list as single integers, so visited check can be done

    visited = set()
    inequivalent = []

    for sub_idx, sub_key in zip(substitution_indices, all_keys):
        if sub_key in visited: # check if configuration is already in an earlier orbit
            continue                                       

        # orbit of this subset under every symmetry operation
        orbit_keys = np.unique(encode_base_n_atoms(np.sort(perms[:, sub_idx], axis=1)))
        visited.update(orbit_keys.tolist())
        inequivalent.append((sub_idx, len(orbit_keys)))

    assert sum(m for _, m in inequivalent) == len(substitution_indices)
    print(f"{len(substitution_indices)} subsets -> {len(inequivalent)} inequivalent configurations")

    # Stage 4
    struct_dict = {}
    for n, (al_pos, mult) in enumerate(inequivalent):
        ineq_atoms = atoms.copy()
        ineq_atoms.symbols[al_pos] = "Al"
        ineq_atoms.set_initial_magnetic_moments(
                [magmom if s == "Fe" else 0.0 for s in ineq_atoms.get_chemical_symbols()]
            )
        cell = (ineq_atoms.cell[:], ineq_atoms.get_scaled_positions(), ineq_atoms.numbers)
        sym = spglib.get_symmetry_dataset(cell, symprec=symprec)
        struct_dict[n] = {
            "atoms": ineq_atoms,
            "spacegroup": sym.international.replace("/", "_"),
            "al_idx": list(map(int, al_pos)),
            "number": sym.number,
            "multiplicity": mult,
            "formula": formula,
            "Al ratio": n_subs / n_atoms * 100
            }

    # Print results
    print(f"{'#':>4}  {'space group':<12} {'no.':>4}  {'mult.':>6}  Al indices")
    print("-" * 50)
    for n, d in sorted(struct_dict.items(), key=lambda kv: -kv[1]["multiplicity"]):
        print(f"{n:>4}  {d['spacegroup']:<12} {d['number']:>4}  "
            f"{d['multiplicity']:>6}  {list(d['al_idx'])}")

    return struct_dict

def write_results_ineq_supercells(struct_dict: dict, output_dir: str):
    """Write results and cif files for inequivalent cif files."""
    os.makedirs(output_dir, exist_ok=True)
    

    ranked = sorted(
        struct_dict.values(),
        key=lambda d: (d["multiplicity"], -d["number"], d["al_idx"]),
    )
    width = len(str(len(ranked) - 1))
    renumbered = {}
    for rank, d in enumerate(ranked):
        tag = f"{d['formula']}_{rank:0{width}d}_{d["spacegroup"]}_m{d['multiplicity']}"
        write_cif(d["atoms"], os.path.join(output_dir, f"{tag}.cif"))
        renumbered[rank] = {**d, "filename": tag}

    save_nested_dict_csv(struct_dict, os.path.join(output_dir, "summary.csv"), index_label="Symmetry rank")

if __name__ == "__main__":
    structs = gen_sym_ineq_supercells(2, 2, (2 * V_PER_ATOM_AL + 6* V_PER_ATOM_FE) / 8, cell_type="primitive")
    write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\2x2x2")