import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spglib

from crystal_builder import *
from utils import *

def structure_to_cell(structure):
    """Convert an ASE Atoms object to the (lattice, positions, numbers) tuple spglib expects. (Claude)"""
    atoms = structure.copy()

    # Sanity check: there should be Al in the structure already
    al_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Al"]
    if not al_idx:
        raise ValueError("No Al atom found in structure")

    return (
        np.asarray(atoms.cell, dtype=float).tolist(),
        np.asarray(atoms.get_scaled_positions(), dtype=float).tolist(),
        np.asarray(atoms.numbers, dtype=int).tolist(),
    )

### PRIMITIVE ###

structure = supercell_primitive_one_al(2, V_PER_ATOM_AL + 7 * V_PER_ATOM_FE)
sym = spglib.get_symmetry_dataset(structure_to_cell(structure), symprec=1e-4)
write_cif(structure, f"CIF Files/primitive_supercells/2x2x2/1Al/Fe7Al_0_{sym.international.replace("/", "_")}.cif")

structs = gen_sym_ineq_supercells(2, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/2x2x2/2Al")

structure = supercell_primitive_one_al(3, V_PER_ATOM_AL + 26 * V_PER_ATOM_FE)
sym = spglib.get_symmetry_dataset(structure_to_cell(structure), symprec=1e-4)
write_cif(structure, f"CIF Files/primitive_supercells/3x3x3/1Al/Fe26Al_0_{sym.international.replace("/", "_")}.cif")

structs = gen_sym_ineq_supercells(3, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/2Al")

structs = gen_sym_ineq_supercells(3, 3, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/3Al")

structs = gen_sym_ineq_supercells(3, 4, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/4Al")

structs = gen_sym_ineq_supercells(3, 5, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/5Al")

structs = gen_sym_ineq_supercells(3, 6, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/6Al", max_cifs=330)

structure = supercell_primitive_one_al(4, V_PER_ATOM_AL + 63 * V_PER_ATOM_FE)
sym = spglib.get_symmetry_dataset(structure_to_cell(structure), symprec=1e-4)
write_cif(structure, f"CIF Files/primitive_supercells/4x4x4/1Al/Fe63Al_0_{sym.international.replace("/", "_")}.cif")

### CONVENTIONAL ###

structs = gen_sym_ineq_supercells(2, 2, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/2x2x2/2Al")

structure = supercell_conventional_one_al(2, V_PER_ATOM_AL + 15 * V_PER_ATOM_FE)
sym = spglib.get_symmetry_dataset(structure_to_cell(structure), symprec=1e-4)
write_cif(structure, f"CIF Files/conventional_supercells/2x2x2/1Al/Fe15Al_0_{sym.international.replace("/", "_")}.cif")

structs = gen_sym_ineq_supercells(2, 3, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/2x2x2/3Al")

structs = gen_sym_ineq_supercells(2, 4, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/2x2x2/4Al")

structure = supercell_conventional_one_al(3, V_PER_ATOM_AL + 53 * V_PER_ATOM_FE)
sym = spglib.get_symmetry_dataset(structure_to_cell(structure), symprec=1e-4)
write_cif(structure, f"CIF Files/conventional_supercells/3x3x3/1Al/Fe53Al_0_{sym.international.replace("/", "_")}.cif")

structs = gen_sym_ineq_supercells(3, 2, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/2Al")

structs = gen_sym_ineq_supercells(3, 3, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/3Al")

structs = gen_sym_ineq_supercells(3, 4, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/4Al")

structs = gen_sym_ineq_supercells(3, 5, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/5Al", max_cifs=100)

structs = gen_sym_ineq_supercells(3, 6, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/6Al", max_cifs=100)

structure = supercell_conventional_one_al(4, V_PER_ATOM_AL + 127 * V_PER_ATOM_FE)
sym = spglib.get_symmetry_dataset(structure_to_cell(structure), symprec=1e-4)
write_cif(structure, f"CIF Files/conventional_supercells/4x4x4/1Al/Fe127Al_0_{sym.international.replace("/", "_")}.cif")
