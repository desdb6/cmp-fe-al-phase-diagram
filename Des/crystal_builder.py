"""
Crystal structure generation for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 04/11/2026
"""

from ase.visualize import view
from ase import Atoms
from ase.build import bulk
from ase.io import write

def write_cif(atoms: ase.Atoms, path: str) -> None:
    """Generate a cif file from an Atoms object."""
    write(path, atoms)


def vis_atoms(atoms: ase.Atoms) -> None:
    """Visualize Atoms object."""
    view(atoms)


def fe7al(lattice_parameter: float) -> ase.Atoms:
    """Generate an Atoms object for an Fe7Al structure"""
    atoms = bulk('Fe', 'bcc', a=lattice_parameter) * (2, 2, 2)
    atoms[0].symbol = 'Al'
    return atoms


def bcc_fe_conventional(vol: float, magmom: float = 0.5) -> ase.Atoms:
    """Generate a conventional BCC iron atoms object"""
    lattice_parameter = vol ** (1/3)
    atoms = bulk('Fe', 'bcc', a=lattice_parameter, cubic=True)
    atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    return atoms

def bcc_fe_primitive(vol: float, magmom: float = 0.5) -> ase.Atoms:
    """Generate a primitive BCC iron atoms object"""
    lattice_parameter = (2 * vol) ** (1/3)
    atoms = bulk('Fe', 'bcc', a=lattice_parameter)
    atoms.set_initial_magnetic_moments([magmom] * len(atoms))
    return atoms

def fcc_al_conventional(vol: float) -> ase.atoms:
    """Generate a conventional FCC alimunium atoms object."""
    lattice_parameter = vol ** (1/3)
    atoms = bulk('Al', 'fcc', a=lattice_parameter, cubic=True)
    return atoms

def fcc_al_primitive(vol: float) -> ase.atoms:
    """Generate a primitive FCC alimunium atoms object."""
    lattice_parameter = (4 * vol) ** (1/3)
    atoms = bulk('Al', 'fcc', a=lattice_parameter, cubic=True)
    return atoms

def d03_fe3al(lattice_parameter: float, magmom: float = 0.5) -> ase.Atoms:
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

def supercell_primitive_one_al(n: int, vol: float, magmom: float = 0.5) -> ase.Atoms:
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

def supercell_conventional_one_al(n: int, vol: float, magmom: float = 0.5) -> ase.Atoms:
    """
    Generate an nxnxn supercell of conventional bcc Fe unit cells,
    where one atom has been changed for Al.
    """
    lattice_parameter_single_cell = (vol) ** (1/3) / n # Volume times two because this cell is not cubic, check the volume formula
    atoms = bulk("Fe", crystalstructure="bcc", cubic=True, a=lattice_parameter_single_cell) * (n, n, n)
    atoms[0].symbol = "Al"
    atoms.set_initial_magnetic_moments(
        [magmom if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()]
    )
    return atoms

if __name__ == "__main__":
    structure = supercell_primitive_one_al(3, 55 * 27)
    write_cif(structure, "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/primitive_test.cif")

    structure = supercell_conventional_one_al(3, 55 * 27)
    write_cif(structure, "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/conventional_test.cif")