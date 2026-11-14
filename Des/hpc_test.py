from crystal_builder import *
from calculations import *

structure = supercell_conventional_one_al(n=2, vol=15*V_PER_ATOM_FE + V_PER_ATOM_AL)
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=PROJECT_ROOT / "hpc_test/supercell")
