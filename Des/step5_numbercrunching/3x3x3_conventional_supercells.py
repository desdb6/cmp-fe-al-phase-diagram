import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *
from calculations import *

structure = supercell_conventional_one_al(3, vol = 53*V_PER_ATOM_FE + V_PER_ATOM_AL)
write_cif(structure, r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF_Files\conventional_supercells\3x3x3\1Al\Fe53Al1_0_Pm3m.cif")