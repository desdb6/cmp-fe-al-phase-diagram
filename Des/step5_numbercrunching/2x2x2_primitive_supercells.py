import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *
from calculations import *

# Fe3Al
P4_mmm_structure = read_cif(r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\2x2x2\2Al\Fe6Al2_0_P4_mmm_m12.cif")
run_calculation(P4_mmm_structure, SCF_SETTINGS_HIGH, calc_dir="C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\Des\step5_numbercrunching\primitive\2x2x2\2Al\P4_mmm")

R3m_structure = read_cif(r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\2x2x2\2Al\Fe6Al2_1_R-3m_m16.cif")
run_calculation(R3m_structure, SCF_SETTINGS_HIGH, calc_dir="C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\Des\step5_numbercrunching\primitive\2x2x2\2Al\R-3m")
