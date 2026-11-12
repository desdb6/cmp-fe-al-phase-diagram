import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *
from calculations import *

# Fe7AL
structure = supercell_conventional_one_al(2, vol=15*V_PER_ATOM_FE + V_PER_ATOM_AL)
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/1AL")

# # Fe7Al
# structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF Files/conventional_supercells/2x2x2/2Al/Fe14Al2_0_Im-3m_m8.cif")
# run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/2Al/Im-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF Files/conventional_supercells/2x2x2/2Al/Fe14Al2_3_R-3m_m64.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/2Al/R-3m")

# # Fe13Al3
# structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF Files/conventional_supercells/2x2x2/3Al/Fe13Al3_0_Pm-3m_m16.cif")
# run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/3Al/Pm-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF Files/conventional_supercells/2x2x2/3Al/Fe13Al3_5_Cmm2_m192.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/3Al/Cmm2")

# # Fe3Al
# structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF Files/conventional_supercells/2x2x2/4Al/Fe12Al4_00_Fm-3m_m4.cif") # DO3
# run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Fm-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF Files/conventional_supercells/2x2x2/4Al/Fe12Al4_17_Cm_m384.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/C_m384")