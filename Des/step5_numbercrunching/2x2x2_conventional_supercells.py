import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *
from calculations import *

# Fe15AL
structure = supercell_conventional_one_al(2, vol=15*V_PER_ATOM_FE + V_PER_ATOM_AL)
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/1Al")

# Fe7Al
structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/2Al/Fe14Al2_0_Im-3m_m8.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/2Al/Im-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/2Al/Fe14Al2_3_R-3m_m64.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/2Al/R-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/2Al/Fe14Al2_1_P4_mmm_m24.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/2Al/P4_mmm_1")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/2Al/Fe14Al2_2_P4_mmm_m24.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/2Al/P4_mmm_2")

# Fe13Al3
structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/3Al/Fe13Al3_0_Pm-3m_m16.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/3Al/Pm-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/3Al/Fe13Al3_1_P4_mmm_m48.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/3Al/P4_mmm_1")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/3Al/Fe13Al3_2_P4_mmm_m48.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/3Al/P4_mmm_2")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/3Al/Fe13Al3_5_Cmm2_m192.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/3Al/Cmm2")

# Fe3Al
structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_00_Fm-3m_m4.cif") # DO3
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Fm-3m")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_17_Cm_m384.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/C_m384")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_01_P4_mmm_m12.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/P4_mmm_m12_0-2-4-6")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_02_P4_mmm_m12.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/P4_mmm_m12_0-2-12-14")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_04_Pm-3m_m16.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Pm-3m_m16")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_07_P4_2_mmc_m48.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/P4_2_mmc_m48")

structure = read_cif("/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_08_P4_mmm_m48.cif")
run_calculation(structure, VC_RELAX_SETTINGS_LOW, calc_dir=r"/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/P4_mmm_m48_0-2-4-14")