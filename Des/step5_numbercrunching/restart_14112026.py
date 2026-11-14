import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *
from calculations import *

OLD_DIRS = [
    "/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Fe12Al4_01_P4_mmm_m12",
    "/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Fe12Al4_02_P4_mmm_m12",
    "/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Fe12Al4_05_P4_2_mmc_m48",
    "/home/des/cmp-fe-al-phase-diagram/Des/step5_numbercrunching/conventional/2x2x2/4Al/Fe12Al4_06_P4_mmm_m48"
]

CIFS = [
    "/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_01_P4_mmm_m12.cif",
    "/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_02_P4_mmm_m12.cif",
    "/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_05_P4_2_mmc_m48.cif",
    "/home/des/cmp-fe-al-phase-diagram/CIF_Files/conventional_supercells/2x2x2/4Al/Fe12Al4_06_P4_mmm_m48.cif"
]

for old_dir, ref_cif in zip(OLD_DIRS, CIFS):
    print(f"Restarting {old_dir}")
    reference_structure = read_cif(ref_cif)
    restart_vcrelax(reference_structure, VC_RELAX_SETTINGS_LOW, old_dir=old_dir, new_dir=old_dir + "/restart")
    print(f"Succesfully finished {old_dir}")