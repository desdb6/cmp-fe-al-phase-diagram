import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calculations import *
from crystal_builder import *

primitive_fe7al = supercell_primitive_one_al(n=2, vol=7*V_PER_ATOM_FE + V_PER_ATOM_AL)
conventional_fe15al = supercell_conventional_one_al(n=2, vol=15*V_PER_ATOM_FE + V_PER_ATOM_AL)
primitive_fe26al = supercell_primitive_one_al(n=3, vol=26*V_PER_ATOM_FE + V_PER_ATOM_AL)
conventional_fe53al = supercell_primitive_one_al(n=3, vol=53*V_PER_ATOM_FE + V_PER_ATOM_AL)
scf_settings_high = QESettings(
    calc_type='scf',
    ecutwfc=134,
    ecutrho=1200,
    kmesh=(9, 9, 9)
)
vc_relax_settings_high = QESettings(
    calc_type='vc-relax',
    ecutwfc=134,
    ecutrho=1200,
    kmesh=(9, 9, 9),
    target_press=0,
    forc_conv_thr=5e-4
)
scf_settings_low = QESettings(
    calc_type='scf',
    ecutwfc=120,
    ecutrho=800,
    kmesh=(6, 6, 6)
)
vc_relax_settings_low = QESettings(
    calc_type='vc-relax',
    ecutwfc=120,
    ecutrho=800,
    kmesh=(6, 6, 6),
    target_press=0,
    forc_conv_thr=5e-4
)
full_analysis(primitive_fe7al, vc_relax_settings_high, scf_settings_high, "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/primitive/12.5/full_analysis_test")
print("fe7al done")
full_analysis(conventional_fe15al, vc_relax_settings_high, scf_settings_high, "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/conventional/6.25/full_analysis_test")
print("fe15al done")
full_analysis(primitive_fe26al, vc_relax_settings_low, scf_settings_low, "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/primitive/3.70/full_analysis_test")
print("fe26al done")
full_analysis(conventional_fe53al, vc_relax_settings_low, scf_settings_low, "/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/conventional/1.85/full_analysis_test")
print("fe53al done")
