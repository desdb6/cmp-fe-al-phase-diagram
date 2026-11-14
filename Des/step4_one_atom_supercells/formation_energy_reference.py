import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calculations import *
from crystal_builder import *
from utils import *

# BCC Fe
structure = bcc_fe_primitive(vol=V_PER_ATOM_FE) # Primitive is cheaper to compute 
settings = QESettings( # Very high precision settings
    calc_type='vc-relax',
    ecutwfc=142,
    ecutrho=1500,
    kmesh=(11, 11, 11),
    target_press=0,
    forc_conv_thr=5e-4
)

results = run_calculation(structure, settings, calc_dir=PROJECT_ROOT / "Formation_Energies/bccFe", overwrite=True)

# FCC Al
structure = fcc_al_primitive(vol=V_PER_ATOM_AL) # Primitive is cheaper to compute 
settings = QESettings( # Very high precision settings
    calc_type='vc-relax',
    ecutwfc=142,
    ecutrho=1500,
    kmesh=(11, 11, 11),
    target_press=0,
    forc_conv_thr=5e-4
)

results = run_calculation(structure, settings, calc_dir=PROJECT_ROOT / "Formation_Energies/fccAl", overwrite=True)
