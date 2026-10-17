import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os, csv
from time import time
import numpy as np
from ase.io import read
from ase.units import GPa
from ase.calculators.espresso import Espresso, EspressoProfile

from utils import *
from convergence_scf import single_sfc

KMESH = (5, 5, 5)
ECUTWFC = 90
RATIOS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for ratio in RATIOS:
    single_sfc(ECUTWFC, ECUTWFC * ratio, KMESH, "/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/ecutrho")