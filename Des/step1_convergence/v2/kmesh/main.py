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

KMESHES = [
    (1, 1, 1),
    (3, 3, 3),
    (4, 4, 4),
    (5, 5, 5),
    (6, 6, 6),
    (7, 7, 7),
    (9, 9, 9),
    (11, 11, 11),
    ]
ECUTWFC = 90
ECUTRHO = 1080

for kmesh in KMESHES:
    single_sfc(ECUTWFC, ECUTRHO, kmesh, "/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/kmesh")