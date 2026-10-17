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
ECUTWFCS = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

for ecutwfc in ECUTWFCS:
    single_sfc(ecutwfc, ecutwfc * 12, KMESH, "/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/ecutwfc")