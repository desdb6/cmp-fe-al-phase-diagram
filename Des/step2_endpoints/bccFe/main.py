import numpy as np

from scf import single_scf, single_relax
from qe_utils import *
from plotting import plot_csv

EXP_V = 23.55

settings = QESettings(ecutwfc=120, ecutrho=1440, kmesh=(6, 6, 6))

# atoms = atoms_bcc_Fe(lattice_parameter=EXP_V ** (1/3))
# write_cif(atoms, "/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/bccFe", "structure")

# for v in np.arange(22, 24, 0.02):
#     single_scf(v, settings, "/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/bccFe/run2", rm_calc_dir=True)

eos_fit = EoSfit("/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/bccFe/run2/results.csv")
eos_fit.plot("/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/bccFe/run2/EoS_bm_fit")

# plot_csv(
#     "/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/bccFe/run2/results.csv",
#     x="volume (Ang**3)",
#     y="moment (muB)",
#     title="Magnetic moment as a function of unit cell volume",
#     xlabel=r"Volume (Å$^3$)")
# plt.show()

# single_relax(EXP_V, settings, "/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/bccFe/vcrelax")