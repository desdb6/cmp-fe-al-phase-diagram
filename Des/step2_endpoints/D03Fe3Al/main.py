import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from qe_utils import *
from plotting import plot_csv
from scf import single_scf, single_relax

EXP_V_BCC = 23.55

settings = QESettings(ecutwfc=90, ecutrho=1080, kmesh=(5, 5, 5))

# Generate and write a cif file
atoms = atoms_d03_fe3al(lattice_parameter=(EXP_V_BCC) ** (1/3))
write_cif(atoms, path="/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/D03Fe3Al", filename="D03Fe3Al")

# Run a single vcrelax calculation to find the equilibrium volume
vc_relax_vol, _ = single_relax(
    volume=EXP_V_BCC * 8,
    settings=settings,
    output_folder="/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/D03Fe3Al/vcrelax",
    )
    
# Run multiple single scf calculations with 10% volume variation
scf_dir = "/home/des/cmp-fe-al-phase-diagram/Des/step2_endpoints/D03Fe3Al/scf_series"
volumes = vc_relax_vol * np.linspace(0.9, 1.1, 9)
for v in volumes:
    single_scf(v,
    settings=settings,
    output_folder=scf_dir,
    )

# Plot the energy and magnetic moment as functions of the volume
plot_csv(
    scf_dir + "/results.csv",
    x="volume (Ang**3)",
    y="energy (Ry)",
    xlabel=r"Volume (Å$^3$)",
    ylabel="Energy (Ry)",
    title=r"Energy per unit cell of $D0_3-Fe_3Al$ structure as a function of unit cell volume",
    filename=scf_dir + "/energy_plot.png"
)

plot_csv(
    scf_dir + "/results.csv",
    x="volume (Ang**3)",
    y="moment (muB)",
    xlabel=r"Volume (Å$^3$)",
    ylabel=r"Magnetic moment (\mu B)",
    title=r"Energy per unit cell of $D0_3-Fe_3Al$ structure as a function of unit cell volume",
    filename=scf_dir + "/moment_plot.png"
)

# Perform a EoS fit, save and plot the results
eos_fit = EoSfit(csv_path=scf_dir + "/results.csv")
eos_fit.plot(filename=scf_dir + "/eos_plot.png")
eos_fit.write_results()