"""
Calculations file for quantum espresso scripting

Author: Des De Borger
Email: des.deborger@student.uantwerpen.be
Last modified: 04/11/2026
"""

from pathlib import Path
import csv
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

from ase.calculators.espresso import Espresso, EspressoProfile
from ase.eos import EquationOfState

from parsers import *
from utils import *
from crystal_builder import *

# Pseudopotential paths
PSEUDO_DIR = "/home/des/cmp-fe-al-phase-diagram/Pseudopotentials"
PSEUDOS = {"Fe": "Fe.paw.pbe.z_16.ld1.psl.v0.2.1",
           "Al": "Al.us.pbe.z_3.ld1.psl.v1.0.0-low"}

# Dataclass for quantum espresso settings
@dataclass(frozen=True)
class QESettings:
    """General settings for Quantum Espresso calculations."""
    calc_type: str                 # Calculation type: "scf", "relax", "vc-relax"
    ecutwfc: float                 # Wavefunction basis set energy cutoff (Ry)
    ecutrho: float                 # Density basis set energy cutoff (Ry)
    kmesh: tuple[int, int, int]    # Kmesh density

    # General settings
    smearing: str = "mv"           # Smearing type
    degauss: float = 0.02          # Smearing width (Ry)
    conv_thr: float = 1e-8         # SCF convergence threshold (Ry)
    mixing_beta: float = 0.3       # SCF mixing factor
    electron_maxstep: int = 200    # Maximum number of SCF iterations

    # Calculation specific settings
    forc_conv_thr: float | None = None  # Atomic force threshold (Ry/au) for relax and vc-relax
    etot_conv_thr: float = 1e-5         # Energy threshold (Ry) for relax and vc-relax
    target_press: float | None = None   # Target pressure (kbar) for vc-relax
    press_conv_thr: float = 0.5         # Pressure threshold (kbar) for vc-relax
    cell_dofree: str = "all"            # Cell degrees of freedom for vc-relax

    def __post_init__(self):
        # Calculation type check
        if self.calc_type not in ["scf", "relax", "vc-relax"]:
            raise ValueError("Error, please input a valid calculation type (scf, relax, vc-relax)")

        # Type-specific settings checks
        if self.calc_type in ["relax", "vc-relax"] and self.forc_conv_thr is None:
            raise ValueError(f"Error, please input a force threshold for {self.calc_type} calculation.")
        if self.calc_type == "vc-relax" and self.target_press is None:
            raise ValueError(f"Error, please input a target pressure for {self.calc_type} calculation.")

        # Convert numpy scalars to plain Python types, QE cannot parse np.float64(...)
        object.__setattr__(self, "ecutwfc", float(self.ecutwfc))
        object.__setattr__(self, "ecutrho", float(self.ecutrho))
        object.__setattr__(self, "kmesh", tuple(int(k) for k in self.kmesh))

    def to_input_data(self) -> dict:
        data = {
            "control": {"calculation": self.calc_type},
            "system": {
                "ecutwfc": self.ecutwfc,
                "ecutrho": self.ecutrho,
                "occupations": "smearing",
                "smearing": self.smearing,
                "degauss": self.degauss,
            },
            "electrons": {
                "conv_thr": self.conv_thr,
                "mixing_beta": self.mixing_beta,
                "electron_maxstep": self.electron_maxstep,
            },
        }
        if self.calc_type in ["relax", "vc-relax"]:
            data["control"]["forc_conv_thr"] = self.forc_conv_thr
            data["control"]["etot_conv_thr"] = self.etot_conv_thr
        if self.calc_type == "vc-relax":
            data["cell"] = {
                "press": self.target_press,
                "press_conv_thr": self.press_conv_thr,
                "cell_dofree": self.cell_dofree,
            }
        return data

# Class for EoS fits
class EoSfit:
    """Fit an equation of state to volume (A^3) and energy (Ry) data."""

    def __init__(
        self,
        energies,
        volumes,
        output_path,
        **read_kwargs,
    ):
        self.V = volumes   # A^3
        self.E = energies   # Ry
        self.output_path = output_path
        self.fit()
        self.write_results()
        self.plot()

    def fit(self, eos=("birchmurnaghan", "murnaghan", "vinet")):
        """Return a DataFrame with V0, E0, B0 for each EOS form."""
        rows = []
        for name in eos:
            # Inputs in A^3 and Ry -> outputs in A^3, Ry, Ry/A^3
            v0, e0, B = EquationOfState(self.V, self.E, eos=name).fit()
            rows.append({
                "eos": name,
                "V0": v0,
                "E0": e0,
                "B0": B * RY_PER_ANG3_TO_KBAR_CONV,
            })
        self.results = pd.DataFrame(rows)
        return self.results

    def write_results(self):
        """Write a csv file with the results of the fit."""
        self.results.to_csv(self.output_path + "/eos_results.csv", index=False)

    def plot(self, eos="birchmurnaghan"):
        """Plot data and fit with correct axis labels (A^3, Ry)."""
        e = EquationOfState(self.V, self.E, eos=eos)
        e.fit()
        _, e0, v0, B, x, y, v, en = e.getplotdata()  # B is in Ry/A^3 here

        fig, ax = plt.subplots()
        ax.plot(v, en, "o", label="data")
        ax.plot(x, y, "-", label=eos)
        ax.set_xlabel(r"Volume (Å$^3$)")
        ax.set_ylabel("Energy (Ry)")
        ax.legend()
        ax.grid(alpha=0.5)
        ax.set_title(
            rf"$V_0$ = {v0:.2f} Å$^3$,  $E_0$ = {e0:.2f} Ry,  "
            rf"$B_0$ = {B * RY_PER_ANG3_TO_KBAR_CONV:.2f} kbar",
            fontsize=8,
        )

        fig.savefig(self.output_path + "/eos_plot.png", dpi=200, bbox_inches="tight")
        return fig, ax
        
def parse_qe_settings(structure: Atoms, settings: QESettings, calc_dir: str) -> Espresso:
    """
    Function to parse settings in QESettings object to Espresso object to generate calculator.
    """
    profile = EspressoProfile(command="mpirun -np 1 /home/des/q-e/bin/pw.x", pseudo_dir=PSEUDO_DIR)

    # Settings from the dataclass (depends on calc_type)
    input_data = settings.to_input_data()

    # Always compute stress and forces, needed for pressure and EoS analysis
    input_data["control"]["tstress"] = True
    input_data["control"]["tprnfor"] = True

    return Espresso(
        profile=profile,
        directory=calc_dir,
        pseudopotentials={s: PSEUDOS[s] for s in set(structure.get_chemical_symbols())},
        input_data=input_data,
        kpts=settings.kmesh,
    )

def save_result_csv(result: dict, output_path) -> None:
    """Save the results dict of a single calculation to a csv file. The 'atoms' entry is skipped. (Claude)"""
    columns = [k for k in result if k != "atoms"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerow([result[k] for k in columns])

def run_calculation(structure: Atoms, settings: QESettings, calc_dir: str,
                    overwrite: bool = False) -> dict:
    """
    Run (or load) a QE calculation in calc_dir and return the results as a dict.
    """
    calc_dir = Path(calc_dir)
    output = calc_dir / "espresso.pwo"

    done = output.exists() and "JOB DONE" in output.read_text()
    if done and not overwrite:
        print(f"Skipping {calc_dir}, already done")
    else:
        calc_dir.mkdir(parents=True, exist_ok=True)
        atoms = structure.copy()
        atoms.calc = parse_qe_settings(atoms, settings, str(calc_dir))
        atoms.get_potential_energy()      # this triggers pw.x

    # Always read from file, for relax/vc-relax this is the final structure
    final = read(output, format="espresso-out")

    results = {
        "energy": read_energy(calc_dir), # Ry
        "volume": read_volume(calc_dir), # Ang**3
        "pressure": read_pressure(calc_dir), # kbar
        "magnetization": read_total_magnetization(calc_dir), # muB/cell
        "time": read_time(calc_dir), # s
        "atoms": final,
    }

    save_result_csv(results, calc_dir / "results.csv")
    return results

def convergence_test(
    structure: Atoms,
    calc_dir: str,
    ecutwfc_list: list,
    ecutrho_list: list,
    kmesh_list: list,
    rho_wfc_ratio: int = 6,
    kmesh_dense: int = 7,
    var_to_converge: str = "pressure",
    ecutwfc_tol = 5e-1,
    ecutrho_tol = 5e-1,
    kmesh_tol = 5e-1
    ):
    """
    Perform an automatic convergence test on a given structure, for settings within given ranges.
    This function will consecutively converge the ecutwfc, ecutrho and kmesh settings.
    """
    if var_to_converge not in ["energy", "pressure", "magnetization"]:
        raise ValueError("Variable to converge error, choose an appropriate variable (energy, pressure, magnetization)")

    Path(calc_dir).mkdir(parents=True, exist_ok=True) # Make the calculation directory if it does not exist

    def check_convergence(x_list, results, tol, xlabel, filename):
        """Find the convergence point of var_to_converge and plot energy, pressure and magnetization. (Claude)"""
        y_conv = [results[x][var_to_converge] for x in x_list]
        conv = find_convergence(x_list, y_conv, tol)

        times = [results[x]["time"] for x in x_list]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for a, key in zip(ax, UNIT_DICT):
            y = [results[x][key] for x in x_list]
            a.plot(x_list, y, "-", color="gray", alpha=0.5, zorder=1)
            sc = a.scatter(x_list, y, c=times, cmap="viridis", zorder=2)
            if conv is not None:
                a.axvline(conv, color="red", ls="--", alpha=0.5)
            a.set_xlabel(xlabel)
            a.set_ylabel(f"{key} ({UNIT_DICT[key]})")
            a.grid(alpha=0.5)
        fig.colorbar(sc, ax=ax, label="time (s)")
        fig.savefig(filename, dpi=200, bbox_inches="tight")

        if conv is None:
            print(f"{xlabel} did not converge for {var_to_converge}, choosing last value.")
            conv = x_list[-1]
        return conv

    # Stage 1: ecutwfc
    results = {}
    for ecutwfc in ecutwfc_list:
        ecutrho = ecutwfc * rho_wfc_ratio # Set ecutrho to given ratio
        output_dir = f"{calc_dir}/ecutwfc/{ecutwfc}_{ecutrho}_{kmesh_dense}"

        settings = QESettings(
            calc_type="scf",
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            kmesh=(kmesh_dense, kmesh_dense, kmesh_dense)
        )
        results[ecutwfc] = run_calculation(structure, settings, output_dir)
    conv_ecutwfc = check_convergence(ecutwfc_list, results, ecutwfc_tol,
                                     "ecutwfc (Ry)", f"{calc_dir}/ecutwfc/ecutwfc_convergence.png")
    save_nested_dict_csv(results, f"{calc_dir}/ecutwfc/results.csv", index_label="ecutwfc")

    # Stage 2: ecutrho
    results = {}
    for ecutrho in ecutrho_list:
        output_dir = f"{calc_dir}/ecutrho/{conv_ecutwfc}_{ecutrho}_{kmesh_dense}"

        settings = QESettings(
            calc_type="scf",
            ecutwfc=conv_ecutwfc,
            ecutrho=ecutrho,
            kmesh=(kmesh_dense, kmesh_dense, kmesh_dense)
        )
        results[ecutrho] = run_calculation(structure, settings, output_dir)
    conv_ecutrho = check_convergence(ecutrho_list, results, ecutrho_tol,
                                     "ecutrho (Ry)", f"{calc_dir}/ecutrho/ecutrho_convergence.png")
    save_nested_dict_csv(results, f"{calc_dir}/ecutrho/results.csv", index_label="ecutrho")

    # Stage 3: kmesh
    results = {}
    for kmesh in kmesh_list:
        output_dir = f"{calc_dir}/kmesh/{conv_ecutwfc}_{conv_ecutrho}_{kmesh}"

        settings = QESettings(
            calc_type="scf",
            ecutwfc=conv_ecutwfc,
            ecutrho=conv_ecutrho,
            kmesh=(kmesh, kmesh, kmesh)
        )
        results[kmesh] = run_calculation(structure, settings, output_dir)
    conv_kmesh = check_convergence(kmesh_list, results, kmesh_tol,
                                   "kmesh", f"{calc_dir}/kmesh/kmesh_convergence.png")
    save_nested_dict_csv(results, f"{calc_dir}/kmesh/results.csv", index_label="kmesh")

    # Saving results
    with open(f"{calc_dir}/converged_settings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "unit"])
        writer.writerow(["ecutwfc", conv_ecutwfc, "Ry"])
        writer.writerow(["ecutrho", conv_ecutrho, "Ry"])
        writer.writerow(["kmesh", conv_kmesh, "per direction"])
        writer.writerow(["variable converged", var_to_converge, UNIT_DICT[var_to_converge]])
        writer.writerow(["tolerance ecutwfc", ecutwfc_tol, UNIT_DICT[var_to_converge]])
        writer.writerow(["tolerance ecutrho", ecutrho_tol, UNIT_DICT[var_to_converge]])
        writer.writerow(["tolerance kmesh", kmesh_tol, UNIT_DICT[var_to_converge]])

    print("Final converged settings:")
    print(f"  ecutwfc: {conv_ecutwfc} Ry")
    print(f"  ecutrho: {conv_ecutrho} Ry")
    print(f"  kmesh:   {conv_kmesh} x {conv_kmesh} x {conv_kmesh}")

    return {"ecutwfc": conv_ecutwfc, "ecutrho": conv_ecutrho, "kmesh": conv_kmesh}

def scf_volume_sweep(
    structure: ase.Atoms,
    settings: QESettings,
    calc_dir: str,
    starting_volume: float,
    volume_lower_bound: float = 0.96,
    volume_upper_bound: float = 1.06,
    n_volumes: int = 10
    ):
    """Performs a volume sweep."""
    if settings.calc_type != "scf":
        raise ValueError("Please input an scf settings file to perform a volume sweep.")

    # Volume sweep
    ratios = np.linspace(volume_lower_bound, volume_upper_bound, n_volumes)
    volumes = ratios * starting_volume

    results = {}
    for volume, ratio in zip(volumes, ratios):
        rescaled_structure = set_volume(structure, volume)
        output_dir = calc_dir + f"/volume_{ratio:2g}"
        results[ratio] = run_calculation(rescaled_structure, settings, output_dir)
    save_nested_dict_csv(results, f"{calc_dir}/results.csv", index_label="ratio")

    return results

def full_analysis(
    structure: ase.Atoms,
    settings_vcrelax: QESettings,
    settings_sweep: QESettings,
    calc_dir: str
    ):
    """
    Perform a full analysis in 4 stages:
    1. vc-relax calculation to find equilibrium volume
    2. volume sweep to build E(V) curve
    3. Birch-Murghanan EoS fit
    4. Calculate formation energy
    """

    # Stage 1: vc-relax
    vc_relax_results = run_calculation(structure, settings_vcrelax, calc_dir + "/vc_relax")

    # Stage 2: volume sweep
    sweep_results = scf_volume_sweep(structure, settings_sweep, calc_dir + "/volume_sweep", vc_relax_results["volume"])

    # Stage 3: Birch-Murghanan fit
    energies = np.array([r["energy"] for r in sweep_results.values()])
    volumes  = np.array([r["volume"] for r in sweep_results.values()])
    eos_fit = EoSfit(energies, volumes, calc_dir)

    # Stage 4: formation energy
    row = eos_fit.results.set_index("eos").loc["birchmurnaghan"]
    e0_cell = row["E0"]
    v0_cell = row["V0"]
    b0 = row["B0"]

    refs = get_formation_energies()               # Ry per atom
    symbols = structure.get_chemical_symbols()
    n_atoms = len(symbols)
    e_ref = sum(refs[s] for s in symbols)         # also works for Fe-Al mixtures

    e_form_ry = (e0_cell - e_ref) / n_atoms
    e_form_ev = e_form_ry * Ry                    # Ry comes from ase.units via utils

    summary = {
        "formula": structure.get_chemical_formula(),
        "n_atoms": n_atoms,
        "V0 (A**3/atom)": v0_cell / n_atoms,
        "B0 (kbar)": b0,
        "E0 (Ry/atom)": e0_cell / n_atoms,
        "E_form (Ry/atom)": e_form_ry,
        "E_form (eV/atom)": e_form_ev,
    }
    save_result_csv(summary, Path(calc_dir) / "formation_energy.csv")
    return summary

if __name__ == "__main__":
    primitive_fe7al = supercell_primitive_one_al(n=2, vol=7*V_PER_ATOM_FE + V_PER_ATOM_AL)
    conventional_fe15al = supercell_conventional_one_al(n=2, vol=15*V_PER_ATOM_FE + V_PER_ATOM_AL)
    primitive_fe26al = supercell_primitive_one_al(n=3, vol=26*V_PER_ATOM_FE + V_PER_ATOM_AL)
    conventional_fe53al = supercell_primitive_one_al(n=3, vol=53*V_PER_ATOM_FE + V_PER_ATOM_AL)
    scf_settings_high = QESettings(
        calc_type='scf',
        ecutwfc=120,
        ecutrho=1500,
        kmesh=(6, 6, 6)
    )
    vc_relax_settings_high = QESettings(
        calc_type='vc-relax',
        ecutwfc=120,
        ecutrho=1500,
        kmesh=(6, 6, 6),
        target_press=0,
        forc_conv_thr=5e-4
    )
    scf_settings_low = QESettings(
        calc_type='scf',
        ecutwfc=120,
        ecutrho=1500,
        kmesh=(4, 4, 4)
    )
    vc_relax_settings_low = QESettings(
        calc_type='vc-relax',
        ecutwfc=120,
        ecutrho=1500,
        kmesh=(4, 4, 4),
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
