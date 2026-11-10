import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calculations import *
from crystal_builder import *
from utils import *

# Primitive BCC Fe
structure = bcc_fe_primitive(vol=get_equilibrium_volumes()["Fe"])
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5, 7, 9, 11, 13, 15]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/bcc_fe/convergence_testing/primitive",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=12,
    var_to_converge="pressure",
    ecutwfc_tol=5e-1,
    ecutrho_tol=5e-1,
    kmesh_tol=5e-1
    )

# Conventional BCC Fe
structure = bcc_fe_conventional(vol=2*get_equilibrium_volumes()["Fe"])
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5, 7, 9, 11, 13, 15]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/bcc_fe/convergence_testing/conventional",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=12,
    var_to_converge="pressure",
    ecutwfc_tol=5e-1,
    ecutrho_tol=5e-1,
    kmesh_tol=5e-1
    )

# Primitive FCC Al
structure = fcc_al_primitive(vol=get_equilibrium_volumes()["Al"])
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5, 7, 9, 11, 13, 15]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/fcc_al/convergence_testing/primitive",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=12,
    var_to_converge="pressure",
    ecutwfc_tol=5e-1,
    ecutrho_tol=5e-1,
    kmesh_tol=5e-1
    )

# Conventional FCC Al
structure = fcc_al_conventional(vol=4*get_equilibrium_volumes()["Al"])
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5, 7, 9, 11 ,13, 15]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/fcc_al/convergence_testing/conventional",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=12,
    var_to_converge="pressure",
    ecutwfc_tol=5e-1,
    ecutrho_tol=5e-1,
    kmesh_tol=5e-1
    )

# Primitive 2x2x2
structure = supercell_primitive_one_al(n=2, vol=7*V_PER_ATOM_FE+V_PER_ATOM_AL)
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5, 7, 9]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/primitive/2x2x2_convergence_test",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=6
    )

# Conventional 2x2x2
structure = supercell_conventional_one_al(n=2, vol=15*V_PER_ATOM_FE+V_PER_ATOM_AL)
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5, 7, 9]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/conventional/2x2x2_convergence_test",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=6
    )

# Primitive 3x3x3
structure = supercell_primitive_one_al(n=3, vol=26*V_PER_ATOM_FE+V_PER_ATOM_AL)
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/primitive/3x3x3_convergence_test",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=4
    )

# Conventional 3x3x3
structure = supercell_conventional_one_al(n=3, vol=53*V_PER_ATOM_FE+V_PER_ATOM_AL)
ecutwfc_list = list(np.round(np.linspace(80, 150, 10)))
ecutrho_list = list(np.round(np.linspace(800, 1500, 10)))
kmesh_list = [1, 2, 3, 5]

convergence_test(
    structure=structure,
    calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/conventional/3x3x3_convergence_test",
    ecutwfc_list=ecutwfc_list,
    ecutrho_list=ecutrho_list,
    kmesh_list=kmesh_list,
    kmesh_dense=4
    )


# # Primitive 4x4x4
# structure = supercell_primitive_one_al(n=4, vol=63*V_PER_ATOM_FE+V_PER_ATOM_AL)
# ecutwfc_list = list(np.round(np.linspace(40, 100, 8)))
# ecutrho_list = list(np.round(np.linspace(500, 1000, 8)))
# kmesh_list = [1, 2]

# convergence_test(
#     structure=structure,
#     calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/primitive/4x4x4_convergence_test",
#     ecutwfc_list=ecutwfc_list,
#     ecutrho_list=ecutrho_list,
#     kmesh_list=kmesh_list,
#     kmesh_dense=2
#     )

# # Conventional 4x4x4
# structure = supercell_conventional_one_al(n=4, vol=127*V_PER_ATOM_FE+V_PER_ATOM_AL)
# ecutwfc_list = list(np.round(np.linspace(40, 100, 8)))
# ecutrho_list = list(np.round(np.linspace(500, 1000, 8)))
# kmesh_list = [1, 2]

# convergence_test(
#     structure=structure,
#     calc_dir="/home/des/cmp-fe-al-phase-diagram/Des/step4_one_atom_supercells/conventional/4x4x4_convergence_test",
#     ecutwfc_list=ecutwfc_list,
#     ecutrho_list=ecutrho_list,
#     kmesh_list=kmesh_list,
#     kmesh_dense=2
#     )
