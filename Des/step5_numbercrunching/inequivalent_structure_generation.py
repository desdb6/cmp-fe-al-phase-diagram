import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *

### PRIMITIVE ###

structs = gen_sym_ineq_supercells(2, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/2x2x2/2Al")

structs = gen_sym_ineq_supercells(3, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/2Al")

structs = gen_sym_ineq_supercells(3, 3, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/3Al")

structs = gen_sym_ineq_supercells(3, 4, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/4Al")

structs = gen_sym_ineq_supercells(3, 5, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/5Al")

structs = gen_sym_ineq_supercells(3, 6, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/primitive_supercells/3x3x3/6Al", max_cifs=100)

### CONVENTIONAL ###

structs = gen_sym_ineq_supercells(2, 2, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/2x2x2/2Al")

structs = gen_sym_ineq_supercells(2, 3, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/2x2x2/3Al")

structs = gen_sym_ineq_supercells(2, 4, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/2x2x2/4Al")

structs = gen_sym_ineq_supercells(3, 2, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/2Al")

structs = gen_sym_ineq_supercells(3, 3, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/3Al")

structs = gen_sym_ineq_supercells(3, 4, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/4Al")

structs = gen_sym_ineq_supercells(3, 5, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/5Al", max_cifs=100)

structs = gen_sym_ineq_supercells(3, 6, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"CIF Files/conventional_supercells/3x3x3/6Al", max_cifs=100)
