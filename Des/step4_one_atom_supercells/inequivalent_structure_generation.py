import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_builder import *

### PRIMITIVE ###

structs = gen_sym_ineq_supercells(2, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\2x2x2\2Al")

structs = gen_sym_ineq_supercells(3, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\3x3x3\2Al")

structs = gen_sym_ineq_supercells(3, 3, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\3x3x3\3Al")

structs = gen_sym_ineq_supercells(3, 4, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\3x3x3\4Al")

structs = gen_sym_ineq_supercells(3, 5, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\3x3x3\5Al")

structs = gen_sym_ineq_supercells(4, 2, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\4x4x4\2Al")

structs = gen_sym_ineq_supercells(4, 3, cell_type="primitive")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\primitive_supercells\4x4x4\3Al")

### CONVENTIONAL ###

structs = gen_sym_ineq_supercells(2, 2, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\2x2x2\2Al")

structs = gen_sym_ineq_supercells(2, 3, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\2x2x2\3Al")

structs = gen_sym_ineq_supercells(2, 4, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\2x2x2\4Al")

structs = gen_sym_ineq_supercells(2, 5, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\2x2x2\5Al")

structs = gen_sym_ineq_supercells(3, 2, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\3x3x3\2Al")

structs = gen_sym_ineq_supercells(3, 3, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\3x3x3\3Al")

structs = gen_sym_ineq_supercells(3, 4, cell_type="conventional")
write_results_ineq_supercells(structs, output_dir=r"C:\Users\desde\Documents\GitHub\cmp-fe-al-phase-diagram\CIF Files\conventional_supercells\3x3x3\4Al")
