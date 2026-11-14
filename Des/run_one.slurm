#!/bin/bash
#SBATCH --job-name=relax_cifs_FeAl
#SBATCH --account=ap_course_2400wetcmp
#SBATCH --partition=zen2
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=1-12
#SBATCH --output=%x.%A_%a.out

module purge
module load calcua/2025a
module load SciPy-bundle/2025.06-gfbf-2025a
module load ASE/3.25.0-gfbf-2025a
module load matplotlib/3.10.3-gfbf-2025a
module load QuantumESPRESSO/7.5-foss-2025a
source $VSC_DATA/venvs/feal/bin/activate

export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export CMP_ROOT=$VSC_SCRATCH
export PW_COMMAND="srun pw.x -nk 4"      # 16 MPI ranks, 4 k-point pools

cd "$CMP_ROOT"

INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cif_list.txt)
if [ -z "$INPUT_FILE" ]; then
    echo "No CIF found for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
OUTPUT_DIR="${INPUT_FILE%.cif}"

python hpc_run_one.py --cif "$INPUT_FILE" --output_dir "$OUTPUT_DIR"