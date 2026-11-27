#!/bin/bash
#SBATCH --job-name=relax_pressures
#SBATCH --account=ap_course_2400wetcmp
#SBATCH --partition=zen2
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=des.deborger@student.uantwerpen.be
# NOTE: --array is set by submit_pressures.sh (0 .. N_POINTS-1)

# Usage (normally via submit_pressures.sh):
#   sbatch --array=0-$((N-1)) relax_pressures.sh CIF P_MAX [N_POINTS=8] [SETTINGS=low] [BULK_MODULUS_GPA]

CIF="$1"; P_MAX="$2"
N_POINTS="${3:-8}"
SETTINGS="${4:-low}"
BULK_MODULUS="$5"

if [ -z "$CIF" ] || [ -z "$P_MAX" ]; then
    echo "Usage: $0 CIF P_MAX [N_POINTS=8] [SETTINGS=low] [BULK_MODULUS_GPA]" >&2
    exit 1
fi

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
export PW_COMMAND="srun pw.x -nk 4"

cd "$CMP_ROOT"

OUTPUT_DIR="${CIF%.cif}"

EXTRA=()
if [ -n "$BULK_MODULUS" ]; then
    EXTRA+=(--bulk_modulus_guess "$BULK_MODULUS")
fi

python hpc_run_one_bulkmodulus.py \
    --cif "$CIF" \
    --output_dir "$OUTPUT_DIR" \
    --p_max "$P_MAX" --n_points "$N_POINTS" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --settings "$SETTINGS" \
    "${EXTRA[@]}"