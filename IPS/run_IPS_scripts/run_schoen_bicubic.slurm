#!/bin/bash
#SBATCH --job-name=schoen_bicubic_sampling
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/output_files/ips_sampling/ips_sampling_schoen_bicubic_%A_%a.log
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=20G
#SBATCH --partition=short

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2
export CUDA_VISIBLE_DEVICES=""

REGION_LIST=(1 3 5 11)

NUM_REGIONS=${REGION_LIST[$SLURM_ARRAY_TASK_ID]}
POINT_NUM=20000
OUTPUT_DIR="/home/habjan.e/CY_metric/data/ips_mathematica_output/schoen_bicubic/schoen_bicubic_reg${NUM_REGIONS}/"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "NUM_REGIONS = ${NUM_REGIONS}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"

module load mathematica/14.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate IPSenv

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/run_IPS_scripts/run_schoen_bicubic.py \
  --output-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}" \
  --total-points "${POINT_NUM}"

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/Euler_number.py \
  --input-dir "${OUTPUT_DIR}"
