#!/bin/bash
#SBATCH --job-name=torus_sampling
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/output_files/ips_sampling/ips_sampling_torus_%A_%a.log
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

REGION_LIST=(1 2 3 5 11 19)

NUM_REGIONS=${REGION_LIST[$SLURM_ARRAY_TASK_ID]}
POINT_NUM=1000
OUTPUT_DIR="/home/habjan.e/CY_metric/data/ips_mathematica_output/torus/torus_reg${NUM_REGIONS}/"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "NUM_REGIONS = ${NUM_REGIONS}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"

module load mathematica/14.1.0
WolframKernel -script /home/habjan.e/CY_metric/ips_sampling/IPS/run_IPS_scripts/run_torus.m \
  --output-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}" \
  --total-points "${POINT_NUM}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate IPSenv

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/cymetric_compatibility.py \
  --input-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}"