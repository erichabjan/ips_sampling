#!/bin/bash
#SBATCH --job-name=quintic_1reg
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/output_files/1reg_output/ips_sampling_quintic_1reg_%A_%a.log
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=20G
#SBATCH --partition=short

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2
export CUDA_VISIBLE_DEVICES=""

REGION_LIST=(1 1 1 1 1)

NUM_REGIONS=${REGION_LIST[$SLURM_ARRAY_TASK_ID]}
POINT_NUM=10000
OUTPUT_DIR="/home/habjan.e/CY_metric/data/ips_mathematica_output/quintic/quintic_reg${NUM_REGIONS}_run${SLURM_ARRAY_TASK_ID}/"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "NUM_REGIONS = ${NUM_REGIONS}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"

module load mathematica/14.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate IPSenv

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/run_IPS_scripts/run_quintic.py \
  --output-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}" \
  --total-points "${POINT_NUM}"

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/Euler_number.py \
  --input-dir "${OUTPUT_DIR}"

### Run the exact same code but for the points sampled by cymetric
CYMETRIC_OUTPUT_DIR="/home/habjan.e/CY_metric/data/cymetric_output/quintic/quintic_reg${NUM_REGIONS}_run${SLURM_ARRAY_TASK_ID}/"
IPS_METADATA="${OUTPUT_DIR}/metadata.json"

echo "CYMETRIC_OUTPUT_DIR = ${CYMETRIC_OUTPUT_DIR}"

python3 /home/habjan.e/CY_metric/ips_sampling/cymetric/quintic/quintic_sampling.py \
  --ips-metadata "${IPS_METADATA}" \
  --output-dir "${CYMETRIC_OUTPUT_DIR}" \
  --num-points "${POINT_NUM}" \
  --num-regions "${NUM_REGIONS}"

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/Euler_number.py \
  --input-dir "${CYMETRIC_OUTPUT_DIR}"
