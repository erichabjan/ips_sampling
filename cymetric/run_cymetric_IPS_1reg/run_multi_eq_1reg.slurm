#!/bin/bash
#SBATCH --job-name=multi_eq_1reg
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/output_files/1reg_output/ips_sampling_multi_eq_1reg_%A_%a.log
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

REGION_LIST=(1 1 1 1 1)

NUM_REGIONS=${REGION_LIST[$SLURM_ARRAY_TASK_ID]}
POINT_NUM=10000
OUTPUT_DIR="/home/habjan.e/CY_metric/data/ips_mathematica_output/multi_eq/multi_eq_reg${NUM_REGIONS}_run${SLURM_ARRAY_TASK_ID}/"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "NUM_REGIONS = ${NUM_REGIONS}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"

module load mathematica/14.1.0
WolframKernel -script /home/habjan.e/CY_metric/ips_sampling/IPS/run_IPS_scripts/run_multi_eq.m \
  --output-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}" \
  --total-points "${POINT_NUM}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate IPSenv

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/cymetric_compatibility.py \
  --input-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}"

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/Euler_number.py \
  --input-dir "${OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}"

### Run the exact same code but for the points sampled by cymetric
CYMETRIC_OUTPUT_DIR="/home/habjan.e/CY_metric/data/cymetric_output/multi_eq/multi_eq_reg${NUM_REGIONS}_run${SLURM_ARRAY_TASK_ID}/"
IPS_METADATA="${OUTPUT_DIR}/metadata_${NUM_REGIONS}.json"

echo "CYMETRIC_OUTPUT_DIR = ${CYMETRIC_OUTPUT_DIR}"

python3 /home/habjan.e/CY_metric/ips_sampling/cymetric/multi_eq/multi_eq_sampling.py \
  --ips-metadata "${IPS_METADATA}" \
  --output-dir "${CYMETRIC_OUTPUT_DIR}" \
  --num-points "${POINT_NUM}" \
  --num-regions "${NUM_REGIONS}"

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/cymetric_compatibility.py \
  --input-dir "${CYMETRIC_OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}"

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/Euler_number.py \
  --input-dir "${CYMETRIC_OUTPUT_DIR}" \
  --num-regions "${NUM_REGIONS}"
