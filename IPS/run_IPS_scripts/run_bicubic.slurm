#!/bin/bash
#SBATCH --job-name=bicubic_sampling
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/output_files/ips_sampling/ips_sampling_bicubic_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

NUM_REGIONS=1
POINT_NUM=1000
OUTPUT_DIR="/home/habjan.e/CY_metric/data/ips_mathematica_output/bicubic/bicubic_reg${NUM_REGIONS}/"

module load mathematica/14.1.0
WolframKernel -script /home/habjan.e/CY_metric/ips_sampling/IPS/run_IPS_scripts/run_bicubic.m \
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
