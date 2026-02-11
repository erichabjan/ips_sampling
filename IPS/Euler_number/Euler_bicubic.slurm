#!/bin/bash
#SBATCH --job-name=Euler_bicubic
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/output_files/Euler_number/Euler_number_bicubic_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate IPSenv

python3 /home/habjan.e/CY_metric/ips_sampling/IPS/Euler_number/Euler_bicubic.py
