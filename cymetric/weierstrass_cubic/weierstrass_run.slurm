#!/bin/bash
#SBATCH --job-name=weierstrass_cubic
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/slurm_logs/weierstrass_cubic.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate cymetric

python3 /home/habjan.e/CY_metric/sandbox_notebooks/weierstrass_cubic/weierstrass_sampling.py