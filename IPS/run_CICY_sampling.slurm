#!/bin/bash
#SBATCH --job-name=cicy_ips_sampling
#SBATCH --output=/home/habjan.e/CY_metric/sandbox_notebooks/slurm_logs/cicy_ips_%A_%a.log
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G

#SBATCH -p reservation
#SBATCH --reservation=m_testing

module load mathematica/14.1.0

# Print module information
echo "Loaded modules:"
module list

# Print Mathematica version
echo "Mathematica version:"
math -version

# Set number of parallel kernels
export MATHEMATICA_KERNELS=${SLURM_CPUS_PER_TASK}

cd /home/habjan.e/CY_metric/ips_sampling

echo "Checking for required files:"
ls -lh PointGeneratorMathematicaCICYIPS.m run_CICY_IPS.m

# Run the Mathematica script
echo "Starting CICY IPS sampling..."
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

if command -v wolframscript &> /dev/null; then
    echo "Using wolframscript..."
    wolframscript -file run_CICY_IPS.m
else
    echo "Using math -script..."
    math -script run_CICY_IPS.m
fi

echo "End time: $(date)"
echo "The code has finished running"