#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:30:00 
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=baselines.%j.log
module load GCCcore/12.3.0 && module load Python/3.11.3 && module load GCC/12.3.0 && module load OpenMPI/4.1.5 && module load PyTorch/2.1.2-CUDA-12.1.1 && source .venv/bin/activate && ./baselines_diff_models.sh
