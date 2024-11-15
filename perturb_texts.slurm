#!/bin/bash
#SBATCH --job-name=perturb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00 
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --output=perturb_texts.%j.log
module load GCCcore/12.3.0 && module load Python/3.11.3 && module load GCC/12.3.0 && module load OpenMPI/4.1.5 && module load PyTorch/2.1.2-CUDA-12.1.1 && source .venv/bin/activate && time python perturb_texts.py --dataset_file ai_xsum/falcon-7b_xsum.json --cache /scratch/user/dylanharden3/AITextDetector/dataset
