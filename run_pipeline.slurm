#!/bin/bash

#SBATCH --job-name=track2cllg
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=16G
#SBATCH --time=04:00:00

# move to your project directory
cd ~/Track2Cllg

# activate your environment
source ~/keyavenv/bin/activate

# optional check
python --version
nvidia-smi

# run your script
python run_pipeline.py

