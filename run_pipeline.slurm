#!/bin/bash

#SBATCH --job-name=track2cllg
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=32G
#SBATCH --time=04:00:00

# create logs directory
mkdir -p logs

# move to project directory
cd /WAVE/projects2/track2college/Track2Cllg

# activate virtual environment
source /WAVE/projects2/track2college/keyavenv/bin/activate

# debug info
echo "Running on $(hostname)"
which python
python --version
nvidia-smi

# run pipeline
python run_ingestion.py
