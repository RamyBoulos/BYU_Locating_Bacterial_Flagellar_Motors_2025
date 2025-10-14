#!/bin/bash
#SBATCH --nodes=1              # request 1 node
#SBATCH --cpus-per-task=6      # use 6 threads per task
#SBATCH --gres=gpu:1          # request 1 GPU per node
#SBATCH --time=01:00:00 
#SBATCH --mem=32GB             
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kerem.inci@mailbox.tu-dresden.de
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=train.py



cd /home/h4/kein254g/BYU_Locating_Bacterial_Flagellar_Motors_2025
source .venv/bin/activate
mkdir -p logs

# Your entry point (example)
srun python3 -m src.inference

# submission of job
# sbatch ./jobs/train_gpu.sbatch 
