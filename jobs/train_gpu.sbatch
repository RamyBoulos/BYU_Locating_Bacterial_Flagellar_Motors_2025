#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00                # was 01:00:00; give training room
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kerem.inci@mailbox.tu-dresden.de
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /home/h4/kein254g/BYU_Locating_Bacterial_Flagellar_Motors_2025
source .venv/bin/activate
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Your entry point (example)
srun python3 -m src.train
