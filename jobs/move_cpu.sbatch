#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=move.py
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kerem.inci@mailbox.tu-dresden.de
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/kein254g/BYU_Locating_Bacterial_Flagellar_Motors_2025
source .venv/bin/activate
mkdir -p logs
srun python3 -m src.move

# sbatch ./jobs/move_cpu.sbatch  for submission
# scancel JOBID  for cancellation