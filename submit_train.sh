#!/bin/bash
#SBATCH --job-name=train3d
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --output=/data/horse/ws/rabo074f-team_project/logs/train_%j.out
#SBATCH --error=/data/horse/ws/rabo074f-team_project/logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ramy_magdy_lamei.boulos@mailbox.tu-dresden.de
#SBATCH --account=p_scads

# Activate the environment
source ~/.bashrc
module load Python/3.11.5
export PATH="$HOME/.local/bin:$PATH"

cd /data/horse/ws/rabo074f-team_project/BYU_Locating_Bacterial_Flagellar_Motors_2025

# Run the training script with poetry (training job)
PYTHONPATH=. poetry run python src/modeling_3d/train.py --patch_size 32 64 64 --epochs 1