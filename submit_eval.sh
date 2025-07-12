#!/bin/bash
#SBATCH --job-name=eval3d
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/data/horse/ws/rabo074f-team_project/logs/eval_%j.out
#SBATCH --error=/data/horse/ws/rabo074f-team_project/logs/eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ramy_magdy_lamei.boulos@mailbox.tu-dresden.de
#SBATCH --account=p_scads

# Activate the environment
source ~/.bashrc
module load Python/3.11.5
export PATH="$HOME/.local/bin:$PATH"

cd /data/horse/ws/rabo074f-team_project/BYU_Locating_Bacterial_Flagellar_Motors_2025

# Run the evaluation script with poetry
PYTHONPATH=. poetry run python src/modeling_3d/eval.py --checkpoint checkpoints/unet3d_epoch_5.pth