#!/bin/bash
#SBATCH --job-name=train3d
#SBATCH --nodes=1
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=96:00:00
#SBATCH --output=/data/horse/ws/rabo074f-team_project/BYU2/logs/train_%j.out
#SBATCH --error=/data/horse/ws/rabo074f-team_project/BYU2/logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ramy_magdy_lamei.boulos@mailbox.tu-dresden.de
#SBATCH --account=p_scads

# -------------------------
# Setup environment
# -------------------------
source ~/.bashrc
module load Python/3.11.5
export PATH="$HOME/.local/bin:$PATH"

# -------------------------
# Navigate to project dir
# -------------------------
cd /data/horse/ws/rabo074f-team_project/BYU2

# -------------------------
# Run training
# -------------------------
poetry run python train_validate_motor_only.py