#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=generate_labels.py
#SBATCH --mail-type=start,end
#SBATCH --mail-user=kerem.inci@mailbox.tu-dresden.de
#SBATCH --output=logs/%x_%j.out   # stdout
#SBATCH --error=logs/%x_%j.err    # stderr

# OpenMP env (Python bunu thread sayısı olarak kullanacak)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Proje köküne git
cd /home/kein254g/BYU_Locating_Bacterial_Flagellar_Motors_2025

# Ortamı yükle (modül/venv; kendi cluster’ına göre değiştir)
# module load python/3.10
source .venv/bin/activate

# Log klasörü garanti
mkdir -p logs

# Çalıştır (dataset tipini parametreleştir)
# Örnek: train
srun python3 -m src.generate_labels

# İstersen diğer setler arka arkaya:
# srun python -m src.move --dataset valid
# srun python -m src.move --dataset test
