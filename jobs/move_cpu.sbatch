#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=move.py
#SBATCH --mail-type=start,end
#SBATCH --mail-user=kerem.inci@mailbox.tu-dresden.de

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
srun python3 -m src.move

# İstersen diğer setler arka arkaya:
# srun python -m src.move --dataset valid
# srun python -m src.move --dataset test
