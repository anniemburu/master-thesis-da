#!/bin/bash
#SBATCH --job-name=wrn_drv
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/wrangling
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis
srun python3 wrangle_drive.py
