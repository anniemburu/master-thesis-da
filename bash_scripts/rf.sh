#!/bin/bash
#SBATCH --job-name=mult_rf_v1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/experiments
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis
srun python3 random_forest_exp.py
