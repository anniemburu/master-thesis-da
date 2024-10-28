#!/bin/bash
#SBATCH --job-name=openml
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/wrangling
source activate MasterThesis
srun python3 data_download.py
