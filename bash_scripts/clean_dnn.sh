#!/bin/bash
#SBATCH --job-name=dnn_cln
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/cleaning
source activate MasterThesis
srun python3 cleaning_dnn.py