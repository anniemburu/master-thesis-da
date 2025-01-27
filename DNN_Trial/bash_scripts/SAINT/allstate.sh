#!/bin/bash
#SBATCH --job-name=SAINT_Brazillian_V2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TabSurvey2
srun python3 train.py --config brazillian_houses --model_name SAINT --optimize_hyperparameters --n_trials 1 --epochs 100 --batch_size 64 --val_batch_size 128




