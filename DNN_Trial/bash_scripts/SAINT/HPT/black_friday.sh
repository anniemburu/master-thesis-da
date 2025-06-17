#!/bin/bash
#SBATCH --job-name=SAINT_black_friday_final_reg
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUDL
#SBATCH --gres=gpu:1
#SBATCH --account=long

cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TabSurvey

srun python3 train.py --config config/black_friday.yml --model_name SAINT --objective regression --optimize_hyperparameters --epochs 100 --batch_size 16 --val_batch_size 32




