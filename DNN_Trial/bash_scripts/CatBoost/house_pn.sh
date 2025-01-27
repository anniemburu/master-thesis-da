#!/bin/bash
#SBATCH --job-name=CatBoost_HPN_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TabSurvey2
srun python3 train.py --config config/house_prices_nominal.yml --model_name CatBoost --optimize_hyperparameters --n_trials 100  --epochs 100
