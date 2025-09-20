#!/bin/bash
#SBATCH --job-name=XGBoost_Mercedes_final_reg
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=wekesa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TabSurvey
srun python3 train.py --config config/mercedes_benz.yml --model_name XGBoost --objective regression --optimize_hyperparameters --epochs 100
