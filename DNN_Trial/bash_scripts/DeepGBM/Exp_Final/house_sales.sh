#!/bin/bash
#SBATCH --job-name=DeepGBM_house_sales_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TabSurvey
srun python3 train.py --config config/house_sales.yml --model_name DeepGBM --optimize_hyperparameters --batch_size 64 --val_batch_size 128
