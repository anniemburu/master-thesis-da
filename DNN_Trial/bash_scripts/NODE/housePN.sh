#!/bin/bash
#SBATCH --job-name=NODE_HPN_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Test4Node
srun python3 train.py --config config/house_prices_nominal.yml --model_name NODE --optimize_hyperparameters --n_trials 50 --epochs 100 --batch_size 32 --val_batch_size 64

