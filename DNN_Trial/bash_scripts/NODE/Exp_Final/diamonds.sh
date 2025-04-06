#!/bin/bash
#SBATCH --job-name=NODE_Diamonds_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Tab4Node
srun python3 train.py --config config/diamonds.yml --model_name NODE --optimize_hyperparameters --n_trials 20 --epochs 100 --batch_size 32 --val_batch_size 64


