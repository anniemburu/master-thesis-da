#!/bin/bash
#SBATCH --job-name=NODE_Mercedes_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Test4Node
srun python3 train.py --config config/mercedes_benz.yml --model_name NODE --optimize_hyperparameters --n_trials 45 --epochs 100

