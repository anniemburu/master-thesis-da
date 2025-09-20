#!/bin/bash
#SBATCH --job-name=NODE_Diamonds_BR
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUDL
#SBATCH --gres=gpu:1
#SBATCH --account=long

cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Tab4Node
srun python3 train.py --config config/diamonds.yml --model_name NODE 


