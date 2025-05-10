#!/bin/bash
#SBATCH --job-name=FTTransformer_BF_reg
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUDL
#SBATCH --gres=gpu:1
#SBATCH --account=long


cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TabSurvey
srun python3 train.py --config config/black_friday.yml --model_name FTTransformer --objective regression --optimize_hyperparameters --n_trials 5 --epochs 100 --batch_size 8 --val_batch_size 32

