#!/bin/bash
#SBATCH --job-name=TabNet_BF_BR2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


cd ~/Master_Thesis/master-thesis-da/DNN_Trial
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TabSurvey
srun python3 train.py --config config/black_friday.yml --model_name TabNet --optimize_hyperparameters --n_trials 18 --epochs 100 --batch_size 64 --val_batch_size 128

