#!/bin/bash
#SBATCH --job-name=SAINT_Odyssey_final_reg
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUDL
#SBATCH --gres=gpu:1
#SBATCH --account=long

CONFIGS=(
    config/abalone.yml
    config/house_sales.yml
    config/mip_2016.yml
)

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training SAINT Vesion 1 with Dataset: %s \n\n'  "$config" 
    printf "\n\n----------------------------------------------------------------------------\n"

    cd ~/Master_Thesis/master-thesis-da/DNN_Trial
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate TabSurvey
    srun python3 train.py --config "$config" --model_name SAINT --objective regression --optimize_hyperparameters --epochs 100 --batch_size 32 --val_batch_size 64
done
