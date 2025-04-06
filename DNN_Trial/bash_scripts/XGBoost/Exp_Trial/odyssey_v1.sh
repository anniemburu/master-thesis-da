#!/bin/bash
#SBATCH --job-name=XGBoost_Odyssey_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

CONFIGS=(
    config/brazillian_houses.yml
    config/house_sales.yml
)

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training XGBoost Vesion 1 with Dataset: %s \n\n'  "$config" 
    printf "\n\n----------------------------------------------------------------------------\n"

    cd ~/Master_Thesis/master-thesis-da/DNN_Trial
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate TabSurvey
    srun python3 train.py --config "$config" --model_name XGBoost 
done
