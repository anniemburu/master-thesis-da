#!/bin/bash
#SBATCH --job-name=XGBoost_Titans_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

CONFIGS=(
    "config/sat11.yml"
    "config/diamonds.yml"
    "config/house_prices_nominal.yml"
    "config/mercedes_benz.yml"
    "config/allstate.yml")

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training XGBoost Vesion 1 with Dataset: %s \n\n'  "$config" 
    printf "\n\n----------------------------------------------------------------------------\n"

    cd ~/Master_Thesis/master-thesis-da/DNN_Trial
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate TabSurvey
    srun python3 train.py --config "$config" --model_name XGBoost --optimize_hyperparameters
done



