#!/bin/bash
#SBATCH --job-name=RandomForest_Olympus_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

CONFIGS=("config/boston.yml"
        "config/socmob.yml"
        "config/sensory.yml"
        "config/moneyball.yml"
        "config/black_friday.yml"
          )

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training RandomForest Vesion 1 with Dataset: %s \n\n'  "$config" 
    printf "\n\n----------------------------------------------------------------------------\n"

    cd ~/Master_Thesis/master-thesis-da/DNN_Trial
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate TabSurvey
    srun python3 train.py --config "$config" --model_name RandomForest --optimize_hyperparameters
done



