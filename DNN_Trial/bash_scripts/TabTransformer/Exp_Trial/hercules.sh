#!/bin/bash
#SBATCH --job-name=TabTransformer_Hercules
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

CONFIGS=(
    config/abalone.yml
    config/allstate.yml
    config/black_friday.yml
    config/boston.yml
    config/brazillian_houses.yml
    #config/diamonds.yml
    config/house_prices_nominal.yml
    config/house_sales.yml
    config/mercedes_benz.yml
    config/mip_2016.yml
    config/moneyball.yml
    config/sat11.yml
    config/sensory.yml
    config/socmob.yml
)

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training VIME Vesion 1 with Dataset: %s \n\n'  "$config" 
    printf "\n\n----------------------------------------------------------------------------\n"

    cd ~/Master_Thesis/master-thesis-da/DNN_Trial
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate TabSurvey
    srun python3 train.py --config "$config" --model_name TabTransformer --objective probabilistic_regression

done
