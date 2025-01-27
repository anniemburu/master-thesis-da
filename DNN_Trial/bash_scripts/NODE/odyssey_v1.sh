#!/bin/bash
#SBATCH --job-name=NODE_Odyssey_V1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mburu@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

CONFIGS=(
    config/brazillian_houses.yml
    config/abalone.yml
    #config/nyc_taxi.yml
    #config/house_sales.yml
    #config/mip_2016.yml
)

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training NODE Vesion 1 with Dataset: %s \n\n'  "$config" 
    printf "\n\n----------------------------------------------------------------------------\n"

    cd ~/Master_Thesis/master-thesis-da/DNN_Trial
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate Test4Node
    srun python3 train.py --config "$config" --model_name NODE --optimize_hyperparameters --n_trials 40 --epochs 100 --batch_size 64 --val_batch_size 128

done
