CONFIGS=(
    ../../config/abalone.yml
    ../../config/allstate.yml
    ../../config/black_friday.yml
    ../../config/boston.yml
    ../../config/brazillian_houses.yml
    ../../config/diamonds.yml
    ../../config/house_prices_nominal.yml
    ../../config/house_sales.yml
    ../../config/mercedes_benz.yml
    ../../config/mip_2016.yml
    ../../config/moneyball.yml
    ../../config/sat11.yml
    ../../config/sensory.yml
    ../../config/socmob.yml
)

for config in "${CONFIGS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training XGBoost Vesion Remote with Dataset: %s \n\n'  "$config"
    printf 'Current Directory: %s \n\n' "$(pwd)"
    printf "\n\n----------------------------------------------------------------------------\n"

    cd /Users/johnmburu/Desktop/Master\ Thesis/master-thesis-da/DNN_Trial/
    printf 'Current Directory: %s \n\n' "$(pwd)"
    #printf 'Current Directory: %s \n\n' "$(pwd)"
    #source ~/miniconda3/etc/profile.d/conda.sh
    #conda activate TabSurvey
    #python3 /Users/johnmburu/Desktop/Master\ Thesis/master-thesis-da/DNN_Trial/train.py --config "$config" --model_name XGBoost --optimize_hyperparameters --objective regression --n_trials 5  --epochs 100 
    python3 train.py --config "$config" --model_name XGBoost --objective regression --epochs 100 > output_tests.log 2>&1
done