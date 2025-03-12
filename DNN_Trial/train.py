import logging
import sys
import numpy as np
import pandas as pd

import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import update_yaml, save_results_to_file, save_hyperparameters_to_file, save_loss_to_file, get_output_path
from utils.parser import get_parser, get_given_parameters_parser
from utils.visualization import loss_vizualization
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split
import warnings
warnings.filterwarnings("ignore")

#Calculates Freedman-Diaconis Rule
def freedman_diaconis(y):
    #calc IQR
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    #calc bin width
    n = len(y)
    bin_width = 2 * (iqr / (n ** (1/3)))

    #calc num of bins
    data_range = np.max(y) - np.min(y)
    num_bins = int(np.round(data_range / bin_width))

    return num_bins

# Sturges' Rule
def sturges(y): 
    n = len(y)
    num_bins = 1 + int(np.log2(n))

    return num_bins

def bin_finder(args, y):
    if args.y_distribution == "normal" :
        bins = sturges(y)
    elif args.y_distribution == "skewed" or args.y_distribution == "bimodial":
        bins = freedman_diaconis(y)
    else:
        raise NotImplementedError("Distribution" + args.y_distribution + "is not yet implemented.")

    return bins

def cross_validation(model, X, y, args, visual=False, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression" or args.objective == "probabilistic_regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Bin the target variable
        if args.objective == "probabilistic_regression":
            args.num_bins = bin_finder(args, y_train)

            if args.y_distribution == "bimodial":
                strategy = 'kmeans'
            else:
                strategy = 'quantile'
            
            binning = KBinsDiscretizer(n_bins=args.num_bins, encode='ordinal', strategy=strategy)
            y_train = binning.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = binning.transform(y_test.reshape(-1, 1)).flatten()
            args.num_classes = args.num_bins

            y_train = y_train.astype(int)  # For NumPy arrays
            y_test = y_test.astype(int)
        

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
        
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)

        #print(f"Probabilities: {curr_model.prediction_probabilities}")
        

        #save the losses
        print(f"State of save is {save_model} b4 loss saving")
        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)
            print('Saved Losses')
            
        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(f'{sc.get_results()} \n \n')
    # Best run is saved to file
    if save_model:
        print("Saving model.....")
        print("Results After CV:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    print("Finished cross validation")

    #visualization

    if visual:
        losses = losses_history(args)
        loss_vizualization(args, losses)

    #print(get_output_path(args, filename="logging", file_type = None))
    return sc, (train_timer.get_average_time(), test_timer.get_average_time())

def losses_history(args):
    path = get_output_path(args, filename="", directory='logging',file_type = None)
    print(f"Loss path :{path}")
    folds = 5
    loss_dict = {
        'train' : [],
        'val' : []
    }

    for i in np.arange(folds):
        loss_path = path + f'loss_{i}.txt' #changed this
        val_loss_path = path + f'val_loss_{i}.txt' #changed this

        loss_file = np.loadtxt(loss_path)
        val_loss_file = np.loadtxt(val_loss_path)

        loss_dict['train'].append(list(loss_file))
        loss_dict['val'].append(list(val_loss_file))


    return loss_dict


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args, visual=False, save_model=False)#Dont save model during HPT

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time) #saved after every trial
        print(f"Hyperparam was saved!!! Hurrah!!!")

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction=args.direction, #changed this
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    print("Best parameters After Trials:", study.best_trial.params)

    ##Save the best parameters
    update_yaml(args.dataset, args.model_name, study.best_trial.params)
    print("Parameters saved to YAML file!!!")

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)
    cross_validation(model, X, y, args, visual=True, save_model=True)
    


def main_once(args):
    print("Train model with given hyperparameters")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    sc, time = cross_validation(model, X, y, args, visual=True)
    print(sc.get_results())
    print(time)




if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        main_once(arguments)
