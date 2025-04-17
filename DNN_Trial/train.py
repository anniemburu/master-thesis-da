import logging
import sys
import numpy as np
import pandas as pd
import copy

import optuna 

from models import str2model
from utils.load_data import load_data
from utils.data_encoding import encoding
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import update_yaml, save_results_to_file, save_hyperparameters_to_file, save_loss_to_file, get_output_path, save_regularization_to_file
from utils.parser import get_parser, get_given_parameters_parser
from utils.visualization import loss_vizualization
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split
import warnings
warnings.filterwarnings("ignore")


def cross_validation(model, X, y, args, visual=False, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.frequency_reg:
        #Need to Clean here
        X,y,frequency_map = encoding(args, X, y)
    else:
        #print("Doing encoding : WE ARE IN TRAIN.PY")
        X,y = encoding(args, X, y)

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary" or args.objective == "probabilistic_regression":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")


    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Fold {i+1}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #print("Before encoding...")
        #print(X_train[:5,:])
        #print(X_test[:5,:])

        #print("After encoding : : WE ARE IN TRAIN.PY")
        #Check Valuesprint("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        """print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"num_features :{args.num_features}")
        print(f"num_classes : {args.num_classes}")
        print(f"cat_idx : {args.cat_idx}")
        print(f"nominal_idx : {args.nominal_idx}")
        print(f"ordinal_idx : {args.ordinal_idx}")
        print(f"num_idx : {args.num_idx}")
        print(f"cat_dims : {args.cat_dims}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")"""
        
        

        # Create a new unfitted version of the model
        curr_model = model.clone()
        print(curr_model.params)

        # Train model
        train_timer.start()
        if args.frequency_reg: ## For frequency regularization
            loss_history, val_loss_history, lambda_reg_history = curr_model.fit(X_train, y_train, X_test, y_test, frequency_map)
        else:
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
        if save_model and args.model_name != "TabPFN":
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)
            if args.frequency_reg:
                save_regularization_to_file(args, lambda_reg_history, "lambda_reg", extension=i)
            print('Saved Losses and Regularization')
        
        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± \n")
        print("B4 Evaluation")
        print(f"Number of classes : {args.num_classes}")
        print(f"Class label len :{len(args.bin_alt)}")
        print(f"Class labels : {args.bin_alt}")
        print(f"Unique y_true : {len(np.unique(y_test))}")
        print(f"Unique train : {len(np.unique(y_train))}\n")
        print(f"Prediction shape : {curr_model.predictions.shape}")
        print(f"Probabilities shape : {curr_model.prediction_probabilities.shape} \n")
        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± \n")
        

        #y_test = bin_shifter(args,y_train,y_test)
        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities,labels=np.unique(y_train))
        print("After Evaluation")

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
        args_cp = copy.deepcopy(self.args)

        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, args_cp)

        # Create model
        model = self.model_name(trial_params, args_cp)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, args_cp, visual=False, save_model=False)#Dont save model during HPT

        save_hyperparameters_to_file(args_cp, trial_params, sc.get_results(), time) #saved after every trial
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

    print("I am in Main Once")

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    print("Almost Cross Validating")
    print(f"Model Name: {args.model_name}")
    if args.model_name == "TabPFN":
        sc, time = cross_validation(model, X, y, args, visual=False, save_model=True)
    else:
        sc, time = cross_validation(model, X, y, args, visual=True, save_model=True)
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
