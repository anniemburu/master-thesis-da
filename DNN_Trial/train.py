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

#Calculates Freedman-Diaconis Rule
def freedman_diaconis(y, args):
    #calc IQR
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    #calc bin width
    n = len(y) // args.num_splits
    bin_width = 2 * (iqr / (n ** (1/3)))

    #calc num of bins
    data_range = np.max(y) - np.min(y)
    num_bins = int(np.round(data_range / bin_width))

    return num_bins

# Sturges' Rule
def sturges(y,args): 
    n = len(y) // args.num_splits
    num_bins = 1 + int(np.log2(n))

    return num_bins

def bin_finder(args, y):
    if args.y_distribution == "normal" :
        bins = sturges(y, args)
    elif args.y_distribution == "skewed" or args.y_distribution == "bimodial":
        bins = freedman_diaconis(y,args)
    else:
        raise NotImplementedError("Distribution" + args.y_distribution + "is not yet implemented.")

    return bins

"""def bin_shifter(args, y_train, y_test):
    def find_earliest_gap(arr):
        expected = 0
        for num in arr:
            if num != expected:
                return expected
            expected += 1

        return expected
    
    def shift_array(arr, shift_value, max_diff):
        shifted_arr = []
        existing_values = set(arr)  # Track existing values to avoid duplicates

        for x in arr:
            if x > shift_value:
                shifted_arr.append(x - max_diff)
            else:
                shifted_arr.append(x)

        return np.array(shifted_arr)

    #train_unique, train_len = np.unique(y_train), len(np.unique(y_train))
    #test_unique, test_len = np.unique(y_test), len(np.unique(y_test))
    #comb = np.union1d(train_unique, test_unique)

    comb = np.unique(np.concatenate([y_train, y_test]))
    comb_len = len(comb)

    max_class = np.max(y_train)
    max_diff = np.abs(comb_len - max_class)

    if max_diff == 0: max_diff = 1 #shift by one if zero
 
    print(f"MAX : {max_class} , Max diff : {max_diff}, LEN : {comb_len} , BINS : {args.num_bins}")

    if comb_len != args.num_bins: #means some classes are missing
        print("WE ARE HERE")
        
        if comb_len == np.max(comb): # when the max class is the same as number of classes ##added or np.max(comb) > comb_len
            print("IN THE SHIIIIIT")
            #y_train = np.where(y_train == max_class, max_class - max_diff, y_train)
            #y_test  = np.where(y_test == max_class, max_class - max_diff, y_test)

            #find the gaps.. missing classes
            train_gap = find_earliest_gap(y_train)
            test_gap = find_earliest_gap(y_test)

            y_train_shift = shift_array(y_train, train_gap, max_diff)
            y_test_shift = shift_array(y_test, test_gap, max_diff)

            print(f"Length orig Train: {len(y_train_shift)}, Length shift : {len(y_train)}")
            print(f"Length orig Test: {len(y_test_shift)}, Length shift : {len(y_test)}")

            # Ensure same length
            if len(np.unique(y_train_shift)) != len(np.unique(y_train)):
                print("Warning: Length mismatch in y_train_shift!")

            if len(np.unique(y_test_shift)) != len(np.unique(y_test)):
                print("Warning: Length mismatch in y_test_shift!")


            args.num_classes = comb_len
            args.bin_alt = sorted(np.unique(np.concatenate([y_train_shift, y_test_shift])).tolist())

            #print(f"Train after shift : {np.unique(y_train)}")
            #print(f"Test after shift : {np.unique(y_test)}")
            #print(f"Num Classes after shift : {args.num_classes}")
            print(f"Bin alt after shift : {args.bin_alt} \n")
            print(f"Train after shift I : {np.unique(y_train_shift)},  Length : {len(np.unique(y_train_shift))}")
            print(f"Test after shift I : {np.unique(y_test_shift)}, Length : {len(np.unique(y_test_shift))} \n")

            return y_train_shift, y_test_shift
        
        elif comb_len < np.max(comb):
            print("IN THE SHIIIIIT II")
            #First shift
            train_gap = find_earliest_gap(y_train)
            test_gap = find_earliest_gap(y_test)

            y_train_shift = shift_array(y_train, train_gap, max_diff)
            y_test_shift = shift_array(y_test, test_gap, max_diff)

            #Second shift
            train_gap2 = find_earliest_gap(y_train_shift)
            test_gap2 = find_earliest_gap(y_test_shift)

            y_train_shift2 = shift_array(y_train, train_gap2, max_diff)
            y_test_shift2 = shift_array(y_test, test_gap2, max_diff)

            print(f"Train after shift II : {np.unique(y_train_shift2)},  Length : {len(np.unique(y_train_shift2))}")
            print(f"Test after shift II : {np.unique(y_test_shift2)}, Length : {len(np.unique(y_test_shift2))} \n")

            return y_train_shift2, y_test_shift2
        
        else:
            args.num_classes = comb_len
            args.bin_alt = [x for x in range(comb_len)]

            return y_train, y_test

    else:
        args.num_classes = comb_len
        args.bin_alt = [x for x in range(comb_len)]

        return y_train, y_test"""

def bin_shifter(args, y_train, y_test):
    """
    Shifts class labels so that they are contiguous (without gaps).
    """
    def get_contiguous_labels(arr):
        """ Renumber labels to remove gaps """
        unique_vals = np.unique(arr)
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_vals)}
        return np.vectorize(mapping.get)(arr), mapping

    # Get contiguous labels
    comb = np.unique(np.concatenate([y_train, y_test]))
    comb_len = len(comb)

    if comb_len != args.num_bins:
        print("WE ARE IN THE GUTTERS!!!!!")
        y_train_shift, train_mapping = get_contiguous_labels(y_train)
        y_test_shift = np.vectorize(train_mapping.get)(y_test)  # Apply same mapping to test

        # Update arguments
        args.num_classes = len(np.unique(y_train_shift))  # Set correct number of classes
        args.bin_alt = sorted(list(np.unique(y_train_shift)))  # Ensure proper bin numbering

        print(f"Final Train Labels: {np.unique(y_train_shift)}, Length: {len(np.unique(y_train_shift))}")
        print(f"Final Test Labels: {np.unique(y_test_shift)}, Length: {len(np.unique(y_test_shift))}")
        print(f"Final Num Classes: {args.num_classes}")
        print(f"Final Bin Labels: {args.bin_alt}")

        return y_train_shift, y_test_shift

    else:
        print("No need to shift labels.")
        args.bin_alt = [x for x in range(args.num_bins)]
        return y_train, y_test

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

    #temp hold
    num_features_temp = args.num_features
    num_classes_temp = args.num_classes
    cat_idx_temp = args.cat_idx
    nominal_idx_temp = args.nominal_idx
    ordinal_idx_temp = args.ordinal_idx
    num_idx_temp = args.num_idx
    cat_dims_temp = args.cat_dims
    bin_alt_temp = args.bin_alt

    args_temps = {
        'num_features' : num_features_temp,
        'num_classes' : num_classes_temp,
        'cat_idx' : cat_idx_temp,
        'nominal_idx' : nominal_idx_temp,
        'ordinal_idx' : ordinal_idx_temp,
        'num_idx' : num_idx_temp,
        'cat_dims' : cat_dims_temp,
        'bin_alt' :  bin_alt_temp
    }

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Fold {i+1}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #print("Before encoding...")
        #print(X_train[:5,:])
        #print(X_test[:5,:])

        #Check Values
        for key, value in args_temps.items():
            print(f"{key} : {value}")

        if args.frequency_reg:
            #Need to Clean here
            X_train,y_train,X_test,y_test,frequency_map = encoding(args, X_train, y_train, X_test, y_test, args_temps)
        else:
            #print("Doing encoding : WE ARE IN TRAIN.PY")
            X_train,y_train,X_test,y_test = encoding(args, X_train, y_train, X_test, y_test, args_temps)

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

            #print(f"Number of bins: {args.num_bins}")
            print(f"Number of Classes B4 Bin Verifier: {args.num_classes}")
            print(f"Unique values in y_train: {np.unique(y_train), len(np.unique(y_train))}")
            print(f"Unique values in y_test: {np.unique(y_test), len(np.unique(y_test))}")

            y_train = y_train.astype(int)  # For NumPy arrays
            y_test = y_test.astype(int)

            #Rectify bin
            y_train, y_test = bin_shifter(args, y_train, y_test)
            print("VERIFY SHIFT")
            print(f"Train after shift : {np.unique(y_train)}, Length : {len(np.unique(y_train))}")
            print(f"Test after shift : {np.unique(y_test)}, Length : {len(np.unique(y_test))}")
            print(f"Number of Classes After Bin Verifier: {args.num_classes}")
        

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

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
        if save_model:
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
        print(f"Unique y_true : {np.unique(y_test)} \n")
        print(f"Prediction shape : {curr_model.predictions.shape}")
        print(f"Probabilities shape : {curr_model.prediction_probabilities.shape} \n")
        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± \n")
        

        #y_test = bin_shifter(args,y_train,y_test)
        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)
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
