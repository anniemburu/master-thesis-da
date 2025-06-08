import logging
import sys
import numpy as np
import random
import pandas as pd
import copy
import time
import torch
import yaml
import os

import optuna 

from models import str2model
from utils.load_data import load_data
from utils.data_encoding import encoding
from utils.scorer import get_scorer , RegScorer
from utils.timer import Timer
from utils.io_utils import update_yaml, save_results_to_file, save_matrix_to_file, save_arrays_to_file, save_hyperparameters_to_file, save_hyperparameters_to_file_inner,save_loss_to_file, get_output_path, save_regularization_to_file
from utils.parser import get_parser, get_given_parameters_parser
from utils.visualization import loss_vizualization
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import resample

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")
#path ="/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets"

#files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

#print("Files in folder:", files)

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

def bin_finder(args,y):
    #args.binning = "sturges"

    if args.binning == "freedman":
        bins = freedman_diaconis(y)
    elif args.binning == "sturges":
        bins = sturges(y)

    return bins

def impute_missing_test(train,test):
    missing_arr = [x for x in np.unique(test) if x not in np.unique(train)] #missing vals

    
    if len(missing_arr) > 0: #there is missing array
        results = []
        for x in missing_arr:
            gr_array = [a for a in np.unique(train) if a > x]
            if len(gr_array) > 0:
                rep_val = min(gr_array)
            else:
                ls_array = [a for a in np.unique(train) if a < x]
                rep_val = max(ls_array)
            results.append(rep_val)

        for old, new in zip(missing_arr,results):
            test = np.where(test == old, new, test)
        
        return test

    else:
        return test
    
def binning(args, y, y_val):
    #Bin the target variable
    if args.objective == "probabilistic_regression":
        args.num_bins = bin_finder(args, y)

        """if args.y_distribution == "bimodial":
            strategy = 'kmeans'
        else:
            strategy = 'quantile'"""
         # Use quantile binning for all cases
        
        binning = KBinsDiscretizer(n_bins=args.num_bins, encode='ordinal', strategy=args.strategy, subsample=200000)
        y = binning.fit_transform(y.reshape(-1, 1)).flatten()
        y_val = binning.transform(y_val.reshape(-1, 1)).flatten()

        if args.num_bins < 3:
            print("Make Multiclass")
            args.num_bins += args.num_bins + 1

        args.num_classes = args.num_bins

        print(f"Number of bins: {args.num_bins}")
        print(f"Number of Classes B4 Bin Verifier: {args.num_classes}")
        print(f"Unique values in y: {np.unique(y), len(np.unique(y))}")
        print(f"Unique values in y_val: {np.unique(y_val), len(np.unique(y_val))}")

        y = y.astype(int)  # For NumPy arrays
        y_val = y_val.astype(int)

        #Rectify bin
        y, y_val = bin_shifter(args, y, y_val)

        #bin_edges = binning.bin_edges_[0]


        print("VERIFY SHIFT")
        #print(f"Bin Edges: {bin_edges}") 
        print(f"Y shape : {y.shape} , Y_val shape : {y_val.shape}")
        print(f"Train after shift : {np.unique(y)}, Length : {len(np.unique(y))}")
        print(f"Number of Classes After Bin Verifier: {args.num_classes} \n\n")

    return y , y_val #, bin_edges

def bin_shifter(args, y_train, y_val):
    """
    Shifts class labels so that they are contiguous (without gaps).
    """

    y_val = impute_missing_test(y_train,y_val) #missing y classes
    
    def get_contiguous_labels(arr):
        """ Renumber labels to remove gaps """
        unique_vals = np.unique(arr)
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_vals)}
        return np.vectorize(mapping.get)(arr), mapping

    # Get contiguous labels
    comb = np.unique(np.concatenate([y_train, y_val]))
    comb_len = len(comb)

    if comb_len != args.num_bins:
        print("WE ARE IN THE GUTTERS!!!!!")
        y_train_shift, train_mapping = get_contiguous_labels(y_train)
        y_val_shift = np.vectorize(train_mapping.get)(y_val)  # Apply same mapping to test

        # Update arguments
        args.num_classes = len(np.unique(y_train_shift))  # Set correct number of classes
        args.bin_alt = sorted(list(np.unique(y_train_shift)))  # Ensure proper bin numbering

        print(f"Final Train Labels Length: {len(np.unique(y_train_shift))}")
        print(f"Final Test Labels Length: {len(np.unique(y_val_shift))}")
        print(f"Final Num Classes: {args.num_classes}")
        print(f"Final Bin Labels: {args.bin_alt}")

        return y_train_shift, y_val_shift

    else:
        print("No need to shift labels.")
        args.bin_alt = [x for x in range(args.num_bins)]
        return y_train, y_val

def binning(args, y, y_val):
    #Bin the target variable
    print(" WE ARE IN BINNING FUNCTION")
    
    args.num_bins = bin_finder(args, y)

    """if args.y_distribution == "bimodial":
        strategy = 'kmeans'
    else:
        strategy = 'quantile'"""
        # Use quantile binning for all cases
    
    binning = KBinsDiscretizer(n_bins=args.num_bins, encode='ordinal', strategy=args.strategy, subsample=200000)
    y = binning.fit_transform(y.reshape(-1, 1)).flatten()
    y_val = binning.transform(y_val.reshape(-1, 1)).flatten()

    if args.num_bins < 3:
        print("Make Multiclass")
        args.num_bins += args.num_bins + 1

    args.num_classes = args.num_bins

    print(f"Number of bins: {args.num_bins}")
    print(f"Number of Classes B4 Bin Verifier: {args.num_classes}")
    print(f"Unique values in y: {np.unique(y), len(np.unique(y))}")
    print(f"Unique values in y_val: {np.unique(y_val), len(np.unique(y_val))}")

    y = y.astype(int)  # For NumPy arrays
    y_val = y_val.astype(int)

    #Rectify bin
    y, y_val = bin_shifter(args, y, y_val)

    #bin_edges = binning.bin_edges_[0]


    print("VERIFY SHIFT")
    #print(f"Bin Edges: {bin_edges}") 
    print(f"Y shape : {y.shape} , Y_val shape : {y_val.shape}")
    print(f"Train after shift : {np.unique(y)}, Length : {len(np.unique(y))}")
    print(f"Number of Classes After Bin Verifier: {args.num_classes} \n\n")

    return y , y_val #, bin_edges

def count_classes(model_name, parameters, X_train, y_train, X_test, y_test, args, visual=False, save_model=True):
    """
    Count the number of unique classes in the target variable.
    """

    X_train_original = X_train.copy()
    y_train_original = y_train.copy()
    X_test_original = X_test.copy()
    y_test_original = y_test.copy()

    print("B4 encoding")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"num_features :{args.num_features}")
    print(f"num_classes : {args.num_classes}")
    print(f"cat_idx : {args.cat_idx}")
    print(f"nominal_idx : {args.nominal_idx}")
    print(f"ordinal_idx : {args.ordinal_idx}")
    print(f"num_idx : {args.num_idx}")
    print(f"cat_dims : {args.cat_dims}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

    X_train, y_train, X_test, y_test = encoding(args, X_train, y_train, X_test, y_test)

    print("After encoding : : WE ARE IN TRAIN.PY")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"num_features :{args.num_features}")
    print(f"num_classes : {args.num_classes}")
    print(f"cat_idx : {args.cat_idx}")
    print(f"nominal_idx : {args.nominal_idx}")
    print(f"ordinal_idx : {args.ordinal_idx}")
    print(f"num_idx : {args.num_idx}")
    print(f"cat_dims : {args.cat_dims}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")


    y_train_class, y_test_class = binning(args, y_train, y_test)

    print("After Binning : ")

    print(f"Dataset: {args.dataset} , Bins: {args.binning}, Train Classes: {np.unique(y_train_class)}, Test Classes: {np.unique(y_test_class)}")

    with open("bin_finder.txt", "a") as f:
        f.write(
            f"Dataset: {args.dataset} , Bins: {args.binning}, "
            f"Train Classes: {len(np.unique(y_train_class))}, "
            f"Test Classes: {len(np.unique(y_test_class))}\n"
        )
    print("ALL DONE!!!!")


def main(args):
    """
    Main function to run the binning process.
    """
    print("--- Running Nested Cross-Validation with Predefined Hyperparameters ---")
    # Load data - should we load test data here? Nested CV typically uses the *full* dataset (X, y)
    # and the outer folds act as the test sets. If you have a separate final holdout set,
    # that would be used *after* nested CV determines the best approach/model.
    # Let's assume we use the main dataset X, y for nested CV.

    print(f"Dataset: {args.dataset}, Model: {args.model_name}, Objective: {args.objective}")

    X, X_test, y, y_test = load_data(args, is_test=True) # Use the main dataset

    model_name = str2model(args.model_name)
    #print(f"Params for {args.model_name}: {args.parameters}")

    parameters = args.parameters[args.dataset][args.model_name]
    
    #model_cls = model_name (parameters, args)
    #print(f"Model Params After fitting: {model_cls.params}")

    args.save_results = True # Ensure results are saved

    # Check if parameters are defined
    """if not hasattr(args, 'parameters') or \
       args.dataset not in args.parameters or \
       args.model_name not in args.parameters[args.dataset]:
        print(f"ERROR: Predefined parameters not found for dataset '{args.dataset}' and model '{args.model_name}' in config.")
        return
  """
    
    print("Started Classification Module ...... ")
    print(f"Parameters Used HERE: {parameters}")
    count_classes(model_name, parameters, X, y, X_test, y_test, args, visual=False, save_model=True)

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Make sure to add outer_splits argument
    base_parser = get_parser() # Get your base parser
    base_parser.add_argument('--outer_splits', type=int, default=3, help='Number of outer folds for nested cross-validation')
    base_parser.add_argument('--save_results', action='store_true', default=True, help='Save aggregated results file') # Control saving final results
    base_parser.add_argument('--no_save_results', action='store_false', dest='save_results')

    # Initial parse to check mode
    temp_args, unknown = base_parser.parse_known_args()

    
    # Re-parse with the parser that includes predefined parameters
    parser = get_given_parameters_parser() # Assumes this parser loads yaml/dict params
    parser.add_argument('--outer_splits', type=int, default=3, help='Number of outer folds for nested cross-validation')
    parser.add_argument('--save_results', action='store_false', default=False, help='Save aggregated results file')
    parser.add_argument('--no_save_results', action='store_false', dest='save_results')
    
    arguments = parser.parse_args()
    print("Running Mode: Nested CV with Predefined Hyperparameters")
    print(f"Arguements: {arguments}")
    main(arguments)

