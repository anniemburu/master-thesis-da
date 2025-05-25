import logging
import sys
import numpy as np
import random
import pandas as pd
import copy
import time
import torch
import yaml

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Calculates Freedman-Diaconis Rule
def freedman_diaconis(y, args):
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
def sturges(y,args): 
    n = len(y) 
    num_bins = 1 + int(np.log2(n))

    return num_bins

def bin_finder(args, y):
    """if args.y_distribution == "normal" :
        bins = sturges(y, args)
    elif args.y_distribution == "skewed" or args.y_distribution == "bimodial":
        bins = freedman_diaconis(y,args)
    else:
        raise NotImplementedError("Distribution" + args.y_distribution + "is not yet implemented.")
    """
    """if args.model_name == "NODE":
        #start with Sturges' Rule
        bins = sturges(y, args)
    else:"""
    
    #args.binning = "sturges"

    if args.binning == "freedman":
        bins = freedman_diaconis(y, args)
    elif args.binning == "sturges":
        bins = sturges(y,args)

    return bins

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

def custom_class_weights(y):
    """
    Calculate custom class weights based on the distribution of classes in y.
    This is useful for handling imbalanced datasets.
    """
    unique_classes = np.unique(y)
    unique_classes.sort()

    class_weights = compute_class_weight('balanced',classes = unique_classes,y = y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weights

# --- Bin Means --- *
def pred_map(train_pred, train_classes):

    #print(f"train_pred:{train_pred}")

    train_pred= np.array(train_pred)
    train_classes = np.array(train_classes)

    unique_classes = np.unique(train_classes)
    bin_mean = {}

    # Calculate mean for each class
    for cls in unique_classes:
        mask = (train_classes == cls)  # Boolean mask for current class
        bin_mean[cls] = train_pred[mask].mean()

    #print("Mean per class:")
    #for cls, mean_val in bin_mean.items():
        #print(f"Class {cls}: {mean_val:.3f}")

    bin_mean = {int(k): float(v) for k, v in bin_mean.items()}
    print(f"Bin Mean: {bin_mean}")

    return bin_mean

# --- Bin Mapping to Reg---
def reg_pred(y, reg_mapping):
    #reg_mapping = dict(reg_mapping)
    print(f"In reg pred func and reg_mapping is {type(reg_mapping)}")
    print(reg_mapping)
    return np.vectorize(reg_mapping.get)(y)

# --- Inner Loop: Hyperparameter Evaluation using Standard CV ---
# This function evaluates a *single* set of hyperparameters using standard k-fold CV
# It's called by the Optuna Objective function within each outer fold.
def evaluate_hyperparameters_cv(model_prototype, trial_params, X_train_outer, y_train_outer, args, fold_prefix="inner"):
    """
    Evaluates a given model prototype with fixed hyperparameters using k-fold CV
    on the provided training data (outer fold's training set).

    Args:
        model_prototype: An unfitted model instance with specific hyperparameters set.
        X_train_outer: Features for the outer loop's training set.
        y_train_outer: Target for the outer loop's training set.
        args: Configuration arguments (num_splits for inner CV, etc.).
        fold_prefix: String identifier for logging (e.g., "inner" or "outer_X_inner").

    Returns:
        Average score (objective metric) over the inner folds.
        Average train time.
        Average inference time.
    """
    ## Make a copyy
    args_temp = copy.deepcopy(args)

    inner_sc = get_scorer(args) # Use a fresh scorer for this inner evaluation

    inner_train_timer = Timer()
    inner_eval_timer = Timer()

    # Use args.num_splits for the *inner* cross-validation
    if args.objective == "regression" or args.objective == "probabilistic_regression":
        inner_kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        inner_kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError(f"Objective {args.objective} not implemented for inner CV.")
    
    #Keep copies of the original args
    # This is important to ensure that the original args are not modified during the inner loop
    
    fold_scores = [] # To calculate average score later

    for i, (train_inner_idx, val_inner_idx) in enumerate(inner_kf.split(X_train_outer, y_train_outer)):
        print(f"  {fold_prefix} Fold {i+1}/{args.num_splits}")

        # --- Create inner fold data ---
        X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
        y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]

        # --- Make a DEEP COPY of args for this specific fold to avoid state pollution ---
        # Encoding and binning can modify args (num_features, num_classes, cat_dims, etc.)
        args = copy.deepcopy(args_temp)

        # --- DEBUGG---
        print(f"Before Debugging -- Inner Fold {i+1}")  
        print(f"DEBUG Inner Fold {i+1}: num_features: {args.num_features}")
        print(f"DEBUG Inner Fold {i+1}: num_classes: {args.num_classes}")
        print(f"DEBUG Inner Fold {i+1}: cat_idx: {args.cat_idx}")
        print(f"DEBUG Inner Fold {i+1}: nominal_idx: {args.nominal_idx}")
        print(f"DEBUG Inner Fold {i+1}: ordinal_idx: {args.ordinal_idx}")
        print(f"DEBUG Inner Fold {i+1}: num_idx: {args.num_idx}")
        print(f"DEBUG Inner Fold {i+1}: cat_dims: {args.cat_dims}")
        print(f"DEBUG Inner Fold {i+1}: bin_alt: {args.bin_alt}")

        # --- Preprocessing: Fit on Inner Train, Transform Inner Train & Val ---
        # Important: Encoding and Binning fit *only* on X_train_inner, y_train_inner
        #try:
        if args.frequency_reg:
            X_train_inner_proc, y_train_inner_proc, X_val_inner_proc, y_val_inner_proc, frequency_map = encoding(
                args, X_train_inner, y_train_inner, X_val_inner, y_val_inner) # Pass original state for reference if needed by encoding
        else:
            X_train_inner_proc, y_train_inner_proc, X_val_inner_proc, y_val_inner_proc = encoding(
                args, X_train_inner, y_train_inner, X_val_inner, y_val_inner) # Pass original state
            
        # Binning
        if args.objective == "probabilistic_regression":
            y_train_inner_proc, y_val_inner_proc = binning(args, y_train_inner_proc, y_val_inner_proc) # Modifies args inplace

        """except Exception as e:
            print(f"ERROR during preprocessing in inner fold {i+1}: {e}")
            print(f"Skipping inner fold {i+1} due to preprocessing error.")
            # Decide how to handle: skip fold? return NaN? For now, skip.
            continue # Skip to next inner fold
       """
        try:
            class_weights = custom_class_weights(y_train_inner_proc)
        except Exception as e:
            print(f"Y unique : {np.unique(y_train_inner_proc)}")
            print(f"ERROR calculating class weights in inner fold {i+1}: {e}")
            print(f"Skipping inner fold {i+1} due to class weight calculation error.")
            continue

        print(f"After Debugging -- Inner Fold {i+1}") 
        print(f"DEBUG Inner Fold {i+1}: num_features: {args.num_features}")
        print(f"DEBUG Inner Fold {i+1}: num_classes: {args.num_classes}")
        print(f"DEBUG Inner Fold {i+1}: cat_idx: {args.cat_idx}")
        print(f"DEBUG Inner Fold {i+1}: nominal_idx: {args.nominal_idx}")
        print(f"DEBUG Inner Fold {i+1}: ordinal_idx: {args.ordinal_idx}")
        print(f"DEBUG Inner Fold {i+1}: num_idx: {args.num_idx}")
        print(f"DEBUG Inner Fold {i+1}: cat_dims: {args.cat_dims}")
        print(f"DEBUG Inner Fold {i+1}: bin_alt: {args.bin_alt}")

        # --- Model Training and Evaluation for the Inner Fold ---
        # Create a new model instance from the prototype FOR THIS FOLD
        # This ensures no state leaks between inner folds
        #curr_model = model_prototype.clone()
        try:
            curr_model = model_prototype(trial_params, args) # Use deepcopy to ensure a fresh instance
        except Exception as e:
            print(f"ERROR instantiating model in inner fold {i+1} AFTER preprocessing: {e}")
            print(f"Trial Params: {trial_params}")
            print(f"Args Fold State: num_features={args.num_features}, num_classes={args.num_classes}, cat_dims={args.cat_dims}")
            print("Skipping inner fold {i+1} due to model instantiation error.")
            continue # Skip to next inner fold

        # Train model
        inner_train_timer.start()
        #try:
        if args.frequency_reg:
            # We don't typically save loss/reg history during HPO CV
            _ = curr_model.fit(X_train_inner_proc, y_train_inner_proc, X_val_inner_proc, y_val_inner_proc, frequency_map)
        else:
    
            print("**** Data b4 Fitting ****")
            #print(f"X_train_inner_proc: {X_train_inner_proc[:10,:]}")
            #print(f"y_train_inner_proc: {y_train_inner_proc[:10]}")
            #print(f"X_val_inner_proc WHAT: {X_val_inner_proc[:10,:]}")
            #print(f"y_val_inner_proc: {y_val_inner_proc[:10]}")
            #print(f"Looking for Issue in XGB: {args.objective}")
            #print(f"Still running??")

            if args.weighted_loss:
                print(f"Class Weights Applied: {class_weights}")
                # Use class weights for loss function
                _,_ = curr_model.fit(X_train_inner_proc, y_train_inner_proc, X_val_inner_proc, y_val_inner_proc, class_weights = class_weights)
            else:
                print(f"Class Weights Ddnt Apply")
                _,_ = curr_model.fit(X_train_inner_proc, y_train_inner_proc, X_val_inner_proc, y_val_inner_proc)
    

        """except Exception as e:
             print(f"ERROR!!! during model fitting in inner fold {i+1}: {e}")
             print(f"The Model: {curr_model} , Model Type: {type(curr_model)}")
             print(f"Model params: {curr_model.params}")
             print(f"Args for fold: num_classes={args.num_classes}, bin_alt={args.bin_alt}")
             print(f"Train data shapes: X={X_train_inner_proc.shape}, y={y_train_inner_proc.shape}")
             print(f"Validation data shapes : X_val = {X_val_inner_proc.shape} , y_val shape = {y_val_inner_proc.shape}")
             print(f"Y type : {type(y_train_inner_proc)}, Y_val type : {type(y_val_inner_proc)}")
             print(f"Unique y: {np.unique(y_train_inner_proc)}")
             print(f"Skipping inner fold {i+1} due to fitting error.")
             inner_train_timer.end() # Still record time spent
             continue # Skip to next inner fold
    """
        inner_train_timer.end()

        # Test model
        inner_eval_timer.start()
        try:
            predictions = curr_model.predict(X_val_inner_proc) # Get predictions directly
            #probabilities = curr_model.prediction_probabilities # Assumes model stores probabilities
            #probababilities = curr_model.predict_proba(X_val_inner_proc) # Get probabilities
            #print(f"Predictions Tesst: {predictions}")
            #print(f"Prediction Probs : {curr_model.prediction_probabilities}")
        except Exception as e:
             print(f"ERROR during prediction in inner fold {i+1}: {e}")
             print(f"Skipping inner fold {i+1} due to prediction error.")
             inner_eval_timer.end()
             continue # Skip to next inner fold
        inner_eval_timer.end()

        # Evaluate predictions for this inner fold
        # Use the unique labels *from the inner training set* for evaluation if needed by metric
       
        try:
            if args.objective == "probabilistic_regression":   
                inner_sc.eval(y_val_inner_proc, predictions, curr_model.prediction_probabilities, labels=np.unique(y_train_inner_proc))
            else:
                inner_sc.eval(y_val_inner_proc, predictions, curr_model.prediction_probabilities)

            # Store the objective score for averaging later
            fold_scores.append(inner_sc.get_objective_result()) # Add score for this fold

            print(f"Inner Fold {i+1} Results: {inner_sc.get_results()}")
        
        except Exception as e:
            print(f"ERROR during evaluation in inner fold {i+1}: {e}")
            print(f"y_true shape: {y_val_inner_proc.shape}, unique: {np.unique(y_val_inner_proc)}")
            print(f"predictions shape: {predictions.shape}, unique: {np.unique(predictions)}")
            print(f"y_val_inner_proc shape: {y_val_inner_proc.shape}")
            print(f"predictions: {predictions[:10]}")
            if hasattr(curr_model, 'prediction_probabilities') and curr_model.prediction_probabilities is not None:
                print(f"probabilities: {curr_model.prediction_probabilities[:10]}")
            else:
                print("probabilities: None")
            # print(f"probabilities shape: {probabilities.shape}")
            print(f"eval labels: {np.unique(y_train_inner_proc)}")
            print(f"Skipping inner fold {i+1} due to evaluation error.")
            continue # Skip to next inner fold


        # --- Aggregate Inner CV Results ---
    if not fold_scores: # Handle case where all inner folds failed
        print(f"Warning: No inner folds completed successfully for HPO trial.")
        # Return a value indicating failure (e.g., NaN, +/- infinity depending on direction)
        return float('inf') if args.direction == 'minimize' else float('-inf'), 0, 0

    avg_score = np.mean(fold_scores)
    avg_train_time = inner_train_timer.get_average_time()
    avg_eval_time = inner_eval_timer.get_average_time()

    # print(f"  Finished Inner CV for HPO Trial. Avg Score: {avg_score:.4f}")

    return inner_sc, avg_score, (avg_train_time, avg_eval_time)

# --- Nested Cross-Validation Function ---
def nested_cross_validation(model_cls, X, y, args, optimize_params=True):
    """
    Performs nested cross-validation.

    Args:
        model_cls: The model class (e.g., SAINT, TabNetModel).
        X: Full feature dataset.
        y: Full target dataset.
        args: Configuration arguments.
        optimize_params (bool): If True, run Optuna (inner loop) to find params per outer fold.
                                If False, use predefined params from args.parameters.

    Returns:
        Aggregated scorer object with results across outer folds.
        Aggregated average train time across outer folds.
        Aggregated average inference time across outer folds.
    """
    ## Args Copy
    args_temp_nested = copy.deepcopy(args)

    print(f"--- Starting Nested Cross-Validation ({args.outer_splits} Outer Folds, {args.num_splits} Inner Folds) ---")
    print(f"Hyperparameter Optimization per Outer Fold: {optimize_params}")

    outer_sc = get_scorer(args) # Aggregated scorer for outer folds
    outer_train_times = []
    outer_test_times = []
    all_outer_fold_results = [] # Store detailed results per outer fold
    best_params_per_fold = [] # Store the best params found for each outer fold
    loss_per_fold = {} # Store the loss per outer fold


    if args.objective == "regression" or args.objective == "probabilistic_regression":
        outer_kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)

    elif args.objective == "classification" or args.objective == "binary":
        outer_kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)

    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")
    
    #Keep copies of the original args
    # This is important to ensure that the original args are not modified during the inner loop
    """ 
    original_full_args_state = {
            'num_features': copy.deepcopy(args.num_features),
            'num_classes': copy.deepcopy(args.num_classes),
            'cat_idx': copy.deepcopy(args.cat_idx),
            'nominal_idx': copy.deepcopy(args.nominal_idx),
            'ordinal_idx': copy.deepcopy(args.ordinal_idx),
            'num_idx': copy.deepcopy(args.num_idx),
            'cat_dims': copy.deepcopy(args.cat_dims),
            'bin_alt': copy.deepcopy(args.bin_alt)
        }
    """

    # --- Outer Loop Execution ---
    for i, (train_outer_idx, test_outer_idx) in enumerate(outer_kf.split(X, y)):
        print(f"\n--- Outer Fold {i+1}/{args.outer_splits} ---")

        # --- Create outer fold data ---
        X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
        y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

        # --- Make a DEEP COPY of args for this specific outer fold ---
        args = copy.deepcopy(args_temp_nested)

        best_params_for_fold = None
        inner_cv_time = 0 # Time spent in HPO for this fold

        #if optimize_params:
        print(f"Starting Hyperparameter Optimization for Outer Fold {i+1}...")
        inner_loop_start_time = time.time()

        # Use an in-memory study for each outer fold to keep HPO separate
        study = optuna.create_study(direction=args.direction,
                                    study_name=f"{args.model_name}_{args.dataset}_outer{i+1}",
                                    storage=None) # In-memory storage

        objective = Objective(args, model_cls, X_train_outer, y_train_outer, i)

        #try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1) # n_jobs=1 unless Objective and preprocessing are thread-safe
        best_params_for_fold = study.best_trial.params
        best_inner_score = study.best_value
        print(f"Best Hyperparameters found for Outer Fold {i+1}: {best_params_for_fold}")
        print(f"Best Inner CV Score : {best_inner_score:.4f}")

        """except Exception as e:
            print(f"ERROR during Optuna optimization for outer fold {i+1}: {e}")
            print("Skipping outer fold due to HPO error.")
            best_params_per_fold.append(None) # Record failure
            all_outer_fold_results.append(None)
            continue # Skip to the next outer fold
        """
        inner_cv_time = time.time() - inner_loop_start_time
        print(f"Hyperparameter Optimization for Outer Fold {i+1} finished in {inner_cv_time:.2f} seconds.")

        """ 
        else: # Use predefined hyperparameters
        print(f"Using predefined hyperparameters for Outer Fold {i+1}.")
        try:
            best_params_for_fold = args.parameters[args.dataset][args.model_name]
            print(f"Predefined Params: {best_params_for_fold}")
        except KeyError:
            print(f"ERROR: Predefined parameters not found for dataset '{args.dataset}' and model '{args.model_name}' in config.")
            print("Skipping outer fold.")
            best_params_per_fold.append(None) # Record failure
            all_outer_fold_results.append(None)
            continue # Skip to the next outer fold
        """
        best_params_per_fold.append(best_params_for_fold) # Store params for this fold

        

        # --- Preprocessing: Fit on Outer Train, Transform Outer Train & Outer Test ---
        # Reset args state potentially modified during HPO before preprocessing again
        # Use the clean args and the original state reference
        """args.num_features = copy.deepcopy(original_full_args_state['num_features'])
        args.num_classes = copy.deepcopy(original_full_args_state['num_classes'])
        args.cat_idx = copy.deepcopy(original_full_args_state['cat_idx'])
        args.nominal_idx = copy.deepcopy(original_full_args_state['nominal_idx'])
        args.ordinal_idx = copy.deepcopy(original_full_args_state['ordinal_idx'])
        args.num_idx = copy.deepcopy(original_full_args_state['num_idx'])
        args.cat_dims = copy.deepcopy(original_full_args_state['cat_dims'])
        args.bin_alt = copy.deepcopy(original_full_args_state['bin_alt'])
        #args.num_bins = copy.deepcopy(original_full_args_state['num_bins'])"""

        print(f"Before Debugging -- Outer Fold {i+1}")  
        
        print(f"DEBUG Outer Fold {i+1}: num_features: {args.num_features}")
        print(f"DEBUG Outer Fold {i+1}: num_classes: {args.num_classes}")
        print(f"DEBUG Outer Fold {i+1}: cat_idx: {args.cat_idx}")
        print(f"DEBUG Outer Fold {i+1}: nominal_idx: {args.nominal_idx}")
        print(f"DEBUG Outer Fold {i+1}: ordinal_idx: {args.ordinal_idx}")
        print(f"DEBUG Outer Fold {i+1}: num_idx: {args.num_idx}")
        print(f"DEBUG Outer Fold {i+1}: cat_dims: {args.cat_dims}")
        print(f"DEBUG Outer Fold {i+1}: bin_alt: {args.bin_alt}")

        try:
            if args.frequency_reg:
                X_train_outer_proc, y_train_outer_proc, X_test_outer_proc, y_test_outer_proc, frequency_map_outer = encoding(
                    args, X_train_outer, y_train_outer, X_test_outer, y_test_outer) # Pass original state for reference if needed by encoding
            else:
                X_train_outer_proc, y_train_outer_proc, X_test_outer_proc, y_test_outer_proc = encoding(
                    args, X_train_outer, y_train_outer, X_test_outer, y_test_outer)

            if args.objective == "probabilistic_regression":
                # Binning 
                y_train_outer_proc, y_test_outer_proc = binning(args, y_train_outer_proc, y_test_outer_proc) # Modifies args inplace

            # Update the final model instance with potentially modified args (like num_classes)
            #final_model.args = args # Assuming model uses self.args

        except Exception as e:
            print(f"ERROR during preprocessing for final model training in outer fold {i+1}: {e}")
            print(f"Skipping outer fold {i+1} due to preprocessing error.")
            all_outer_fold_results.append(None)
            continue # Skip to next outer fold
        
        try:
            class_weights = custom_class_weights(y_train_outer_proc)
        except Exception as e:
            print(f"Y unique : {np.unique(y_train_outer_proc)}")
            print(f"ERROR calculating class weights in outer fold {i+1}: {e}")
            print(f"Skipping outer fold {i+1} due to class weight calculation error.")
            all_outer_fold_results.append(None)
            continue
        print(f"After Debugging -- Outer Fold {i+1}")  
        
        print(f"DEBUG Outer Fold {i+1}: num_features: {args.num_features}")
        print(f"DEBUG Outer Fold {i+1}: num_classes: {args.num_classes}")
        print(f"DEBUG Outer Fold {i+1}: cat_idx: {args.cat_idx}")
        print(f"DEBUG Outer Fold {i+1}: nominal_idx: {args.nominal_idx}")
        print(f"DEBUG Outer Fold {i+1}: ordinal_idx: {args.ordinal_idx}")
        print(f"DEBUG Outer Fold {i+1}: num_idx: {args.num_idx}")
        print(f"DEBUG Outer Fold {i+1}: cat_dims: {args.cat_dims}")
        print(f"DEBUG Outer Fold {i+1}: bin_alt: {args.bin_alt}")

        # --- Final Model Training on Outer Train Set ---
        try:
            print(f"Training final model for Outer Fold {i+1}...")
            final_model = model_cls(best_params_for_fold, args) # Use args copy
        except Exception as e:
            print(f"ERROR instantiating final model in outer fold {i+1} AFTER preprocessing: {e}")
            print(f"Best Params: {best_params_for_fold}")
            print(f"Args Fold State: num_features={args.num_features}, num_classes={args .num_classes}, cat_dims={args.cat_dims}")
            print("Skipping outer fold {i+1} due to model instantiation error.")
            all_outer_fold_results.append(None)
            continue # Skip to next outer fold

        # Train final model for this outer fold
        outer_fold_train_timer = Timer()
        outer_fold_train_timer.start()
        try:
            if args.frequency_reg:
                loss_history, val_loss_history, lambda_reg_history = final_model.fit(
                    X_train_outer_proc, y_train_outer_proc, X_test_outer_proc, y_test_outer_proc, frequency_map_outer) # Use test set for validation monitoring if desired
            else:
                if args.weighted_loss:
                    loss_history, val_loss_history = final_model.fit(
                        X_train_outer_proc, y_train_outer_proc, X_test_outer_proc, y_test_outer_proc, class_weights = class_weights)
                else:
                 loss_history, val_loss_history = final_model.fit(
                    X_train_outer_proc, y_train_outer_proc, X_test_outer_proc, y_test_outer_proc) # Use test set for validation monitoring if desired
        except Exception as e:
            print(f"ERROR during final model fitting in outer fold {i+1}: {e}")
            print(f"Model params: {final_model.params}")
            print(f"Args for fold: num_classes={args.num_classes}, bin_alt={args.bin_alt}")
            print(f"Input shapes: X={X_train_outer_proc.shape}, y={y_train_outer_proc.shape}")
            print(f"Unique y: {np.unique(y_train_outer_proc)}")
            print(f"Skipping outer fold {i+1} due to fitting error.")
            outer_fold_train_timer.end()
            all_outer_fold_results.append(None)
            continue # Skip to next outer fold

        outer_fold_train_timer.end()
        outer_train_times.append(outer_fold_train_timer.get_average_time())

        # --- Final Model Evaluation on Outer Test Set ---
        print(f"Evaluating final model on Outer Test Set for Fold {i+1}...")
        outer_fold_test_timer = Timer()
        outer_fold_test_timer.start()
        try:
            predictions_outer = final_model.predict(X_test_outer_proc)
            #probabilities_outer = final_model.prediction_probabilities
        except Exception as e:
             print(f"ERROR during final prediction in outer fold {i+1}: {e}")
             print(f"Skipping outer fold {i+1} due to prediction error.")
             outer_fold_test_timer.end()
             all_outer_fold_results.append(None)
             continue # Skip to next outer fold
        outer_fold_test_timer.end()
        outer_test_times.append(outer_fold_test_timer.get_average_time())

        # Evaluate predictions for this outer fold
        # Use the unique labels *from the outer training set* for evaluation

        fold_scorer = get_scorer(args) # Use a scorer for this fold only
        try:
            if args.objective == "probabilistic_regression":
                fold_scorer.eval(y_test_outer_proc, predictions_outer, final_model.prediction_probabilities, labels=np.unique(y_train_outer_proc))
            else:
                fold_scorer.eval(y_test_outer_proc, predictions_outer, final_model.prediction_probabilities)

            fold_results = fold_scorer.get_results()
            fold_loss = fold_scorer.get_objective_result() # Get the objective score for this fold
            print(f"Outer Fold {i+1} Results: {fold_results}")
            all_outer_fold_results.append(fold_results) # Store results dict
            loss_per_fold[i] = [fold_loss, best_inner_score, best_params_for_fold] # outer

            """
            # --- Save fold-specific artifacts (optional) ---
            if args.save_model and args.model_name != "TabPFN":
                fold_suffix = f"outer{i+1}"
                 # Save losses if they were returned by fit
                if 'loss_history' in locals() and loss_history is not None:
                    save_loss_to_file(args, loss_history, "loss", extension=fold_suffix)
                if 'val_loss_history' in locals() and val_loss_history is not None:
                    save_loss_to_file(args, val_loss_history, "val_loss", extension=fold_suffix)
                if args.frequency_reg and 'lambda_reg_history' in locals() and lambda_reg_history is not None:
                    save_regularization_to_file(args, lambda_reg_history, "lambda_reg", extension=fold_suffix)

                 # Save model and predictions for this fold
                 # Use a method that saves based on fold index
                final_model.save_model_and_predictions(y_test_outer_proc, i, filename_suffix=f"_outer{i+1}")
                print(f"Saved losses, model, and predictions for Outer Fold {i+1}")
            """

        except Exception as e:
            print(f"ERROR during evaluation or saving for outer fold {i+1}: {e}")
            print(f"y_true shape: {y_test_outer_proc.shape}, unique: {np.unique(y_test_outer_proc)}")
            print(f"predictions shape: {predictions_outer.shape}, unique: {np.unique(predictions_outer)}")
            all_outer_fold_results.append(None) # Indicate failure for this fold
            continue

    best_loss_dict = [loss_per_fold[i][0] for i in range(len(loss_per_fold))]
    best_hp_idx = best_loss_dict.index(min(best_loss_dict))
    best_params = loss_per_fold[best_hp_idx][2]


    # --- Aggregate Results Across Outer Folds ---
    print("\n--- Nested Cross-Validation Summary ---")
    successful_folds = [res for res in all_outer_fold_results if res is not None]
    num_successful_folds = len(successful_folds)

    print(f"Outer Fold Results :{all_outer_fold_results}")
    print(f"Successful Outer Folds: {successful_folds}")
    print(f"Number of Successful Outer Folds: {num_successful_folds}/{args.outer_splits}")
    print(f"")

    if num_successful_folds == 0:
        print("ERROR: No outer folds completed successfully.")
        # Return dummy/error values
        return outer_sc, 0, 0 # Return the empty scorer

    # Average metrics
    aggregated_results = {}
    if successful_folds:
        first_res = successful_folds[0]
        for key in first_res.keys():
            # Check if the value is numeric before averaging
            try:
                values = [fold_res[key] for fold_res in successful_folds if fold_res and key in fold_res]
                # Filter out non-numeric types if necessary, or handle potential errors
                numeric_values = [v for v in values if isinstance(v, (int, float, np.number))]
                if numeric_values and "std" not in key:
                    aggregated_results[key] = np.mean(numeric_values)
                    aggregated_results[f"{key}_std"] = np.std(numeric_values)
                #else:
                     # Handle cases where the key exists but has no numeric values (e.g., contains strings or lists)
                     #aggregated_results[key] = 'N/A (non-numeric)' # Or copy the first instance, etc.
                     #aggregated_results[f"{key}_std"] = 'N/A'
            except (KeyError, TypeError, ValueError) as e:
                print(f"Warning: Could not aggregate metric '{key}'. Error: {e}. Setting to N/A.")
                aggregated_results[key] = 'N/A'
                aggregated_results[f"{key}_std"] = 'N/A'


    avg_total_train_time = np.mean(outer_train_times) if outer_train_times else 0
    avg_total_inference_time = np.mean(outer_test_times) if outer_test_times else 0

    print(f"Aggregated Results over {num_successful_folds}/{args.outer_splits} Outer Folds:")
    for key, value in aggregated_results.items():
        if isinstance(value, (float, np.float_)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"Average Outer Fold Train Time: {avg_total_train_time:.4f} sec")
    print(f"Average Outer Fold Inference Time: {avg_total_inference_time:.4f} sec")

    # Update the main scorer object with aggregated results (optional, depends on scorer design)
    # Alternatively, just return the dictionary
    #outer_sc.results = aggregated_results # Example: Store aggregated results in the scorer

    # Save aggregated results and best parameters found (average or per fold)
    best_loss_dict = [loss_per_fold[i][0] for i in range(len(loss_per_fold))]
    best_hp_idx = best_loss_dict.index(min(best_loss_dict))
    best_params = loss_per_fold[best_hp_idx][2]
    args.save_results = True # Set to True to save results
   
    if args.save_results:
        print("Saving model.....")
        print("Results After CV Manual:", aggregated_results)
        print("Train time:", avg_total_train_time)
        print("Inference time:", avg_total_inference_time)
        save_results_to_file(args, aggregated_results,
                             avg_total_train_time, avg_total_inference_time,
                             best_params,"train_model")
        update_yaml(args.dataset, args.model_name,best_params)
        print("Aggregated results saved.")

    print("Loss per Fold", loss_per_fold)



    # Optional: Visualization of losses across outer folds (needs adaptation)
    # if args.visual and args.save_model and args.model_name != "TabPFN":
    #     try:
    #         # losses_history function needs update to load outer fold losses
    #         losses = losses_history_nested(args) # Implement this function
    #         loss_vizualization(args, losses) # May need update too
    #     except Exception as e:
    #         print(f"Could not generate loss visualization: {e}")

    return aggregated_results, avg_total_train_time, avg_total_inference_time

#calc using train data
def mean_per_bin(y_true, y_class):
    df = pd.DataFrame({'y_true': y_true, 'class': y_class})
    return df.groupby('class')['y_true'].mean()


def test_model(model, parameters, X_train, y_train, X_test, y_test, args, visual=False, save_model=False):
    # Record some statistics and metrics
    if args.class_comp:
        #set obj to be regression
        args.objective = "regression" #rem to rreturn it back to norm
        sc = get_scorer(args)
        args.objective = "probabilistic_regression" #rem to rreturn it back to norm
    else:
        sc = get_scorer(args)

    train_timer = Timer()
    test_timer = Timer()

    n_repeats = args.outer_splits
    mse_scores = []
    r2_scores = []

    args_test_temp = copy.deepcopy(args)

    orig_objective = args.objective

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("TESTING MODEL")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    #make copies of the original data
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()
    X_test_original = X_test.copy()
    y_test_original = y_test.copy()

    for seed in range(5):
        print("In Test Model, Seed is ", seed)
        print(f"--- Test Run {seed+1}/{n_repeats} ---")
        #seed_update = np.random.randint(seed, 100000)
        set_seed(seed)
        
    
        #make sure to use the original data
        X_train, y_train = X_train_original.copy(), y_train_original.copy()
        X_test, y_test = X_test_original.copy(), y_test_original.copy()

        X_train, y_train = resample(X_train, y_train, random_state=seed)

        args = copy.deepcopy(args_test_temp) # Reset args for each test run

        args.test_seed = seed #for future use

        if args.frequency_reg:
            #Need to Clean here
            X_train, y_train, X_test ,y_test,frequency_map = encoding(args, X_train, y_train, X_test, y_test)
            
        else:
            #print("Doing encoding : WE ARE IN TRAIN.PY")
            X_train, y_train, X_test, y_test = encoding(args, X_train, y_train, X_test, y_test)


        if args.dataset == "House_Prices_Nominal" and args.model_name == "FTTransformer":
            x_cat_train = X_train[:, args.ordinal_idx]
            x_cat_test = X_test[:, args.ordinal_idx]

            all_cats = np.concatenate([x_cat_train, x_cat_test], axis=0)    
            args.cat_dims = [int(np.max(all_cats[:, i])) + 1 for i in range(all_cats.shape[1])]
            print(f"BEBUDDING DIMS")
            for i, cat_dim in enumerate(args.cat_dims):
                max_val = int(all_cats[:, i].max())
                if max_val >= cat_dim:
                    print(f"⚠️ Column {i}: max_val {max_val} >= cat_dim {cat_dim} → Mismatch!")
                else:
                    #print(f"✅ Column {i}: max_val {max_val} < cat_dim {cat_dim} → Match!")
                    pass
       

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

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("BINNING INIT")

        #Acquire Bins
        if args.objective == "probabilistic_regression":
            y_train_class, y_test_class = binning(args, y_train, y_test)
            bin_mean = mean_per_bin(y_train, y_train_class)
            print(f"Bin Mean: {bin_mean}")
        else:
            train_class, test_class = binning(args, y_train, y_test)
            print("Binning Done")
        
        print("BINNING END")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n")

        model_name2 = str2model(args.model_name) #load new model

        # Create a new unfitted version of the model
        curr_model = model_name2(parameters, args) # Use args copy
        #curr_model = model.clone() # Use deepcopy to ensure a fresh instance
        print(f"Model Parameters: {curr_model.params}")
        print(f"Parameters Passed: {parameters}")
        print(curr_model.params)



        for attr in dir(model):
            if isinstance(getattr(model, attr), torch.nn.Module):
                print(f"{attr}: {type(getattr(model, attr))}")

        # Train model
        train_timer.start()
        if args.frequency_reg: ## For frequency regularization
            if args.weighted_loss:
                class_weights = custom_class_weights(y_train_class)
                print(f"Class Weights Applied: {class_weights}")
                loss_history, test_loss_history, lambda_reg_history = curr_model.fit(X_train, y_train_class, X_test, y_test_class, frequency_map=frequency_map,class_weights = class_weights)
            
                save_regularization_to_file(args, lambda_reg_history, "lambda_reg", extension=seed)
            else:
                print(f"Class Weights DDNT APPLY")
                loss_history, test_loss_history, lambda_reg_history = curr_model.fit(X_train, y_train_class, X_test, y_test_class, frequency_map=frequency_map)
    
                save_regularization_to_file(args, lambda_reg_history, "lambda_reg", extension=seed)
                #loss_history, test_loss_history, lambda_reg_history = curr_model.fit(X_train, y_train_class, X_test, y_test_class, frequency_map)
        else:
            if args.objective == "probabilistic_regression":
                if args.weighted_loss:
                    class_weights = custom_class_weights(y_train_class)
                    print(f"Class Weights Applied: {class_weights}")
                    loss_history, test_loss_history = curr_model.fit(X_train, y_train_class, X_test, y_test_class, class_weights = class_weights) 
                else:
                    print(f"Class Weights DDNT APPLY")
                    loss_history, test_loss_history = curr_model.fit(X_train, y_train_class, X_test, y_test_class)
            else:
                loss_history, test_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  #regression problems

        train_timer.end()

        # Test model
        test_timer.start()
        prediction = curr_model.predict(X_test)
        probabilities = curr_model.prediction_probabilities

        print(f"Prediction shape : {prediction.shape}")
        #print(f"Probabilities shape : {probabilities.shape} \n")
        #print(f"Y true: {y_test_class[:10]}")
        #print(f"Prediction : {prediction[:10]}")
        #print(f"Probabilities : {probabilities[:10]} \n")
        #print(f"Mean bin : {bin_mean}, Type : {type(bin_mean)}, shape : {bin_mean.shape}")
        test_timer.end()

        print(f"Prediction shape : {prediction.shape}")
        print(f"Prediction Results : {prediction[:10]}")

        # Save model weights and the truth/prediction pairs for traceability
        #curr_model.save_model_and_predictions(y_test_class,seed)

        #print(f"Probabilities: {curr_model.prediction_probabilities}")

        #save the losses
        print(f"State of save is {save_model} b4 loss saving")
       
        if save_model and args.model_name != "TabPFN":
            save_loss_to_file(args, loss_history, "loss", extension=seed)
            save_loss_to_file(args, test_loss_history, "test_loss", extension=seed)
            if args.frequency_reg:
                save_regularization_to_file(args, lambda_reg_history, "lambda_reg", extension=seed)
            print('Saved Losses and Regularization')
        
        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± \n")
        print("B4 Evaluation")
        print(f"Number of classes : {args.num_classes}")
        #print(f"Class label len :{len(args.bin_alt)}")
        #print(f"Class labels : {args.bin_alt}")
        #print(f"Unique y_true : {len(np.unique(y_test_class))}")
        #print(f"Unique train : {len(np.unique(y_train_class))}\n")
        print(f"Prediction shape : {curr_model.predictions.shape}")
        #print(f"Probabilities shape : {curr_model.prediction_probabilities.shape} \n")
        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± \n")
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if args.class_comp:
            print("In the gulag")
            #set obj to be regression
            #args.objective = "regression" #rem to rreturn it back to norm
            #sc = get_scorer(args)

            #get the bin means

            #y_train_pred = [bin_mean.get(cls, np.nan) for cls in y_train_class]
            y_test_pred = [bin_mean.get(cls, np.nan) for cls in prediction]

            matrix_name = f"matrix_{seed}"
            array_name = f"array_{seed}"
            target_stacks = np.column_stack((y_test_class, prediction, y_test_pred))


            #save em
            #save_matrix_to_file(args, probabilities, matrix_name, filetype = 'csv')
            #save_arrays_to_file(args, target_stacks, array_name)


            print(f"y_test_pred : {y_test_pred[:10]}, y_test_class : {y_test_class[:10]}")

            #can apply weighed ones if i want!!!
            y_test_pred = np.array([float(x) for x in y_test_pred])
            #y_test_exp = probabilities @ bin_mean
            #print(f"y_test_pred shape : {y_test_pred.shape}, y_test_exp shape : {y_test_exp.shape}")
            
            #evaluate
            error_results = sc.eval(y_test, y_test_pred,curr_model.prediction_probabilities)
            #error_results = sc.eval(y_test, y_test_exp, curr_model.prediction_probabilities)
            print(f"ERRORS: {error_results}")
            print(f'{sc.get_results()} \n \n')

            get_results = sc.get_results()

            #Append Scores
            #mse_scores.append(error_results['MSE'])         
            #r2_scores.append(error_results['R2'])
            args.objective = orig_objective # Reset objective to original

            #print(f"MSE SCORES on seed {seed}: {mse_scores}, R2 SCORES: {r2_scores}")

        else:
            print("We are not in Class comp")
            # Compute scores on the output
            if args.objective == "probabilistic_regression":
                # Use the binning labels for evaluation
                error_results = sc.eval(y_test_class, prediction, curr_model.prediction_probabilities, labels=np.unique(y_train_class))
            else:
                error_results = sc.eval(y_test, prediction, curr_model.prediction_probabilities)

            print("After Evaluation")

            print(f"ERRORS: {error_results}")

            print(f'{sc.get_results()} \n \n')

            get_results = sc.get_results()
    """
    if args.class_comp:
        mse_mean = np.mean(mse_scores)
        mse_std = np.std(mse_scores)

        r2_mean = np.mean(r2_scores)
        r2_std = np.std(r2_scores)

        get_results = {"MSE - mean": mse_mean,
                "MSE - std": mse_std,
                "R2 - mean": r2_mean,
                "R2 - std": r2_std}
        
        print(f"Final MSE SCORES: {mse_scores}, Final R2 SCORES: {r2_scores}")
    """
    # Best run is saved to file
    if save_model:
        print("Saving model.....")
        print("Results After CV:", get_results)
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, get_results,
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             curr_model.params, "test_model")

    print("Finished cross validation")

    #visualization

    if visual:
        print("Visualizing Happening") 
        losses = loss_hist(args, type='test')
        loss_vizualization(args, losses, type='test')  

    #print(get_output_path(args, filename="logging", file_type = None))
    return sc, (train_timer.get_average_time(), test_timer.get_average_time()), get_results #, (X_train, y_train_class, X_test, y_test_class), args

def loss_hist(args, type):
    path = get_output_path(args, filename="", directory='logging',file_type = None)

    print(f"Type of task : {type}")
    if type == 'train':
        loss_dict = {
            'train' : [],
            'val' : []
        }

        for i in np.arange(args.outer_splits):
            loss_path = path + f'loss_{i}.txt' #changed this
            val_loss_path = path + f'val_loss_{i}.txt' #changed this

            loss_file = np.loadtxt(loss_path)
            val_loss_file = np.loadtxt(val_loss_path)

            loss_dict['train'].append(list(loss_file))
            loss_dict['val'].append(list(val_loss_file))
    else:
        loss_dict = {
            'train' : [],
            'test' : []
        }

        for i in np.arange(args.outer_splits):
            loss_path = path + f'loss_{i}.txt' #changed this
            test_loss_path = path + f'test_loss_{i}.txt' #changed this

            loss_file = np.loadtxt(loss_path)
            test_loss_file = np.loadtxt(test_loss_path)

            loss_dict['train'].append(list(loss_file))
            loss_dict['test'].append(list(test_loss_file))

    return loss_dict


class Objective(object):
    def __init__(self, args, model_cls, X_outer_train, y_outer_train, outer_fold_num):
        # Store the model *class* (not instance)
        self.model_cls = model_cls

        # Store the outer fold's training data
        self.X_outer_train = X_outer_train
        self.y_outer_train = y_outer_train

        # Store args (a copy might be safer if Objective modifies it)
        self.args = copy.deepcopy(args) # Use a copy for safety
        self.outer_fold_num = outer_fold_num

    def __call__(self, trial):
        # Make a deep copy of args FOR THIS TRIAL to avoid interference between trials
        args_trial = copy.deepcopy(self.args)

        # Define hyperparameters to optimize for this trial
        try:
            trial_params = self.model_cls.define_trial_parameters(trial, args_trial) # Pass args_trial
        except Exception as e:
             print(f"ERROR defining trial parameters: {e}")
             # Report failure to Optuna
             raise optuna.TrialPruned(f"Parameter definition failed: {e}")
        

        #model prototype with the sampled hyperparameters
        # Do not fit it yet!
        #try:
        #    model = self.model_cls(trial_params, args_trial)
        #except Exception as e:
        #    print(f"ERROR creating model instance with params {trial_params}: {e}")
        #    raise optuna.TrialPruned(f"Model instantiation failed: {e}")
        
        # Evaluate these hyperparameters using inner CV on the outer training data
        fold_prefix = f"outer_{self.outer_fold_num+1}_inner"
        #try:
        # Pass the *unfitted* model prototype
        inner_sc, avg_score, time = evaluate_hyperparameters_cv(
            self.model_cls, trial_params, self.X_outer_train, self.y_outer_train, args_trial, fold_prefix=fold_prefix
        )
        """except Exception as e:
            print(f"ERROR during inner cross-validation (evaluate_hyperparameters_cv) for trial {trial.number}: {e}")
            # Report failure to Optuna
            raise optuna.TrialPruned(f"Inner CV failed: {e}")
        """
        # Optuna needs to know if the trial failed completely (e.g., all inner folds failed)
        if (args_trial.direction == 'minimize' and avg_score == float('inf')) or \
           (args_trial.direction == 'maximize' and avg_score == float('-inf')):
            print(f"Trial {trial.number} failed completely during inner CV.")
            # Report failure, Optuna will handle based on direction
            raise optuna.TrialPruned("All inner CV folds failed.")

        try:
            print(f"Trial Results B4 Saving: {inner_sc.get_results()}")
            print(f"Score: {inner_sc.get_objective_result()}")
            
            save_hyperparameters_to_file_inner(args_trial, trial_params, inner_sc.get_results(), time)

        except Exception as e:
            print(f"ERROR saving hyperparameters to file coz inner_cs issues: {e}")
            #print(f"Results: {inner_sc.get_results()}")
            #print(f"Score: {inner_sc.get_objective_result()}")
            # Report failure to Optuna
            #raise optuna.TrialPruned(f"Saving hyperparameters failed: {e}")

        
        return avg_score  # inner_sc.get_objective_result() ## return the mean score of the loss


def main(args):
    print("--- Running Nested Cross-Validation with Hyperparameter Optimization ---")

    X, y = load_data(args, is_test=False) # Load full dataset
    model_cls = str2model(args.model_name) # Get model class

    # Run nested CV with optimization enabled
    agg_scorer, avg_train_time, avg_test_time = nested_cross_validation(
        model_cls, X, y, args, optimize_params=True
    )

    print("\n--- Nested Cross-Validation Finished ---")
    print("Aggregated Results:", agg_scorer)
    print(f"Avg Outer Fold Train Time: {avg_train_time:.4f} sec")
    print(f"Avg Outer Fold Inference Time: {avg_test_time:.4f} sec")


def main_once(args):
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
    sc, timer, results = test_model(model_name, parameters, X, y, X_test, y_test, args, visual=False, save_model=True)

    print("\n--- Testin on Final Model Finished ---")
    print("Aggregated Results:", results)
    print(f"Avg Outer Fold Train Time: {timer[0]:.4f} sec")
    print(f"Avg Outer Fold Inference Time: {timer[1]:.4f} sec")

    

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Make sure to add outer_splits argument
    base_parser = get_parser() # Get your base parser
    base_parser.add_argument('--outer_splits', type=int, default=3, help='Number of outer folds for nested cross-validation')
    base_parser.add_argument('--save_results', action='store_true', default=True, help='Save aggregated results file') # Control saving final results
    base_parser.add_argument('--no_save_results', action='store_false', dest='save_results')

    # Initial parse to check mode
    temp_args, unknown = base_parser.parse_known_args()

    if temp_args.optimize_hyperparameters:
        # Re-parse with the full parser if optimizing
        parser = get_parser()
        parser.add_argument('--outer_splits', type=int, default=3, help='Number of outer folds for nested cross-validation')
        parser.add_argument('--save_results', action='store_false', default=False, help='Save aggregated results file')
        parser.add_argument('--no_save_results', action='store_false', dest='save_results')
        arguments = parser.parse_args()
        print("Running Mode: Nested CV with Hyperparameter Optimization")
        print(f"Arguements: {arguments}")
        main(arguments)
    else:
        # Re-parse with the parser that includes predefined parameters
        parser = get_given_parameters_parser() # Assumes this parser loads yaml/dict params
        parser.add_argument('--outer_splits', type=int, default=3, help='Number of outer folds for nested cross-validation')
        parser.add_argument('--save_results', action='store_false', default=False, help='Save aggregated results file')
        parser.add_argument('--no_save_results', action='store_false', dest='save_results')
        
        arguments = parser.parse_args()
        print("Running Mode: Nested CV with Predefined Hyperparameters")
        print(f"Arguements: {arguments}")
        main_once(arguments)

    
