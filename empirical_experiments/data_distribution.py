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
from utils.io_utils import update_yaml, save_results_to_file, save_hyperparameters_to_file, save_hyperparameters_to_file_inner,save_loss_to_file, get_output_path, save_regularization_to_file
from utils.parser import get_parser, get_given_parameters_parser
from utils.visualization import loss_vizualization
from sklearn.preprocessing import KBinsDiscretizer

from scipy.stats import skew, kurtosis, shapiro, normaltest
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import os
print(os.getcwd())

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

def detect_distribution(y, plot=False):
    y = np.asarray(y).flatten()
    s = skew(y)
    k = kurtosis(y)
    shap = shapiro(y)
    norm = normaltest(y)

    # Estimate density and count peaks
    y_vals = np.linspace(np.min(y), np.max(y), 1000).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=(np.std(y) / 20)).fit(y.reshape(-1, 1))
    log_dens = kde.score_samples(y_vals)
    peaks, _ = find_peaks(np.exp(log_dens))

    print(f"Skewness: {s}, Kurtosis: {k}, Shapiro p-value: {shap}, Normal p-value: {norm}")
    print(f"Number of peaks: {len(peaks)}")

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.kdeplot(y, shade=True)
        plt.title("KDE with Peak Detection")
        for peak in y_vals[peaks]:
            plt.axvline(peak, color='red', linestyle='--', alpha=0.6)
        plt.show()

    # Classification logic
    if len(peaks) >= 2:
        return "bimodal"
    elif abs(s) < 0.5 and abs(k) < 1:
        return "normal"
    else:
        return "skewed"
    

# Load the configuration file
parser = get_parser()
arguments = parser.parse_args()

X, X_test, y, y_test = load_data(arguments, is_test=True)

print("X shape: ", X.shape)
print("y shape: ", y.shape)

distrib_train = detect_distribution(y, plot=False)
distrib_test = detect_distribution(y_test, plot=False)

print(f"Detected distribution for {distrib_train, distrib_test} ")



