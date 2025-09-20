import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import numpy as np
import math
from scipy.stats import skew, kurtosis, shapiro, normaltest
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import os
from sklearn.model_selection import train_test_split
print(os.getcwd())

path = "datasets"
files = os.listdir(path)
files = [f for f in files if f.endswith('.csv')]
labels = [name.split("-", 1)[1] for name in files]
labels = [name.split(".")[0] for name in labels]
print(f"Labels: {labels}")


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
    

def load_data(path, files):

    for file in files:
        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±")
        print(f"Loading {file}...")
        data = pd.read_csv(os.path.join(path, file))
        
        y = data.iloc[:, -1]

        distrib = detect_distribution(y, plot=False)

        print(f"Detected distribution for {file}: {distrib} ")


        print("±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± \n \n")


def plot_kdes_in_grid(files, grid_cols=3, labels=None):
    num_files = len(files)
    grid_rows = math.ceil(num_files / grid_cols)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 3))
    axes = axes.flatten()  # Make indexing easier

    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(path, file))
        y_col = df.columns[-1]

        X = df.drop(columns=[y_col])
        y = df[y_col]

        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=40)
        #y_col = df.iloc[:, -1]
        
        print(f"Y column for {file}: {y_col}")
        label = labels[i] if labels else Path(file).stem
        sns.kdeplot(y_, ax=axes[i], shade=True, fill=True, color='blue', label='Train')
        sns.kdeplot(y_test, ax=axes[i], shade=True, fill=True, color='red', label='Test')

        axes[i].set_title(label)
        axes[i].set_xlabel(y_col)
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].grid(True)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    fig.savefig("y_distribution.png", dpi=300, bbox_inches="tight")  # Saves as PNG with 300 DPI
    print(f"Plot saved as {path}")

load_data(path, files)

plot_kdes_in_grid(files, grid_cols=3, labels=labels)