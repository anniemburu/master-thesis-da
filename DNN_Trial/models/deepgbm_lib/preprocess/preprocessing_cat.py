import pandas as pd
import numpy as np

import models.deepgbm_lib.config as config

'''

Preprocess the input data for the CatNN - Bucketize the numerical features

'''


class CatEncoder(object):

    """
        cat_col: list of column names of categorical data
        num_col: list of column names of numerical data
    """
    
    def __init__(self, cat_col, num_col):
        
        self.cat_col = cat_col
        self.num_col = num_col
        
        self.feature_columns = cat_col+num_col

        self.keep_values = {}
        self.num_bins = {}

    def fit_transform(self, X):

        print("Preprocess data for CatNN...")
        for idx in self.num_col:
            # Bucketize numeric features

            #print(f"THE X Idx : \n {X[:, idx]} \n\n")
            #print(f"Column {idx} data type: {X[:, idx].dtype} \n")
            #print(f"Column {idx} sample data: {X[:, idx]} \n \n \n")

            #nan_mask = np.isnan(X[:, idx])
            #print(f"Rows with NaN in column {idx}: {nan_mask} \n")
            ttypes = [type(x) for x in X[:, idx]]
            
            print(f"Data types: \n {set(ttypes)}")

            if X[:, idx].dtype == 'object': 
                X[:, idx] = pd.to_numeric(X[:, idx], errors='coerce') #change to int and change wierd to NAN
                X[:, idx] = pd.Series(X[:, idx]).fillna(X[:, idx].mean()).values
            ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #if np.issubdtype(X[:, idx].dtype, np.number):
            #    out, bins = pd.qcut(X[:, idx], config.config['bins'], labels=False, retbins=True, duplicates='drop')
            #    X[:, idx] = np.nan_to_num(out, nan=0).astype("int")
            #    self.num_bins[idx] = bins

        X = X.astype("int")
        # Get feature sizes (number of different features in every column)
        feature_sizes = []

        for idx in self.feature_columns:
            feature_sizes.append(X[:, idx].max()+1)

        return X, feature_sizes

    def transform(self, X):

        for idx in self.num_col:
            # Bucketize numeric features
            out = pd.cut(X[:, idx], self.num_bins[idx], labels=False, include_lowest=True)
            X[:, idx] = np.nan_to_num(out, nan=0).astype("int")

        return X
        