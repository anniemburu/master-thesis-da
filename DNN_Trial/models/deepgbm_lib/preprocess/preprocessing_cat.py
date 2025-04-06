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

    '''
    Check why shit is failing
    '''
    # Apply pd.to_numeric to check if conversion works
    def is_numeric(self, arr):
        try:
            # Try converting the array to numeric, non-numeric values will become NaN
            numeric_array = np.array([pd.to_numeric(x, errors='coerce') for x in arr])
            return np.isnan(numeric_array).any()  # Check if any element is NaN (non-numeric)
        except Exception as e:
            return True  # If an error occurs, the array has non-numeric values
        
    def check_non_numeric(self, arr):
        non_numeric_found = False
        for value in arr.flatten():  # Flatten to check all values in the array
            if not isinstance(value, (int, float, np.number)):  # Check for non-numeric types
                non_numeric_found = True
                break
        return non_numeric_found

    def fit_transform(self, X):
        print(F'X type b4: {X.dtype}')
        X = X.astype(float)
        print(F'X type after: {X.dtype}')

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

        print(F'X type b4: {X.dtype}')
        X = X.astype(float)
        print(F'X type after: {X.dtype}')

        for idx in self.num_col:
            # Bucketize numeric features
            out = pd.cut(X[:, idx], self.num_bins[idx], labels=False, include_lowest=True)
            X[:, idx] = np.nan_to_num(out, nan=0).astype("int")

        return X
        