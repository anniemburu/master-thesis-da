import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd
import os


def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q

# Get colnames
def get_colnames(data, idx):
    return data.columns[idx]

#Get column index
def get_colidx(data, colnames):
    col_idx = []
    for col in colnames:
        col_idx.append(data.columns.get_loc(col))

    return col_idx

def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    ####~~~~~~~~~~~~~~~~~~~~ SPECIFY HOW DATASETS ARE LOADED AND STUFF ~~~~~~~~~~~~~~~~~ 
    if args.dataset == "Boston":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/531-boston/raw_data.csv')
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/531-boston.csv')
        label_col = "MEDV"

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Socmob":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/541-socmob/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/541-socmob.csv')
        label_col = 'counts_for_sons_current_occupation'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()


    elif args.dataset == "Sensory":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/546-sensory/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/546-sensory.csv')
        label_col = 'Score'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Moneyball":
        #df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41021-Moneyball/raw_data.csv') #CLUSTER
        df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41021-Moneyball.csv')
        label_col = 'RS'

        cat_cols = get_colnames(df, args.cat_idx) #categorical columns

        #drop cols
        drop_cols = get_colidx(df, args.dropna_idx) #get the columns to drop
        df.drop(columns=drop_cols, inplace=True)
        args.cat_idx = get_colidx(df, cat_cols) #update index of categorical columns

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()
    #####################################################################################

    print("Dataset loaded!")
    print(X.shape)

    # Preprocess target 
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)


    num_idx = []
    args.cat_dims = []

    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])

            # Setting this?
            args.cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)

    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        new_x2 = X[:, num_idx]
        X = np.concatenate([new_x1, new_x2], axis=1)

        #change the num of features after one hot encoding;
        args.num_features = X.shape[1]
        print(f"args.num_features: {args.num_features}")
        print("New Shape:", X.shape)

    return X, y
