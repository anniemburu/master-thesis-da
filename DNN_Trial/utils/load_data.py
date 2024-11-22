import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder

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

def get_catidx(args):
    #Populate args.cat_idx
    if args.ordinal_encode & args.one_hot_encode:
        cat_idx = sorted(args.nominal_idx + args.ordinal_idx)
    elif args.ordinal_encode:
        cat_idx = args.ordinal_idx
    else:
        cat_idx = args.nominal_idx
    
    return cat_idx

def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    ####~~~~~~~~~~~~~~~~~~~~ SPECIFY HOW DATASETS ARE LOADED AND STUFF ~~~~~~~~~~~~~~~~~ 
    if args.dataset == "Boston":
        #df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/531-boston/raw_data.csv')
        df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/531-boston.csv')
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

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        ord_cols = get_colnames(df, args.ordinal_idx) # ordinal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.ordinal_idx = get_colidx(df, ord_cols)
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Black_Friday" :
        #df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41540-black_friday/raw_data.csv') #CLUSTER
        df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41540-black_friday.csv')
        label_col = 'Purchase'
        args.cat_idx = get_catidx(args)
        print(f'Unique Age : {df["Age"].unique()}')

        print(f'Dataset B4: {df.head()} \n ')

        df.loc[df['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = 4
        df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()
    #####################################################################################

    print("Dataset loaded!")
    print(X.shape)
    print(f'unique {np.unique(X[:, [1]])}')

    # Preprocess target 
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)


    num_idx = []
    args.cat_dims = []
    args.cat_idx = get_catidx(args) 
    
    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            #Only Nominal
            if args.nominal_idx and i in args.nominal_idx:
                le = LabelEncoder()
                X[:, i] = le.fit_transform(X[:, i])

                # Setting this?
                args.cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)
    #print(f"num_idx : {num_idx}")
    
    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

        print(f"X after scaling: \n {X[:6, ]} \n")

    if args.one_hot_encode:
        print(f'Norminal idx ; {args.nominal_idx}')
        print(f'Ordinal idx ; {args.ordinal_idx}')
        print(f'Numerical idx ; {num_idx}')
        
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.nominal_idx])
        new_x2 = X[:, num_idx]

        if args.ordinal_encode:
            ord_len = len(args.ordinal_idx)
            new_ord = X[:, args.ordinal_idx]
            args.ordinal_idx = [x for x in range(ord_len)] #update ordinal idx
            X = np.concatenate([new_ord, new_x1, new_x2], axis=1)

        else:
            X = np.concatenate([new_x1, new_x2], axis=1)

        #change the num of features after one hot encoding;
        args.num_features = X.shape[1]
        print(f"args.num_features: {args.num_features}")
        print("New Shape:", X.shape)
        

    # Ordinal Encode
    if args.ordinal_encode:
        if args.dataset == "Black_Friday":
            ordinal_encoder = OrdinalEncoder(categories=[[None,'0-17','18-25','26-35','36-45','46-50','51-55','55+']])
            X[:, args.ordinal_idx] = ordinal_encoder.fit_transform(X[:, args.ordinal_idx])
            

    return X, y
