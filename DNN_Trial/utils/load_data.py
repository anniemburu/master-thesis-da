import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
#print(f" Panda Version: {pd.__version__}")
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

def load_data(args, is_test=False):
    import pandas as pd
    #print(f" Panda Version: {pd.__version__}")
    print("Loading dataset " + args.dataset + "...")

    ####~~~~~~~~~~~~~~~~~~~~ SPECIFY HOW DATASETS ARE LOADED AND STUFF ~~~~~~~~~~~~~~~~~ 
    if args.dataset == "Boston":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/531-boston/raw_data.csv')
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/531-boston.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/531-boston.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/531-boston.csv')
        label_col = "MEDV"

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Socmob":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/541-socmob/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/541-socmob.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/541-socmob.csv') #UBUNTU
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/541-socmob.csv')
        label_col = 'counts_for_sons_current_occupation'

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Sensory":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/546-sensory/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/546-sensory.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/546-sensory.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/546-sensory.csv')
        
        label_col = 'Score'

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Moneyball":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41021-Moneyball/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41021-Moneyball.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/41021-Moneyball.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/41021-Moneyball.csv')
        label_col = 'RS'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(inplace=True)

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Black_Friday" :
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41540-black_friday/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41540-black_friday.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/41540-black_friday.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/41540-black_friday.csv')
        label_col = 'Purchase'

        #Age
        df['Age'] = df['Age'].astype(str)

        #Occupation
        df['Occupation'] = df['Occupation'].astype(str)
       
        #City_Years
        df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(str)
        
        #MS
        df['Marital_Status'] = df['Marital_Status'].astype(str)
        

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()


    elif args.dataset == "SAT11":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41980-SAT11-HAND-runtime-regression/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41980-SAT11-HAND-runtime-regression.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/41980-SAT11-HAND-runtime-regression.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/41980-SAT11-HAND-runtime-regression.csv')
        label_col = 'runtime'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(inplace=True) #drop missing

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Diamonds":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42225-diamonds/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42225-diamonds.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42225-diamonds.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42225-diamonds.csv')
        label_col = 'price'

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "House_Prices_Nominal":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42563-house_prices_nominal/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42563-house_prices_nominal.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42563-house_prices_nominal.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42563-house_prices_nominal.csv')
        label_col = 'SalePrice'

        #nulls
        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        ord_cols = get_colnames(df, args.ordinal_idx)
        miss_cat_cols = get_colnames(df, args.miss_cat_idx)
        miss_num_cols = get_colnames(df, args.miss_num_idx)
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.ordinal_idx = get_colidx(df, ord_cols)
        args.num_features = df.shape[1] - 1 #update number of features

        #fill nulls
        df[miss_cat_cols] = df[miss_cat_cols].fillna("None")

        for idx in miss_num_cols:
            median_val = df[idx].median()
            df[idx] = df[idx].fillna(median_val)

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Mercedes_Benz":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42570-Mercedes_Benz_Greener_Manufacturing/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42570-Mercedes_Benz_Greener_Manufacturing.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42570-Mercedes_Benz_Greener_Manufacturing.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42570-Mercedes_Benz_Greener_Manufacturing.csv')
        label_col = 'y'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(inplace=True)

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Allstate_Claims":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42571-Allstate_Claims_Severity/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42571-Allstate_Claims_Severity.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42571-Allstate_Claims_Severity.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42571-Allstate_Claims_Severity.csv')
        label_col = 'loss'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(inplace=True)

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Brazillian_Houses":
        #df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42688-Brazilian_houses/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42688-Brazilian_houses.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42688-Brazilian_houses.csv')
        df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42688-Brazilian_houses.csv')
        label_col = 'total_(BRL)'

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Abalone":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42726-abalone/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42726-abalone.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42726-abalone.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42726-abalone.csv')
        label_col = 'Class_number_of_rings'

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

        print("In Abalone")
        print(df.info())

    elif args.dataset == "NYC_Taxi":
        pass
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42729-nyc-taxi-green-dec-2016/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42729-nyc-taxi-green-dec-2016.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42729-nyc-taxi-green-dec-2016.csv')
        df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/546-sensory.csv')
        label_col = 'tip_amount'

        #drop nulls
        df.dropna(inplace=True)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "House_Sales":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42731-house_sales/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42731-house_sales.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/42731-house_sales.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/42731-house_sales.csv')
        label_col = 'price'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(inplace=True)

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "MIP":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/43071-MIP-2016-regression/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/43071-MIP-2016-regression.csv')
        #df = pd.read_csv('/home/wambo/Desktop/Master Thesis/datasets/3071-MIP-2016-regression.csv')
        #df = pd.read_csv('/Users/johnmburu/Desktop/Master Thesis/master-thesis-da/datasets/43071-MIP-2016-regression.csv')
        label_col = 'PAR10'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(inplace=True)

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    

    print("Dataset loaded! \n")
    #print(f"X b4 encoding : {X[0]} \n")
    print(X.shape)
    #print(f"Data Type of X: {type(X)}")
    #print(f"Nominal Idx: {args.nominal_idx}")
    #print(f"Ordinal Idx: {args.ordinal_idx}")
    #print(f"Cat Dims: {args.cat_dims} \n \n")
    #print(f"Normonal Idx: {args.nominal_idx}")

    #Split to train and Split

    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=40)

    print(f"X : {X_[:5,:]}")
    print(f"X : {y_[:5,:]}")

    if is_test:
        return X_, X_test, y_, y_test
    else:
        return X_, y_
