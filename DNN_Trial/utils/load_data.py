import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
print(f" Panda Version: {pd.__version__}")
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
    import pandas as pd
    print(f" Panda Version: {pd.__version__}")
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
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41021-Moneyball/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41021-Moneyball.csv')
        label_col = 'RS'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Black_Friday" :
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41540-black_friday/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41540-black_friday.csv')
        label_col = 'Purchase'

        df.loc[df['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = 4
        df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()


    elif args.dataset == "SAT11":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/41980-SAT11-HAND-runtime-regression/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/41980-SAT11-HAND-runtime-regression.csv')
        label_col = 'runtime'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop
        df.dropna(axis=1, inplace=True) #drop missing

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Diamonds":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42225-diamonds/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42225-diamonds.csv')
        label_col = 'price'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "House_Prices_Nominal":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42563-house_prices_nominal/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42563-house_prices_nominal.csv')
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

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Mercedes_Benz":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42570-Mercedes_Benz_Greener_Manufacturing/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42570-Mercedes_Benz_Greener_Manufacturing.csv')
        label_col = 'y'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Allstate_Claims":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42571-Allstate_Claims_Severity/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42571-Allstate_Claims_Severity.csv')
        label_col = 'loss'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Brazillian_Houses":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42688-Brazilian_houses/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42688-Brazilian_houses.csv')
        label_col = 'total_(BRL)'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Abalone":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42726-abalone/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42726-abalone.csv')
        label_col = 'Class_number_of_rings'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "NYC_Taxi":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42729-nyc-taxi-green-dec-2016/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42729-nyc-taxi-green-dec-2016.csv')
        label_col = 'tip_amount'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "House_Sales":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/42731-house_sales/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/42731-house_sales.csv')
        label_col = 'price'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "MIP":
        df = pd.read_csv('/home/mburu/Master_Thesis/master-thesis-da/datasets/43071-MIP-2016-regression/raw_data.csv') #CLUSTER
        #df = pd.read_csv('/Users/wambo/Desktop/Master Thesis/master-thesis-da/datasets/43071-MIP-2016-regression.csv')
        label_col = 'PAR10'

        norm_cols = get_colnames(df, args.nominal_idx) #nominal cols
        drop_cols = get_colnames(df, args.dropna_idx) #get the columns to drop

        df.drop(columns=drop_cols, inplace=True) #drop

        args.nominal_idx = get_colidx(df, norm_cols) #update index of norm columns
        args.num_features = df.shape[1] - 1 #update number of features 

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    

    print("Dataset loaded! \n")
    print(f"X b4 encoding : {X[0]} \n")
    print(X.shape)
    print(f"Data Type of X: {type(X)}")
    print(f"Nominal Idx: {args.nominal_idx}")
    print(f"Ordinal Idx: {args.ordinal_idx}")
    print(f"Cat Dims: {args.cat_dims} \n \n")
    print(f"Normonal Idx: {args.nominal_idx}")
    

    # Preprocess target 
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)


    num_idx = []
    args.cat_dims = []
    args.cat_idx = get_catidx(args)
    print(f"Cat Idx Part II: {args.cat_idx} ")
    print(f"ENDE \n \n")

    #####################################################################################
    # NO Encoding for XGBoost, CatBoost, LightGBM
    if args.model_name == "XGBoost" or args.model_name == "CatBoost" or args.model_name == "LightGBM":
        args.one_hot_encode = False
        args.ordinal_encode = False
        print(f'No one Hot for this Baby!!! \n')

    #~~~~~~~~~~~~~~~~~~~~~~~~~

    
    # Preprocess  Nominal data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            #Only Nominal
            if args.model_name == "XGBoost" or args.model_name == "CatBoost" or args.model_name == "LightGBM":
                le = LabelEncoder()
                X[:, i] = le.fit_transform(X[:, i])
                args.cat_dims.append(len(le.classes_))
            else:
                if args.nominal_idx and i in args.nominal_idx:
                    le = LabelEncoder()
                    X[:, i] = le.fit_transform(X[:, i])

                    # Setting this?
                    args.cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)

    args.num_idx = num_idx #update num_idx
    
    
    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])


    if args.one_hot_encode:
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
        #args.cat_idx = get_catidx(args)
        #args.cat_idx = args.ordinal_idx  ##coz the norminal are now int....
        if args.ordinal_encode:
            args.cat_idx = args.ordinal_idx
        else:
            args.cat_idx = None
            
        print("One Hot Encoding...")
        print(f"args.num_features: {args.num_features}")
        print(f"args.cat_idx: {args.cat_idx}")
        print("New Shape:", X.shape)
        

    # Ordinal Encode
    if args.ordinal_encode:
        if args.dataset == "Black_Friday":
            ordinal_encoder = OrdinalEncoder(categories=[[None,'0-17','18-25','26-35','36-45','46-50','51-55','55+']])
            X[:, args.ordinal_idx] = ordinal_encoder.fit_transform(X[:, args.ordinal_idx])

        elif args.dataset == "Diamonds":
            categories = [
                            [None, 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],  # For 'cut'
                            [None, 'J', 'I', 'H', 'G', 'F', 'E', 'D'],  # For 'color'
                            [None, 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # For 'clarity'
                        ]

            # Create the OrdinalEncoder
            encoder = OrdinalEncoder(categories=categories, dtype=int)

            # Fit and transform the data
            X[:, args.ordinal_idx] = encoder.fit_transform(X[:, args.ordinal_idx])

        elif args.dataset == "House Prices Nominal":
            categories = [
                    [None,'Grvl', 'Pave'],
                    ['None', 'Grvl', 'Pave'],
                    [None, 'Low','Bnk','HLS','Lvl'],
                    [None,'NoSeWa','AllPub'],
                    [None,'Inside', 'FR2', 'FR3', 'Corner', 'CulDSac'],
                    [None,'Sev', 'Mod', 'Gtl'],
                    [None,'RRNe','RRNn','RRAe','RRAn','Artery','Feedr','Norm','PosN','PosA'],
                    [None,'RRNn','RRAe','RRAn','Artery','Feedr','Norm','PosN','PosA'],
                    [None, '1Fam','TwnhsE', 'Twnhs', 'Duplex', '2fmCon'],
                    [None,"1Story", "1.5Unf","SFoyer","SLvl","1.5Fin", "2Story","2.5Unf","2.5Fin"],
                    [None,"Flat", "Shed", "Gambrel", "Mansard", "Gable","Hip"],
                    [None,'Roll','Tar&Grv','Membran','CompShg','WdShngl','WdShake','Metal','ClyTile'],
                    [None,'CBlock','AsphShn','ImStucc','AsbShng','Plywood','Wd Sdng','WdShing','MetalSd','VinylSd','HdBoard','Stucco','BrkComm','CemntBd','BrkFace','Stone'],
                    [None, 'Other','CBlock','AsphShn','ImStucc','AsbShng','Plywood','Wd Sdng','Wd Shng','MetalSd','VinylSd','HdBoard','Stucco','Brk Cmn','CmentBd','BrkFace','Stone'],
                    ['None', 'BrkCmn','BrkFace','Stone'],
                    [None,'Fa','TA','Gd','Ex'],
                    [None,'Po','Fa','TA','Gd','Ex'],
                    [None,'Wood', 'Slab', 'BrkTil','CBlock', 'Stone','PConc'],
                    ['None','Fa','TA','Gd','Ex'],
                    ['None','Po','Fa','TA','Gd'],
                    ['None', 'No','Mn', 'Av', 'Gd'],
                    ['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
                    ['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
                    [None,'OthW','Grav','Wall','Floor', 'GasW','GasA'],
                    ['None','Po','Fa','TA','Gd','Ex'],
                    ['None', 'N', 'Y'],
                    ['None', 'Mix','FuseP','FuseF', 'FuseA','SBrkr'],
                    [None,'Fa','TA','Gd','Ex'],
                    [None, 'Sev','Maj1','Maj2','Min1','Min2','Mod','Typ'],
                    ['None','Po','Fa','TA','Gd','Ex'],
                    ['None','CarPort', 'Detchd','Basment','2Types','BuiltIn','Attchd'],
                    ['None','Unf','RFn','Fin'],
                    ['None','Po','Fa','TA','Gd','Ex'],
                    ['None','Po','Fa','TA','Gd','Ex'],
                    ['None', 'N','P','Y'],
                    ['None','Fa','Gd','Ex'],
                    ['None','MnWw','MnPrv','GdWo','GdPrv'],
                    ['None','Othr','Shed','Gar2','TenC'],
                    [None, 'Oth','COD', 'ConLD','ConLw','ConLI','Con','WD','CWD','New'],
                    [None,'Abnorml','AdjLand','Family','Alloca','Partial','Normal']
                ]
            # Create the OrdinalEncoder
            encoder = OrdinalEncoder(categories=categories, dtype=int)

            # Fit and transform the data
            X[:, args.ordinal_idx] = encoder.fit_transform(X[:, args.ordinal_idx])
        
        elif args.dataset == "Brazillian Houses":

            encoder = OrdinalEncoder(categories=[[None,'not furnished','furnished']])

            # Fit and transform the data
            X[:, args.ordinal_idx] = encoder.fit_transform(X[:, args.ordinal_idx])

            print("OE Done!!! \n")

    return X, y
