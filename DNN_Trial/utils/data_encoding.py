
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
import numpy as np
from collections import defaultdict

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

def frequency_mapper(X_onehot, onehot_encoder):
    # Initialize a dictionary to store frequencies
    frequency_map = defaultdict(float)

    # Iterate over the one-hot encoded columns and compute frequencies
    for i in range(X_onehot.shape[1]):
        category_name = onehot_encoder.get_feature_names_out()[i]  # Get the category name
        frequency = np.mean(X_onehot[:, i])  # Frequency = mean of the one-hot encoded column
        frequency_map[category_name] = frequency

    return frequency_map

#Calculates Freedman-Diaconis Rule
def freedman_diaconis(y, args):
    #calc IQR
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    #calc bin width
    n = len(y) // args.num_splits
    bin_width = 2 * (iqr / (n ** (1/3)))

    #calc num of bins
    data_range = np.max(y) - np.min(y)
    num_bins = int(np.round(data_range / bin_width))

    return num_bins

# Sturges' Rule
def sturges(y,args): 
    n = len(y) // args.num_splits
    num_bins = 1 + int(np.log2(n))

    return num_bins

def bin_finder(args, y):
    if args.y_distribution == "normal" :
        bins = sturges(y, args)
    elif args.y_distribution == "skewed" or args.y_distribution == "bimodial":
        bins = freedman_diaconis(y,args)
    else:
        raise NotImplementedError("Distribution" + args.y_distribution + "is not yet implemented.")

    return bins

def bin_shifter(args, y):
    """
    Shifts class labels so that they are contiguous (without gaps).
    """
    
    def get_contiguous_labels(arr):
        """ Renumber labels to remove gaps """
        unique_vals = np.unique(arr)
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_vals)}
        return np.vectorize(mapping.get)(arr), mapping

    # Get contiguous labels
    #comb = np.unique(np.concatenate([y, y_test]))
    comb_len = len(y)

    if comb_len != args.num_bins:
        print("WE ARE IN THE GUTTERS!!!!!")
        y_train_shift, train_mapping = get_contiguous_labels(y)
        #y_test_shift = np.vectorize(train_mapping.get)(y_test)  # Apply same mapping to test

        # Update arguments
        args.num_classes = len(np.unique(y_train_shift))  # Set correct number of classes
        args.bin_alt = sorted(list(np.unique(y_train_shift)))  # Ensure proper bin numbering

        print(f"Final Train Labels Length: {len(np.unique(y_train_shift))}")
        #print(f"Final Test Labels Length: {len(np.unique(y_test_shift))}")
        print(f"Final Num Classes: {args.num_classes}")
        print(f"Final Bin Labels: {args.bin_alt}")

        return y_train_shift

    else:
        print("No need to shift labels.")
        args.bin_alt = [x for x in range(args.num_bins)]
        return y

def encoding(args, X,y):
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("BEFORE ANY ENCODING")
    print(f"num_features :{args.num_features}")
    print(f"num_classes : {args.num_classes}")
    print(f"cat_idx : {args.cat_idx}")
    print(f"nominal_idx : {args.nominal_idx}")
    print(f"ordinal_idx : {args.ordinal_idx}")
    print(f"num_idx : {args.num_idx}")
    print(f"cat_dims : {args.cat_dims}")
    print(f"bin_alt : {args.bin_alt} \n\n")
    print(f"X shape : {X.shape}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    ## DO for TabPFN
    if args.model_name == "TabPFN":
        args.one_hot_encode = False
        args.ordinal_encode = False

        if args.nominal_idx is not None and args.ordinal_idx is not None:
            args.cat_idx = sorted(args.nominal_idx + args.ordinal_idx)
        elif args.ordinal_idx is not None:
            args.cat_idx = args.ordinal_idx
        else:
            args.cat_idx = args.nominal_idx

        for idx in args.cat_idx:
            le = LabelEncoder()
            X[:, idx] = le.fit_transform(X[:, idx])

        X[:, args.cat_idx] = X[:, args.cat_idx].astype(float)

    # Preprocess target 
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

    num_idx = [] # Index of numerical features
    args.cat_dims = [] # dimensions for categorical features
    args.cat_idx = get_catidx(args) # Index of categorical features

    #print(f"Nominal Index : {args.nominal_idx}")
    #print(f"Ordinal Index : {args.ordinal_idx}")
    #print(f"Cat Idx : {args.cat_idx}")
    #print(f"Cat dims : {args.cat_dims}")

    #print(f"X shape before encoding : {X.shape}")
    #print(f"X before encoding : {X[:10,:]}")
    
   
    # NO Encoding for XGBoost, CatBoost, LightGBM
    if args.model_name == "XGBoost" or args.model_name == "CatBoost" or args.model_name == "LightGBM":
        args.one_hot_encode = False
        args.ordinal_encode = False
        print(f'No one Hot for this Baby!!! \n')

    
    # Preprocess  Nominal data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:

            #Only Nominal
            if args.model_name == "XGBoost" or args.model_name == "CatBoost" or args.model_name == "LightGBM":
                le = LabelEncoder()
                X[:, i] = le.fit_transform(X[:, i])
                args.cat_dims.append(len(le.classes_))
            else:
                if args.ordinal_idx and i in args.ordinal_idx:
                    le = LabelEncoder()
                    #X[:, i] = le.fit_transform(X[:, i])
                    le.fit_transform(X[:, i])

                    # Gets number of unique classes per ordinal feature
                    #Covers future cases with None
                    if np.any(X[:, i] == "None"):
                        args.cat_dims.append(len(le.classes_))
                    else:
                        args.cat_dims.append(len(le.classes_)+1)

        else:
            num_idx.append(i)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n")
    print("AFTER SEPARATING CATEGORICALS AND NUMERICALS")
    print(f"Numerical Index V1 : {num_idx}")
    print(f"Cat Dims V1 : {args.cat_dims}")
    print(f"Cat Idx V1 : {args.cat_idx} \n \n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n")

    # Encode the numerical features
    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
       


    #Encode Nominal Features
    if args.one_hot_encode:
        print("One Hot Encoding...")
        #print(f"Nominal Index : {args.nominal_idx}")
        #print(f"Ordinal Index : {args.ordinal_idx}")
        #print(f"Numerical Index : {num_idx}")
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.nominal_idx])
        new_x2 = X[:, num_idx]



        if args.ordinal_encode:
            ord_len = len(args.ordinal_idx)
            new_ord = X[:, args.ordinal_idx]
           
            
            #print(f"Ordinal Idx Updated: {args.ordinal_idx}")
            X = np.concatenate([new_ord, new_x1, new_x2], axis=1)
            #X_val = np.concatenate([new_ord_val, new_x1_val, new_x2_val], axis=1)

            args.ordinal_idx = [x for x in range(ord_len)] #update ordinal idx
            args.nominal_idx = [x+len(args.ordinal_idx) for x in range(new_x1.shape[1])]  #Update Nominal idx
            args.num_idx = [x for x in range(X.shape[1])][-len(num_idx):]

        else:
            X = np.concatenate([new_x1, new_x2], axis=1)
            #X_val = np.concatenate([new_x1_val, new_x2_val], axis=1)

            args.num_idx = [x for x in range(X.shape[1])][-len(num_idx):]
            args.nominal_idx = [x for x in range(new_x1.shape[1])]

        #change the num of features after one hot encoding;
        args.num_features = X.shape[1] #here is the issue
        #args.cat_idx = get_catidx(args)
        #args.cat_idx = args.ordinal_idx  ##coz the norminal are now int...
        
        

        """
        We have encoded nominal features. Therefore categorical data now is if we have 
        odinal features.
        """
        if args.ordinal_encode:
            args.cat_idx = args.ordinal_idx
        else:
            args.cat_idx = None

        freqency_map = frequency_mapper(new_x1, ohe) #mapping only OHE

        #print("One Hot Encoding...")
        #print("New Shape:", X.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("After OHE")
    print(f"Numerical Index V2 : {num_idx} \n\n")
    print(f"OHE Idx : {args.nominal_idx}\n\n")
    print(f"Ordinal Idx V2: {args.ordinal_idx}\n\n")
    print(f"Cat Dims V2 : {args.cat_dims}")
    print(f"Cat Idx V2 : {args.cat_idx} \n \n")
    print(f"Train: {X[:10,:5]} \n \n ")
    print(f"Val : {X.shape} \n \n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
       
        

    # Ordinal Encode
    if args.ordinal_encode:
        print("Ordinal Encoding...")
        if args.dataset == "Black_Friday":
            #print(f"Ordinal Index b4 using OE : {args.ordinal_idx}")
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

        elif args.dataset == "House_Prices_Nominal":
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
            #X_val[:, args.ordinal_idx] = encoder.transform(X_val[:, args.ordinal_idx])
        
        elif args.dataset == "Brazillian_Houses":

            encoder = OrdinalEncoder(categories=[[None,'not furnished','furnished']])

            # Fit and transform the data
            X[:, args.ordinal_idx] = encoder.fit_transform(X[:, args.ordinal_idx])
            #X_val[:, args.ordinal_idx] = encoder.transform(X_val[:, args.ordinal_idx])

            #print("OHE Done!!! \n")

    #Do for TabNet
    

        
        
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("After ORDINAL")
    print(f"Numerical Index V2 : {num_idx} \n\n")
    print(f"OHE Idx : {args.nominal_idx}\n\n")
    print(f"Ordinal Idx V2: {args.ordinal_idx}\n\n")
    print(f"Cat Dims V2 : {args.cat_dims}")
    print(f"Cat Idx V2 : {args.cat_idx} \n \n")
    print(f"Train: {X[:10,:]} \n \n ")
    print(f"Val : {X.shape} \n \n")
    print("FINISHED ENCODING")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("BINNING INIT")
    y = binning(args, y)
    print(f"y after binning: {y[:10]}")
    print("BINNING END")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n")

    if args.frequency_reg:
        #return X, y, X_val, y_val, freqency_map
        return X, y, freqency_map
    else:
        #return X, y, X_val, y_val
        return X, y
    
def binning(args, y):
    #Bin the target variable
    if args.objective == "probabilistic_regression":
        args.num_bins = bin_finder(args, y)

        if args.y_distribution == "bimodial":
            strategy = 'kmeans'
        else:
            strategy = 'quantile'
        
        binning = KBinsDiscretizer(n_bins=args.num_bins, encode='ordinal', strategy=strategy)
        y = binning.fit_transform(y.reshape(-1, 1)).flatten()
        #y_test = binning.transform(y_test.reshape(-1, 1)).flatten()
        args.num_classes = args.num_bins

        #print(f"Number of bins: {args.num_bins}")
        print(f"Number of Classes B4 Bin Verifier: {args.num_classes}")
        print(f"Unique values in y: {np.unique(y), len(np.unique(y))}")
        #print(f"Unique values in y_test: {np.unique(y_test), len(np.unique(y_test))}")

        y = y.astype(int)  # For NumPy arrays
        #y_test = y_test.astype(int)

        #Rectify bin
        y = bin_shifter(args, y)


        print("VERIFY SHIFT")
        print(f"Train after shift : {np.unique(y)}, Length : {len(np.unique(y))}")
        print(f"Number of Classes After Bin Verifier: {args.num_classes}")

    return y
    

    # X_train, X_val, y, y_val = train_test_split(X_train, y, test_size=0.05, random_state=args.seed)