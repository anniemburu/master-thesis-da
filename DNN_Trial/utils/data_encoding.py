
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

def encoding(args, X_train, y_train, X_val, y_val, args_temps):
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print("INSIDE ENCODING")
    #print("Checking if the args dict holds..... \n")

    #for key, value in args_temps.items():
        #print(f"{key} : {value}")


    #print("See if it updates our args")
    #Reset after ever fold
    args.num_features = args_temps["num_features"]
    args.num_classes = args_temps["num_classes"]
    args.cat_idx = args_temps["cat_idx"]
    args.nominal_idx = args_temps["nominal_idx"]
    args.ordinal_idx = args_temps["ordinal_idx"]
    args.num_idx = args_temps["num_idx"]
    args.cat_dims = args_temps["cat_dims"]
    args.bin_alt = args_temps["bin_alt"]
    
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
    print(f"X shape : {X_train.shape}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Preprocess target 
    if args.target_encode:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)

    num_idx = [] # Index of numerical features
    args.cat_dims = [] # dimensions for categorical features
    args.cat_idx = get_catidx(args) # Index of categorical features

    #print(f"Nominal Index : {args.nominal_idx}")
    #print(f"Ordinal Index : {args.ordinal_idx}")
    #print(f"Cat Idx : {args.cat_idx}")
    #print(f"Cat dims : {args.cat_dims}")

    #print(f"X_train shape before encoding : {X_train.shape}")
    #print(f"X_train before encoding : {X_train[:10,:]}")
    

   
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
                X_train[:, i] = le.fit_transform(X_train[:, i])
                args.cat_dims.append(len(le.classes_))
            else:
                if args.ordinal_idx and i in args.ordinal_idx:
                    le = LabelEncoder()
                    #X[:, i] = le.fit_transform(X[:, i])
                    le.fit_transform(X_train[:, i])

                    # Gets number of unique classes per ordinal feature
                    #Covers future cases with None
                    if np.any(X_train[:, i] == "None"):
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
        X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
        X_val[:, num_idx] = scaler.transform(X_val[:, num_idx])


    #Encode Nominal Features
    if args.one_hot_encode:
        print("One Hot Encoding...")
        #print(f"Nominal Index : {args.nominal_idx}")
        #print(f"Ordinal Index : {args.ordinal_idx}")
        #print(f"Numerical Index : {num_idx}")
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X_train[:, args.nominal_idx])
        new_x2 = X_train[:, num_idx]

        #X_val
        new_x1_val = ohe.transform(X_val[:, args.nominal_idx])
        new_x2_val = X_val[:, num_idx]


        if args.ordinal_encode:
            ord_len = len(args.ordinal_idx)
            new_ord = X_train[:, args.ordinal_idx]
            new_ord_val = X_val[:, args.ordinal_idx]

            args.ordinal_idx = [x for x in range(ord_len)] #update ordinal idx
            #print(f"Ordinal Idx Updated: {args.ordinal_idx}")
            X_train = np.concatenate([new_ord, new_x1, new_x2], axis=1)
            X_val = np.concatenate([new_ord_val, new_x1_val, new_x2_val], axis=1)

            #Update Nominal idx
            args.nominal_idx = [x+len(args.ordinal_idx) for x in range(new_x1.shape[1])]

        else:
            X_train = np.concatenate([new_x1, new_x2], axis=1)
            X_val = np.concatenate([new_x1_val, new_x2_val], axis=1)

            args.nominal_idx = [x for x in range(new_x1.shape[1])]

        #change the num of features after one hot encoding;
        args.num_features = X_train.shape[1] #here is the issue
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
        #print("New Shape:", X_train.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("After OHE")
    print(f"Numerical Index V2 : {num_idx} \n\n")
    print(f"OHE Idx : {args.nominal_idx}\n\n")
    print(f"Ordinal Idx V2: {args.ordinal_idx}\n\n")
    print(f"Cat Dims V2 : {args.cat_dims}")
    print(f"Cat Idx V2 : {args.cat_idx} \n \n")
    print(f"Train: {X_train[:10,:5]} \n \n ")
    print(f"Val : {X_train.shape} \n \n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
       
        

    # Ordinal Encode
    if args.ordinal_encode:
        if args.dataset == "Black_Friday":
            #print(f"Ordinal Index b4 using OE : {args.ordinal_idx}")
            ordinal_encoder = OrdinalEncoder(categories=[[None,'0-17','18-25','26-35','36-45','46-50','51-55','55+']])
            X_train[:, args.ordinal_idx] = ordinal_encoder.fit_transform(X_train[:, args.ordinal_idx])
            X_val[:, args.ordinal_idx] = ordinal_encoder.transform(X_val[:, args.ordinal_idx])

        elif args.dataset == "Diamonds":
            categories = [
                            [None, 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],  # For 'cut'
                            [None, 'J', 'I', 'H', 'G', 'F', 'E', 'D'],  # For 'color'
                            [None, 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # For 'clarity'
                        ]

            # Create the OrdinalEncoder
            encoder = OrdinalEncoder(categories=categories, dtype=int)

            # Fit and transform the data
            X_train[:, args.ordinal_idx] = encoder.fit_transform(X_train[:, args.ordinal_idx])
            X_val[:, args.ordinal_idx] = encoder.transform(X_val[:, args.ordinal_idx])

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
            X_train[:, args.ordinal_idx] = encoder.fit_transform(X_train[:, args.ordinal_idx])
            X_val[:, args.ordinal_idx] = encoder.transform(X_val[:, args.ordinal_idx])
        
        elif args.dataset == "Brazillian_Houses":

            encoder = OrdinalEncoder(categories=[[None,'not furnished','furnished']])

            # Fit and transform the data
            X_train[:, args.ordinal_idx] = encoder.fit_transform(X_train[:, args.ordinal_idx])
            X_val[:, args.ordinal_idx] = encoder.transform(X_val[:, args.ordinal_idx])

            #print("OHE Done!!! \n")

    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("After ORDINAL")
    print(f"Numerical Index V2 : {num_idx} \n\n")
    print(f"OHE Idx : {args.nominal_idx}\n\n")
    print(f"Ordinal Idx V2: {args.ordinal_idx}\n\n")
    print(f"Cat Dims V2 : {args.cat_dims}")
    print(f"Cat Idx V2 : {args.cat_idx} \n \n")
    print(f"Train: {X_train[:10,:]} \n \n ")
    print(f"Val : {X_train.shape} \n \n")
    print("FINISHED ENCODING")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if args.frequency_reg:
        return X_train, y_train, X_val, y_val, freqency_map
    else:
        return X_train, y_train, X_val, y_val