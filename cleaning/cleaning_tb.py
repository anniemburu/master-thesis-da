# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import os

#dataset path
path = './datasets'

#datasets names
folder_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


#important info of the datasets
variable_dict = {
    "531-boston" : {'norm_cat' : ['CHAS'], 'ord_cat': None, 'target' : 'MEDV', 'drop_cols' : None},
    "541-socmob" : {'norm_cat' : ['fathers_occupation', 'sons_occupation', 'family_structure', 'race'],
                    'ord_cat': None,
                    'target' : 'counts_for_sons_current_occupation', 'drop_cols' : None},
    "546-sensory" : {'norm_cat' : ['Occasion', 'Judges', 'Interval', 'Sittings','Position', 'Squares',
                                    'Rows', 'Columns', 'Halfplot', 'Trellis', 'Method'],
                     'ord_cat': None, 'target' : 'Score', 'drop_cols' : None},
    "41021-Moneyball" : {'norm_cat' : ['Team', 'League','Playoffs', 'G'],
                         'ord_cat': None, 'target' : 'RS',
                         'drop_cols' : ['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']},
    "41540-black_friday" : {'norm_cat' : ['Gender', 'Occupation', 'City_Category', 'Marital_Status',
                                          'Product_Category_1','Product_Category_2', 'Product_Category_3'],
                            'ord_cat': ['Age'], 'target' : 'Purchase',
                            'drop_cols' : None},
    "41980-SAT11-HAND-runtime-regression" : {'norm_cat' : ['algorithm'], 'ord_cat': None,
                                             'target' : 'runtime', 'drop_cols': ['row_id']},
    "42225-diamonds" : {'norm_cat' : None, 'ord_cat': ['cut', 'color', 'clarity'], 'drop_cols': None,
                        'target' : 'price'},
    "42563-house_prices_nominal" : {'norm_cat' : ['MSZoning','LotShape', 'Neighborhood'],
                                    'ord_cat': ['Street', 'Alley', 'Utilities','LotConfig', 'LandSlope','LandContour','Condition1', 'Condition2',
                                                'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
                                                'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                                                'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType','GarageFinish',
                                                'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                                                'SaleCondition'],
                                    'target' : 'SalePrice', 'drop_cols':['Id']},
    "42570-Mercedes_Benz_Greener_Manufacturing" : {'norm_cat' : ['X0','X1','X2','X3','X4','X5','X6','X8'],
                                                   'ord_cat': None,
                                                   'target' : 'y',
                                                   'drop_cols':['ID']},
    "42571-Allstate_Claims_Severity" : {'norm_cat' : [f'cat{i}' for i in range(1,117)], 'ord_cat': None,
                                        'target' : 'loss', 'drop_cols':['id']},
    "42688-Brazilian_houses" : {'norm_cat' : ['city', 'animal'], 'ord_cat': ['furniture'], 'target' : 'total_(BRL)', 'drop_cols': None},
    "42726-abalone" : {'norm_cat' : ['Sex'], 'ord_cat': None, 'target' : 'Class_number_of_rings', 'drop_cols': None},
    "42728-Airlines_DepDelay_10M" : {'norm_cat' : ['Month','DayOfWeek','UniqueCarrier', 'Origin', 'Dest'],
                                     'ord_cat': None, 'target' : 'DepDelay', 'drop_cols': None},
    "42729-nyc-taxi-green-dec-2016" : {'norm_cat' : ['VendorID', 'store_and_fwd_flag','PULocationID', 'DOLocationID', 'trip_type'],
                                       'ord_cat': None, 'target' : 'tip_amount', 'drop_cols': None},
    "42731-house_sales" : {'norm_cat' : ['zipcode'], 'ord_cat': None , 'target' : 'price', 'drop_cols':['id']},
    "43071-MIP-2016-regression" : {'norm_cat' : ['algorithm','runstatus'], 'ord_cat': None, 'target' : 'PAR10', 'drop_cols':['instance_id']},

}

class data_cleaning:
    def __init__(self, dataset_list, variable_dict):
        self.dataset_list = dataset_list
        self.variable_dict = variable_dict


    def missing_duplicates(self, data, dataset):
        """
            Function drops any duplicates or missing values in the dataset.

            Parameters:
            data (DataFrame): Data to be cleaned.
            dataset (str): Name of the dataset.

            Returns:
            DataFrame: Returns the cleaned dataset as a dataframe.
        """
        ## Handling NULLS

        ### @42563-house_prices_nominal
        if(dataset == '42563-house_prices_nominal'):
            #groups
            missing_categoricals = {
                                    'Alley' : 'None', 'MasVnrType': 'None','BsmtQual': 'None',
                                    'BsmtCond': 'None', 'BsmtExposure': 'None', 'BsmtFinType1': 'None',
                                    'BsmtFinType2': 'None', 'Electrical': 'None','FireplaceQu': 'None', 'GarageType': 'None',
                                    'GarageFinish' : 'None', 'GarageQual' : 'None', 'GarageCond' : 'None',
                                    'PoolQC' : 'None', 'Fence' : 'None', 'MiscFeature': 'None'
                                    }

            missing_numericals = {
                                'LotFrontage': data['LotFrontage'].median(),
                                'MasVnrArea': data['MasVnrArea'].median(),
                                'GarageYrBlt' : data['GarageYrBlt'].median()
                                 }

            #replacements
            data.fillna(missing_categoricals, inplace=True) #categoricals
            data.fillna(missing_numericals, inplace=True) #numericals

        else:
            #general care
            pass

        ##drop useless
        if(self.variable_dict[dataset]['drop_cols'] is not None):
            data.drop(self.variable_dict[dataset]['drop_cols'], axis=1, inplace=True)

        ##drop duplicates and nulls
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)

        ## CHANGE COLUMN DTYPE
        if (dataset == '41540-black_friday'):
            data.loc[data['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = 4
            data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype(int)
        else:
            pass

        return data


    # ENCODING AND STANDARDIZATON
    def encoding_standardization(self, train, test, dataset):
        """
            Function drops any duplicates or missing values in the dataset.

            Parameters:
            train (DataFrame): train data to be cleaned.
            test (DataFrame): test data to be cleaned.
            datset (str): Name of the dataset.

            Returns:
            DataFrame: Returns the cleaned dataset as a dataframe.
        """
        nominal_cat = self.variable_dict[dataset]['norm_cat']
        ordinal_cat = self.variable_dict[dataset]['ord_cat']

        try:
            if((nominal_cat is not None) & (ordinal_cat is not None)):
                categorical_cols = nominal_cat + ordinal_cat
            elif(nominal_cat is not None):
                categorical_cols = nominal_cat
            else:
                categorical_cols = ordinal_cat
        except:
            pass

        #numerical cols
        numerical_cols = [col for col in train.columns if col not in categorical_cols]



        ## ORDINAL
        if ordinal_cat is not None:
            #@41540-black_friday
            if(dataset == '41540-black_friday'):
                ordinal_encoder = OrdinalEncoder(categories=[[None,'0-17','18-25','26-35','36-45','46-50','51-55','55+']])
                train[ordinal_cat] = ordinal_encoder.fit_transform(train[ordinal_cat])
                test[ordinal_cat] = ordinal_encoder.transform(test[ordinal_cat])

            elif(dataset == '42225-diamonds'):
                categories = [[None,'Fair','Good', 'Very Good', 'Premium', 'Ideal'],
                              [None,'J','I','H','G','F','E','D'],
                              [None,'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']]

                #preprocessor
                pre_processor = ColumnTransformer(transformers=[('orlEncdr_with_map',
                                        Pipeline(steps=[('orlEncdr_with_map', OrdinalEncoder(categories = categories, dtype=int))]),
                                       ordinal_cat)])
                train[ordinal_cat] = pre_processor.fit_transform(train[ordinal_cat])
                test[ordinal_cat] = pre_processor.transform(test[ordinal_cat])

            ### @42563-house_prices_nominal
            elif(dataset == '42563-house_prices_nominal'):
                categories = [
                    [None,'Grvl', 'Pave'],
                    ['None', 'Grvl', 'Pave'],
                    [None,'NoSeWa','AllPub'],
                    [None,'Inside', 'FR2', 'FR3', 'Corner', 'CulDSac'],
                    [None,'Sev', 'Mod', 'Gtl'],
                    [None, 'Low','Bnk','HLS','Lvl'],
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

                #preprocessor
                pre_processor = ColumnTransformer(transformers = [('orlEncdr_with_map',
                                        Pipeline(steps=[('orlEncdr_with_map', OrdinalEncoder(categories = categories, dtype=int))]),
                                       ordinal_cat)])
                train[ordinal_cat] = pre_processor.fit_transform(train[ordinal_cat])
                test[ordinal_cat] = pre_processor.transform(test[ordinal_cat])

            elif(dataset == '42688-Brazilian_houses'):
                ordinal_encoder = OrdinalEncoder(categories=[[None,'not furnished','furnished']])
                train[ordinal_cat] = ordinal_encoder.fit_transform(train[ordinal_cat])
                test[ordinal_cat] = ordinal_encoder.transform(test[ordinal_cat])


            else:
                pass
        else:
            pass #no ordinal variables

        ## NORMINAL
        if nominal_cat is not None:
            ##one hot encoding

            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            #encode
            train_encoded_features = encoder.fit_transform(train[nominal_cat])
            test_encoded_features = encoder.transform(test[nominal_cat])


            #encoded column
            train_encoded_df = pd.DataFrame(train_encoded_features,
                                            columns = encoder.get_feature_names_out(nominal_cat), index = train.index)

            test_encoded_df = pd.DataFrame(test_encoded_features,
                                            columns = encoder.get_feature_names_out(nominal_cat), index = test.index)

            #combine encoded and original
            train_encoded = pd.concat([train, train_encoded_df],  axis=1)
            test_encoded = pd.concat([test, test_encoded_df], axis=1)


            #drop the encoded columns
            train = train_encoded.drop(nominal_cat, axis=1)
            test = test_encoded.drop(nominal_cat, axis=1)

        else:
            pass

        ## NUMERICAL
        # Create a StandardScaler object

        if(len(numerical_cols) > 0):
            #skip mercedes coz it's sparse
            if(dataset == '42570-Mercedes_Benz_Greener_Manufacturing'):
                pass
            else:
                scaler = StandardScaler()

                # Fit and transform the numerical features
                train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
                test[numerical_cols] = scaler.transform(test[numerical_cols])
        else:
            pass

        return train, test

    def fit(self):
        #dataset paths
        data_path = './datasets'

        for dataset in self.dataset_list:
            print(f"We are in the dataset : {dataset}")

            #dataset
            data_df = pd.read_csv(f'{data_path}/{dataset}/raw_data.csv')

            #remove missing
            data_df = self.missing_duplicates(data_df,dataset)

            #X and y
            X = data_df.drop(self.variable_dict[dataset]['target'], axis=1)
            y = data_df[self.variable_dict[dataset]['target']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train, X_test = self.encoding_standardization(X_train, X_test, dataset)

            #concat
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            #save as csv
            X_train.to_csv(f'{data_path}/{dataset}/train.csv', index=False)
            X_test.to_csv(f'{data_path}/{dataset}/test.csv', index=False)

clean = data_cleaning(folder_names, variable_dict)

#action item
clean.fit()