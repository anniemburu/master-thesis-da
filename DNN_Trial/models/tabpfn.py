from models.basemodel import BaseModel
from tabpfn import TabPFNClassifier
from sklearn.metrics import log_loss , mean_squared_error
import numpy as np

class TabPFN(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        self.params = params
        
        if self.args.objective == "classification" or self.args.objective == "probabilistic_regression":
            self.model = TabPFNClassifier()
        else:
            raise ValueError("Objective must be either 'regression' or 'classification' or 'probabilistic_regression'.")
        

    def fit(self, X, y, X_val=None, y_val=None):

        self.model.fit(X, y)

        if self.args.objective == "classification" or self.args.objective == "probabilistic_regression":
            train_pred = self.model.predict_proba(X)
            train_loss = log_loss(y, train_pred)
            
        elif self.args.objective == "regression":
            #train_pred = self.model.predict(X)
            #train_loss = mean_squared_error(y, train_pred)
            pass
            
        valid_loss = []

        if X_val is not None and y_val is not None:
            if self.args.objective == "classification" or self.args.objective == "probabilistic_regression":
                val_pred = self.model.predict_proba(X_val)
                val_loss = log_loss(y_val, val_pred)
            
            elif self.args.objective == "regression":
                val_pred = self.model.predict(X_val)
                val_loss = mean_squared_error(y_val, val_pred)

            valid_loss.append(val_loss)
        
        print(f"Train Loss: {[train_loss]}, Valid Loss: {valid_loss}")

        return [train_loss], valid_loss
    
    def predict(self, X):
        return super().predict(X)
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    
    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "N_ensemble_configurations": trial.suggest_int("N_ensemble_configurations", 16, 64),
            "max_iters": trial.suggest_int("max_iters", 50, 500)
        }

        return params
