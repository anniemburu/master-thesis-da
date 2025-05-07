from rtdl_revisiting_models import MLP as RTDL_MLP, ResNet, FTTransformer


import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
#import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from models.basemodel_torch import BaseModelTorch

from utils.io_utils import get_output_path
from tqdm.std import tqdm

warnings.resetwarnings()

class ResMLP(BaseModelTorch):
   def __init__(self, params, args):
      super().__init__(params, args)

      self.params = params
      self.args = args

      self.params['d_in'] = args.num_features
      self.params['d_out'] = args.num_classes

      #model
      self.model = ResNet(
         d_in = self.params['d_in'],
         d_out = self.params['d_out'],
         n_blocks = self.params['n_blocks'],
         d_block = self.params['d_block'],
         d_hidden_multiplier = self.params['d_hidden_multiplier'],
         dropout1 = self.params['dropout1'],
         dropout2 = self.params['dropout2']
      ).to(self.device)

      # Optimizer
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)

      print("On Device:", self.device)


        
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):

      print("Types in B4 Model")
      print(f"X : {type(X)}, y : {type(y)}, X : {type(X_val)}, X : {type(X_val)} \n")
      print(f"In fit Nump iss")


      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.args.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.args.objective == 'regression' else torch.long).to(self.device)

      print(f"X : {type(X)}, y : {type(y)}, X : {type(X_val)}, X : {type(X_val)} \n")

      train_dataset = TensorDataset(X, y)
      loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            pin_memory=True
        )

      # Training loop
      self.model.train()
      loss_history = []
      val_history = []

      for epoch in range(self.args.epochs):
         epoch_loss = 0
         for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            #print(f"Batch X: {type(batch_X)} \n")
            #print(f"Batch y: {type(batch_y)} \n")
                
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = self._compute_loss(outputs, batch_y, class_weights)

            loss.backward()
            self.optimizer.step()

            #Calc and Store training loss
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)

            val_loss = self._evaluate(X_val, y_val, class_weights)
            val_history.append(val_loss)

            # Early stopping
            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
                
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break

         return loss_history , val_history

   def _compute_loss(self, outputs, targets, class_weights):
        if self.args.objective == 'regression':
            return nn.MSELoss()(outputs, targets)
        elif self.args.objective == "probabilistic_regression" or self.args.objective == "classification":
            if self.args.weighted_loss:
               return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets)
            else:
               return nn.CrossEntropyLoss()(outputs, targets)
            
   def predict(self, X):
      self.model.eval()
      X = torch.tensor(X, dtype=torch.float32).to(self.device)
      
      with torch.no_grad():
         outputs = self.model(X).squeeze()

         if self.args.objective == 'regression':
            self.predictions = outputs.detach().cpu().numpy()
            
         else:
            #print(f"Output in curr mod: {outputs.cpu().numpy()}")
            #return outputs.argmax(dim=1).cpu().numpy()
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
         #print(f"Predictions in curr mod: {self.predictions}")
         return self.predictions
      
   def predict_proba(self, X):
      if self.args.objective == 'regression':
         raise NotImplementedError("Method only available for classification tasks")
      else:
         self.model.eval()
         #X = torch.tensor(X, dtype=torch.float32).to(self.device)
         X = X.clone().detach().to(torch.float32).to(self.device)
         output = torch.softmax(self.model(X), dim=1)
         probabilities = output.detach().cpu().numpy()

         self.prediction_probabilities = probabilities
         #print(f"Probabilities in curr mod: {self.prediction_probabilities}")
         return self.prediction_probabilities

   def _evaluate(self, X, y, class_weights):
         self.model.eval()
         #X = torch.tensor(X, dtype=torch.float32).to(self.device)
         X = X.clone().detach().to(torch.float32).to(self.device)

         with torch.no_grad():
            outputs = self.model(X).squeeze()
            loss = self._compute_loss(outputs, y, class_weights).item()
         return loss

   

    
   @classmethod
   def define_trial_parameters(cls, trial, args):
      params = {
         "d_in": args.num_features,
         "d_out": args.num_classes,
         "n_blocks": trial.suggest_int("n_blocks", 2, 6, log=True),
         "d_block": trial.suggest_categorical('d_block', [64, 128, 256]),
         "d_hidden": trial.suggest_int("d_hidden", 2, 10, log=True),
         "d_hidden_multiplier" : trial.suggest_float("d_hidden_multiplier", 1.0, 5.0, log=True),
         "dropout1" : trial.suggest_float("dropout1", 1e-8,0.5, log=False),
         "dropout2" : trial.suggest_float("dropout2", 1e-8, 0.3, log=False),
      }
      return params
   

'''
      MPL Model : 
'''

class MLP(BaseModelTorch):
   def __init__(self, params, args):
      super().__init__(params, args)

      self.params = params
      self.args = args

      self.params['d_in'] = args.num_features
      self.params['d_out'] = args.num_classes

      #model
      self.model = RTDL_MLP(
         d_in = self.params['d_in'],
         d_out = self.params['d_out'],
         n_blocks = self.params['n_blocks'],
         d_block = self.params['d_block'],
         dropout = self.params['dropout']
      ).to(self.device)

      # Optimizer
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)

      print("On Device:", self.device)


        
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):
       #self.model = ResNet().to(self.device)
       #optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)

      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.args.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.args.objective == 'regression' else torch.long).to(self.device)

      train_dataset = TensorDataset(X, y)
      loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            pin_memory=True
        )

      # Training loop
      self.model.train()
      loss_history = []
      val_history = []

      for epoch in range(self.args.epochs):
         epoch_loss = 0
         for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            #(f"Batch X: {type(batch_X)} \n")
            #print(f"Batch y: {type(batch_y)} \n")
                
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = self._compute_loss(outputs, batch_y, class_weights)

            loss.backward()
            self.optimizer.step()

            #Calc and Store training loss
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)

            val_loss = self._evaluate(X_val, y_val, class_weights)
            val_history.append(val_loss)

            # Early stopping
            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
                
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break

         return loss_history , val_history

   def _compute_loss(self, outputs, targets, class_weights):
        if self.args.objective == 'regression':
            return nn.MSELoss()(outputs, targets)
        elif self.args.objective == "probabilistic_regression" or self.args.objective == "classification":
            if self.args.weighted_loss:
               return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets)
            else:
               return nn.CrossEntropyLoss()(outputs, targets)
            
   def predict(self, X):
      self.model.eval()
      print(f"X in predict: {type(X)}")
      X = torch.tensor(X, dtype=torch.float32).to(self.device)
      
      with torch.no_grad():
         outputs = self.model(X).squeeze()

         if self.args.objective == 'regression':
            self.predictions = outputs.detach().cpu().numpy()
            
         else:
            #print(f"Output in curr mod: {outputs.cpu().numpy()}")
            #return outputs.argmax(dim=1).cpu().numpy()
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
         #print(f"Predictions in curr mod: {self.predictions}")
         return self.predictions
      
   def predict_proba(self, X):
      if self.args.objective == 'regression':
         raise NotImplementedError("Method only available for classification tasks")
      else:
         self.model.eval()
         print(f"X in predict proba: {type(X)}")

         #X = torch.tensor(X, dtype=torch.float32).to(self.device)
         X = X.clone().detach().to(torch.float32).to(self.device)
         output = torch.softmax(self.model(X), dim=1)
         probabilities = output.detach().cpu().numpy()

         self.prediction_probabilities = probabilities
         #print(f"Probabilities in curr mod: {self.prediction_probabilities}")
         return self.prediction_probabilities

   def _evaluate(self, X, y, class_weights):
         self.model.eval()
         print(f"X in evaluate: {type(X)}")

         #X = torch.tensor(X, dtype=torch.float32).to(self.device)
         X = X.clone().detach().to(torch.float32).to(self.device)

         with torch.no_grad():
            outputs = self.model(X).squeeze()
            loss = self._compute_loss(outputs, y, class_weights).item()
         return loss
    
   @classmethod
   def define_trial_parameters(cls, trial, args):
      params = {
         "d_in": args.num_features,
         "d_out": args.num_classes,
         "n_blocks": trial.suggest_int("n_blocks", 2, 6, log=True),
         "d_block": trial.suggest_categorical('d_block', [64, 128, 256]),
         "dropout" : trial.suggest_float("dropout", 1e-8, 0.3, log=False),
      }
      return params
   
class FTTransformerWrapper(BaseModelTorch):
   def __init__(self, params, args):
      super().__init__(params, args)

      self.params = params
      self.args = args

      #self.params['d_in'] = args.num_features
      #self.params['d_out'] = args.num_classes

  
      self.model = FTTransformer(
         n_cont_features= len(self.args.num_idx) if self.args.num_idx is not None else 0,
         cat_cardinalities= self.args.cat_dims if self.args.cat_dims is not None else [],
         d_out = self.args.num_classes,
         **FTTransformer.get_default_kwargs(),
                     ).to(self.device)

      # Optimizer
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])

      print("On Device:", self.device)


        
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):
      print(f"Train Data b4 training...")
      print(f"X: {type(X)} , y : {type(y), }, X_val: {type(X_val)} , y : {type(y_val)}\n")
      
      #Convert to NP array
      X = np.asarray(X, dtype=np.float32)
      y = np.asarray(y, dtype=np.float32 if self.args.objective == 'regression' else np.int64)
      X_val = np.asarray(X_val, dtype=np.float32) if X_val is not None else None
      y_val = np.asarray(y_val, dtype=np.float32 if self.args.objective == 'regression' else np.int64) if y_val is not None else None


      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.args.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.args.objective == 'regression' else torch.long).to(self.device)

      print(f"Train Data after...")
      print(f"X: {type(X)} , y : {type(y), }, X_val: {type(X_val)} , y : {type(y_val)}\n")

      train_dataset = TensorDataset(X, y)
      loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            pin_memory=True
        )

      # Training loop
      self.model.train()
      loss_history = []
      val_history = []

      for epoch in range(self.args.epochs):
         epoch_loss = 0
         for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            #print(f"Batch X: {type(batch_X)} \n")
            #print(f"Batch y: {type(batch_y)} \n")
                
            self.optimizer.zero_grad()
            x_cont, x_cat = self._split_inputs(batch_X)

            outputs = self.model(x_cont, x_cat)
            #outputs = self.model(batch_X).squeeze()
            loss = self._compute_loss(outputs, batch_y, class_weights)

            loss.backward()
            self.optimizer.step()

            #Calc and Store training loss
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)

            val_loss = self._evaluate(X_val, y_val, class_weights)
            val_history.append(val_loss)

            # Early stopping
            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
                
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break

         return loss_history , val_history

   def _compute_loss(self, outputs, targets, class_weights):
        if self.args.objective == 'regression':
            return nn.MSELoss()(outputs, targets)
        elif self.args.objective == "probabilistic_regression" or self.args.objective == "classification":
            if self.args.weighted_loss:
               return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets)
            else:
               return nn.CrossEntropyLoss()(outputs, targets)
            
   def predict(self, X):
      self.model.eval()
      X = torch.tensor(X, dtype=torch.float32).to(self.device)
      
      with torch.no_grad():
         x_cont, x_cat = self._split_inputs(X)

         outputs = self.model(x_cont, x_cat)
         #outputs = self.model(X).squeeze()

         if self.args.objective == 'regression':
            self.predictions = outputs.detach().cpu().numpy()
            
         else:
            #print(f"Output in curr mod: {outputs.cpu().numpy()}")
            #return outputs.argmax(dim=1).cpu().numpy()
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
         #print(f"Predictions in curr mod: {self.predictions}")
         return self.predictions
      
   def predict_proba(self, X):
      if self.args.objective == 'regression':
         raise NotImplementedError("Method only available for classification tasks")
      else:
         self.model.eval()
         x_cont, x_cat = self._split_inputs(X)

         outputs = torch.softmax(self.model(x_cont, x_cat), dim=1)
         probabilities = outputs.detach().cpu().numpy()

         self.prediction_probabilities = probabilities
         #print(f"Probabilities in curr mod: {self.prediction_probabilities}")
         return self.prediction_probabilities

   def _evaluate(self, X, y, class_weights):
         self.model.eval()
         X = torch.tensor(X, dtype=torch.float32).to(self.device)

         with torch.no_grad():
            x_cont, x_cat = self._split_inputs(X)

            outputs = self.model(x_cont, x_cat)
            loss = self._compute_loss(outputs, y, class_weights).item()
         return loss
   
   def _split_inputs(self, X):
      print(f"I am in _split input")
      #print(f"Num : {self.args.num_idx}, Cats : {self.args.cat_idx}")

      x_cont = X[:, self.args.num_idx] if self.args.num_idx else None
      x_cat = X[:, self.args.cat_idx] if self.args.cat_idx else np.empty((X.shape[0], 0))

      #x_cat = torch.tensor(x_cat, dtype=torch.int64).to(self.device)
      #x_cont = torch.tensor(x_cont, dtype=torch.float32).to(self.device)
      if x_cat is not None:
         x_cat = x_cat.long().to(self.device)
      if x_cont is not None:
         x_cont = x_cont.float().to(self.device)
 
      #print(f"x_cont: {type(x_cont)} , x_cat: {type(x_cat)} \n")
      return x_cont, x_cat

    
   @classmethod
   def define_trial_parameters(cls, trial, args):
      params = {
         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
      }
      return params

