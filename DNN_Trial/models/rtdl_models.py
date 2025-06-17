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
import sklearn.utils
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

from models.basemodel_torch import BaseModelTorch

from utils.io_utils import get_output_path
from tqdm.std import tqdm

from train import freedman_diaconis, sturges, bin_finder, bin_shifter, impute_missing_test, binning

warnings.resetwarnings()

class ResMLP(BaseModelTorch):
   def __init__(self, params, args):
      super().__init__(params, args)

      self.params = params
      self.args = args

      if self.args.objective == "ordinal_regression":
         self.n_bins = self.args.num_bins
         self.params['d_out'] = self.n_bins - 1
      else:
         self.params['d_out'] = args.num_classes

      self.params['d_in'] = args.num_features

      self.lambda_reg =  torch.nn.Parameter(torch.log(torch.tensor(0.01))) # Initialize with 0.01  #FRR

      self.bin_means = None
      self.discretizer = None

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
      self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + [self.lambda_reg], lr=3e-4, weight_decay=1e-5)

      """print("On Device:", self.device)
      print(f"Model: {self.model}")
      for name, param in self.model.named_parameters():
         print(name, param.shape, param.view(-1)[0].item())
      
      for name, param in self.model.state_dict().items():
         print("In here")
         print(name, param.shape, param.view(-1)[0].item())
      
      print(f"Wights for first layer: {self.model.input_projection.weight}")
      print(f"Weight shape: {self.model.input_projection.weight.shape}")
     """
       

   def seed_worker(self, worker_id):
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      random.seed(worker_seed)

#  Generator for DataLoader
   def get_generator(self, seed):
      g = torch.Generator()
      g.manual_seed(seed)
      return g
        
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):

      print("Types in B4 Model")
      print(f"X : {type(X)}, y : {type(y)}, X : {type(X_val)}, X : {type(X_val)} \n")
      print(f"In fit Nump iss")

      y_original = y.copy()  # Store original y for ordinal regression
      y_val_original = y_val.copy() if y_val is not None else None

      if self.args.objective == "ordinal_regression": 
         #create bins
         y_train_binned, y_val_binned = binning(self.args, y)

         self.bin_means = np.array([y[y_train_binned == k].mean() for k in range(self.n_bins)])

         y = self._create_ordinal_targets(y_train_binned)
         #y_val = self._create_ordinal_targets(y_val_binned)


      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.args.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device) if X_val is not None else None
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.args.objective == 'regression' else torch.long).to(self.device) if y_val is not None else None
      class_weights = class_weights.to(self.device) if class_weights is not None else None

      print(f"X : {type(X)}, y : {type(y)}, X : {type(X_val)}, X : {type(X_val)} \n")

      if self.args.optimize_hyperparameters is None:
         print(f"Seed in MLP: {self.args.test_seed}")
         g = self.get_generator(self.args.test_seed) #seed gen

         train_dataset = TensorDataset(X, y)
         loader = DataLoader(
               train_dataset, 
               batch_size=self.args.batch_size, 
               shuffle=True, 
               pin_memory=True,
               num_workers=4,
               worker_init_fn=self.seed_worker,
               generator=g
         )
      else:
         print("hyperparameter optimization")
         train_dataset = TensorDataset(X, y)
         loader = DataLoader(
               train_dataset, 
               batch_size=self.args.batch_size, 
               shuffle=True,
         )

      # Training loop
      self.model.train()
      loss_history = []
      val_history = []
      lambda_history = []

      for epoch in range(self.args.epochs):
         epoch_loss = 0
         epoch_loss_val = 0
         for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            #print(f"Batch X: {type(batch_X)} \n")
            #print(f"Batch y: {type(batch_y)} \n")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Frequency regularization term
            if frequency_map is not None:
               if self.args.ordinal_encode:
                  idx_buff = len(self.args.ordinal_idx)
               else:
                  idx_buff = 0  # Ordinal feat are stored b4 OHE

               weights = self.model.input_projection.weight # Get weights for one-hot encoded features
      
               penalty = 0.0
               for i, col in enumerate(frequency_map.keys()):
                  penalty += torch.sum(torch.abs(weights[:, i + idx_buff])) / (frequency_map[col] + 1e-8)  #i add len ordinal
               penalty *=  torch.exp(self.lambda_reg)   # Use the learnable lambda_reg
            else:
               penalty = 0.0
               
                #~~~~
                
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            if self.args.objective == "ordinal_regression":
               loss = self.ordinal_loss(outputs, batch_y) + penalty #add penalty
            else:
               loss = self._compute_loss(outputs, batch_y, class_weights) + penalty #add penalty

            loss.backward()
            self.optimizer.step()

            #Calc and Store training loss
            epoch_loss += loss.item()

            if X_val is not None:
               if self.args.objective == "ordinal_regression":
                  with torch.no_grad():
                     val_pred = self.predict(X_val)
                     val_loss = mean_squared_error(y_val, val_pred)
               else:
                  val_loss = self._evaluate(X_val, y_val, class_weights)
            
               epoch_loss_val += val_loss

         avg_loss = epoch_loss / len(loader)
         avg_loss_val = epoch_loss_val / len(loader)
         lambda_history.append(torch.exp(self.lambda_reg).item())
         
         if X_val is not None:
            loss_history.append(avg_loss)
            val_history.append(avg_loss_val)
         
            # Early stopping
            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
                  
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break

      if self.args.frequency_reg:
         return loss_history , val_history, lambda_history
      else:
         return loss_history , val_history

   def _create_ordinal_targets(self, y_binned):
      """
      Create ordinal targets from binned targets.
      Converts binned targets into a binary matrix where each column represents a bin.
      #Convert bin indices to cumulative binary targets.
      """
      targets = np.zeros((len(y_binned), self.n_bins-1))
      for i, bin_idx in enumerate(y_binned):
         if bin_idx > 0:
               targets[i, :int(bin_idx)] = 1.0
      return torch.tensor(targets, dtype=torch.float32)
    
      
   def ordinal_loss(self, outputs, targets):
      """
      Calculate the ordinal regression loss.
      Weighted binary cross-entropy loss that respects ordinal relationships
      Penalizes distant misclassifications more than adjacent ones

      Args:
          outputs: Model predictions.
          targets: True labels.
      Returns:
          Ordinal regression loss.
      """
      true_bins = targets.sum(dim=1)  # Get the true bin indices
      threshold_idx = torch.arange(1, self.n_bins, device=self.device) # Exclude the last bin
      weights = torch.abs(true_bins.unsqueeze(1) - threshold_idx)  # Calculate weights based on distance to true bin

      # Calculate the binary cross-entropy loss for each bin (Weighted BCE Loss)
      bce = nn.BCEWithLogitsLoss(reduction='none')
      #print(f"Type of outputs: {type(outputs)}, Type of target : {type(targets)}" )  
      #print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
      #print(f"Outputs: {outputs}, Targets: {targets}")
      loss = bce(outputs, targets.float())

      return (loss * weights).mean()  # Average loss across all samples
   
   def _ordinal_to_continuous(self, outputs):
      """Convert threshold outputs to continuous predictions"""
      probs = torch.sigmoid(outputs).cpu().numpy()
      bin_probs = np.zeros((len(probs), self.n_bins))
      
      bin_probs[:, 0] = 1 - probs[:, 0]  # P(bin0)
      for k in range(1, self.n_bins-1):
         bin_probs[:, k] = probs[:, k-1] - probs[:, k]  # P(bin_k)
      bin_probs[:, -1] = probs[:, -1]  # P(last_bin)
      
      return bin_probs @ self.bin_means  # Weighted average

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

         if self.args.objective == 'ordinal_regression':
            self.predictions = self._ordinal_to_continuous(outputs)

         elif self.args.objective == 'regression':
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

         if self.args.objective == 'ordinal_regression':
            outputs = self.model(X)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()

            bin_probs = np.zeros((len(probs), self.n_bins))
            bin_probs[:, 0] = 1 - probs[:, 0]
            for k in range(1, self.n_bins-1):
                bin_probs[:, k] = probs[:, k-1] - probs[:, k]
            bin_probs[:, -1] = probs[:, -1]

            self.prediction_probabilities = bin_probs

         else:
            output = torch.softmax(self.model(X), dim=1)
            probabilities = output.detach().cpu().numpy()

            self.prediction_probabilities = probabilities
            #print(f"Probabilities in curr mod: {self.prediction_probabilities}")
         return self.prediction_probabilities

   def _evaluate(self, X, y, class_weights):
      if self.args.objective == "ordinal_regression":
         y_pred = self.predict(X)
         loss = mean_squared_error(y.cpu().numpy(), y_pred)
         
      else:
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
         "n_blocks": trial.suggest_categorical("n_blocks", [2, 4, 6]),
         "d_block": trial.suggest_categorical('d_block', [64, 128, 256]),
         "d_hidden": trial.suggest_categorical("d_hidden", [1.0, 2.0, 3.0, 4.0, 5.0]),
         "d_hidden_multiplier" : trial.suggest_categorical("d_hidden_multiplier", [1.0, 2.0, 3.0, 4.0, 5.0]),
         "dropout1" : trial.suggest_categorical("dropout1", [1e-8, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
         "dropout2" : trial.suggest_categorical("dropout2", [1e-8, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
      }
      return params 
   
   @classmethod
   def define_grid_parameters(cls):
      search_space = {
         "n_blocks": [2, 4, 6],
         "d_block": [64, 128, 256],
         "d_hidden": [1.0, 2.0, 3.0, 4.0, 5.0],
         "d_hidden_multiplier": [1.0, 2.0, 3.0, 4.0, 5.0],
         "dropout1": [1e-8, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
         "dropout2": [1e-8, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
      }

      return search_space

'''
      MPL Model : 
'''

class MLP(BaseModelTorch):
   def __init__(self, params, args):
      super().__init__(params, args)

      self.params = params
      self.args = args

      if self.args.objective == "ordinal_regression":
         self.n_bins = self.args.num_bins
         self.params['d_out'] = self.n_bins - 1
      else:
         self.params['d_out'] = args.num_classes

      self.params['d_in'] = args.num_features

      self.lambda_reg =  torch.nn.Parameter(torch.log(torch.tensor(0.01))) # Initialize with 0.01  #FRR

      self.bin_means = None
      self.discretizer = None


      #model
      self.model = RTDL_MLP(
         d_in = self.params['d_in'],
         d_out = self.params['d_out'],
         n_blocks = self.params['n_blocks'],
         d_block = self.params['d_block'],
         dropout = self.params['dropout']
      ).to(self.device)

      # Add lambda_reg as a trainable parameter

      # Optimizer
      self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + [self.lambda_reg], lr=3e-4, weight_decay=1e-5)

      print("On Device:", self.device)
      """
      print(f"Model: {self.model}")
      for name, param in self.model.named_parameters():
         print(name, param.shape, param.view(-1)[0].item())
      
      for name, param in self.model.state_dict().items():
         print("In here")
         print(name, param.shape, param.view(-1)[0].item())
      
      print(f"Wights for first layer: {self.model.blocks[0].linear.weight}")
      print(f"Weight shape: {self.model.blocks[0].linear.weight.shape}")
      """

   def seed_worker(self, worker_id):
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      random.seed(worker_seed)

#  Generator for DataLoader
   def get_generator(self, seed):
      g = torch.Generator()
      g.manual_seed(seed)
      return g

        
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):
       #self.model = ResNet().to(self.device)
       #optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
      y_original = y.copy()  # Store original y for ordinal regression
      y_val_original = y_val.copy() if y_val is not None else None

      if self.args.objective == "ordinal_regression": 
         #create bins
         y_train_binned, y_val_binned = binning(self.args, y)
         
         self.bin_means = np.array([y[y_train_binned == k].mean() for k in range(self.n_bins)])

         print(f"Y origi : {y[:10]}")
         print(f"Binned : {y_train_binned[:10]}")
         print(f"Bin Mean : {self.bin_means}")
         
         y = self._create_ordinal_targets(y_train_binned)
         #y_val = self._create_ordinal_targets(y_val_binned)


      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.args.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device) if X_val is not None else None
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.args.objective == 'regression' else torch.long).to(self.device) if y_val is not None else None
      class_weights = class_weights.to(self.device) if class_weights is not None else None
      
      if self.args.optimize_hyperparameters is None:
         print(f"Seed in MLP: {self.args.test_seed}")
         g = self.get_generator(self.args.test_seed) #seed gen

         train_dataset = TensorDataset(X, y)
         loader = DataLoader(
               train_dataset, 
               batch_size=self.args.batch_size, 
               shuffle=True, 
               pin_memory=True,
               num_workers=4,
               worker_init_fn=self.seed_worker,
               generator=g
         )
      else:
         print("hyperparameter optimization")
         train_dataset = TensorDataset(X, y)
         loader = DataLoader(
               train_dataset, 
               batch_size=self.args.batch_size, 
               shuffle=True,
         )

      # Training loop
      self.model.train()
      loss_history = []
      val_history = []
      lambda_history = []

      for epoch in range(self.args.epochs):
         epoch_loss = 0
         epoch_loss_val = 0
         for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            #(f"Batch X: {type(batch_X)} \n")
            #print(f"Batch y: {type(batch_y)} \n")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Frequency regularization term
            if frequency_map is not None:
               if self.args.ordinal_encode:
                  idx_buff = len(self.args.ordinal_idx)
               else:
                  idx_buff = 0  # Ordinal feat are stored b4 OHE

               weights = self.model.blocks[0].linear.weight # Get weights for one-hot encoded features
      
               penalty = 0.0
               for i, col in enumerate(frequency_map.keys()):
                  penalty += torch.sum(torch.abs(weights[:, i + idx_buff])) / (frequency_map[col] + 1e-8)  #i add len ordinal
               penalty *=  torch.exp(self.lambda_reg)   # Use the learnable lambda_reg
            else:
               penalty = 0.0
               
                #~~~~  
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()

            if self.args.objective == "ordinal_regression":
               loss = self.ordinal_loss(outputs, batch_y) + penalty #add penalty
            else:
               loss = self._compute_loss(outputs, batch_y, class_weights) + penalty #add penalty

            loss.backward()
            self.optimizer.step()
         
            if X_val is not None:
               if self.args.objective == "ordinal_regression":
                  with torch.no_grad():
                     val_pred = self.predict(X_val)
                     val_loss = mean_squared_error(y_val, val_pred)
               else:
                  val_loss = self._evaluate(X_val, y_val, class_weights)

               epoch_loss_val += val_loss

            #Calc and Store training and val loss
            epoch_loss += loss.item()
            

         avg_loss = epoch_loss / len(loader)
         loss_history.append(avg_loss)

         lambda_history.append(torch.exp(self.lambda_reg).item())

         if X_val is not None:
            avg_loss_val = epoch_loss_val / len(loader)
            val_history.append(avg_loss_val)
            
            # Early stopping 
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break

            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
         

      if self.args.frequency_reg:
         return loss_history , val_history, lambda_history
      else:
         return loss_history , val_history
   
   def _create_ordinal_targets(self, y_binned):
      """
      Create ordinal targets from binned targets.
      Converts binned targets into a binary matrix where each column represents a bin.
      #Convert bin indices to cumulative binary targets.
      """
      targets = np.zeros((len(y_binned), self.n_bins-1))
      for i, bin_idx in enumerate(y_binned):
         if bin_idx > 0:
               targets[i, :int(bin_idx)] = 1.0
      return torch.tensor(targets, dtype=torch.float32)
    
      
   def ordinal_loss(self, outputs, targets):
      """
      Calculate the ordinal regression loss.
      Weighted binary cross-entropy loss that respects ordinal relationships
      Penalizes distant misclassifications more than adjacent ones

      Args:
          outputs: Model predictions.
          targets: True labels.
      Returns:
          Ordinal regression loss.
      """
      true_bins = targets.sum(dim=1)  # Get the true bin indices
      threshold_idx = torch.arange(1, self.n_bins, device=self.device) # Exclude the last bin
      weights = torch.abs(true_bins.unsqueeze(1) - threshold_idx)  # Calculate weights based on distance to true bin

      # Calculate the binary cross-entropy loss for each bin (Weighted BCE Loss)
      bce = nn.BCEWithLogitsLoss(reduction='none')
      #print(f"Type of outputs: {type(outputs)}, Type of target : {type(targets)}" )  
      #print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
      #print(f"Outputs: {outputs}, Targets: {targets}")
      loss = bce(outputs, targets.float())

      return (loss * weights).mean()  # Average loss across all samples
   
   def _ordinal_to_continuous(self, outputs):
      """Convert threshold outputs to continuous predictions"""
      probs = torch.sigmoid(outputs).cpu().numpy()
      bin_probs = np.zeros((len(probs), self.n_bins))

      print(f"We are in pred ordinal to continuous")
      print(f"Probs in pred ordinal to continuous: {probs.shape}")
      print(f"Probs in pred ordinal to continuous: {probs}")
      print(f"bin prob: {bin_probs}")
      print(f"Bin means in pred ordinal to continuous: {self.bin_means}")
      
      bin_probs[:, 0] = 1 - probs[:, 0]  # P(bin0)
      for k in range(1, self.n_bins-1):
         bin_probs[:, k] = probs[:, k-1] - probs[:, k]  # P(bin_k)
      bin_probs[:, -1] = probs[:, -1]  # P(last_bin)
      
      return bin_probs @ self.bin_means  # Weighted average
      

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
         #print("WE ARE IN THE PREDICT METHOD")
         #print(f"Outputs in curr mod: {outputs.shape}")
         #print(f"Outputs in curr mod: {outputs.cpu().numpy()}")

         if self.args.objective == 'ordinal_regression':
            self.predictions = self._ordinal_to_continuous(outputs)
            print(f"Predictions in curr mod: {self.predictions.shape}")
            print(f"Predictions in curr mod: {self.predictions}")
         elif self.args.objective == 'regression':
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

         if self.args.objective == 'ordinal_regression':
            outputs = self.model(X)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()

            bin_probs = np.zeros((len(probs), self.n_bins))
            bin_probs[:, 0] = 1 - probs[:, 0]
            for k in range(1, self.n_bins-1):
                bin_probs[:, k] = probs[:, k-1] - probs[:, k]
            bin_probs[:, -1] = probs[:, -1]

            self.prediction_probabilities = bin_probs

         else:
            output = torch.softmax(self.model(X), dim=1)
            probabilities = output.detach().cpu().numpy()

            self.prediction_probabilities = probabilities
            #print(f"Probabilities in curr mod: {self.prediction_probabilities}")
         return self.prediction_probabilities

   def _evaluate(self, X, y, class_weights):
      if self.args.objective == "ordinal_regression":
         y_pred = self.predict(X)
         loss = mean_squared_error(y.cpu().numpy(), y_pred)
      else:
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
         "n_blocks": trial.suggest_categorical("n_blocks", [2, 4, 6]),
         "d_block": trial.suggest_categorical('d_block', [64, 128, 256]),
         "dropout" : trial.suggest_categorical("dropout", [1e-8, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
      }
      return params
   
   @classmethod
   def define_grid_parameters(cls):
      search_space = {
         "n_blocks": [2, 4, 6],
         "d_block": [64, 128, 256],
         "dropout": [1e-8, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
      }
      return search_space
   
class FTTransformerWrapper(BaseModelTorch):
   def __init__(self, params, args):
      super().__init__(params, args)

      self.params = params
      self.args = args

      self.lambda_reg =  torch.nn.Parameter(torch.log(torch.tensor(0.01))) # Initialize with 0.01  #FRR

      #self.params['d_in'] = args.num_features
      #self.params['d_out'] = args.num_classes
      if self.args.objective == "ordinal_regression":
         self.n_bins = self.args.num_bins
         d_out = self.n_bins - 1

      else:
         d_out = args.num_classes

      self.bin_means = None
      self.discretizer = None

      if self.args.optimize_hyperparameters:
         self.model = FTTransformer(
         n_cont_features= len(self.args.num_idx) if self.args.num_idx is not None else 0,
         cat_cardinalities= self.args.cat_dims if self.args.cat_dims is not None else [],
         d_out = d_out,
         **FTTransformer.get_default_kwargs(),
                     ).to(self.device)
      else:
         kwargs = FTTransformer.get_default_kwargs()
         kwargs.update({
            "n_blocks": self.params["n_blocks"],
            "d_block": self.params["d_block"],
            "attention_n_heads": self.params["attention_n_heads"],
            "attention_dropout": self.params["attention_dropout"],
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": self.params["ffn_d_hidden_multiplier"],
            "ffn_dropout": self.params["ffn_dropout"],
            "residual_dropout": self.params["residual_dropout"],
         })

         self.model = FTTransformer(
            n_cont_features= len(self.args.num_idx) if self.args.num_idx is not None else 0,
            cat_cardinalities= self.args.cat_dims if self.args.cat_dims is not None else [],
            d_out = d_out,
            **kwargs,
                        ).to(self.device)

      # Optimizer
      self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + [self.lambda_reg], lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])

      print("On Device:", self.device)
      print(f"Model: {self.model}")
      """for name, param in self.model.named_parameters():
         print(name, param.shape, param.view(-1)[0].item())"""
      
   def seed_worker(self, worker_id):
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      random.seed(worker_seed)

#  Generator for DataLoader
   def get_generator(self, seed):
      g = torch.Generator()
      g.manual_seed(seed)
      return g   
   
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):
      print(f"Train Data b4 training...")
      print(f"X: {type(X)} , y : {type(y), }, X_val: {type(X_val)} , y : {type(y_val)}\n")
      
      if self.args.objective == "ordinal_regression": 
         #create bins
         y_train_binned, y_val_binned = binning(self.args, y)
         
         self.bin_means = np.array([y[y_train_binned == k].mean() for k in range(self.n_bins)])

         print(f"Y origi : {y[:10]}")
         print(f"Binned : {y_train_binned[:10]}")
         print(f"Bin Mean : {self.bin_means}")
         
         y = self._create_ordinal_targets(y_train_binned)
         #y_val = self._create_ordinal_targets(y_val_binned)

      #Convert to NP array
      X = np.asarray(X, dtype=np.float32)
      y = np.asarray(y, dtype=np.float32 if self.args.objective == 'regression' else np.int64)
      X_val = np.asarray(X_val, dtype=np.float32) if X_val is not None else None
      y_val = np.asarray(y_val, dtype=np.float32 if self.args.objective == 'regression' else np.int64) if y_val is not None else None
      class_weights = class_weights.to(self.device) if class_weights is not None else None

      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.args.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device) if X_val is not None else None
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.args.objective == 'regression' else torch.long).to(self.device) if y_val is not None else None


      print(f"Train Data after...")
      print(f"X: {type(X)} , y : {type(y), }, X_val: {type(X_val)} , y : {type(y_val)}\n")

      if self.args.optimize_hyperparameters is None:
         print(f"Seed in MLP: {self.args.test_seed}")
         g = self.get_generator(self.args.test_seed) #seed gen


         train_dataset = TensorDataset(X, y)
         loader = DataLoader(
               train_dataset, 
               batch_size=self.args.batch_size, 
               shuffle=True, 
               pin_memory=True,
               num_workers=4,
               worker_init_fn=self.seed_worker,
               generator=g
         )
      else:
         print("hyperparameter optimization")
         train_dataset = TensorDataset(X, y)
         loader = DataLoader(
               train_dataset, 
               batch_size=self.args.batch_size, 
               shuffle=True, 
         )
      # Training loop
      self.model.train()
      loss_history = []
      val_history = []
      lambda_history = []

      for epoch in range(self.args.epochs):
         epoch_loss = 0
         epoch_loss_val = 0

         for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            #print(f"Batch X: {type(batch_X)} \n")
            #print(f"Batch y: {type(batch_y)} \n")
            if frequency_map is not None:
               penalty = 0.0
               for i, col in enumerate(frequency_map.keys()):
                  weights = self.model.cat_embeddings.embeddings[i].weight # Get weights for one-hot encoded features
                  #weights = torch.tensor(weights, dtype=torch.float32)

                  # Get frequencies for this feature — shape: (num_categories_i,)
                  #freq = frequency_map[col].to(weights.device) + 1e-8  # avoid division by zero
                  freq = frequency_map[col] + 1e-8  # avoid division by zero

                  # L1 norm across embedding dimensions for each category
                  l1_norms = torch.sum(torch.abs(weights), dim=1)  # shape: (num_categories_i,)

                  # Apply frequency-based scaling
                  penalty += torch.sum(l1_norms / freq)
      
                  #penalty += torch.sum(torch.abs(weights[:, i + idx_buff])) / (frequency_map[col] + 1e-8)  #i add len ordinal
               penalty *=  torch.exp(self.lambda_reg)   # Use the learnable lambda_reg
            else:
               penalty = 0.0
               
                #~~~~
                
            self.optimizer.zero_grad()
            x_cont, x_cat = self._split_inputs(batch_X)
            outputs = self.model(x_cont, x_cat)

            if self.args.objective == "ordinal_regression":
               loss = self.ordinal_loss(outputs, batch_y) + penalty #add penalty
            else:
               #outputs = self.model(batch_X).squeeze()
               loss = self._compute_loss(outputs, batch_y, class_weights)

            loss.backward()
            self.optimizer.step()

            if X_val is not None:
               if self.args.objective == "ordinal_regression":
                  with torch.no_grad():
                     val_pred = self.predict(X_val)
                     val_loss = mean_squared_error(y_val, val_pred)
               else:
                  val_loss = self._evaluate(X_val, y_val, class_weights)

               epoch_loss_val += val_loss

            #Calc and Store training loss
            epoch_loss += loss.item()
            
            
         avg_loss = epoch_loss / len(loader)
         avg_loss_val = epoch_loss_val / len(loader)
         lambda_history.append(torch.exp(self.lambda_reg).item())

         if X_val is not None:
            loss_history.append(avg_loss)
            val_history.append(avg_loss_val)
         
            # Early stopping
            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
                  
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break
         

      if self.args.frequency_reg:
         return loss_history , val_history, lambda_history
      else:
         return loss_history , val_history

   def _create_ordinal_targets(self, y_binned):
      """
      Create ordinal targets from binned targets.
      Converts binned targets into a binary matrix where each column represents a bin.
      #Convert bin indices to cumulative binary targets.
      """
      targets = np.zeros((len(y_binned), self.n_bins-1))
      for i, bin_idx in enumerate(y_binned):
         if bin_idx > 0:
               targets[i, :int(bin_idx)] = 1.0
      return torch.tensor(targets, dtype=torch.float32)
    
      
   def ordinal_loss(self, outputs, targets):
      """
      Calculate the ordinal regression loss.
      Weighted binary cross-entropy loss that respects ordinal relationships
      Penalizes distant misclassifications more than adjacent ones

      Args:
          outputs: Model predictions.
          targets: True labels.
      Returns:
          Ordinal regression loss.
      """
      true_bins = targets.sum(dim=1)  # Get the true bin indices
      threshold_idx = torch.arange(1, self.n_bins, device=self.device) # Exclude the last bin
      weights = torch.abs(true_bins.unsqueeze(1) - threshold_idx)  # Calculate weights based on distance to true bin

      # Calculate the binary cross-entropy loss for each bin (Weighted BCE Loss)
      bce = nn.BCEWithLogitsLoss(reduction='none')
      #print(f"Type of outputs: {type(outputs)}, Type of target : {type(targets)}" )  
      #print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
      #print(f"Outputs: {outputs}, Targets: {targets}")
      loss = bce(outputs, targets.float())

      return (loss * weights).mean()  # Average loss across all samples
   
   def _ordinal_to_continuous(self, outputs):
      """Convert threshold outputs to continuous predictions"""
      probs = torch.sigmoid(outputs).cpu().numpy()
      bin_probs = np.zeros((len(probs), self.n_bins))

      print(f"We are in pred ordinal to continuous")
      print(f"Probs in pred ordinal to continuous: {probs.shape}")
      print(f"Probs in pred ordinal to continuous: {probs}")
      print(f"bin prob: {bin_probs}")
      print(f"Bin means in pred ordinal to continuous: {self.bin_means}")
      
      bin_probs[:, 0] = 1 - probs[:, 0]  # P(bin0)
      for k in range(1, self.n_bins-1):
         bin_probs[:, k] = probs[:, k-1] - probs[:, k]  # P(bin_k)
      bin_probs[:, -1] = probs[:, -1]  # P(last_bin)
      
      return bin_probs @ self.bin_means  # Weighted average


   def _compute_loss(self, outputs, targets, class_weights):
        if self.args.objective == 'regression':
            return nn.MSELoss()(outputs.squeeze(-1), targets)
        elif self.args.objective == "probabilistic_regression" or self.args.objective == "classification":
            if self.args.weighted_loss:
               return nn.CrossEntropyLoss(weight=class_weights)(outputs.squeeze(-1), targets)
            else:
               return nn.CrossEntropyLoss()(outputs.squeeze(-1), targets)
            
   def predict(self, X):
      self.model.eval()
      X = torch.tensor(X, dtype=torch.float32).to(self.device)
      
      with torch.no_grad():
         x_cont, x_cat = self._split_inputs(X)

         outputs = self.model(x_cont, x_cat)
         #outputs = self.model(X).squeeze()

         if self.args.objective == 'ordinal_regression':
            self.predictions = self._ordinal_to_continuous(outputs)

         elif self.args.objective == 'regression':
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

         if self.args.objective == 'ordinal_regression':
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            x_cont, x_cat = self._split_inputs(X)
            outputs = self.model(x_cont, x_cat)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Calculate bin probabilities
            bin_probs = np.zeros((len(probs), self.n_bins))
            bin_probs[:, 0] = 1 - probs[:, 0]
            for k in range(1, self.n_bins-1):
                bin_probs[:, k] = probs[:, k-1] - probs[:, k]
            bin_probs[:, -1] = probs[:, -1]
            
            self.prediction_probabilities = bin_probs 
            
         else:
            x_cont, x_cat = self._split_inputs(X)
            outputs = torch.softmax(self.model(x_cont, x_cat), dim=1)
            probabilities = outputs.detach().cpu().numpy()

            self.prediction_probabilities = probabilities
         #print(f"Probabilities in curr mod: {self.prediction_probabilities}")

         return self.prediction_probabilities

   def _evaluate(self, X, y, class_weights):
         if self.args.objective == "ordinal_regression":
            y_pred = self.predict(X)
            
            loss = mean_squared_error(y.cpu().numpy(), y_pred)

         else:
            self.model.eval()
            X = torch.tensor(X, dtype=torch.float32).to(self.device)

            with torch.no_grad():
               x_cont, x_cat = self._split_inputs(X)

               outputs = self.model(x_cont, x_cat)
               loss = self._compute_loss(outputs, y, class_weights).item()

         return loss
   
   def _split_inputs(self, X):
      x_cont = X[:, self.args.num_idx] if self.args.num_idx else None

      if self.args.dataset == "House_Prices_Nominal" and self.args.model_name == "FTTransformer":
         x_cat = X[:, self.args.ordinal_idx] if self.args.ordinal_idx else None
      else:
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
         "learning_rate": trial.suggest_categorical("learning_rate", 1e-5, 1e-2, ),
         "weight_decay": trial.suggest_categorical("weight_decay", 1e-6, 1e-2),
         "n_blocks": trial.suggest_categorical("n_blocks", [2, 4, 6]),
         "d_block": trial.suggest_categorical('d_block', [64, 128, 256]),
         "attention_n_heads": trial.suggest_categorical("attention_n_heads", [1, 2, 4]),
         "attention_dropout": trial.suggest_categorical("attention_dropout", [0.0, 0.1, 0.2]),
         "ffn_d_hidden_multiplier": trial.suggest_categorical("ffn_d_hidden_multiplier", [1.0, 2.0, 3.0, 4.0, 5.0]),
         "ffn_dropout": trial.suggest_categorical("ffn_dropout", [0.0, 0.1, 0.2]),
         "residual_dropout": trial.suggest_categorical("residual_dropout", [0.0, 0.1, 0.2]),
      }
      return params

   @classmethod
   def define_grid_parameters(cls):
      search_space = {
         "n_blocks": [2, 4, 6],
         "d_block": [64, 128, 256],
         "attention_n_heads": [1, 2, 4],
         "attention_dropout": [0.0, 0.1, 0.2],
         "ffn_d_hidden_multiplier": [1.0, 2.0, 3.0, 4.0, 5.0],
         "ffn_dropout": [0.0, 0.1, 0.2],
         "residual_dropout": [0.0, 0.1, 0.2]
      }

      return search_space
