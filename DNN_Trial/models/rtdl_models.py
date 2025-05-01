from rtdl_revisiting_models import MLP, ResNet, FTTransformer

import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
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
      self.model = ResNet().to(self.device)

      # Optimizer
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)

      print("On Device:", self.device)


        
   def fit(self, X, y, X_val=None, y_val=None, frequency_map=None, class_weights=None):
       #self.model = ResNet().to(self.device)
       #optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)

      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32 if self.objective == 'regression' else torch.long)
      X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
      y_val = torch.tensor(y_val, dtype=torch.float32 if self.objective == 'regression' else torch.long).to(self.device)

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
                
            self.optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = self._compute_loss(outputs, batch_y, class_weights)

            loss.backward()
            self.optimizer.step()

            #Calc and Store training loss
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)

            val_loss = self._evaluate(X_val, y_val)
            val_history.append(val_loss)

            # Early stopping
            if min(val_history) == val_history[-1]:
               best_model = self.model.state_dict()
                
            if len(val_history) - val_history.index(min(val_history)) > self.args.early_stopping_rounds:
               break

         return loss_history , val_history

   def _compute_loss(self, outputs, targets, class_weights):
        if self.objective == 'regression':
            return nn.MSELoss()(outputs, targets)
        elif self.objective == "probabilistic_regression" or self.objective == "classification":
            if self.args.weighted_loss:
               return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets)
            else:
               return nn.CrossEntropyLoss()(outputs, targets)
            
   def predict(self, X):
      self.model.eval()
      
      with torch.no_grad():
         outputs = self.model(X).squeeze()

         if self.args.objective == 'regression':
            return outputs.cpu().numpy()

         else:
            return outputs.argmax(dim=1).cpu().numpy()
         
   def _evaluate(self, X, y):
         self.model.eval()
         with torch.no_grad():
            outputs = self.model(X).squeeze()
            loss = self._compute_loss(outputs, y).item()
         return loss

   def predict_proba(self, X):
      if self.args.objective == 'regression':
         raise NotImplementedError("Method only available for classification tasks")
      else:
         self.model.eval()
         output = torch.softmax(self.model(X), dim=1)
         return output.cpu().numpy

    
   @classmethod
   def define_trial_parameters(cls, trial, args):
      params = {
         "n_blocks": trial.suggest_int("max_depth", 2, 6, log=True),
         "d_block": trial.suggest_categorical('d_block', [64, 128, 256]),
         "d_hidden": trial.suggest_int("max_depth", 2, 10, log=True),
         "d_hidden_multiplier" : trial.suggest_float("lambda", 0, 5.0, log=True),
         "dropout1" : trial.suggest_float("lambda", 1e-8,0.5, log=True),
         "dropout2" : trial.suggest_float("lambda", 1e-8, 0.3, log=True),
      }
      return params


class MLP(BaseModelTorch):
   pass

class FTTransformer(BaseModelTorch):
   pass

