from models.basemodel import BaseModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

import numpy as np

from utils.io_utils import get_output_path


class BaseModelTorch(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.device = self.get_device()
        #self.gpus = args.gpu_ids if args.use_gpu and torch.cuda.is_available() and args.data_parallel else None
        self.lambda_reg = nn.Parameter(torch.tensor(0.01))  # Learnable lambda_reg

    def to_device(self):
        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  # + ''.join(str(i) + ',' for i in self.args.gpu_ids)[:-1]
            else:
                device = 'cuda'
        else:
            device = 'cpu'

        return torch.device(device)

    def get_device(self):
        print("In get_device")
        """
        Determine the device to use (CPU or GPU).
        """
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device('cuda')  # PyTorch automatically maps to the first visible GPU
        
        return torch.device('cpu')

    def fit(self, X, y, X_val=None, y_val=None, frequency_map=None):
        if self.args.frequency_reg:
            optimizer = optim.AdamW(list(self.model.parameters()) + [self.lambda_reg], lr=self.params["learning_rate"])
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "probabilistic_regression":
            loss_func = nn.CrossEntropyLoss()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []
        lambda_reg_history = []  # To track lambda_reg values

        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):
                out = self.model(batch_X.to(self.device))
                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Frequency regularization term
                if frequency_map is not None:
                    if self.args.ordinal_encode :
                        idx_buff = len(self.args.ordinal_idx)
                    else:
                        idx_buff = 0  # Ordinal feat are stored b4 OHE

                    weights = self.model_semi.input_layer.weight  # Get weights for one-hot encoded features
                    penalty = 0.0
                    for i, col in enumerate(frequency_map.keys()):
                        penalty += torch.sum(torch.abs(weights[:, i + idx_buff])) / (frequency_map[col] + 1e-8)  #i add len ordinal
                    penalty *= self.model_semi.lambda_reg  # Use the learnable lambda_reg
                else:
                    penalty = 0.0

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                
                loss = loss_func(out, batch_y.to(self.device)) + penalty
                loss_history.append(loss.item()) #training loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Save lambda_reg value for this epoch
            lambda_reg_history.append(self.lambda_reg.item())

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                out = self.model(batch_val_X.to(self.device))
                

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            if self.args.frequency_reg:
                print("Epoch %d, Val Loss: %.5f, Lambda: %.5f" % (epoch, val_loss, self.lambda_reg.item())) 
            else:
                print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
        if self.args.frequency_reg:
            return loss_history, val_loss_history, lambda_reg_history
        else:
            return loss_history, val_loss_history , []

    def predict(self, X):
        if self.args.objective == "regression":
            self.predictions = self.predict_helper(X)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)
        
        ## For Prob Reg
        if self.args.objective == "probabilistic_regression":
            probas = np.clip(probas, 1e-5, 1)
            probas /= probas.sum(axis=1, keepdims=True)

        self.prediction_probabilities = probas

        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
