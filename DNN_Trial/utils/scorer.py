from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score
from scipy.special import rel_entr
import torch
import torch.nn as nn

import numpy as np


def get_scorer(args):
    if args.objective == "regression":
        return RegScorer()
    elif args.objective == "probabilistic_regression":
        return ProbRegScorer()
    elif args.objective == "classification":
        return ClassScorer()
    elif args.objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError("No scorer for \"" + args.objective + "\" implemented")


class Scorer:

    """
        y_true: (n_samples,)
        y_prediction: (n_samples,) - predicted classes
        y_probabilities: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    """
    def eval(self, y_true, y_prediction, y_probabilities):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_objective_result(self):
        raise NotImplementedError("Has be implemented in the sub class")


class RegScorer(Scorer):

    def __init__(self):
        self.mses = []
        self.r2s = []

    # y_probabilities is None for Regression
    def eval(self, y_true, y_prediction, y_probabilities):
        mse = mean_squared_error(y_true, y_prediction)
        r2 = r2_score(y_true, y_prediction)

        self.mses.append(mse)
        self.r2s.append(r2)

        return {"MSE": mse, "R2": r2}

    def get_results(self):
        mse_mean = np.mean(self.mses)
        mse_std = np.std(self.mses)

        r2_mean = np.mean(self.r2s)
        r2_std = np.std(self.r2s)

        return {"MSE - mean": mse_mean,
                "MSE - std": mse_std,
                "R2 - mean": r2_mean,
                "R2 - std": r2_std}

    def get_objective_result(self):
        return np.mean(self.mses)
    
class ProbRegScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        #self.kld_loss = []
        self.aucs = []
        self.accs = []
        self.f1s = []

    def eval(self, y_true, y_prediction, y_probabilities):
        #print(f"Unique classes in y_true: {np.unique(y_true)}")
        #print(f"Unique classes in y_pred: {np.unique(y_probabilities)}") 
        y_probabilities = torch.tensor(y_probabilities)
        y_true = torch.tensor(y_true, dtype=torch.long)
        y_logits = torch.log(y_probabilities + 1e-9)

        # CrossEntropyLoss 
        loss_fn = nn.CrossEntropyLoss()

        logloss = loss_fn(y_logits, y_true)
        auc = roc_auc_score(y_true, y_probabilities, multi_class='ovo', average="macro")
        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="weighted")  # use here macro or weighted?

        #self.loglosses.append(logloss)
        self.loglosses.append(round(logloss.item(), 5))
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}
        
        # KLDivergence
        #y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=y_probabilities.shape[1]).type(torch.float32) 
        #y_logits_k = torch.clip(y_probabilities, 1e-6, 1)
        #y_true_k = torch.clip(y_true_onehot, 1e-6, 1)

        #kld_loss_fn = nn.KLDivLoss(reduction='batchmean')
        #kld_loss = kld_loss_fn(torch.log(y_logits_k), y_true_k) 

        #print(f"Log Loss: {logloss.item()} KLD Loss: {kld_loss.item()}")


        
        #self.kld_loss.append(round(kld_loss.item(), 4))
        
        #return {"Log Loss": logloss, "KL Divergence": kld_loss}
        #return {"Log Loss": logloss}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std}
    
    def get_objective_result(self):
        return np.mean(self.loglosses)


class ClassScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []

    def eval(self, y_true, y_prediction, y_probabilities):
        logloss = log_loss(y_true, y_probabilities)
        # auc = roc_auc_score(y_true, y_probabilities, multi_class='ovr')
        auc = roc_auc_score(y_true, y_probabilities, multi_class='ovo', average="macro")

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="weighted")  # use here macro or weighted?

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std}

    def get_objective_result(self):
        return np.mean(self.loglosses)


class BinScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []

    def eval(self, y_true, y_prediction, y_probabilities):
        logloss = log_loss(y_true, y_probabilities)
        auc = roc_auc_score(y_true, y_probabilities[:, 1])

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="micro")  # use here macro or weighted?

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std}

    def get_objective_result(self):
        return np.mean(self.aucs)
