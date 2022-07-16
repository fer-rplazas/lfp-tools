from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.nn import BCEWithLogitsLoss, NLLLoss
import torchmetrics
from functools import partial
from copy import deepcopy

metrics = {
    "acc": lambda labels, preds, scores: torchmetrics.functional.accuracy(
        preds, labels
    ),
    # "auc": lambda labels, preds, scores: partial(
    #     torchmetrics.functional.auroc, num_classes=2
    # )(scores, labels),
    "mcc": lambda labels, preds, scores: 0.0
    if partial(torchmetrics.functional.matthews_corrcoef, num_classes=2)(
        preds, labels
    ).isnan()
    else partial(torchmetrics.functional.matthews_corrcoef, num_classes=2)(
        preds, labels
    ),
    "bal_acc": lambda labels, preds, scores: balanced_accuracy_score(
        labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
    ),
    "f1": lambda labels, preds, scores: f1_score(
        labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
    ),
}

metrics_fixed = deepcopy(metrics)

metrics["compound"] = lambda labels, preds, scores: sum(
    [fun(labels, preds, scores) for fun in metrics_fixed.values()]
)


class Metric:
    value: float
    std: Optional[float] = None

    def __init__(self, cls_type):
        self.cls_type = cls_type

    def accumulate(self, metric_list):

        vals = np.array([el.value for el in metric_list])
        self.value, self.std = vals.mean(axis=0), vals.std(axis=0)

        return self


class myAccuracy(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = accuracy_score(y, y_hat)
        return self


class myAUC(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = roc_auc_score(y, y_score)
        return self


class myPrecision(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = precision_score(y, y_hat)
        return self


class myRecall(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = recall_score(y, y_hat)
        return self


class myF1(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = f1_score(y, y_hat)
        return self


class myMCC(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = matthews_corrcoef(y, y_hat)
        if np.isnan(self.value):
            self.value = 0.0
        return self


class myBalancedAccuracy(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = balanced_accuracy_score(y, y_hat)
        return self


class myConfMat(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = confusion_matrix(y, y_hat)
        return self


class myConfMatNorm(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        self.value = confusion_matrix(y, y_hat, normalize="true")
        return self


class myRocCurve(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr = roc_curve(y, y_score, drop_intermediate=False)[:2]
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.value = interp_tpr
        return self


class myLosses(Metric):
    def __init__(self, cls_type=None):
        super().__init__(cls_type=cls_type)

    def compute(self, y, y_hat, y_score, losses):

        if self.cls_type == "CNN":
            self.value = losses
        elif self.cls_type == "SVM":
            # TODO: Incorporate Hinge loss to better represent SVM?
            self.value = BCEWithLogitsLoss(reduction="none")(
                torch.tensor(y_score), torch.tensor(y.astype(float))
            ).numpy()

        elif self.cls_type == "LDA":
            self.value = NLLLoss(reduction="none")(
                torch.tensor(losses.astype(float)), torch.tensor(y)
            ).numpy()
        else:
            self.value = None
            raise Warning(
                "MyLosses object cannot recognize the classifier type (and thus cannot assign the `.value` attribute)"
            )
        return self


class Scorer:
    def __init__(
        self,
        exclude_metrics=None,
        ds_type="train",
        cls_type=None,
    ):
        self.cls_type = cls_type
        self.exclude_metrics = exclude_metrics
        # if exclude_metrics is not None:
        #     [
        #         self.metrics_names.remove(metric)
        #         if metric in self.metrics_names
        #         else None
        #         for metric in exclude_metrics
        #     ]
        self.ds_type = ds_type
        self.n_runs = 1

        self.metrics = {
            "acc": myAccuracy(cls_type=self.cls_type),
            "AUC": myAUC(cls_type=self.cls_type),
            "precision": myPrecision(cls_type=self.cls_type),
            "recall": myRecall(cls_type=self.cls_type),
            # "conf_mat": myConfMat(cls_type=self.cls_type),
            # "conf_mat_norm": myConfMatNorm(cls_type=self.cls_type),
            # "losses": myLosses(cls_type=self.cls_type),
            # "roc_curve": myRocCurve(cls_type=self.cls_type),
            "balanced_accuracy": myBalancedAccuracy(cls_type=self.cls_type),
            "f1": myF1(cls_type=self.cls_type),
            "mcc": myMCC(cls_type=self.cls_type),
        }

    def __repr__(self):
        aid = "\n"
        exclude = ["losses", "conf_mat", "conf_mat_norm", "roc_curve"]

        return f"""Scorer Object --- ds_type = {self.ds_type} -- cls_type: {self.cls_type} -- n_runs = {self.n_runs}
            
    {aid.join([f"{k}: {v.value:.2f} +/- {[v.std if v.std is not None else np.nan][0]:.2f}" for (k,v) in self.metrics.items() if k not in exclude])}

        """

    def __str__(self):
        return self.__repr__()

    def get_scores(self, y, y_hat, y_scores, losses=None, times=None):

        self.times = times
        y, y_hat, y_scores = np.array(y), np.array(y_hat), np.array(y_scores)

        self.y, self.y_hat = y, y_hat

        for _, metric in self.metrics.items():
            metric.compute(y, y_hat, y_scores, losses)

        # self.accuracy = (
        #     accuracy_score(y, y_hat) if "accuracy" in self.metrics_names else None
        # )
        # self.AUC = roc_auc_score(y, y_scores) if "AUC" in self.metrics_names else None
        # self.precision = (
        #     precision_score(y, y_hat) if "precision" in self.metrics_names else None
        # )
        # self.recall = recall_score(y, y_hat) if "recall" in self.metrics_names else None
        # self.f1 = f1_score(y, y_hat) if "f1" in self.metrics_names else None

        # self.conf_mat = (
        #     confusion_matrix(y, y_hat) if "conf_mat" in self.metrics_names else None
        # )
        # self.conf_mat_norm = (
        #     confusion_matrix(y, y_hat, normalize="true")
        #     if "conf_mat_norm" in self.metrics_names
        #     else None
        # )

        # if "roc_curve" in self.metrics_names:
        #     mean_fpr = np.linspace(0, 1, 100)
        #     fpr, tpr = roc_curve(y, y_scores, drop_intermediate=False)[:2]
        #     interp_tpr = np.interp(mean_fpr, fpr, tpr)
        #     interp_tpr[0] = 0.0
        #     self.roc_curve = interp_tpr
        # else:
        #     self.roc_curve = None

        # if "losses" in self.metrics_names:
        #     self.losses = (
        #         losses
        #         if losses is not None
        #         else BCELoss(reduction="none")(
        #             Sigmoid()(torch.tensor(y_scores)), torch.tensor(y.astype(float))
        #         ).numpy()
        # )
        # assert (
        #     np.all(self.losses > 0) if self.losses is not None else True
        # ), "Error: Some losses are negative"

        return self

    def accumulate(self, scores_list):
        self.n_runs = len(scores_list)
        self.ds_type = scores_list[0].ds_type

        self.times = scores_list[0].times
        self.y = scores_list[0].y

        y_hats = np.stack([el.y_hat for el in scores_list])
        self.y_hat = {
            "mean": np.mean(y_hats, axis=0),
            "std": np.std(y_hats, axis=0),
        }

        for k, metric in self.metrics.items():
            metric.accumulate([el.metrics[k] for el in scores_list])

        # if "accuracy" in self.metrics_names:
        #     metric = np.array([el.accuracy for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among accuracy scores"
        #     self.accuracy = {"mean": np.mean(metric), "std": np.std(metric)}
        # else:
        #     self.accuracy = None

        # if "AUC" in self.metrics_names:
        #     metric = np.array([el.AUC for el in scores_list])
        #     assert not np.any(metric == None), "None values were found among AUC scores"
        #     self.AUC = {"mean": np.mean(metric), "std": np.std(metric)}
        # else:
        #     self.AUC = None

        # if "precision" in self.metrics_names:
        #     metric = np.array([el.precision for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among precision scores"
        #     self.precision = {"mean": np.mean(metric), "std": np.std(metric)}
        # else:
        #     self.precision = None

        # if "recall" in self.metrics_names:
        #     metric = np.array([el.recall for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among recall scores"
        #     self.recall = {"mean": np.mean(metric), "std": np.std(metric)}
        # else:
        #     self.recall = None

        # if "f1" in self.metrics_names:
        #     metric = np.array([el.f1 for el in scores_list])
        #     assert not np.any(metric == None), "None values were found among f1 scores"
        #     self.f1 = {"mean": np.mean(metric), "std": np.std(metric)}
        # else:
        #     self.f1 = None

        # if "conf_mat" in self.metrics_names:
        #     metric = np.stack([el.conf_mat for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among conf_mat scores"
        #     self.conf_mat = {
        #         "mean": np.mean(metric, axis=0),
        #         "std": np.std(metric, axis=0),
        #     }
        # else:
        #     self.conf_mat = None

        # if "conf_mat_norm" in self.metrics_names:
        #     metric = np.stack([el.conf_mat_norm for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among conf_mat_norm scores"
        #     self.conf_mat_norm = {
        #         "mean": np.mean(metric, axis=0),
        #         "std": np.std(metric, axis=0),
        #     }
        # else:
        #     self.conf_mat_norm = None

        # if "roc_curve" in self.metrics_names:
        #     metric = np.stack([el.roc_curve for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among roc_curve scores"
        #     self.roc_curve = {
        #         "mean": np.mean(metric, axis=0),
        #         "std": np.std(metric, axis=0),
        #     }
        # else:
        #     self.roc_curve = None

        # if "losses" in self.metrics_names:
        #     metric = np.stack([el.losses for el in scores_list])
        #     assert not np.any(
        #         metric == None
        #     ), "None values were found among losses scores"
        #     self.losses = {
        #         "mean": np.mean(metric, axis=0),
        #         "std": np.std(metric, axis=0),
        #     }
        # else:
        #     self.losses = None

        return self


class CnnScores:
    def __init__(self, train_scores, valid_scores):
        self.train, self.valid = (
            train_scores,
            valid_scores,
        )

    def __repr__(self):
        return f"""Score Summary Object
        Train: {self.train}
        Valid: {self.valid}
        """


class SvmScores:
    def __init__(self, train_scores, valid_scores, svm_coef=None):
        self.train, self.valid = train_scores, valid_scores

        if svm_coef is not None:
            self.svm_coef, self.lin_svm_acc = (
                svm_coef[0],
                svm_coef[1],
            )

    def __repr__(self):
        return f"""Score Summary Object
        Train: {self.train}
        Valid: {self.valid}
        """
