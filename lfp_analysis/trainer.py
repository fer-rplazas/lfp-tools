from fastai.metrics import BalancedAccuracy, RocAuc, RocAucBinary
from omegaconf.omegaconf import OmegaConf
import torch
from torch.utils.data.dataloader import DataLoader
import wandb
from typing import Callable, Optional

from fastai.vision.all import (
    Learner,
    nn,
    accuracy,
    EarlyStoppingCallback,
    CrossEntropyLossFlat,
)

from .score import Scorer, CnnScores


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Trainer:

    learner: Learner
    model: nn.Module
    dataloader: DataLoader
    loss: Callable

    model_dir = "."

    def __init__(
        self,
        dls,
        model: nn.Module,
        loss=CrossEntropyLossFlat(),
        wd=1.0,
        log_wandb=False,
        experiment=None,
        device=DEVICE,
    ):

        self.model, self.loss, self.dls = model, loss, dls.to(DEVICE)
        self.wd = wd
        cbs = []
        self.device = device

        self.learner = Learner(
            self.dls,
            self.model,
            metrics=[accuracy, BalancedAccuracy(), RocAucBinary()],
            loss_func=self.loss,
            cbs=cbs,
            wd=float(self.wd),
        )
        self.learner.to(self.device)

        self.learner.recorder.train_metrics = True

        self.log_wandb = log_wandb
        self.experiment = experiment

        if self.log_wandb is not False:
            self.run = wandb.init()
            # wandb.config(OmegaConf.to_container(cfg))

    def train(self, n_epochs=35, lr_div=3 * 1e-3, wd: Optional[float] = None):

        wd = wd if wd is not None else self.wd

        self.learner.fit_one_cycle(n_epochs, lr_div, wd=wd)

        # self.learner.fit_one_cycle(n_epochs, lr_div)
        # self.learner.fit_one_cycle(n_epochs, lr_div / 2)
        # self.learner.fit_one_cycle(n_epochs, lr_div / 4)
        # self.learner.fit_one_cycle(n_epochs, lr_div / 8)
        # self.learner.add_cb(EarlyStoppingCallback(min_delta=0.001, patience=3))

        # # self.learner.fit_one_cycle(14, 10e-4)
        # # self.learner.fit_one_cycle(25, 5 * 10e-5)
        # self.learner.fit_one_cycle(35, 10e-5)
        # self.learner.fit_one_cycle(35, 3 * 10e-6)
        # self.learner.fit_one_cycle(35, 10e-6)
        # self.learner.fit_one_cycle(35, 10e-7)

        # [self.learner.remove_cb(cb) for cb in self.learner.cbs[3:]]

    def save_model(self, dir=None):
        if dir is not None:
            self.learner.model_dir = dir

        self.learner.save("model")

    def score(self):
        # TODO: Depending on environment, sometimes sequential access to get_preds freezes!
        # Train:
        # times_train = self.data_df[self.data_df["is_valid"] == False]["t"].values
        y_scores, y, losses = self.learner.get_preds(ds_idx=0, with_loss=True)
        y_hat = torch.argmax(y_scores, -1).detach().numpy()
        y_scores = y_scores[:, 1].detach().numpy()

        train_scores = Scorer(ds_type="train", cls_type="CNN").get_scores(
            y, y_hat, y_scores, losses.detach().numpy(), times=None
        )

        # Valid:
        # times_valid = self.data_df[self.data_df["is_valid"] == True]["t"].values
        y_scores, y = self.learner.get_preds(with_loss=False)
        y_hat = torch.argmax(y_scores, -1).detach().numpy()
        y_scores = y_scores[:, 1].detach().numpy()

        valid_scores = Scorer(cls_type="CNN").get_scores(
            y, y_hat, y_scores, losses=None, times=None
        )

        self.scores = CnnScores(train_scores, valid_scores)

        return self.scores
