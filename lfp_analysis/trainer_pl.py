import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from mrmr import mrmr_classif

# from torchaudio.transforms import Resample

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchmetrics
from torchinfo import summary

from sklearn.metrics import balanced_accuracy_score, f1_score

import os
from pathlib import Path
from functools import partial
from copy import deepcopy
from scipy.signal import decimate

from typing import Optional

import wandb

from .arch import *

from lfp_analysis.data import PatID, Task, Stim
from lfp_analysis.score import metrics
from lfp_analysis.svm import window_data

DEFAULT_DEVICE = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

CHECKPOINT_PATH = Path("saved_models")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class DataModule(pl.LightningDataModule):
    def __init__(self, signals: np.ndarray, windower, feature_extractor, bs=32):

        super().__init__()

        self.signals, self.windower, self.feature_extractor = (
            signals,
            windower,
            feature_extractor,
        )
        self.bs = bs

    def setup(
        self,
        stage: Optional[str] = None,
        feat_selection: Optional[list[int]] = None,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
    ):

        # Window data, extract features:
        self.windowed = window_data(
            self.signals,
            self.windower.windower_.df.id_start.values,
            self.windower.windower_.df.id_end.values,
        )
        self.features = self.feature_extractor(self.windowed)
        if means is None:
            self.mean = self.features[
                self.windower.windower_.df["is_valid"] == False, :
            ].mean(axis=0, keepdims=True)
        else:
            self.mean = means
        if stds is None:
            self.std = self.features[
                self.windower.windower_.df["is_valid"] == False, :
            ].std(axis=0, keepdims=True)
        else:
            self.std = stds
        self.features = (self.features - self.mean) / self.std

        X_train = self.features[self.windower.windower_.df["is_valid"] == False, :]
        y_train = self.windower.windower_.df[
            self.windower.windower_.df["is_valid"] == False
        ]["label"].values

        if feat_selection is None:
            feat_selection = mrmr_classif(
                X=pd.DataFrame(X_train), y=pd.DataFrame(y_train), K=24
            )

        self.features = self.features[:, feat_selection]

        # self.features = (
        #     self.features - self.features.mean(axis=0, keepdims=True)
        # ) / self.features.std(axis=0, keepdims=True)

        self.data = torch.tensor(self.windowed).float()

        self.train_data = []
        for jj, row in self.windower.df[
            self.windower.df["is_valid"] == False
        ].iterrows():
            x = torch.stack([self.data[jj] for jj in row["idx_win"]])
            y = [
                (label, torch.tensor(self.features[jj]).float())
                for label, jj in zip(row["label"], row["idx_win"])
            ]
            self.train_data.append((x, y))

        self.valid_data = []
        for jj, row in self.windower.df[
            self.windower.df["is_valid"] == True
        ].iterrows():
            x = torch.stack([self.data[jj] for jj in row["idx_win"]])
            y = [
                (label, torch.tensor(self.features[jj]).float())
                for label, jj in zip(row["label"], row["idx_win"])
            ]
            self.valid_data.append((x, y))

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.bs, shuffle=True, drop_last=True
        )

    def valid_dataloader(self):
        return DataLoader(
            self.valid_data, batch_size=self.bs, shuffle=False, drop_last=True
        )


class ARModule(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        use_mixup=True,
        w_losses: list[int] = [0.5, 0.5],
    ):

        super().__init__()

        self.save_hyperparameters()

        self.model = create_model(model_name, model_hparams)

        self.l_cat = nn.BCEWithLogitsLoss()
        self.l_regress = nn.MSELoss()
        self.w_losses = w_losses

        # self.example_input_array = torch.zeros((1,model_hparams['convs_hparams']['n_in'],1535), dtype=torch.float32)
        # import pdb; pdb.set_trace()
        # self.model_summary = summary(self.model, self.example_input_array)

        self.metrics = metrics
        self.use_mixup = use_mixup

    def configure_optimizers(self):

        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=5e-4,
        #     steps_per_epoch=2,
        #     epochs=180,
        # )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 80], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, _):

        x, labels = batch

        cats = torch.cat([lab[0] for lab in labels]).float()
        feats = torch.stack(
            [lab[1] for lab in labels]
        )  # .view(-1, labels[0][1].shape[-1])

        l1, l2 = self.model((x, feats))

        feats = feats.view(-1, labels[0][1].shape[-1])

        if self.use_mixup:

            if l1.shape[0] % 2 != 0:
                l1 = l1[:-1, ...]
                l2 = l2[:-1, ...]

                cats = cats[:-1, ...]
                feats = feats[:-1, ...]

            n = l1.shape[0] // 2
            lambdas = torch.rand(n).to(x.device)

            # import pdb; pdb.set_trace()
            l1_interp = lambdas * l1[:n, ...] + (1 - lambdas) * l1[n:, ...]
            l2_interp = (
                lambdas.unsqueeze(1).expand(-1, feats.shape[-1]) * l2[:n, ...]
                + (1 - lambdas.unsqueeze(1).expand(-1, feats.shape[-1])) * l2[n:, ...]
            )

            cats_interp = lambdas * cats[:n] + (1 - lambdas) * cats[n:]
            feats_interp = (
                lambdas.unsqueeze(1).expand(-1, feats.shape[-1]) * feats[:n, ...]
                + (1 - lambdas.unsqueeze(1).expand(-1, feats.shape[-1]))
                * feats[n:, ...]
            )

            l1, l2 = l1_interp, l2_interp
            cats, feats = cats_interp, feats_interp

        cat_loss = self.l_cat(l1, cats)
        regress_loss = self.l_regress(l2, feats)
        loss = self.w_losses[0] * cat_loss + self.w_losses[1] * regress_loss

        for k, score_fun in metrics.items():
            self.log(
                f"train/{k}",
                float(
                    score_fun(
                        cats.int(), (torch.sigmoid(l1) > 0.5).int(), torch.sigmoid(l1)
                    )
                ),
                on_step=False,
                on_epoch=True,
            )

        self.log("train/loss", loss.item())
        self.log("train/class_loss", cat_loss.item())
        self.log("train/feats_loss", regress_loss.item())

        return loss

    def validation_step(self, batch, _):

        x, labels = batch

        cats = torch.cat([lab[0] for lab in labels]).float()
        feats = torch.stack(
            [lab[1] for lab in labels]
        )  # .view(-1, labels[0][1].shape[-1])

        l1, l2 = self.model((x, feats))

        feats = feats.view(-1, labels[0][1].shape[-1])
        cat_loss = self.l_cat(l1, cats)
        regress_loss = self.l_regress(l2, feats)
        loss = self.w_losses[0] * cat_loss + self.w_losses[1] * regress_loss

        # for k, score_fun in metrics.items():
        #     self.log(
        #         f"valid/{k}",
        #         float(
        #             score_fun(
        #                 cats.int(), (torch.sigmoid(l1) > 0.5).int(), torch.sigmoid(l1)
        #             )
        #         ),
        #         on_step=False,
        #         on_epoch=True,
        #     )

        self.log("valid/loss", loss.item())
        self.log("valid/class_loss", cat_loss.item())
        self.log("valid/feats_loss", regress_loss.item())

        return cats.int(), (torch.sigmoid(l1) > 0.5).int(), torch.sigmoid(l1)

    def validation_epoch_end(self, val_outputs):

        labels, preds, scores = [], [], []
        for item in val_outputs:
            labels.append(item[0])
            preds.append(item[1])
            scores.append(item[2])

        labels, preds, scores = tuple(
            map(lambda x: torch.cat(x, 0), (labels, preds, scores))
        )

        for k, score_fun in self.metrics.items():
            self.log(
                f"valid/{k}",
                float(score_fun(labels, preds, scores)),
            )


class LFPModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss(label_smoothing=0.1)
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros(
            (1, model_hparams["n_in"], 1535), dtype=torch.float32
        )

        self.model_summary = summary(self.model, self.example_input_array.shape)

        self.use_manifold_mixup = True

        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=5e-4,
        #     steps_per_epoch=2,
        #     epochs=180,
        # )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, labels = batch

        if self.use_manifold_mixup:
            if x.shape[0] % 2 != 0:
                x = x[:-1, ...]
                labels = labels[:-1, ...]

            feats = self.model.convs(x)
            n = feats.shape[0] // 2
            lambdas = torch.rand(n)[:, None].to(x.device)

            feats_interp = lambdas * feats[:n, ...] + (1 - lambdas) * feats[n:, ...]
            labels = (
                lambdas.squeeze() * labels[:n, ...]
                + (1 - lambdas.squeeze()) * labels[n:, ...]
            )

            labels = torch.cat([1 - labels.unsqueeze(1), labels.unsqueeze(1)], 1)

            scores = self.model.cls(feats_interp)

        else:
            scores = self.model(x)

        loss = self.loss_module(scores, labels)

        if len(labels.shape) > 1:
            labels = labels.argmax(-1)

        preds = scores.argmax(-1)

        self.log("train/loss", float(loss), on_step=False, on_epoch=True)

        for k, score_fun in self.metrics.items():
            self.log(
                f"train/{k}",
                float(score_fun(labels, preds, scores)),
                on_step=False,
                on_epoch=True,
            )

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, _):
        x, labels = batch
        scores = self.model(x)
        preds = scores.argmax(-1)

        return labels, preds, scores

        # self.logger.experiment.log(
        #     {
        #         "sinc/low_hz": wandb.Histogram(
        #             self.model.conv_init.low_hz_.detach().cpu().numpy()
        #         )
        #     }
        # )
        # self.logger.experiment.log(
        #     {
        #         "sinc/band_hz": wandb.Histogram(
        #             self.model.conv_init.band_hz_.detach().cpu().numpy()
        #         )
        #     }
        # )
        # self.logger.experiment.log(
        #     {
        #         "sinc/gate": wandb.Histogram(
        #             self.model.conv_init.gate.detach().cpu().numpy()
        #         )
        #     }
        # )

    def validation_epoch_end(self, val_outputs):

        labels, preds, scores = [], [], []
        for item in val_outputs:
            labels.append(item[0])
            preds.append(item[1])
            scores.append(item[2])

        labels, preds, scores = tuple(
            map(lambda x: torch.cat(x, 0), (labels, preds, scores))
        )

        loss = self.loss_module(scores, labels)
        self.log("valid/loss", float(loss))

        for k, score_fun in self.metrics.items():
            self.log(
                f"valid/{k}",
                float(score_fun(labels, preds, scores)),
            )

    def test_step(self, batch, batch_idx):
        x, labels = batch
        scores = self.model(x)

        preds = scores.argmax(-1)

        return labels, preds, scores

    def test_epoch_end(self, test_outputs):

        labels, preds, scores = [], [], []
        for item in test_outputs:
            labels.append(item[0])
            preds.append(item[1])
            scores.append(item[2])

        labels, preds, scores = tuple(
            map(lambda x: torch.cat(x, 0), (labels, preds, scores))
        )

        self.logger.experiment.log(
            {
                "vBL/conf_mat": wandb.plot.confusion_matrix(
                    preds=preds.detach().cpu().numpy(),
                    y_true=labels.detach().cpu().numpy(),
                    probs=None,
                    class_names=["0", "1"],
                )
            }
        )

        for k, score_fun in self.metrics.items():
            self.log(
                f"vBL/{k}",
                float(score_fun(labels, preds, scores)),
            )

        return scores


class Distiller(pl.LightningModule):
    def __init__(
        self,
        teacher_path,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        temperature=15,
        alpha=0.1,
        fs_t=2048,
        fs_s=768,
    ):

        super().__init__()

        self.save_hyperparameters()
        self.my_hparams = {
            "optimizer_name": optimizer_name,
            "optimizer_hparams": optimizer_hparams,
        }

        try:
            self.teacher = LFPModule.load_from_checkpoint(teacher_path)
        except TypeError:
            self.teacher = torch.load(teacher_path)
        self.student = create_model(model_name, model_hparams)

        self.temperature = temperature  # Softmax temperature
        self.alpha = alpha  # proportion of hard student CE-Loss in final loss

        self.loss = nn.CrossEntropyLoss()
        self.metrics = metrics
        if fs_t == 2048:
            self.resampler_t = nn.Identity()
        else:
            self.resampler_t = Resample(orig_freq=2048, new_freq=fs_t)

        if fs_s == 2048:
            self.resampler_s = nn.Identity()
        else:
            self.resampler_s = Resample(orig_freq=2048, new_freq=fs_s)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.my_hparams["optimizer_name"] == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.parameters(), **self.my_hparams["optimizer_hparams"]
            )
        elif self.my_hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=5e-4,
        #     steps_per_epoch=2,
        #     epochs=180,
        # )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_t = self.resampler_t(x)
        x_s = self.resampler_s(x)
        # x_s = torch.tensor(decimate(x.detach().cpu().numpy(), q=3).copy()).to("cuda:1")
        # x_s = x
        teacher_logits = self.teacher(x_t)
        student_logits = self.student(x_s)
        student_preds = student_logits.argmax(-1)

        distill_loss = self.loss(
            student_logits / self.temperature,
            (teacher_logits / self.temperature).softmax(-1),
        )
        hard_loss = self.loss(student_logits, y)
        # import pdb

        # pdb.set_trace()

        loss = self.alpha * hard_loss + (1 - self.alpha) * distill_loss

        self.log("train/loss", float(loss), on_step=False, on_epoch=True)

        for k, score_fun in self.metrics.items():
            self.log(
                f"train/{k}",
                float(score_fun(y, student_preds, student_logits)),
                on_step=False,
                on_epoch=True,
            )

        return loss

    def forward(self, x):

        return self.student(x)

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x_s = self.resampler_s(x)
        scores = self.student(x_s)
        preds = scores.argmax(-1)

        return labels, preds, scores

    def validation_epoch_end(self, val_outputs):

        labels, preds, scores = [], [], []
        for item in val_outputs:
            labels.append(item[0])
            preds.append(item[1])
            scores.append(item[2])

        labels, preds, scores = tuple(
            map(lambda x: torch.cat(x, 0), (labels, preds, scores))
        )

        loss = self.loss(scores, labels)
        self.log("valid/loss", float(loss))

        for k, score_fun in self.metrics.items():
            self.log(
                f"valid/{k}",
                float(score_fun(labels, preds, scores)),
            )

    def test_step(self, batch, batch_idx):
        x, labels = batch
        x_s = self.resampler_s(x)
        # x_s = torch.tensor(decimate(x.detach().cpu().numpy(), q=3).copy()).to("cuda:1")
        scores = self.student(x_s)

        preds = scores.argmax(-1)

        return labels, preds, scores

    def test_epoch_end(self, test_outputs):

        labels, preds, scores = [], [], []
        for item in test_outputs:
            labels.append(item[0])
            preds.append(item[1])
            scores.append(item[2])

        labels, preds, scores = tuple(
            map(lambda x: torch.cat(x, 0), (labels, preds, scores))
        )

        self.logger.experiment.log(
            {
                "vBL/conf_mat": wandb.plot.confusion_matrix(
                    preds=preds.detach().cpu().numpy(),
                    y_true=labels.detach().cpu().numpy(),
                    probs=None,
                    class_names=["0", "1"],
                )
            }
        )

        for k, score_fun in self.metrics.items():
            self.log(
                f"vBL/{k}",
                float(score_fun(labels, preds, scores)),
            )

        return scores


class TrainerPL:
    def __init__(
        self,
        train_loader,
        valid_loader,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        logger=None,
        gpu=[1],
    ):

        self.logger = (
            WandbLogger(project="main", name="default")
            if logger is None
            else True
            if logger is False
            else logger
        )

        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.model_name, self.model_hparams = model_name, model_hparams
        self.optimizer_name, self.optimizer_hparams = optimizer_name, optimizer_hparams

        self.gpu = gpu

        self.save_name = (
            self.logger.name if hasattr(self.logger, "name") else self.model_name
        )

        if self.save_name is None:
            self.save_name = "model"

        trainer = pl.Trainer(
            stochastic_weight_avg=False,
            # gradient_clip_val=2.0,
            logger=self.logger,
            default_root_dir=os.path.join(
                CHECKPOINT_PATH, self.save_name
            ),  # Where to save models
            # We run on a single GPU (if possible)
            gpus=self.gpu,
            # How many epochs to train for if no patience is set
            max_epochs=180,
            callbacks=[
                TQDMProgressBar(refresh_rate=1),
                ModelCheckpoint(
                    save_weights_only=True,
                    mode="max",
                    monitor="valid/bal_acc",
                    save_top_k=1,
                ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            ],  # Log learning rate every epoch
        )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate

        self.trainer = trainer

        self.model = LFPModule(
            model_name=self.model_name,
            model_hparams=self.model_hparams,
            optimizer_name=self.optimizer_name,
            optimizer_hparams=self.optimizer_hparams,
        )

        self.logger.log_hyperparams(self.model.hparams)
        self.logger.experiment.log(
            {
                "model_info/macs": self.model.model_summary.total_mult_adds,
                "model_info/n_params": self.model.model_summary.total_params,
            }
        )

    def train_test(self):

        trainer = pl.Trainer(
            stochastic_weight_avg=True,
            # gradient_clip_val=2.0,
            logger=self.logger,
            fast_dev_run=True,
        )
        trainer.fit(self.model, self.train_loader, self.valid_loader)

    def lr_find(self):

        lr_finder = self.trainer.tuner.lr_find(
            self.model, self.train_loader, self.valid_loader
        )
        fig = lr_finder.plot(suggest=True)
        fig.show()

    def train(self):

        # Check whether pretrained model exists. If yes, load it and skip training
        # pretrained_filename = os.path.join(CHECKPOINT_PATH, self.save_name + ".ckpt")
        # if os.path.isfile(pretrained_filename):
        #     print(f"Found pretrained model at {pretrained_filename}, loading...")
        #     # Automatically loads the model with the saved hyperparameters
        #     self.model = LFPModule.load_from_checkpoint(pretrained_filename)
        # else:
        # pl.seed_everything(42)  # To be reproducable

        # self.logger.watch(self.model, log_freq=10, log="all")
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

        self.model = type(self.model).load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

        # Test best model on validation and test set
        val_result = self.trainer.validate(
            self.model, dataloaders=self.valid_loader, verbose=False
        )

    def score(self, dataloader=None):

        dataloader = dataloader if dataloader is not None else self.valid_loader
        return self.trainer.test(model=self.model, dataloaders=dataloader)
