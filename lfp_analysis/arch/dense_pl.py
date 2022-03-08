# %%
from fastai.callback.hook import has_params
import numpy as np
import pandas as pd

import os
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.module import T
import torch.optim as optim
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import torchmetrics

from types import SimpleNamespace
from functools import partial

from .dense import DenseNetFull
from ..windower import *
from ..dataloader import *


DEFAULT_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

CHECKPOINT_PATH = Path("saved_models")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# %%

# %%

# %%

# %% Prepare models:

model_dict = {}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are {model_dict.keys()}'


act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


# %%


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, act_fn, kernel_size=75):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, padding="same", stride=1
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm1d(c_out),
            act_fn(),
            nn.Conv1d(
                c_out, c_out, kernel_size=kernel_size, padding="same", bias=False
            ),
            nn.BatchNorm1d(c_out),
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.upsample = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        x = self.upsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_blocks=[1, 1, 1],
        c_hidden=[16, 32, 64],
        act_fn_name="relu",
        **kwargs,
    ):
        """
        Inputs:
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden

        self.input_net = nn.Sequential(
            nn.InstanceNorm1d(8, affine=True),
            nn.Conv1d(8, c_hidden[0], kernel_size=75, padding="same"),
            nn.BatchNorm1d(c_hidden[0]),
            self.hparams.act_fn(),
        )

        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                blocks.append(
                    ResNetBlock(
                        c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
                        c_out=c_hidden[block_idx],
                        act_fn=self.hparams.act_fn,
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                c_hidden[len(self.hparams.num_blocks) - 1], self.hparams.num_classes
            ),
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=self.hparams.act_fn_name
                )
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


model_dict["ResNet"] = ResNet
# %%
model_dict["DenseNetFull"] = DenseNetFull


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
        self.loss_module = nn.CrossEntropyLoss(label_smoothing=0.4)
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 8, 1535), dtype=torch.float32)

        self.metrics = {
            "acc": torchmetrics.functional.accuracy,
            "auc": partial(torchmetrics.functional.auroc, num_classes=2),
            "mcc": lambda pred, label: 0.0
            if partial(torchmetrics.functional.matthews_corrcoef, num_classes=2)(
                pred, label
            ).isnan()
            else partial(torchmetrics.functional.matthews_corrcoef, num_classes=2)(
                pred, label
            ),
            "bal_acc": lambda pred, label: 0.5
            * (
                torchmetrics.functional.specificity(pred, label)
                + torchmetrics.functional.recall(pred, label)
            ),
            "avg_precision": partial(
                torchmetrics.functional.average_precision, num_classes=2
            ),
            "f1": torchmetrics.functional.f1,
        }

        metrics_fixed = deepcopy(self.metrics)

        self.metrics["compound"] = lambda pred, label: sum(
            [fun(pred, label) for fun in metrics_fixed.values()]
        )

    def forward(self, x):
        # Forward function that is run when visualizing the graph
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
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, labels = batch
        preds = self.model(x)
        loss = self.loss_module(preds, labels)

        self.log("train/loss", float(loss), on_step=False, on_epoch=True)

        for k, score_fun in self.metrics.items():
            self.log(
                f"train/{k}",
                float(score_fun(preds, labels)),
                on_step=False,
                on_epoch=True,
            )

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        preds = self.model(x)
        loss = self.loss_module(preds, labels)

        self.log("valid/loss", loss)

        for k, score_fun in self.metrics.items():
            self.log(
                f"valid/{k}",
                float(score_fun(preds, labels)),
                on_step=False,
                on_epoch=True,
            )


# %%
ResNet()(torch.randn(1, 8, 1535))


# %%


def train_model(model_name, train_loader, val_loader, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(
            CHECKPOINT_PATH, save_name
        ),  # Where to save models
        # We run on a single GPU (if possible)
        gpus=1 if str(DEVICE) == "cuda:0" else 0,
        # How many epochs to train for if no patience is set
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="valid/bal_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
        progress_bar_refresh_rate=1,
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )
    trainer.logger.experiment.add_hparams(
        {"pat": "ET1", "task": "Pegboard"}, {"hparam/accuracy": 10, "hparam/loss": 10}
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = LFPModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LFPModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = LFPModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.validate(model, val_dataloaders=val_loader, verbose=False)
    result = val_result

    return result


# %%
