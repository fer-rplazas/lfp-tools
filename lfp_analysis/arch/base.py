import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class ConvBlock(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size, stride=1, padding=0, groups=1, bias=True
    ):
        super().__init__()

        self.conv = nn.Conv1d(c_in, c_out, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))


class ConvBlock2d(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size, stride=1, padding=0, groups=1, bias=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            c_in, c_out, (kernel_size, kernel_size), stride, padding, groups=groups
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MyLayerNorm(nn.Module):
    def __init__(self, n_chan) -> None:
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros((1, n_chan, 1))
        )  # learnable mean adjustment
        self.gamma = nn.Parameter(torch.ones((1, n_chan, 1)))  # learnable std. scaling
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean((-2, -1), True)
        std = x.std((-2, -1), False, True)

        x = (x - mean) / (std + self.eps)
        return x * self.gamma + self.beta


class SEConv(nn.Module):
    def __init__(self, n_in, n_out, ks, dilation=1, stride=1, padding="same"):

        super().__init__()
        self.conv = nn.Conv1d(n_in, n_out, ks, stride, padding, dilation)
        self.act = nn.SiLU()
        self.norm = nn.BatchNorm1d(n_out)  # MyLayerNorm(n_out)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.map = nn.Sequential(
            nn.BatchNorm1d(n_out),
            nn.Linear(n_out, n_out),
            nn.SiLU(),
            nn.BatchNorm1d(n_out),
            nn.Linear(n_out, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, _, _ = x.shape
        x = self.norm(self.act(self.conv(x)))

        gates = self.map(self.pool(x).view(bs, -1))

        return gates.unsqueeze(-1) * x


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, act_fn, kernel_size=75):
        """
        Inputs:
            c_in - Number of input features
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
            act_fn - Activation class constructor (e.g. nn.ReLU)
            kernel_size - kernel size for nn.Conv1d
        """
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                padding="same",
                stride=1,
                bias=False,
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm1d(c_out),
            act_fn(),
            nn.BatchNorm1d(c_out),
            # nn.Conv1d(
            #     c_out, c_out, kernel_size=kernel_size, padding="same", bias=False
            # ),
            # nn.BatchNorm1d(c_out),
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
