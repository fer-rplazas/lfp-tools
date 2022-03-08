# %%

import torch

import torch.functional as F
import torch.nn as nn

from fastai.vision.all import *

from .sinc import SincConv_multiChan, SincAugmented
from .base import SEConv

# %%


class DenseFeatures(nn.Module):
    def __init__(self, n_in, depth=5, growth_rate=8, also_scalar=False, ks=255):
        super().__init__()

        self.growth_rate = growth_rate
        self.depth = depth
        self.also_scalar = also_scalar

        self.layers = []
        for i in range(self.depth):
            self.layers.append(
                SEConv(n_in + i * self.growth_rate, self.growth_rate, ks)
            )
        self.layers = nn.Sequential(*self.layers)

        self.transition = SEConv(
            n_in + self.depth * self.growth_rate,
            n_in + self.depth * self.growth_rate,
            ks,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if self.also_scalar:
            bs, _, _ = x.shape
            scalars = []

        for layer in self.layers.children():
            x_forward = layer(x)
            x = torch.cat([x_forward, x], 1)
            if self.also_scalar:
                scalars.append(self.pool(x_forward).view(bs, -1))

        if self.also_scalar:
            transited = self.transition(x)
            scalars.append(self.pool(transited).view(bs, -1))
            return transited, torch.cat(scalars, 1)

        return self.transition(x)


class PoolMod(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], 1)


class DenseNetFull(nn.Module):
    def __init__(
        self,
        n_in=8,
        n_out=2,
        depth=1,
        growth_rate=8,
        ks=55,
        sinc_ks=255,
        also_scalar=True,
        use_sinc=True,
    ):
        super().__init__()

        self.also_scalar = also_scalar

        if ks % 2 == 0:
            ks += 1

        if use_sinc:
            # self.conv_init = SEConv(n_in, 10, 255)
            # self.conv_init = SincConv_multiChan(sinc_ks, n_combs=1, use_gate=True)
            self.conv_init = SincAugmented(
                sinc_ks, ks, n_combs=2, n_f=10, use_gate=False
            )
        else:
            self.conv_init = SEConv(n_in, 28, ks)
        n_first_out = 30

        self.norm_input = nn.InstanceNorm1d(n_in, affine=True)
        self.feature_extractor = DenseFeatures(
            n_first_out,
            depth=depth,
            growth_rate=growth_rate,
            also_scalar=self.also_scalar,
            ks=ks,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.n_feats = n_first_out + depth * growth_rate
        self.conv_id = SEConv(self.n_feats, n_first_out, ks)
        self.n_feats = n_first_out

        if self.also_scalar:
            self.n_feats += n_first_out + depth * growth_rate * 2

        self.cls = nn.Sequential(
            nn.BatchNorm1d(self.n_feats),
            nn.Linear(self.n_feats, self.n_feats * 2),
            nn.SiLU(),
            nn.Dropout(
                0.0,
            ),
            nn.BatchNorm1d(self.n_feats * 2),
            nn.Linear(self.n_feats * 2, n_out),
        )

    def forward(self, x):
        bs, _, _ = x.shape
        x = self.norm_input(x)
        x = self.conv_init(x)

        if self.also_scalar:
            x_feats, x_scalar = self.feature_extractor(x)
            x = x + self.conv_id(x_feats)

            feats = torch.cat([self.pool(x).view(bs, -1), x_scalar], 1)
        else:
            x_feats = self.feature_extractor(x)
            x = x + self.conv_id(x_feats)
            feats = self.pool(x).view(bs, -1)

        return self.cls(feats)

    def convs(self, x):

        bs, _, _ = x.shape
        x = self.norm_input(x)
        x = self.conv_init(x)

        if self.also_scalar:
            x_feats, x_scalar = self.feature_extractor(x)
            x = x + self.conv_id(x_feats)
            feats = torch.cat([self.pool(x).view(bs, -1), x_scalar], 1)
        else:
            x_feats = self.feature_extractor(x)
            x = x + self.conv_id(x_feats)
            feats = self.pool(x).view(bs, -1)

        return feats


# %%
