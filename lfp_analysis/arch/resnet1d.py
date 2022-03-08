import os
from pathlib import Path
from math import pi
import numpy as np

from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.vision.all import nn, ConvLayer, F
from torch.nn.modules.activation import ReLU

import wandb

from ..score import Scorer, CnnScores
from .base import *
from .sinc import *


DATA_PATH = Path("./../data")


class ResNetMini(nn.Module):
    def __init__(
        self,
        n_in=8,
        num_classes=2,
        ks=89,
        num_blocks=[1],
        c_hidden=[8, 16, 32],
        act_fn_name="silu",
    ):
        super().__init__()

        if ks % 2 == 0:
            ks = ks + 1

        self.hparams = SimpleNamespace(
            n_in=n_in,
            num_classes=num_classes,
            ks=ks,
            num_blocks=num_blocks,
            c_hidden=c_hidden,
            act_fn=act_fn_by_name[act_fn_name],
        )

        self._create_network()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.InstanceNorm1d(self.hparams.n_in, affine=True),
            nn.Conv1d(
                self.hparams.n_in,
                self.hparams.c_hidden[0],
                kernel_size=self.hparams.ks,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.hparams.c_hidden[0]),
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
                        c_in=self.hparams.c_hidden[
                            block_idx if not subsample else (block_idx - 1)
                        ],
                        act_fn=self.hparams.act_fn,
                        kernel_size=self.hparams.ks,
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.convs = nn.Sequential(self.input_net, self.blocks, self.out_norm)

        # Mapping to classification output
        self.cls = nn.Sequential(
            nn.BatchNorm1d(self.hparams.c_hidden[len(self.hparams.num_blocks) - 1]),
            nn.Linear(
                self.hparams.c_hidden[len(self.hparams.num_blocks) - 1],
                self.hparams.num_classes,
            ),
        )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.out_norm(x)

        return self.cls(x)


class ResNet(nn.Module):
    def __init__(
        self,
        n_in=8,
        num_classes=2,
        num_blocks=[1, 1, 1],
        c_hidden=[16, 32, 64],
        ks=75,
        use_sinc=True,
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
        if ks % 2 == 0:
            ks += 1

        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
            ks=ks,
            use_sinc=use_sinc,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
            n_in=n_in,
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        ks = self.hparams.ks

        if self.hparams.use_sinc:
            self.conv_init = nn.Sequential(
                SincAugmented(
                    255, ks, n_combs=2, n_in=self.hparams.n_in, n_f=10, use_gate=False
                ),
            )
            self.input_net = nn.Sequential(
                nn.InstanceNorm1d(self.hparams.n_in, affine=True),
                self.conv_init,
                nn.BatchNorm1d(30),
                nn.Conv1d(30, c_hidden[0], kernel_size=ks, padding="same"),
                nn.BatchNorm1d(c_hidden[0]),
                nn.ReLU(),
            )
        else:
            self.input_net = nn.Sequential(
                nn.InstanceNorm1d(self.hparams.n_in, affine=True),
                nn.Conv1d(8, c_hidden[0], kernel_size=ks, padding="same", bias=False),
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
                        kernel_size=ks,
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.convs = nn.Sequential(self.input_net, self.blocks, self.out_norm)

        # Mapping to classification output
        self.cls = nn.Sequential(
            nn.BatchNorm1d(c_hidden[len(self.hparams.num_blocks) - 1]),
            nn.Linear(
                c_hidden[len(self.hparams.num_blocks) - 1],
                c_hidden[len(self.hparams.num_blocks) - 1] + 50,
            ),
            self.hparams.act_fn(),
            nn.Dropout(0.0),
            nn.Linear(
                c_hidden[len(self.hparams.num_blocks) - 1] + 50,
                self.hparams.num_classes,
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
        x = self.out_norm(x)
        x = self.cls(x)
        return x


# class BasicConvNet(nn.Module):
#     def __init__(self, n_in, n_out):
#         super().__init__()
#         filters = [16, 32, 64, 128]
#         f_0 = filters[0]
#         self.layers = [ConvBlock(n_in, f_0, 25)]
#         for f in filters[1:]:
#             self.layers.append(ConvBlock(f_0, f, 15, 2))
#             f_0 = f

#         self.conv = nn.Sequential(*self.layers)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Sequential(nn.Dropout(0.75), nn.Linear(filters[-1], n_out))

#     def forward(self, x):

#         return self.classifier(self.pool(self.conv(x)).view(x.shape[0], -1))


# class BasicSincNet(nn.Module):
#     def __init__(self, n_in, n_out):
#         super().__init__()
#         filters = [20, 25, 25, 25]
#         f_0 = filters[0]
#         self.layers = [SincConv_multiChan(251)]
#         for f in filters[1:]:
#             self.layers.append(ResBlock(f_0, f, 51, 1))
#             f_0 = f

#         self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)
#         self.conv = nn.Sequential(*self.layers)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(filters[-1]), nn.Dropout(0.5), nn.Linear(filters[-1], n_out)
#         )

#     def forward(self, x):

#         return self.classifier(
#             self.pool(self.conv(self.norm_input(x))).view(x.shape[0], -1)


class BasicSincNetAug(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        filters = [40, 60, 80, 120, 240]
        f_0 = filters[0]
        self.layers = [SincAugmented(3, 10)]
        for f in filters[1:]:
            self.layers.append(ResBlock(f_0, f, 75, 1))
            f_0 = f

        self.layer_norm = nn.Identity()  # nn.LayerNorm(1535)
        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)

        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(filters[-1]), nn.Dropout(0.5), nn.Linear(filters[-1], n_out)
        )

    def forward(self, x):

        return self.classifier(
            self.pool(self.conv(self.norm_input(self.layer_norm(x)))).view(
                x.shape[0], -1
            )
        )


class BasicSincNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        filters = [20, 35, 35, 35, 35, 25, 25]
        f_0 = filters[0]
        self.layers = [SincConv_multiChan(251)]
        for f in filters[1:]:
            self.layers.append(ResBlock(f_0, f, 51, 1))
            f_0 = f

        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)
        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(filters[-1]), nn.Dropout(0.5), nn.Linear(filters[-1], n_out)
        )

    def forward(self, x):

        return self.classifier(
            self.pool(self.conv(self.norm_input(x))).view(x.shape[0], -1)
        )


class DenseBlock(nn.Module):
    def __init__(self, conv):
        super().__init__()

        self.conv = conv
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        # import pdb; pdb.set_trace()
        bs, _, _ = x.shape

        return self.conv(x), self.pool(x).view(bs, 1, -1)


class BasicSincNetDense(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        filters = [40, 60, 80, 120, 240]
        f_0 = filters[0]
        self.layers = [SincAugmented(3, 10)]
        for f in filters[1:]:
            self.layers.append(DenseBlock(ResBlock(f_0, f, 75, 1)))
            f_0 = f

        self.layer_norm = nn.Identity()  # nn.LayerNorm(1535)
        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)

        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        n_feats = np.sum(np.array(filters))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_feats), nn.Dropout(0.3), nn.Linear(n_feats, n_out)
        )

    # def forward(self, x):

    #     bs, _, _ = x.shape

    #     x = self.norm_input(self.layer_norm(x))
    #     x_dense = []

    #     x = self.layers[0](x)

    #     for layer in self.layers[1:]:
    #         # import pdb; pdb.set_trace()
    #         x, x_scalar = layer(x)
    #         x_dense.append(x_scalar)

    #     x = self.pool(x).view(bs, -1)
    #     # import pdb; pdb.set_trace()

    #     x_dense = torch.cat(x_dense, 2).view(bs, -1)

    #     x = torch.cat([x, x_dense], 1)

    #     return self.classifier(x)

    def forward(self, x):

        bs, _, _ = x.shape

        x = self.norm_input(self.layer_norm(x))
        # x_dense = []

        x = self.layers[0](x)

        x, x_dense = self.layers[1](x)

        for layer in self.layers[2:]:
            # import pdb; pdb.set_trace()
            x, x_scalar = layer(x)
            x_dense = torch.cat([x_dense, x_scalar], 2)

        x = self.pool(x).view(bs, -1)
        # import pdb; pdb.set_trace()

        # x_dense = torch.cat(x_dense,2).view(bs,-1)

        x = torch.cat([x, x_dense.view(bs, -1)], 1)

        return self.classifier(x)


class ResBlock(Module):
    def __init__(self, n_in, n_f, ks, stride=1):
        self.convs = ConvBlock(n_in, n_f, ks, stride, padding="same")
        self.idconv = noop if n_in == n_f else ConvBlock(n_in, n_f, 1)
        self.pool = noop if stride == 1 else nn.AvgPool1d(2, ceil_mode=True)
        self.bn = nn.BatchNorm1d(n_f)

    def forward(self, x):
        return self.bn(nn.SiLU()(self.convs(x) + self.idconv(self.pool(x))))


class BasicResNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        filters = [8, 8, 8, 8, 8]
        f_0 = filters[0]
        self.layers = [ConvBlock(n_in, f_0, 41, padding="same")]
        for f in filters[1:]:
            self.layers.append(ResBlock(f_0, f, 41, 1))
            f_0 = f

        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)
        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(filters[-1]), nn.Dropout(0.5), nn.Linear(filters[-1], n_out)
        )

    def forward(self, x):

        return self.classifier(
            self.pool(self.conv(self.norm_input(x))).view(x.shape[0], -1)
        )


class BasicIsolatedChans(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()

        filters = [8, 8, 8, 8, 8]
        f_0 = filters[0]
        self.layers = [ConvBlock(1, f_0, 41)]
        for f in filters[1:]:
            self.layers.append(ResBlock(f_0, f, 41, 1))
            f_0 = f

        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)
        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(filters[-1] * n_in),
            nn.Dropout(0.5),
            nn.Linear(filters[-1] * n_in, n_out),
        )

    def forward(self, x):

        bs, _, _ = x.shape

        x = self.norm_input(x)

        channel_wise_embed = torch.stack(
            [self.pool(self.conv(xx.unsqueeze(1))) for xx in torch.unbind(x, 1)],
            1,
        )

        return self.classifier(channel_wise_embed.view(bs, -1))


class BasicIsolatedChans_small(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()

        filters = [1, 2, 4]
        f_0 = filters[0]
        self.layers = [ConvBlock(1, f_0, 25)]
        for f in filters[1:]:
            self.layers.append(ResBlock(f_0, f, 25, 1))
            f_0 = f

        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)
        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(filters[-1] * n_in),
            nn.Dropout(0.75),
            nn.Linear(filters[-1] * n_in, n_out),
        )

    def forward(self, x):

        bs, _, _ = x.shape

        x = self.norm_input(x)

        channel_wise_embed = torch.stack(
            [self.pool(self.conv(xx.unsqueeze(1))) for xx in torch.unbind(x, 1)],
            1,
        )

        return self.classifier(channel_wise_embed.view(bs, -1))


class BasicConvNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        filters = [8, 16, 32, 64, 75]
        f_0 = filters[0]
        self.layers = [ConvBlock(n_in, f_0, 75)]
        for f in filters[1:]:
            self.layers.append(ConvBlock(f_0, f, 75, 1))
            f_0 = f

        self.norm_input = nn.BatchNorm1d(8)  # nn.LayerNorm(1535)
        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(filters[-1]), nn.Dropout(0.75), nn.Linear(filters[-1], n_out)
        )

    def forward(self, x):

        return self.classifier(
            self.pool(self.conv(self.norm_input(x))).view(x.shape[0], -1)
        )


class BasicConvNet2d(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        filters1d = [16, 32]
        f_0 = filters1d[0]
        self.layers1d = [ConvBlock(n_in, f_0, 18)]
        for f in filters1d[1:]:
            self.layers.append(ConvBlock(f_0, f, 18, 2))
            f_0 = f

        filters2d = [6, 12, 18]
        f_0 = filters2d[0]
        self.layers2d = [ConvBlock2d(1, f_0, 5)]
        for f in filters2d[1:]:
            self.layers2d.append(ConvBlock2d(f_0, f, 5, 2))
            f_0 = f

        self.conv1d = nn.Sequential(*self.layers1d)
        self.conv2d = nn.Sequential(*self.layers2d)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.75), nn.Linear(self.layers2d[-1], n_out)
        )

    def forward(self, x):

        x_1d = self.conv1d(x).unsqueeze(1)
        x_2d = self.conv2d(x_1d)

        return self.classifier(self.pool(x_2d).view(x.shape[0], -1))


class DotProd2d(nn.Module):
    def __init__(
        self,
        pat_emb_n=8,
        pat_emb_dim=24,
        task_emb_n=3,
        task_emb_dim=24,
        stim_emb_n=2,
        stim_emb_dim=24,
    ):
        super().__init__()

        filters1d = [16, 32]
        f_0 = filters1d[0]
        self.layers1d = [ConvBlock(8, f_0, 18)]
        for f in filters1d[1:]:
            self.layers1d.append(ConvBlock(f_0, f, 18, 2))
            f_0 = f

        filters2d = [6, 12, 18]
        f_0 = filters2d[0]
        self.layers2d = [ConvBlock2d(1, f_0, 5)]
        for f in filters2d[1:]:
            self.layers2d.append(ConvBlock2d(f_0, f, 5, 2))
            f_0 = f

        self.conv1d = nn.Sequential(*self.layers1d)
        self.conv2d = nn.Sequential(*self.layers2d)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.pat_emb = Embedding(pat_emb_n, pat_emb_dim)
        self.task_emb = Embedding(task_emb_n, task_emb_dim)
        self.stim_emb = Embedding(stim_emb_n, stim_emb_dim)
        self.emb_linear = nn.Linear(pat_emb_dim + task_emb_dim + stim_emb_dim, 128)

        self.cls = nn.Sequential(nn.Dropout(0.5), nn.Linear(128 * filters2d[-1], 2))

    def forward(self, x):
        import pdb

        # pdb.set_trace()
        pat_id, task_id, stim_id, waves = x[0], x[1], x[2], x[3]
        bs = waves.shape[0]
        pat_embedded = self.pat_emb(pat_id)
        task_embedded = self.task_emb(task_id)
        stim_embedded = self.task_emb(stim_id)

        wave_feats = self.pool(self.conv2d(self.conv1d(waves).unsqueeze(1)))

        ctx = self.emb_linear(
            torch.cat([pat_embedded, task_embedded, stim_embedded], dim=1)
        )
        gated = torch.matmul(wave_feats.squeeze(-1), ctx.unsqueeze(1)).view(bs, -1)

        return self.cls(gated)


class DotProdBasic(nn.Module):
    def __init__(
        self,
        pat_emb_n=8,
        pat_emb_dim=24,
        task_emb_n=3,
        task_emb_dim=24,
        stim_emb_n=2,
        stim_emb_dim=24,
    ):
        super().__init__()

        filters = [16, 32, 64, 128, 256, 512, 512]
        f_0 = filters[0]
        self.layers = [ConvBlock(8, f_0, 18)]
        for f in filters[1:]:
            self.layers.append(ConvBlock(f_0, f, 18, 2))
            f_0 = f

        self.conv = nn.Sequential(*self.layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.pat_emb = Embedding(pat_emb_n, pat_emb_dim)
        self.task_emb = Embedding(task_emb_n, task_emb_dim)
        self.stim_emb = Embedding(stim_emb_n, stim_emb_dim)

        self.emb_linear = nn.Linear(pat_emb_dim + task_emb_dim + stim_emb_dim, 128)

        self.cls = nn.Sequential(nn.Dropout(0.5), nn.Linear(128 * filters[-1], 2))

    def forward(self, x):
        import pdb

        # pdb.set_trace()
        pat_id, task_id, stim_id, waves = x[0], x[1], x[2], x[3]
        bs = waves.shape[0]
        pat_embedded = self.pat_emb(pat_id)
        task_embedded = self.task_emb(task_id)
        stim_embedded = self.task_emb(stim_id)

        wave_feats = self.pool(self.conv(waves))

        ctx = self.emb_linear(
            torch.cat([pat_embedded, task_embedded, stim_embedded], dim=1)
        )

        gated = torch.matmul(wave_feats, ctx.unsqueeze(1)).view(bs, -1)

        return self.cls(gated)


class ChannelWiseBasic(nn.Module):
    def __init__(self, n_chan):
        super().__init__()
        filters = [16, 32, 64]
        f_0 = filters[0]
        self.layers = [ConvBlock(1, f_0, 18)]
        for f in filters[1:]:
            self.layers.append(ConvBlock(f_0, f, 18, 2))
            f_0 = f

        self.conv = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.75), nn.Linear(filters[-1] * n_chan, 2)
        )

    def forward(self, x):
        bs, _, _ = x.shape

        channel_wise_embed = torch.stack(
            [self.pool(self.conv(xx.unsqueeze(1))) for xx in torch.unbind(x, 1)], 1
        )

        return self.classifier(channel_wise_embed.view(bs, -1))


def _conv_block1d(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf // 4, 1, ndim=1),
        ConvLayer(nf // 4, nf // 4, 23, stride=stride, ndim=1),
        ConvLayer(nf // 4, nf, 1, act_cls=None, norm_type=NormType.BatchZero, ndim=1),
    )


class ResBlock1d(Module):
    def __init__(self, ni, nf, stride=1):
        self.convs = _conv_block1d(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None, ndim=1)
        self.pool = noop if stride == 1 else nn.AvgPool1d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


def _resnet_stem1d(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i + 1], 23, stride=2 if i == 0 else 1, ndim=1)
        for i in range(len(sizes) - 1)
    ] + [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]


class ResNet1d(nn.Sequential):
    def __init__(self, n_in, n_out, layers, expansion=1):
        stem = _resnet_stem1d(n_in, 32, 64)
        self.block_szs = [64, 64, 128, 128, 256, 256, 512, 512]
        for i in range(1, len(self.block_szs)):
            self.block_szs[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(
            *stem,
            *blocks,
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Dropout(0.75),
            nn.Linear(self.block_szs[len(layers)], n_out),
            Flatten(),
        )

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_szs[idx : idx + 2]
        return nn.Sequential(
            *[
                ResBlock1d(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
                for i in range(n_layers)
            ]
        )


class AcrossConfigsResnet(nn.Module):
    def __init__(self, layers, n_chan_features=256):
        super().__init__()

        self.channelProcessor = ResNet1d(1, n_chan_features, layers)
        self.LSTM = nn.LSTM(
            n_chan_features,
            n_chan_features * 2,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(n_chan_features * 4, n_chan_features * 2),
            nn.ReLU(),
            nn.Linear(n_chan_features * 2, n_chan_features * 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(n_chan_features * 2, 2),
            Flatten(),
        )

    def forward(self, x):
        channel_wise_embed = torch.stack(
            [self.channelProcessor(xx.unsqueeze(1)) for xx in torch.unbind(x, 1)], 1
        )

        global_embed = self.fc1(self.LSTM(channel_wise_embed)[0][:, -1, :])
        out = self.classifier(global_embed)

        return out


class conditionedResNet(nn.Module):
    def __init__(
        self,
        layers: list[int] = [1, 1],
        n_conv_feats: int = 128,
        pat_emb_n: int = 8,
        pat_emb_dim: int = 16,
        task_emb_n: int = 3,
        task_emb_dim: int = 16,
        stim_emb_n: int = 2,
        stim_emb_dim=16,
    ):

        super().__init__()

        self.feature_extractor = ResNet1d(8, n_conv_feats, layers)
        self.cls = nn.Sequential(
            nn.Linear(
                n_conv_feats + pat_emb_dim + task_emb_dim + stim_emb_dim,
                n_conv_feats * 2,
            ),
            nn.Linear(n_conv_feats * 2, 2),
            Flatten(),
        )
        self.pat_emb = Embedding(pat_emb_n, pat_emb_dim)
        self.task_emb = Embedding(task_emb_n, task_emb_dim)
        self.stim_emb = Embedding(stim_emb_n, stim_emb_dim)

    def forward(self, x):

        pat_id, task_id, stim_id, waves = x[0], x[1], x[2], x[3]
        pat_embedded = self.pat_emb(pat_id)
        task_embedded = self.task_emb(task_id)
        stim_embedded = self.task_emb(stim_id)

        wave_embedded = self.feature_extractor(waves)

        feat_vector = torch.cat(
            [pat_embedded, task_embedded, stim_embedded, wave_embedded], dim=1
        )

        return self.cls(feat_vector)


###############################################################################
# FastAI pipeline:
##############################################################################


def get_norm_stats(LFP, train_idx):
    return np.mean(LFP[:, :train_end], axis=-1), np.std(LFP[:, :train_end], axis=-1)


def norm_with_stats(LFP, stats):
    means, stds = stats[0], stats[1]
    return (LFP - means[:, np.newaxis]) / stds[:, np.newaxis]


class LFPNormalizer1d(Transform):
    def __init__(self, stats):
        self.means, self.stds = (
            torch.tensor(stats[0]).float(),
            torch.tensor(stats[1]).float(),
        )

        if torch.cuda.is_available():
            self.means = self.means.cuda()
            self.stds = self.stds.cuda()

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return (X - self.means[:, None]) / self.stds[:, None]

    def decodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return X * self.stds[:, None] + self.means[:, None]


class Trainer:
    def __init__(self, log_wandb=True, experiment=None):
        self.log_wandb = log_wandb
        self.experiment = experiment

        if self.log_wandb:
            if experiment is None:
                self.run = wandb.init()
            else:
                self.run = wandb.init(project=experiment)

    def save_model(self):
        self.learn.model_dir = self.model_path

        self.learn.save("model")

    def score(self):

        # Train:
        times_train = self.data_df[self.data_df["is_valid"] == False]["t"].values
        y_scores, y, losses = self.learn.get_preds(ds_idx=0, with_loss=True)
        y_hat = torch.argmax(y_scores, -1).numpy()
        y_scores = y_scores[:, 1].numpy()

        train_scores = Scorer(ds_type="train").get_scores(
            y, y_hat, y_scores, losses.numpy(), times=times_train
        )

        # Valid:
        times_valid = self.data_df[self.data_df["is_valid"] == True]["t"].values
        y_scores, y, losses = self.learn.get_preds(with_loss=True)
        y_hat = torch.argmax(y_scores, -1).numpy()
        y_scores = y_scores[:, 1].numpy()

        valid_scores = Scorer().get_scores(
            y, y_hat, y_scores, losses.numpy(), times=times_valid
        )

        self.scores = CnnScores(train_scores, valid_scores)

        return self.scores


class Trainer1d(Trainer):
    def __init__(self, layers=[1, 1], wd=0.025, log_wandb=True, experiment=None):

        self.layers, self.wd = layers, wd

        super().__init__(log_wandb, experiment)

    def prepare_dls(self, dataset, windower, bs=256):

        if self.log_wandb:
            self.run.name = f"{dataset.pat_id}/{dataset.task}_{dataset.stim}_1d"

        if self.experiment is not None:
            self.model_path = (
                DATA_PATH
                / "results"
                / f"ET{dataset.pat_id}"
                / self.experiment
                / f"{dataset.task}"
                / "trained"
            )
            self.model_path.mkdir(parents=True, exist_ok=True)

        self.dataset, self.windower = dataset, windower

        self.data_df = self.windower.df

        def get_x(row):
            return torch.tensor(
                dataset.LFP.data[:, int(row["id_start"]) : int(row["id_end"])].copy()
            ).float()

        def get_y(row):
            return row["label"]

        def splitter(df):
            train = df.index[df["is_valid"] == 0].tolist()
            valid = df.index[df["is_valid"] == 1].tolist()
            return train, valid

        train_end = self.data_df[self.data_df["is_valid"] == 0]["id_end"].iloc[-1]

        def LFP_block1d():
            return TransformBlock(
                batch_tfms=(
                    LFPNormalizer1d(get_norm_stats(dataset.LFP.data, train_end))
                )
            )

        self.dblock = DataBlock(
            blocks=(LFP_block1d, CategoryBlock),
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        self.dls = self.dblock.dataloaders(self.data_df, bs=bs)

        return self

    def prepare_learner(self, dls=None, wd=None):

        cbs = [WandbCallback()] if self.log_wandb else []

        dls = self.dls if dls is None else dls
        wd = self.wd if wd is None else wd

        loss = LabelSmoothingCrossEntropy(eps=0.2)
        self.resnet = ResNet1d(self.dataset.LFP.data.shape[0], 2, self.layers).cuda()

        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()

        self.learn = Learner(
            dls,
            self.resnet.cuda(),
            metrics=[
                accuracy,
            ],
            loss_func=loss,
            cbs=cbs,
            wd=float(wd),
        )
        self.learn.recorder.train_metrics = True

        return self

    def train(self, n_epochs=10, lr_div=1):

        self.learn.fit_one_cycle(n_epochs, lr_div)
        self.learn.fit_one_cycle(n_epochs, lr_div / 2)
        self.learn.fit_one_cycle(n_epochs, lr_div / 4)
        self.learn.fit_one_cycle(n_epochs, lr_div / 8)
        self.learn.add_cb(EarlyStoppingCallback(min_delta=0.001, patience=3))

        # self.learn.fit_one_cycle(14, 10e-4)
        # self.learn.fit_one_cycle(25, 5 * 10e-5)
        self.learn.fit_one_cycle(35, 10e-5)
        self.learn.fit_one_cycle(35, 3 * 10e-6)
        self.learn.fit_one_cycle(35, 10e-6)
        self.learn.fit_one_cycle(35, 10e-7)

        [self.learn.remove_cb(cb) for cb in self.learn.cbs[3:]]
