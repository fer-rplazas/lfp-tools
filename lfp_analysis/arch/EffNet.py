import torch
import torch.nn as nn
from math import ceil, pi
import torch.nn.functional as F
import numpy as np

from .resnet1d import ConvBlock


class AdMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        """
        AM Softmax Loss
        """
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat(
            [
                torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(labels)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def pad_val(input_size, kernel_size, stride):

    return ceil(((stride - 1) * input_size - stride + kernel_size) / 2)


base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 15],
    # [6, 24, 2, 2, 15],
    # [6, 40, 2, 2, 15],
    # [6, 80, 3, 2, 15],
    # [6, 112, 3, 1, 31],
    # [6, 192, 4, 2, 31],
    # [6, 320, 1, 1, 31],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.bn(self.silu(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # C x H x W -> C x 1 x 1
            nn.Conv1d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, device=x.device) < self.survival_prob
        )
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, in_channels, num_classes):
        super().__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(self.in_channels, channels, 15, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


# class SincConv_multiChan(nn.Module):
#     """Sinc-based convolution
#     Parameters
#     ----------
#     in_channels : `int`
#         Number of input channels. Must be 1.
#     out_channels : `int`
#         Number of filters.
#     kernel_size : `int`
#         Filter length.
#     sample_rate : `int`, optional
#         Sample rate. Defaults to 16000.
#     Usage
#     -----
#     See `torch.nn.Conv1d`
#     Reference
#     ---------
#     Mirco Ravanelli, Yoshua Bengio,
#     "Speaker Recognition from raw waveform with SincNet".
#     https://arxiv.org/abs/1808.00158
#     """

#     def __init__(
#         self,
#         out_channels,
#         kernel_size,
#         sample_rate=16000,
#         in_channels=8,
#         stride=1,
#         padding=0,
#         dilation=1,
#         bias=False,
#         groups=1,
#         min_low_hz=2,
#         min_band_hz=3,
#         n_combs=10,
#         use_gate=True,
#     ):

#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size

#         # Forcing the filters to be odd (i.e, perfectly symmetrics)
#         if kernel_size % 2 == 0:
#             self.kernel_size = self.kernel_size + 1

#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation

#         if bias:
#             raise ValueError("SincConv does not support bias.")
#         if groups > 1:
#             raise ValueError("SincConv does not support groups.")

#         self.sample_rate = sample_rate
#         self.min_low_hz = min_low_hz
#         self.min_band_hz = min_band_hz

#         self.use_gate = use_gate

#         # initialize filterbanks such that they are equally spaced in Mel scale
#         low_hz = 5
#         high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

#         hz = np.linspace(low_hz, high_hz, self.out_channels + 1)

#         hz_ref = np.array([2, 7, 12, 20, 30, 45, 60, 75, 100, 150, 200]).reshape(-1, 1)
#         bws_ref = np.diff(hz_ref, axis=0)

#         hz = np.tile(hz_ref[:-1, :], (n_combs, self.in_channels)) - self.min_low_hz
#         bws = np.tile(bws_ref, (n_combs, self.in_channels)) - self.min_band_hz

#         self.n_filter = hz.shape[0]

#         # filter lower frequency (out_channels, 1)
#         self.low_hz_ = nn.Parameter(torch.Tensor(hz))

#         # filter frequency band (out_channels, 1)
#         self.band_hz_ = nn.Parameter(torch.Tensor(bws))

#         # Gate:
#         self.gate = nn.Parameter(torch.Tensor(np.zeros((self.low_hz_.numel(), 1))))

#         # Hamming window
#         # self.window_ = torch.hamming_window(self.kernel_size)
#         n_lin = torch.linspace(
#             0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
#         )  # computing only half of the window
#         self.window_ = 0.54 - 0.46 * torch.cos(2 * pi * n_lin / self.kernel_size)

#         # (1, kernel_size/2)
#         n = (self.kernel_size - 1) / 2.0
#         self.n_ = (
#             2 * pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
#         )  # Due to symmetry, I only need half of the time axes

#     def forward(self, waveforms):
#         """
#         Parameters
#         ----------
#         waveforms : `torch.Tensor` (batch_size, 1, n_samples)
#             Batch of waveforms.
#         Returns
#         -------
#         features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
#             Batch of sinc filters activations.
#         """

#         self.n_ = self.n_.to(waveforms.device)

#         self.window_ = self.window_.to(waveforms.device)

#         low = self.min_low_hz + torch.abs(self.low_hz_)

#         high = torch.clamp(
#             low + self.min_band_hz + torch.abs(self.band_hz_),
#             self.min_low_hz,
#             self.sample_rate / 2,
#         )
#         band = high - low

#         f_times_t_low = torch.matmul(low.view(-1, 1), self.n_)
#         f_times_t_high = torch.matmul(high.view(-1, 1), self.n_)

#         band_pass_left = (
#             (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
#         ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
#         band_pass_center = 2 * band.view(-1, 1)
#         band_pass_right = torch.flip(band_pass_left, dims=[1])

#         band_pass = torch.cat(
#             [band_pass_left, band_pass_center, band_pass_right], dim=1
#         )

#         band_pass = band_pass / (2 * band.view(-1, 1))

#         if self.use_gate:
#             band_pass = self.gate * band_pass

#         self.filters = (band_pass).view(
#             self.n_filter, self.in_channels, self.kernel_size
#         )

#         return F.conv1d(
#             waveforms,
#             self.filters,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             bias=None,
#             groups=1,
#         )


# class SincConv_multiChan_depthwise(nn.Module):
#     """Sinc-based convolution
#     Parameters
#     ----------
#     in_channels : `int`
#         Number of input channels. Must be 1.
#     out_channels : `int`
#         Number of filters.
#     kernel_size : `int`
#         Filter length.
#     sample_rate : `int`, optional
#         Sample rate. Defaults to 16000.
#     Usage
#     -----
#     See `torch.nn.Conv1d`
#     Reference
#     ---------
#     Mirco Ravanelli, Yoshua Bengio,
#     "Speaker Recognition from raw waveform with SincNet".
#     https://arxiv.org/abs/1808.00158
#     """

#     def __init__(
#         self,
#         out_channels,
#         kernel_size,
#         sample_rate=2048.0,
#         in_channels=1,
#         stride=1,
#         padding=0,
#         dilation=1,
#         bias=False,
#         groups=1,
#         min_low_hz=1,
#         min_band_hz=1,
#     ):

#         super().__init__()

#         self.out_channels = out_channels
#         self.kernel_size = kernel_size

#         # Forcing the filters to be odd (i.e, perfectly symmetrics)
#         if kernel_size % 2 == 0:
#             self.kernel_size = self.kernel_size + 1

#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation

#         if bias:
#             raise ValueError("SincConv does not support bias.")
#         if groups > 1:
#             raise ValueError("SincConv does not support groups.")

#         self.sample_rate = sample_rate
#         self.min_low_hz = min_low_hz
#         self.min_band_hz = min_band_hz

#         # initialize filterbanks such that they are equally spaced in Mel scale
#         low_hz = 5
#         high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

#         hz = np.linspace(low_hz, high_hz, self.out_channels + 1)

#         hz = np.array([2, 7, 12, 20, 30, 45, 60, 75, 100, 150, 200])

#         # filter lower frequency (out_channels, 1)
#         self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

#         # filter frequency band (out_channels, 1)
#         self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

#         # Hamming window
#         # self.window_ = torch.hamming_window(self.kernel_size)
#         n_lin = torch.linspace(
#             0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
#         )  # computing only half of the window
#         self.window_ = 0.54 - 0.46 * torch.cos(2 * pi * n_lin / self.kernel_size)

#         # (1, kernel_size/2)
#         n = (self.kernel_size - 1) / 2.0
#         self.n_ = (
#             2 * pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
#         )  # Due to symmetry, I only need half of the time axes

#     def forward(self, waveforms):
#         """
#         Parameters
#         ----------
#         waveforms : `torch.Tensor` (batch_size, 1, n_samples)
#             Batch of waveforms.
#         Returns
#         -------
#         features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
#             Batch of sinc filters activations.
#         """

#         self.n_ = self.n_.to(waveforms.device)

#         self.window_ = self.window_.to(waveforms.device)

#         low = self.min_low_hz + torch.abs(self.low_hz_)

#         high = torch.clamp(
#             low + self.min_band_hz + torch.abs(self.band_hz_),
#             self.min_low_hz,
#             self.sample_rate / 2,
#         )
#         band = (high - low)[:, 0]

#         f_times_t_low = torch.matmul(low, self.n_)
#         f_times_t_high = torch.matmul(high, self.n_)

#         band_pass_left = (
#             (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
#         ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
#         band_pass_center = 2 * band.view(-1, 1)
#         band_pass_right = torch.flip(band_pass_left, dims=[1])

#         band_pass = torch.cat(
#             [band_pass_left, band_pass_center, band_pass_right], dim=1
#         )

#         band_pass = band_pass / (2 * band[:, None])

#         self.filters = (band_pass).view(self.low_hz_.shape[0], 1, self.kernel_size)

#         self.filters = torch.tile(self.filters, (1, waveforms.shape[1], 1))

#         return F.conv1d(
#             waveforms,
#             self.filters,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             bias=None,
#             groups=1,
#         )


# class SincNet(nn.Module):
#     def __init__(self):

#         super().__init__()
#         # self.sinc = SincConv_multiChan(
#         #     0, 513, sample_rate=2048.0, in_channels=8, use_gate=True, n_combs=1
#         # )
#         self.sinc = SincConv_multiChan_depthwise(100, 513)
#         self.norm_input = nn.Identity()  # nn.LayerNorm(1535)

#         self.conv = nn.Sequential(
#             CNNBlock(10, 32, 15, 2, 0),
#             CNNBlock(32, 64, 15, 2, 0),
#             CNNBlock(64, 128, 15, 2, 0),
#         )
#         self.pool = nn.AdaptiveMaxPool1d(1)

#         self.cls = nn.Sequential(
#             nn.Dropout(0.0),
#             nn.Linear(1 * 128, 100),
#             nn.Dropout(0.0),
#             nn.Linear(100, 2),
#         )

#     def forward(self, X):

#         return self.cls(
#             self.pool(self.conv(self.sinc(self.norm_input(X)))).view(X.shape[0], -1)
#         )
