import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from .base import *


class SincConv_multiChan(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    def __init__(
        self,
        kernel_size,
        sample_rate=2048.0,
        in_channels=8,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=2,
        min_band_hz=3,
        n_combs=2,
        use_gate=True,
    ):

        super().__init__()

        self.freq_scale = 1e4

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        self.use_gate = use_gate

        hz_ref = np.array([2, 7, 12, 20, 30, 45, 60, 75, 100, 150, 200]).reshape(-1, 1)
        bws_ref = np.diff(hz_ref, axis=0)

        hz = np.tile(hz_ref[:-1, :], (n_combs, self.in_channels)) - self.min_low_hz
        bws = np.tile(bws_ref, (n_combs, self.in_channels)) - self.min_band_hz

        self.n_filter = hz.shape[0]

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz) / self.freq_scale)

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(bws) / self.freq_scale)

        # Gate:
        self.gate = nn.Parameter(torch.Tensor(np.zeros((self.low_hz_.numel(), 1))))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        window_ = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size)

        self.register_buffer("window_", window_, persistent=False)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0

        self.register_buffer(
            "n_",
            2 * np.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate,
            persistent=False,
        )
        # self.n_ = (
        #     2 * np.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        # )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        low = self.min_low_hz + torch.abs(self.low_hz_ * self.freq_scale)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_ * self.freq_scale),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = high - low

        f_times_t_low = torch.matmul(low.view(-1, 1), self.n_)
        f_times_t_high = torch.matmul(high.view(-1, 1), self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band.view(-1, 1))

        if self.use_gate:
            band_pass = torch.sigmoid(self.gate) * band_pass

        self.filters = (
            (band_pass)
            .view(self.n_filter, self.in_channels, self.kernel_size)
            .to(waveforms.device)
        )

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincAugmented(nn.Module):
    def __init__(self, sinc_ks, conv_ks, n_combs, n_f, n_in=8, use_gate=True):
        super().__init__()

        self.sinc = SincConv_multiChan(
            sinc_ks, use_gate=use_gate, n_combs=n_combs, padding="same"
        )
        self.conv = ConvBlock(n_in, n_f, conv_ks, padding="same")

    def forward(self, X):

        sinced = self.sinc(X)
        conved = self.conv(X)

        return torch.cat([sinced, conved], 1)
