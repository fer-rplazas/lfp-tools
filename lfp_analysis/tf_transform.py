from textwrap import dedent
from typing import Optional

import mne
import numpy as np
import pandas as pd
from librosa import power_to_db

from matplotlib.colors import Normalize
from scipy.fftpack import dct


class TfTransformer:
    """Time-frequency transformer"""

    def __init__(self, data):
        self.frequencies = np.arange(2, 101, 1)
        self.n_cycles = self.frequencies / 3
        self.LFPs = data

    @classmethod
    def from_importer(cls, signal_bundle):
        return cls(signal_bundle.data)

    def TF(self):
        self.power = mne.time_frequency.tfr_array_morlet(
            self.LFPs[np.newaxis, ...],
            n_cycles=self.n_cycles,
            freqs=self.frequencies,
            decim=1,
            output="power",
            sfreq=2048,
            n_jobs=1,
        ).squeeze()

        return self

    def normalize(self):
        mean = np.nanmean(self.power, axis=-1, keepdims=True)
        self.power = (self.power - mean) / mean

        return self

    def zscore(self):
        mean = np.nanmean(self.power, axis=-1, keepdims=True)
        std = np.nanstd(self.power, axis=-1, keepdims=True)
        self.power = (self.power - mean) / std

        return self


class AxisApplier:
    """Applies function f along first dimension.

    Intended for structures of shape (channels, frequencies, times) where f = f(frequencies,times) (appy to each channel)
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, X, **kwargs):
        for i in range(X.shape[0]):
            X[i] = self.f(X[i], **kwargs)

        return X


def zscore(X):
    return (X - np.mean(X, axis=-1, keepdims=True)) / np.std(X, axis=-1, keepdims=True)


def bl_correct(X):
    mean = np.mean(X, axis=-1, keepdims=True)
    return (X - mean) / mean


def clip(X, std_factor=7):
    return Normalize(vmin=-std_factor, vmax=std_factor, clip=True)(X)


def to_mfcc(S, axis=1):

    return dct(power_to_db(S), axis=axis, type=2)


def rwt_coefficients(f, fs=2, order=6):
    """Recursive wavelet transform filter coefficients at a given frequency.

    Following J. Ren 2010, 2011

    Args:
        f (number): Peak frequency
        fs (int, optional): Sampling frequency. Defaults to 2.
        order (int, optional): Order of the filter. Defaults to 6.

    Raises:
        ValueError: only `order=3` and `order=6` are implemented

    Returns:
        tuple: b,a (b and a lists of rwt filter coefficients)
    """

    #  normalized frequency and phasor
    w = f / fs
    sig = 2 * np.pi / np.sqrt(3)
    wo = 2 * np.pi
    alpha = np.exp(-w * (sig - 1j * wo))

    # array of coefficient
    if order == 3:
        # lambda
        L = [[-1 / 2, 1 / 2, -1 / 3], [1 / 3, 0, -4 / 3], [-1 / 2, -1 / 2, -1 / 3]]
        # beta
        beta = [-4, 6, -4, 1]
    elif order == 6:
        # lambda
        L = [
            [0, 0, 0, 1 / 3, -1 / 6, 1 / 15],
            [0, 0, 0, 2 / 3, -5 / 3, 26 / 15],
            [0, 0, 0, -6 / 3, 0, 22 / 5],
            [0, 0, 0, 2 / 3, 5 / 3, 26 / 15],
            [0, 0, 0, 1 / 3, 1 / 6, 1 / 15],
        ]
        # beta
        beta = [-6, 15, -20, 15, -6, 1]
    else:
        raise ValueError("'order' of the filter must be either 3 or 6")

    # numerator
    b = [
        np.sqrt(f)
        / fs
        * alpha ** (i + 1)
        * sum([Lj * (sig * w) ** (j + 1) for j, Lj in enumerate(Li)])
        for i, Li in enumerate(L)
    ]
    b += [0 + 0j, 0 + 0j]
    # denominator
    a = [1 + 0j] + [bi * alpha ** (i + 1) for i, bi in enumerate(beta)]
    return np.array(b), np.array(a)


def rwt_filterbank(freqs, fs=2048, order=6):

    n_freqs = len(freqs) if isinstance(freqs, list) else freqs.size

    b_s, a_s = (
        np.zeros((n_freqs, order + 1), dtype=np.complex_),
        np.zeros((n_freqs, order + 1), dtype=np.complex_),
    )

    for ii, freq in enumerate(freqs):
        (
            b_s[
                ii,
            ],
            a_s[
                ii,
            ],
        ) = rwt_coefficients(freq, fs, order)

    return b_s, a_s
