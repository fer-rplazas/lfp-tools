import numpy as np
import pandas as pd

from scipy.signal import butter, hilbert, lfilter, periodogram

from numba import njit
from functools import partial
from collections.abc import Callable
from typing import Union


class FeatureExtractor:
    freq_ranges: list[list[int]]
    feat_mat_computer: Callable[
        [np.ndarray, np.ndarray, Union[list, np.ndarray], np.ndarray, np.ndarray],
        np.ndarray,
    ]

    def __init__(self):
        pass

    def get_feats(
        self, data: np.ndarray, df: pd.DataFrame
    ) -> "tuple[pd.DataFrame, np.ndarray]":
        # Label:
        label_mat = df["label"].to_numpy()

        # Set feature names (requires knowing number of LFP channels)
        self.feat_names = []
        for ch in range(data.shape[1]):
            self.feat_names += [f"pow{ch}_{el[0]}_{el[1]}Hz" for el in self.freq_ranges]

        self.feat_names_base = self.feat_names.copy()

        # Pre-allocate feat_mat
        feat_mat = (
            np.zeros((data.shape[0], data.shape[1] * len(self.freq_ranges))) * np.nan
        )

        # compute feat_mat:
        feat_mat = self.feat_mat_computer(feat_mat, data, self.freq_ranges)

        assert (
            np.isnan(feat_mat).any() == False
        ), "feature extraction yields NaN values (check whether filter is unstable)"

        return pd.DataFrame(feat_mat, columns=self.feat_names), label_mat


def tf_feature_extract(feat_mat, data, freq_ranges, idx_start, idx_end, freq_idx=None):

    for ii, (id_start, id_end) in enumerate(zip(idx_start, idx_end)):
        this_data = data[:, :, id_start:id_end]

        for ch in range(data.shape[0]):

            for kk, freq_lims in enumerate(freq_idx):
                feat_mat[ii, (ch * len(freq_ranges)) + kk] = np.mean(
                    this_data[ch, freq_lims[0] : freq_lims[1]]
                )
    return feat_mat


class TfFeatureExtractor(FeatureExtractor):
    def __init__(self, avoid_line_noise=True):
        self.avoid_line_noise = avoid_line_noise

        self.frequencies = np.arange(2, 101, 1)
        if not avoid_line_noise:
            self.freq_ranges = [
                [2, 7],
                [8, 12],
                [13, 20],
                [21, 30],
                [31, 45],
                [46, 55],  # Line noise
                [56, 75],
                [76, 95],
            ]
        else:
            self.freq_ranges = [
                [2, 7],
                [8, 12],
                [13, 20],
                [21, 30],
                [31, 45],
                [56, 75],
                [76, 95],
            ]

        self.freq_idx = np.array(
            [
                [
                    np.where(self.frequencies == el[0]),
                    np.where(self.frequencies == el[1]),
                ]
                for el in self.freq_ranges
            ]
        ).squeeze()
        self.feat_mat_computer = partial(tf_feature_extract, freq_idx=self.freq_idx)


@njit
def fill_feat_mat(feat_mat, win_id, Pxx, n_chan, freq_idx):

    for ch in range(n_chan):
        for kk, freq_lims in enumerate(freq_idx):
            feat_mat[win_id, (ch * freq_idx.shape[0]) + kk] = np.mean(
                Pxx[ch, freq_lims[0] : freq_lims[1]]
            )
    return feat_mat


def periodogram_extractor(feat_mat, data, freq_ranges, fs=2048.0):

    f, _ = periodogram(np.random.rand((data.shape[2])), fs=fs)

    freq_idx = np.zeros_like(np.array(freq_ranges))
    for jj, freq_range in enumerate(freq_ranges):
        for kk in range(2):
            freq_idx[jj, kk] = np.argmin(np.abs(freq_range[kk] - f))

    # Compute periodogram:
    for ii, this_data in enumerate(data):
        f, Pxx = periodogram(
            this_data,
            fs=fs,
        )
        feat_mat = fill_feat_mat(feat_mat, ii, Pxx, data.shape[1], freq_idx)

    return feat_mat


def hilbert_extractor(feat_mat, data, freq_ranges, idx_start, idx_end, fs=2048.0):
    # Allocate:
    filtered_data = np.zeros((data.shape[0], len(freq_ranges), data.shape[-1])) * np.nan

    # Isolate frequency ranges via butterworth filtering:
    for kk in range(data.shape[0]):  # LFP channels
        for jj, freq_range in enumerate(freq_ranges):  # Frequency ranges
            b, a = butter(3, freq_range, btype="bandpass", fs=fs)
            filtered_data[kk, jj, :] = lfilter(b, a, data[kk])

    # Extract spectral power in each window through hilbert and save as feature:
    for ii, (id_start, id_end) in enumerate(zip(idx_start, idx_end)):
        this_data = np.square(np.abs(hilbert(filtered_data[:, :, id_start:id_end])))

        for ch in range(data.shape[0]):

            for kk, _ in enumerate(freq_ranges):
                feat_mat[ii, (ch * len(freq_ranges)) + kk] = np.mean(this_data[ch, kk])

    return feat_mat


class SignalFeatureExtractor(FeatureExtractor):
    def __init__(self, method="periodogram", avoid_line_noise=True, fs=2048.0):

        self.method, self.avoid_line_noise = method, avoid_line_noise

        # Frequency reanges:
        if not avoid_line_noise:
            self.freq_ranges = [
                [2, 7],
                [8, 12],
                [13, 20],
                [21, 30],
                [31, 45],
                [46, 55],  # Line noise
                [56, 75],
                [76, 95],
                [95, 105],  # Line noise
                [106, 145],
                [146, 155],  # Line noise
                [156, 195],
            ]
        else:
            self.freq_ranges = [
                [2, 7],
                [8, 12],
                [13, 20],
                [21, 30],
                [31, 45],
                [56, 75],
                [76, 95],
                [106, 145],
                [156, 195],
            ]

        # Set feat_mat computer:
        if self.method == "periodogram":
            self.feat_mat_computer = periodogram_extractor
        elif self.method == "hilbert":
            self.feat_mat_computer = hilbert_extractor

        self.feat_mat_computer = partial(self.feat_mat_computer, fs=fs)
