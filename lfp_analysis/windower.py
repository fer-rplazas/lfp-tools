from typing import Optional
from textwrap import dedent

import numpy as np
import pandas as pd
from math import ceil
from scipy.stats import mode

from numba import njit


@njit
def frame(array, win_len: float, hop_len: float, pad: bool = False):
    nrows = ((array.size - win_len) // hop_len) + 1
    # Cast to float for division:
    win_len = float(win_len)
    hop_len = float(hop_len)

    if pad:
        n_reps = ceil((1 - (array.size - win_len) / hop_len % 1) * win_len)
        if win_len == hop_len:
            n_reps -= 1
        array = np.concatenate((array, np.repeat(array[-1], n_reps)), 0)

    nrows = ((array.size - win_len) // hop_len) + 1
    n = array.strides[0]

    # Cast back to int:
    win_len = int(win_len)
    hop_len = int(hop_len)
    nrows = int(nrows)
    return np.lib.stride_tricks.as_strided(
        array, shape=(nrows, win_len), strides=(hop_len * n, n)
    )


@njit
def get_idx(idx_out: np.ndarray, t_vector: np.ndarray, t_points: np.ndarray):

    for jj in range(t_points.size):
        idx_out[jj] = int(np.argmin(np.abs(t_vector - t_points[jj])))

    assert np.isnan(idx_out).any() == False, "some indices weren't found"
    return idx_out


class Windower:
    def __init__(
        self,
        label: np.ndarray,
        data_len: int,
        fs_label: float = 2048.0,
        fs_data: float = 2048.0,
        win_len_sec: float = 0.750,
        perc_overlap: float = 0.0,
        n_folds: int = 5,
        fold_id: Optional[int] = 4,
    ):

        self.label, self.fs_label = np.squeeze(label), fs_label
        self.data_len, self.fs_data = data_len, fs_data

        self.win_len_sec, self.perc_overlap = (
            win_len_sec,
            perc_overlap,
        )
        self.n_folds, self.fold_id = n_folds, fold_id

        self.window()

    def __repr__(self):

        if hasattr(self, "df"):

            return dedent(
                f"""\
                Windower object
                    Overall class balance: 
                        {[f'{jj} -> {el}' for jj,el in enumerate(self.df["label"].value_counts())]}
                        {[f'{jj} -> {el:.2f}%' for jj,el in enumerate(self.df["label"].value_counts(normalize=True))]}
                    On Valid:
                        {[f'{jj} -> {el}' for jj,el in enumerate(self.df[self.df["is_valid"]==True]["label"].value_counts())]}
                        {[f'{jj} -> {el:.2f}%' for jj,el in enumerate(self.df[self.df["is_valid"]==True]["label"].value_counts(normalize=True))]}
                """
            )

        else:
            return "Windower Object --- data_df not set, call .window() "

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_dataset(cls, dataset, **kwargs):

        return cls(
            dataset.label.data,
            dataset.LFP.data.shape[-1],
            dataset.label.fs,
            dataset.LFP.fs,
            **kwargs,
        )

    def window(self):

        assert (
            len(self.label) > 60 * self.fs_label
        ), "data is less than 1m long -> check label fs!"

        # Get label for every window:
        window_labels, label_idx = self.process_label(self.label)
        # Get data indices for every window:
        t_label = np.arange(self.label.size) / self.fs_label
        t_data = np.arange(self.data_len) / self.fs_data

        # Timestamps at win_starts and ends
        t_starts, t_ends = t_label[label_idx[0]], t_label[label_idx[-1]]

        idx_data_start = np.zeros_like(t_starts) * np.nan
        idx_data_start = get_idx(idx_data_start, t_data, t_starts).astype(int)

        idx_data_end = np.zeros_like(t_ends[:10]) * np.nan
        idx_data_end = get_idx(idx_data_end, t_data, t_ends[:10]).astype(int)
        idx_data_end = (
            idx_data_start + mode(idx_data_end - idx_data_start[:10])[0].squeeze()
        )
        assert (
            window_labels.size == idx_data_start.size == idx_data_end.size
        ), "Unequal number of frames extracted from label and from signal_idx"
        self.n_windows = window_labels.size

        # Train / Valid split:
        is_valid = self.get_valid_idx()

        self.df = pd.DataFrame(
            {
                "id_start": idx_data_start,
                "id_end": idx_data_end,
                "label": window_labels,
                "is_valid": is_valid,
                "t": t_ends,
            }
        )

        if np.any(self.df["is_valid"] == 1):
            self.valid_idx = (
                self.df[self.df["is_valid"] == 1]["id_start"].iloc[0],
                self.df[self.df["is_valid"] == 1]["id_end"].iloc[-1],
            )
            self.train_idx = np.setdiff1d(
                np.arange(t_data.shape[-1]),
                np.arange(self.valid_idx[0], self.valid_idx[1] + 1),
            )
        else:
            self.train_idx = np.arange(t_data.shape[-1])
            self.valid_idx = None

        return self

    def process_label(self, label_data: np.ndarray, threshold: float = 0.2):

        frame_length = int(round(self.win_len_sec * self.fs_label))
        hop_length = int(round(frame_length * (1 - self.perc_overlap)))

        framed = frame(label_data.squeeze(), frame_length, hop_length, pad=False)
        framed_idx = frame(
            np.arange(label_data.size), frame_length, hop_length, pad=False
        )
        return (np.mean(framed, axis=-1) > threshold).astype(int), (
            framed_idx[:, 0],
            framed_idx[:, -1],
        )

    def make_fold(self, fold_id: int):
        """Update train and valid indicators for new fold_id"""
        self.fold_id = fold_id

        is_valid_new = self.get_valid_idx()

        self.df["is_valid"] = is_valid_new

        return self

    def get_valid_idx(self):

        is_valid = np.zeros(self.n_windows)
        n_per_split = self.n_windows // self.n_folds
        if self.fold_id is not None:  # Make all of the data training data
            is_valid[
                int(self.fold_id * n_per_split) : int((self.fold_id + 1) * n_per_split)
            ] = 1
        return is_valid.astype(int)

    def concatenate(self, other: "Windower"):

        if (self.fs_label != other.fs_label) or (self.fs_data != other.fs_data):
            raise Warning(
                "Some sampling freqs are different in the two 'Windowers' -- proceeding to concatenate anyways."
            )

        if not hasattr(self, "df") or not hasattr(other, "df"):
            raise Exception(
                "Some of the `Windowers` you are trying to concatenate don't have a `df`. Please call .window() first"
            )

        # Manually overwrite df:
        # This is important for the data_indices, especially at the transition between the two datasets

        other.df["id_start"] = other.df["id_start"].values + self.data_len
        other.df["id_end"] = other.df["id_end"].values + self.data_len
        other.df["t"] = other.df["t"].values + (self.label.size - 1) / self.fs_label

        new_windower = Windower(
            np.concatenate((self.label, other.label)),
            self.data_len + other.data_len,
            self.fs_label,
            self.fs_data,
            win_len_sec=self.win_len_sec,
            n_folds=self.n_folds,
            fold_id=None,
        )

        new_windower.df = pd.concat((self.df, other.df), ignore_index=True)

        return new_windower


class WindowerSeq:
    def __init__(
        self,
        label: np.ndarray,
        data_len: int,
        fs_label: float = 2048.0,
        fs_data: float = 2048.0,
        win_len_sec: float = 0.25,
        seq_len: int = 8,
        n_folds: int = 5,
        fold_id: Optional[int] = 4,
        perc_overlap=0.0,
    ):

        self.label, self.fs_label = np.squeeze(label), fs_label
        self.data_len, self.fs_data = data_len, fs_data

        self.win_len_sec, self.seq_len = win_len_sec, seq_len

        self.n_folds, self.fold_id = n_folds, fold_id

        self.windower_ = Windower(
            self.label,
            self.data_len,
            self.fs_label,
            self.fs_data,
            self.win_len_sec,
            perc_overlap,
            self.n_folds,
            self.fold_id,
        )

        self.window()

    def __repr__(self):
        return self.windower_.__repr__()

    def window(self):

        self.windower_.window()

        idx_cont, idx_wind_cont = [], []
        label_cont = []
        is_valid_cont = []
        t_cont = []

        for jj in range(0, len(self.windower_.df) - self.seq_len, 2):
            row = self.windower_.df.iloc[jj]
            # Check whether
            if row["is_valid"] != self.windower_.df.iloc[jj + self.seq_len]["is_valid"]:
                continue
            else:
                idx_int, label_int, idx_wind_int = [], [], []
                for kk in range(self.seq_len):
                    idx_int.append(
                        (
                            int(self.windower_.df.iloc[jj + kk]["id_start"]),
                            int(self.windower_.df.iloc[jj + kk]["id_end"]),
                        )
                    )
                    idx_wind_int.append(self.windower_.df.iloc[jj + kk].name)
                    label_int.append(int(self.windower_.df.iloc[jj + kk]["label"]))

                idx_cont.append(idx_int)
                idx_wind_cont.append(idx_wind_int)
                label_cont.append(label_int)
                is_valid_cont.append(
                    self.windower_.df.iloc[jj + kk]["is_valid"].astype(int)
                )
                t_cont.append(self.windower_.df.iloc[jj + kk]["t"])

        self.df = pd.DataFrame(
            {
                "idx": idx_cont,
                "idx_win": idx_wind_cont,
                "label": label_cont,
                "is_valid": is_valid_cont,
                "t": t_cont,
            }
        )
