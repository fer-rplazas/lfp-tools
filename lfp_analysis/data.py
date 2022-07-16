# __all__ = ['SmrImporter', 'SignalBundle', 'AccData',
#            'LFPData', 'TfTransformer', 'AxisApplier',
#            'zscore', 'bl_correct', 'clip']

from pathlib import Path
from textwrap import dedent
from collections import OrderedDict

from enum import Enum
from typing import Union, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from neo.io import Spike2IO
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import butter, decimate, filtfilt, hilbert, iirnotch


###############################################################################
# Utilities:
###############################################################################
import os

if os.path.split(os.path.split(os.getcwd())[0])[1] == "lfp_analysis":
    # Use absolute path for hydra
    DATA_PATH = Path("/home/fernando/OneDrive/DPhil/DL Decoding/lfp_analysis/data")
else:
    DATA_PATH = Path("./../../data")


class PatID(Enum):
    ET1, ET2, ET3, ET4, ET5, ET6, ET7, ET8 = tuple(range(1, 9))


class Task(Enum):
    Pegboard = 1
    Pouring = 2
    Posture = 3


class Stim(Enum):
    OFF = 1
    ON = 2


def create_pat_task_stim_dict() -> dict[PatID, dict[Task, dict[Stim, None]]]:

    consolidate_channel_dict = OrderedDict()

    for pat_num in PatID:
        consolidate_channel_dict[pat_num.name] = {}

    for pat_num in PatID:
        for task in Task:
            consolidate_channel_dict[pat_num.name][task.name] = {}

    for pat_num in PatID:
        for task in Task:
            for stim in Stim:
                consolidate_channel_dict[pat_num.name][task.name][stim.name] = None

    return consolidate_channel_dict


# Import dataset-specific settings:
from ._manual_data import *

###############################################################################
# Main classes:
###############################################################################


class Patient:
    """Class representing a patient -> holds datasets"""

    pat_id: PatID
    missing_datasets: list[Optional[Task]]
    datasets: dict[PatID, dict[Task, dict[Stim, Optional["Dataset"]]]]

    def __init__(self, pat_id: PatID):
        self.pat_id = pat_id
        self.missing_datasets = (
            []
            if missing_datasets.get(self.pat_id) is None
            else missing_datasets.get(self.pat_id)
        )

        self.datasets = {}
        for task in Task:

            self.datasets[task] = {}

            for stim in Stim:
                if task in self.missing_datasets:
                    self.datasets[task][stim] = None
                else:
                    self.datasets[task][stim] = Dataset(self.pat_id, task, stim)


class Dataset:
    pat_id: PatID
    task: Task
    stim: Stim

    def __init__(self, pat_id, task, stim=Stim.OFF, data_path=DATA_PATH):
        self.pat_id, self.task, self.stim = pat_id, task, stim

        self.LFP_PATH = (
            data_path
            / "preprocessed"
            / f"{self.pat_id.name}"
            / Path(self.task.name + f"_{stim.name.lower()}").with_suffix(".h5")
        )
        self.TF_PATH = (
            DATA_PATH
            / "TF_and_df"
            / f"{self.pat_id.name}"
            / Path(self.task.name + f"_{stim.name.lower()}").with_suffix(".h5")
        )

    def __repr__(self):

        body = "-- data not loaded"

        if hasattr(self, "LFP"):
            body = dedent(
                f"""\

                LFPs:
                {str(self.LFP.__repr__())}
                
                label: 
                {str(self.label.__repr__())}
                """
            )

        elif hasattr(self, "TF"):
            body = f""" TF Data (repr not yet implemented)"""

        return dedent(
            f"""\
            Dataset: {self.pat_id.name} - {self.task.name + '_' + self.stim.name}
            {body}
            
            """
        ).strip()

    def load_1d(
        self,
        drop_chans: bool = True,
        decimate_on_import: Optional[int] = None,
        notch_filter: bool = False,
    ):

        to_drop = None
        if (
            drop_chans
        ):  # Drop channels to consolidate channel config across stim ON and OFF.
            to_drop = consolidate_chan_dict[self.pat_id][self.task][self.stim]
            if to_drop is not None:
                to_drop = list(to_drop) if not isinstance(to_drop, list) else to_drop

        self.LFP = LFPData.from_h5(self.LFP_PATH, key="LFP", drop_channels=to_drop)

        if notch_filter:
            self.LFP.data = notch_filter_data(self.LFP.data, freq=50)
            self.LFP.data = notch_filter_data(self.LFP.data, freq=100)
            self.LFP.data = notch_filter_data(self.LFP.data, freq=200)

        if decimate_on_import is not None:
            self.LFP.decimate(decimate_on_import)

        self.load_label()

        return self

    def load_2d(self):
        self.LFP = TFData.from_h5(self.TF_PATH, "TF")

        self.load_label()

        return self

    def load_label(self):
        self.label = Label.from_h5(self.LFP_PATH, key="label")

        return self

    def concatenate(self, other: "Dataset") -> "Dataset":

        new_lfps = self.LFP.concatenate(other.LFP)
        new_label = self.label.concatenate(other.label)

        new_dataset = Dataset(self.pat_id, self.task, self.stim)
        new_dataset.LFP = new_lfps
        new_dataset.label = new_label

        return new_dataset

    def plot(self):

        ax = self.LFP.plot()

        for jj in range(self.LFP.n_chan):

            self.label.plot(ax=ax, jj=jj, multiplier=3 * self.LFP.data[jj].std())
        return ax


class SmrImporter:
    """Reads signals and metadata (sampling frequency ``fs``, channel names ``ch_names``) from SMR file"""

    def __init__(self, fname: Union[Path, str]):
        analog_signal = (
            Spike2IO(filename=str(fname)).read()[0].segments[0].analogsignals[0]
        )

        self.ch_names = analog_signal.array_annotations["channel_names"].tolist()

        self.ch_dict = {name: i for i, name in enumerate(self.ch_names)}

        self.fs = analog_signal.sampling_rate.magnitude

        self.data = analog_signal.magnitude.T

    def t(self):
        return np.linspace(
            0, (self.data.shape[-1] - 1.0) / self.fs, self.data.shape[-1]
        )

    def free(self):
        if hasattr(self, "data"):
            del self.data

    def __getitem__(self, key):
        return self.data[self.ch_dict[key]]


class SignalBundle:
    """Superclass to represent bundles of signals that are often grouped together (e.g. bipolar LFPs, etc.)"""

    def __init__(
        self,
        data: np.ndarray,
        names: Optional[list[str]] = [],
        fs: float = 2048.0,
        highpassed: bool = False,
        decimated: bool = False,
    ) -> None:
        self.names = names

        if data.size == 0:
            self.data = data
        else:
            if len(data.shape) != 2:  # Check whether data has appropriate dimensions:
                raise ValueError(
                    f"data must be a 2-dimensional array (n_channels x n_samples), instead {len(data.shape)}-dim data was provided"
                )
            self.data = data if data.shape[1] > data.shape[0] else data.T

        self.n_chan = data.shape[0]
        self.fs = float(fs)

        self.highpassed, self.decimated = highpassed, decimated

    def __repr__(self):

        reprStr = f"""
        - Channel Names: {self.names}
        - Shape: {self.data.shape}
        - End Time: {(self.data.shape[-1]-1)/self.fs:.3f} s
        """
        return dedent(reprStr)

    def highpass(self, cutoff_freq: float = 0.5):

        if not self.data.size > 0:
            return self
        if self.highpassed:
            raise Warning(
                "Flag indicates that data has already been highpassed -- proceeding to highpass again."
            )
        b, a = butter(4, cutoff_freq, "hp", fs=self.fs)
        self.data = filtfilt(b, a, self.data, axis=-1)

        self.highpassed = True
        return self

    def channel_apply(self, f, **kwargs):
        if len(self.data.shape) == 1:
            self.data = f(self.data, **kwargs)
            return self

        self.data = np.apply_along_axis(f, -1, self.data, **kwargs)

        return self

    def concatenate(self, other: "SignalBundle"):

        if self.names != other.names:
            raise Warning(
                "You are attempting to concatenate two dataset with different channel names -- proceeding anyways."
            )

        new_data = np.concatenate((self.data, other.data), axis=-1)

        return SignalBundle(
            new_data,
            self.names,
            fs=self.fs,
            highpassed=self.highpassed,
            decimated=self.decimated,
        )

    def limit(
        self, plot: bool = False, figsize: tuple[float, float] = (13, 6), **kwargs
    ):

        if plot:  # Plot before limiting
            time = self.t()

            _, ax = plt.subplots(self.n_chan, 1, sharex=True, figsize=figsize)
            ax = ax.flatten() if self.n_chan > 1 else [ax]

            for i in range(self.n_chan):
                if self.n_chan > 1:
                    ax[i].plot(time, self.data[i], alpha=0.5, label="Before")
                else:
                    ax[i].plot(time, self.data, alpha=0.5, label="Before")

        self.channel_apply(limiter, **kwargs)  # Limit signals

        if plot:  # Plot after limiting
            for i in range(self.n_chan):
                if self.n_chan > 1:
                    ax[i].plot(time, self.data[i], label="After")
                else:
                    ax[i].plot(time, self.data, label="After")

                if self.names is not None:
                    ax[i].set_title(self.names[i])
                ax[i].legend()
            plt.show()

        return self

    def zscore(self):

        self.data = (self.data - np.mean(self.data, axis=-1, keepdims=True)) / np.std(
            self.data, axis=-1, keepdims=True
        )

    def decimate(self, q_factor: int = 10):
        self.data = decimate(self.data, q_factor)
        self.fs = self.fs / q_factor
        self.decimated, self.q_decim = True, q_factor

        return self

    def t(self):
        return np.linspace(0, self.data.shape[-1] / self.fs, self.data.shape[-1])

    def notch(self, **kwargs):

        if not self.data.size > 0:
            return self

        self.data = notch_filter_data(self.data, **kwargs)

        return self

    def plot(
        self,
        figsize: tuple[float, float] = (13, 6),
        ax=None,
        **kwargs,
    ):
        """Plot each signal on a sublot.

        Args:
            figsize (tuple[float], optional): Defaults to (13, 6).
            ax ([type], optional): Axes handle(s) to plot to; if none new axes will be created. Defaults to None.

        Returns:
            [type]: Axes handles
        """
        if ax is None:
            fig, ax = plt.subplots(self.data.shape[0], 1, figsize=figsize, sharex=True)

        for i in range(self.data.shape[0]):
            ax[i].plot(self.t(), self.data[i], **kwargs)

            if self.names is not None:
                ax[i].set_title(self.names[i])

        return ax


class AccData(SignalBundle):
    """Accelerometer SignalBundle"""

    @classmethod
    def from_importer(
        cls, importer, acc_names=["Aclx", "Acly", "Aclz", "Acrx", "Acry", "Acrz"]
    ):
        data = importer.data[[importer.ch_dict[name] for name in acc_names], :]
        return cls(data, acc_names, importer.fs)

    def to_magnitude(self):
        if self.data.shape[1] == 6:
            self.data = np.linalg.norm(self.data[:3,], axis=0) + np.linalg.norm(
                self.data[
                    3:,
                ],
                axis=0,
            )
        else:
            self.data = np.linalg.norm(self.data, axis=0)

        self.names = ["Acc_merged"]

        return self

    def to_label(
        self,
        smoothing=0.250,
        percentile=75,
        use_hilbert=False,
        plot=True,
        figsize=(13, 5),
    ):
        """Generates binary label (movement vs. rest) from accelererometer data by
            1. Uniformally smoothing acc data with `smoothing` seconds
            2. Thresholding according to `percentile`

        Args:
            smoothing (float, optional): Width of smoothing window in seconds. Defaults to 0.250.
            percentile (int, optional): Percentile of (smoothed signal) used to compute threshold. Defaults to 75.
            plot (bool, optional): Flag indicating whether to generate plot. Defaults to True.
            figsize (tuple, optional): figsize, only used when `plot = True`. Defaults to (13, 5).

        Raises:
           RuntimeError: Accelerometer labels have to be computed from single-channel signal. Please call .to_magnitude() method to get single accelerometer channel.

        Returns:
            AccData: Object of class AccData.
        """

        if not self.highpassed:
            _ = self.highpass()

        if len(self.data.shape) > 1:
            raise RuntimeError(
                "Accelerometer labels have to be computed from single-channel signal. Please call .to_magnitude() method to get single accelerometer channel."
            )

        if use_hilbert:
            self.data = np.abs(hilbert(self.data))

        smoothed = uniform_filter1d(self.data, size=int(smoothing * self.fs))

        thresh = np.percentile(smoothed, percentile)
        self.label = (np.abs(smoothed) > thresh).astype(float)

        label_smoothed = uniform_filter1d(self.label, size=int(self.fs))
        # assert np.allclose(self.label, label_smoothed)

        self.label = (label_smoothed > 0.5).astype(float)

        if plot:
            _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

            ax1.plot(self.t(), smoothed)
            ax1.plot(self.t(), self.label * np.max(smoothed) * 0.5)
            ax1.set_title("Smoothed + label")

            ax2.plot(self.t(), self.data)
            ax2.set_title("Raw data")
        return self


class EMGData(SignalBundle):
    @classmethod
    def from_importer(cls, importer, emg_names=["FleF"]):
        data = importer.data[[importer.ch_dict[name] for name in emg_names], :]

        return cls(data, emg_names, importer.fs)

    def to_label(
        self,
        smoothing=0.250,
        percentile=75,
        use_hilbert=True,
        plot=True,
        figsize=(13, 5),
    ):
        """Generates binary label (movement vs. rest) from accelererometer data by
            1. Uniformally smoothing acc data with `smoothing` seconds
            2. Thresholding according to `percentile`

        Args:
            smoothing (float, optional): Width of smoothing window in seconds. Defaults to 0.250.
            percentile (int, optional): Percentile of (smoothed signal) used to compute threshold. Defaults to 75.
            plot (bool, optional): Flag indicating whether to generate plot. Defaults to True.
            figsize (tuple, optional): figsize, only used when `plot = True`. Defaults to (13, 5).

        Raises:
           RuntimeError: Accelerometer labels have to be computed from single-channel signal. Please call .to_magnitude() method to get single accelerometer channel.

        Returns:
            AccData: Object of class AccData.
        """

        if not self.highpassed:
            _ = self.highpass()

        if len(self.data.shape) > 1:
            raise RuntimeError(
                "EMG labels have to be computed from single-channel signal."
            )

        if use_hilbert:
            self.data = np.abs(hilbert(self.data))

        smoothed = uniform_filter1d(self.data, size=int(smoothing * self.fs))

        thresh = np.percentile(smoothed, percentile)
        self.label = (np.abs(smoothed) > thresh).astype(float)

        label_smoothed = uniform_filter1d(self.label, size=int(self.fs))
        # assert np.allclose(self.label, label_smoothed)

        self.label = (label_smoothed > 0.5).astype(float)

        if plot:
            _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

            ax1.plot(self.t(), smoothed)
            ax1.plot(self.t(), self.label * np.max(smoothed) * 0.5)
            ax1.set_title("Smoothed + label")

            ax2.plot(self.t(), self.data)
            ax2.set_title("Raw data")
        return self


class LFPData(SignalBundle):
    """LFP data"""

    @classmethod
    def from_importer(cls, importer, names):
        data = importer.data[[importer.ch_dict[name] for name in names], :]
        return cls(data, names, importer.fs)

    @classmethod
    def from_h5(
        cls, path: Path, key: str = "LFP", drop_channels: Optional[list[int]] = None
    ):

        lfp_data = h5py.File(path, "r")[key][:]
        if drop_channels is not None:
            lfp_data = np.delete(lfp_data, drop_channels, 0)

        return cls(lfp_data, highpassed=True)


class Label(SignalBundle):
    @classmethod
    def from_h5(cls, path: Path, key: str = "label"):

        return cls(np.expand_dims(h5py.File(path, "r")[key][:], 0), names=["label"])

    @classmethod
    def from_array(
        cls, data: np.ndarray, q_decim: Optional[int] = None, fs: float = 2048.0
    ):

        if q_decim is not None:
            data = decimate(data, q_decim)
            fs = fs / q_decim

        return cls(data, fs=fs, decimated=True, names=["label"])

    def plot(
        self,
        ax=None,
        jj: Optional[int] = None,
        multiplier: Optional[float] = None,
        **kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()

        myax = ax[jj] if jj is not None else ax
        mymult = multiplier if multiplier is not None else 1

        myax.plot(self.t(), mymult * np.squeeze(self.data), **kwargs)


class TFData(SignalBundle):
    @classmethod
    def from_h5(cls, path: Path, key: str = "TF"):
        # TODO: Will lead to error, since TF-data has dims (n_channels x n_freqs x n_samples)

        tf_data = h5py.File(path, "r")[key][:]
        return cls(tf_data, fs=2048)


###############################################################################
# Utilities and other functions:
###############################################################################
def notch_filter_data(X, freq=50):

    fs = 2048.0  # Sample frequency (Hz)
    f0 = freq  # Frequency to be removed from signal (Hz)
    Q = 20.0  # Quality factor
    # Design notch filter
    b, a = iirnotch(f0, Q, fs)

    return filtfilt(b, a, X)


def limiter(x, method="std", percentiles=(0.1, 99.99), std_factor=5, values=None):
    """Limits 1-d signal x using one of the following methods:

    - "std" limits signal with values: (-std_factor * x.std(), std_factor * x.std())
    - "values" limits the signal using an upper and lower value specified by the user
    - "percentile" limits signal according to percentiles
    """

    if method == "values":
        if values is None:
            raise ValueError(
                "If you choose 'values' as method, you have to specify the 'values' parameter as a tuple: (lower_bound, higher_bound)"
            )
        else:
            return np.clip(x, values[0], values[1])

    if method == "std":
        std = std_factor * x.std()
        return np.clip(x, -std, std)

    elif method == "percentile":
        ranges = np.percentile(x, percentiles)
        return np.clip(x, -ranges[0], ranges[1])

    else:
        raise ValueError(
            f"`method` has to be one of: ('std', 'percentile') –> you passed {method}"
        )


def limiter2d(x):
    return np.apply_along_axis(limiter, -1, x)


def save_to_h5(path, lfp, label):
    f = h5py.File(path, "w")
    f.create_dataset("LFP", data=lfp)
    f.create_dataset("label", data=label)
    f.close()


def make_folder_structure(DATA_PATH=Path("./../data/")):
    import os

    processing_stages = ["preprocessed", "TF_and_df"]

    for stage in processing_stages:
        for pat in PatID:
            this_path = Path(DATA_PATH / stage / pat)
            if not os.path.exists(this_path):
                os.makedirs(this_path)

    print(os.getcwd())
    return 0
