from fastai.vision.all import (
    Transform,
    TensorCategory,
    DataBlock,
    CategoryBlock,
    TransformBlock,
    L,
    DataLoaders,
)
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Optional

from joblib.externals.loky.backend.context import get_context

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_norm_stats(LFP: torch.Tensor, train_idx: torch.Tensor):
    train_data = LFP[:, train_idx]
    return torch.mean(train_data, -1, keepdim=True), torch.std(
        train_data, -1, keepdim=True
    )


def norm_with_stats(LFP: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]):

    means, stds = stats[0], stats[1]
    return (LFP - means) / (stds + torch.finfo(LFP.dtype).tiny)


def power_transform(LFP: torch.Tensor, train_idx: torch.Tensor, method="power"):

    train_data = LFP[:, train_idx]
    if method == "power":
        trafo = PowerTransformer().fit(train_data.T)
    elif method == "quantile":
        trafo = QuantileTransformer(output_distribution="normal").fit(train_data.T)

    return torch.tensor(trafo.transform(LFP.T).T, dtype=torch.float32), trafo


class LFPNormalizer1d(Transform):
    def __init__(self, stats):
        self.means, self.stds = stats[0].to(DEVICE), stats[1].to(DEVICE)

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return (X - self.means) / (self.stds + torch.finfo(X.dtype).tiny)

    def decodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return X * self.stds + self.means


class LFPDataLoader:
    def __init__(
        self,
        LFP: np.ndarray,
        df: pd.DataFrame,
        bs=32,
        pad_to_8_chan=False,
        include_pat_and_task: bool = False,
        pat_id: Optional[int] = None,
        task_id: Optional[int] = None,
        stim_id: Optional[int] = None,
        use_power_transform: bool = False,
        train_idx: np.ndarray = None,
        device=[0],
    ):

        self.LFP, self.df = torch.tensor(LFP.copy()).type(torch.FloatTensor), df
        self.bs = bs

        if pad_to_8_chan:
            n_chan = LFP.shape[0]
            chans_to_add = 8 - n_chan

            padding = torch.zeros(chans_to_add, LFP.shape[-1])
            self.LFP = torch.vstack([self.LFP, padding])

        if include_pat_and_task:
            assert np.issubdtype(type(pat_id), int), "pat_id is None"
            assert np.issubdtype(type(task_id), int), "task_id is None"
            assert np.issubdtype(type(stim_id), int), "task_id is None"

            def get_x(row):
                return (
                    torch.tensor(pat_id),
                    torch.tensor(task_id),
                    torch.tensor(stim_id),
                    self.LFP[:, int(row["id_start"]) : int(row["id_end"])],
                )

        else:

            def get_x(row):
                return self.LFP[:, int(row["id_start"]) : int(row["id_end"])]

        def get_y(row):
            return float(row["label"])

        def splitter(df):
            train = df.index[df["is_valid"] == 0].tolist()
            valid = df.index[df["is_valid"] == 1].tolist()
            return train, valid

        if train_idx is None:
            if np.any(self.df["is_valid"] == 1):
                valid_idx = (
                    self.df[self.df["is_valid"] == 1]["id_start"].iloc[0],
                    self.df[self.df["is_valid"] == 1]["id_end"].iloc[-1],
                )
                train_idx = torch.tensor(
                    np.setdiff1d(
                        np.arange(LFP.shape[-1]),
                        np.arange(valid_idx[0], valid_idx[1] + 1),
                    ).copy()
                )
            else:
                train_idx = np.arange(LFP.shape[-1])
        else:
            train_idx = torch.tensor(train_idx)

        def LFP_block1d():
            return TransformBlock()

        if use_power_transform:
            self.LFP, self.trafo = power_transform(self.LFP, train_idx)
        else:
            # self.LFP = norm_with_stats(self.LFP, get_norm_stats(self.LFP, train_idx))
            pass

        self.dblock = DataBlock(
            blocks=(LFP_block1d, CategoryBlock),
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        torch.cuda.set_device(device[0])
        self.dls = self.dblock.dataloaders(
            self.df, bs=bs, num_workers=0, device=torch.device(f"cuda:{device[0]}")
        )
        self.dls = self.dls.to(torch.device(f"cuda:{device[0]}"))

        self.ds = self.dblock.datasets(self.df)

    @classmethod
    def from_dataset_and_windower(cls, dataset, windower, **kwargs):
        return cls(
            dataset.LFP.data, windower.df, train_idx=windower.train_idx, **kwargs
        )


class MiniDL:
    def __init__(self, dls, ds):
        self.ds, self.dls = ds, dls


def concatenate_lfp_dataloaders(dataloaders: list[LFPDataLoader], bs=32, device=DEVICE):

    train_ds, valid_ds = L(), L()
    for dl in dataloaders:
        if isinstance(dl, DataLoaders):
            train_ds.extend(dl[0].dataset)
            valid_ds.extend(dl[1].dataset)

            import pdb

            # pdb.set_trace()
            continue
        train_ds.extend(L(dl.ds.train))
        valid_ds.extend(L(dl.ds.valid))

    dataloader = MiniDL(
        DataLoaders.from_dsets(
            train_ds,
            valid_ds,
            bs=bs,
            device=device,
            multiprocessing_context=get_context("loky"),
            num_workers=0,
        ),
        [train_ds, valid_ds],
    )

    return dataloader
