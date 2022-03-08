import numpy as np
import pandas as pd
import yaml
from itertools import chain
from copy import deepcopy
import fcntl

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns
from pathlib import Path
import pickle

from .data import create_pat_task_stim_dict, PatID, Task, Stim

DATA_PATH = Path("./../data")


class DatasetScores:
    def __init__(self, scores=None):
        if scores is None:
            self.scores = create_pat_task_stim_dict()
        else:
            self.scores = scores

    @classmethod
    def from_file(cls, path):

        with open(path, "r") as fp:
            scores = yaml.load(fp, Loader=yaml.UnsafeLoader)

        return cls(scores)

    def dump(self, path):
        with open(path, "w+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            yaml.dump(dict(self.scores), fp)
            fcntl.flock(fp, fcntl.LOCK_UN)

    def __getitem__(self, *keys):

        x = self.scores
        for key in keys:
            x = x[key]
        return x

    def __setitem__(self, *keys, data):

        x = self.scores
        for key in keys[:-1]:
            x = x[key]

        x[keys[-1]] = data

    def __repr__(self):
        return self.scores.__repr__()

    def get(self, metric, reduce="mean"):

        cont = []
        for pat in PatID:
            for task in Task:
                for stim in Stim:
                    temp_data = self.scores[pat.name][task.name][stim.name]
                    if temp_data is not None:
                        cont.append(
                            self.scores[pat.name][task.name][stim.name][metric][reduce]
                        )

        if reduce == "folds":
            cont = list(chain(*cont))

        return np.array(cont)

    def eliminate(self, pat_id, task, stim):
        self.scores[pat_id][task][stim] = None

    def global_performance(self, metric, per_fold=False):

        return np.mean(self.get(metric, reduce="folds" if per_fold else "mean"))


class ClsCompare:

    scores: OrderedDict["str", DatasetScores]

    def __init__(self, scores, key):
        self.scores = OrderedDict([(key, scores)])

        self.update_df()

    def update_df(self):
        self.df = pd.DataFrame()
        for key, value in self.scores.items():
            self.df[key] = pd.json_normalize(value.scores, sep="_", max_level=2).T
        return self

    def __getitem__(self, metric):

        if not isinstance(metric, tuple):
            val = deepcopy(
                self.df.applymap(lambda x: x[metric] if x is not None else None)
            )
        else:
            val = deepcopy(
                self.df.applymap(
                    lambda x: x[metric[0]][metric[1]] if x is not None else None
                )
            )

        self.update_df()
        return val

    def eliminate(self, pat_id, task, stim):

        for k, v in self.scores.items():
            v.eliminate(pat_id, task, stim)

        self.update_df()

    @classmethod
    def from_file(cls, path, key):
        return cls(DatasetScores.from_file(path), key)

    def append(self, scores, key):

        self.scores[key] = scores
        self.update_df()
        return self

    def append_from_file(self, path, key):

        self.scores[key] = DatasetScores.from_file(path)

        self.update_df()
        return self

    def _repr_html_(self):
        return self.df.head().to_html()

    def __repr__(self):
        return repr(self.df)


class Reporter:
    def __init__(
        self,
        svm_scores=None,
        cnn_1d_scores=None,
        cnn_2d_scores=None,
        beta_scores=None,
        theta_scores=None,
        dataset=None,
        windower=None,
        experiment=None,
    ):
        self.svm, self.cnn_1d, self.cnn_2d, self.beta, self.theta = (
            svm_scores,
            cnn_1d_scores,
            cnn_2d_scores,
            beta_scores,
            theta_scores,
        )
        self.names_svm = ["SVM", "b-SVM", "th-SVM"]
        self.names_cnn = ["1D-CNN", "2D-CNN"]

        self.dataset, self.windower = dataset, windower

        self.experiment = experiment

        if self.experiment is not None:
            self.save_path = (
                DATA_PATH
                / "results"
                / f"ET{self.dataset.pat_id}"
                / self.experiment
                / f"{self.dataset.task}"
            )
            self.save_path.mkdir(parents=True, exist_ok=True)

        else:
            self.save_path = (
                DATA_PATH
                / "results"
                / f"ET{self.dataset.pat_id}"
                / f"{self.dataset.task}"
            )
            self.save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_pickle(cls, pat_id, task, experiment):

        rec = pickle.load(
            open(
                DATA_PATH
                / "results"
                / f"ET{pat_id}"
                / experiment
                / f"{task}"
                / "results.p",
                "rb",
            )
        )

        return cls(
            rec["svm"],
            rec["cnn_1d"],
            rec["cnn_2d"],
            rec["beta"],
            rec["theta"],
            rec["dataset"],
            rec["windower"],
            rec["experiment"],
        )

    def save_plots(
        self,
        select=["losses", "conf_mats", "roc_curves", "svm_coef", "accs"],
    ):
        if "losses" in select:
            fig, ax = self.plot_losses()
            fig.savefig(self.save_path / "losses.png")

        if "conf_mats" in select:
            fig, ax = self.plot_conf_mats()
            fig.savefig(self.save_path / "conf_mats.png")

        if "roc_curves" in select:
            fig, ax = self.plot_roc_curves()
            fig.savefig(self.save_path / "roc_curves.png")

        if "svm_coef" in select:
            fig, ax = self.plot_svm_coefs()
            fig.savefig(self.save_path / "svm_coefs.png")

        if "accs" in select:
            fig, ax = self.plot_accs()
            fig.savefig(self.save_path / "accs.png")

    def save(self):
        pickle.dump(self.__dict__, open(self.save_path / "results.p", "wb"))

    def plot_roc_curves(self):

        fig, ax = plt.subplots()

        mean_fpr = np.linspace(0, 1, 100)

        ax.plot([0, 1], [0, 1], color="k", alpha=0.5, linestyle="--", linewidth=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positve Rate")

        for cls, jj, cls_name in zip(
            (self.svm, self.beta, self.theta), (0, 3, 4), self.names_svm
        ):
            ax.plot(
                mean_fpr,
                cls.valid.roc_curve["mean"],
                color=f"C{jj}",
                label=f"{cls_name} (AUC: {cls.valid.AUC['mean']:.2f} +/- {cls.valid.AUC['std']:.2f})",
            )
            ax.fill_between(
                mean_fpr,
                cls.valid.roc_curve["mean"] - cls.valid.roc_curve["std"],
                cls.valid.roc_curve["mean"] + cls.valid.roc_curve["std"],
                alpha=0.2,
                color=f"C{jj}",
            )

        for cls, jj, cls_name in zip(
            (self.cnn_1d, self.cnn_2d), (1, 2), self.names_cnn
        ):
            ax.plot(
                mean_fpr,
                cls.valid.roc_curve,
                color=f"C{jj}",
                label=f"{cls_name} (AUC: {cls.valid.AUC:.2f})",
            )

        ax.legend(loc="lower right")

        return fig, ax

    def plot_conf_mats(self):

        fig, ax = plt.subplots(5, 1, figsize=(4, 18))

        max_val = max(
            [
                np.max(cls.valid.conf_mat_norm["mean"])
                for cls in [self.svm, self.beta, self.theta]
            ]
            + [np.max(cls.valid.conf_mat_norm) for cls in [self.cnn_1d, self.cnn_2d]]
        )
        min_val = min(
            [
                np.min(cls.valid.conf_mat_norm["mean"])
                for cls in [self.svm, self.beta, self.theta]
            ]
            + [np.min(cls.valid.conf_mat_norm) for cls in [self.cnn_1d, self.cnn_2d]]
        )

        for cls, jj, cls_name in zip(
            (self.svm, self.beta, self.theta), (0, 3, 4), self.names_svm
        ):
            means, stds = (
                cls.valid.conf_mat_norm["mean"],
                cls.valid.conf_mat_norm["std"],
            )
            annot = pd.DataFrame(
                [
                    [
                        f"{mean:.1f} +/- {std:.2f}"
                        for mean, std in zip(mean_outer, std_outer)
                    ]
                    for mean_outer, std_outer in zip(means, stds)
                ]
            )

            df = pd.DataFrame(
                cls.valid.conf_mat_norm["mean"], index=[0, 1], columns=[0, 1]
            )
            sns.heatmap(
                df,
                annot=annot,
                vmin=min_val,
                vmax=max_val,
                ax=ax[jj],
                fmt="s",
                cmap="rocket",
            )
            ax[jj].set_title(cls_name)
            ax[jj].set_ylabel("True Label")

        for cls, jj, cls_name in zip(
            (self.cnn_1d, self.cnn_2d), (1, 2), self.names_cnn
        ):
            df = pd.DataFrame(cls.valid.conf_mat_norm, index=[0, 1], columns=[0, 1])
            sns.heatmap(
                df,
                annot=True,
                vmin=min_val,
                vmax=max_val,
                ax=ax[jj],
                fmt=".2f",
                cmap="rocket",
            )
            ax[jj].set_title(cls_name)
            ax[jj].set_ylabel("True Label")

        ax[-1].set_xlabel("Predicted Label")

        return fig, ax

    def plot_svm_coefs(self):

        n_chan = self.dataset.n_chan
        fig, ax = plt.subplots(self.dataset.n_chan, 1)
        comp = 1
        accs = []

        for jj, (cls, cls_id) in enumerate(
            zip([self.svm, self.beta, self.theta], (0, 3, 4))
        ):
            for kk in range(self.dataset.n_chan):
                idx = list(
                    range(
                        kk * (len(cls.svm_coef) // n_chan),
                        ((kk + 1) * (len(cls.svm_coef) // n_chan)),
                    )
                )

                xcoord = np.array(range(len(idx))) / comp
                if jj == 1:
                    xcoord = (np.array(range(len(idx))) + 2) / comp
                ax[kk].errorbar(
                    xcoord,
                    cls.svm_coef["mean"][idx],
                    yerr=cls.svm_coef["std"][idx],
                    linestyle="none",
                    marker="d",
                    c=f"C{cls_id}",
                    alpha=0.5,
                )
                if cls_id == 0:
                    ax[kk].axhline(
                        xmax=np.array(range(len(idx)))[-1] / comp,
                        c="k",
                        linewidth=0.3,
                        linestyle="dashed",
                    )
                ax[kk].set_xticks([])
                ax[kk].set_ylim((-2.5, 2.5))

            accs.append(cls.lin_svm_acc[0])

        idx_svm = list(
            range(
                kk * (len(self.svm.svm_coef) // n_chan),
                ((kk + 1) * (len(self.svm.svm_coef) // n_chan)),
            )
        )
        ax[-1].set_xticks(np.array(range(len(idx_svm))) / comp)
        labels = [
            el[5:-2]
            for el in list(np.array(self.svm.svm_coef.index.values).squeeze()[idx_svm])
        ]
        ax[-1].set_xticklabels(labels, rotation=35)
        ax[0].set_title(
            f"LinearSVM Coefs (Accs All: {accs[0]:.2f}, Beta: {accs[1]:.2f}, Theta: {accs[2]:.2f})"
        )

        return fig, ax

    def plot_accs(self):

        fig, ax = plt.subplots()

        width = 0.35

        accs = [
            cls.valid.accuracy
            for cls in [self.svm, self.cnn_1d, self.cnn_2d, self.beta, self.theta]
        ]
        accs = [el["mean"] if isinstance(el, dict) else el for el in accs]

        aucs = [
            cls.valid.AUC
            for cls in [self.svm, self.cnn_1d, self.cnn_2d, self.beta, self.theta]
        ]
        aucs = [el["mean"] if isinstance(el, dict) else el for el in aucs]

        rects1 = ax.bar(np.array(range(5)) - width / 2, aucs, width, label="AUC")
        rects2 = ax.bar(np.array(range(5)) + width / 2, accs, width, label="Accuracy")

        ax.set_xticks(np.array(range(5)))
        ax.set_xticklabels(
            [el for el in ["SVM", "1D-CNN", "2D-CNN", "b-SVM", "th-SVM"]]
        )
        ax.legend()
        ax.set_ylim((0.5, 1))

        return fig, ax

    def plot_losses(self):

        fig, ax = plt.subplots(6, 1, figsize=(12, 8))

        ax[-1].set_title("LFPs and Label")
        ax[-1].plot(self.dataset.t_decim(), self.dataset.LFP_decim.T, alpha=0.2)
        ax[-1].plot(
            self.dataset.t(),
            self.dataset.label * 0.9 * np.max(self.dataset.LFP_decim),
            c="r",
        )

        for cls, jj, cls_name in zip(
            (self.svm, self.beta, self.theta), (0, 3, 4), self.names_svm
        ):

            ax[jj].set_title(cls_name)
            ax[jj].set_xticks([])
            ax[jj].plot(cls.train.times, cls.train.losses["mean"], color=f"C{jj}")
            ax[jj].plot(cls.valid.times, cls.valid.losses["mean"], color=f"C{jj}")

            ax[jj].plot(
                self.dataset.t(),
                0.6931 * np.ones_like(self.dataset.t()),
                linewidth=0.3,
                linestyle="--",
                c="k",
            )

            ax[jj].add_patch(
                patches.Rectangle(
                    (cls.valid.times[0], 0),
                    width=(cls.valid.times[-1] - cls.valid.times[0]),
                    height=(2),
                    alpha=0.3,
                    color="C6",
                )
            )

            idx = np.where(cls.train.y != (cls.train.y_hat["mean"] > 0.1))
            ax[jj].scatter(
                cls.train.times[idx],
                np.ones_like(idx) * 2,
                marker="*",
                c="r",
                alpha=0.2,
                s=5,
            )

            idx = np.where(cls.valid.y != (cls.valid.y_hat["mean"] > 0.1))
            ax[jj].scatter(
                cls.valid.times[idx],
                np.ones_like(idx) * 2,
                marker="*",
                c="r",
                alpha=0.2,
                s=5,
            )

            ax[jj].set_ylim((0, 2.2))

        for cls, jj, cls_name in zip(
            (self.cnn_1d, self.cnn_2d), (1, 2), self.names_cnn
        ):

            ax[jj].set_title(cls_name)
            ax[jj].set_ylim((0, 3.2))
            ax[jj].set_xticks([])

            ax[jj].plot(cls.train.times, cls.train.losses, color=f"C{jj}")
            ax[jj].plot(cls.valid.times, cls.valid.losses, color=f"C{jj}")

            ax[jj].plot(
                self.dataset.t(),
                0.6931 * np.ones_like(self.dataset.t()),
                linewidth=0.3,
                linestyle="--",
                c="k",
            )

            ax[jj].add_patch(
                patches.Rectangle(
                    (cls.valid.times[0], 0),
                    width=(cls.valid.times[-1] - cls.valid.times[0]),
                    height=(2),
                    alpha=0.3,
                    color="C6",
                )
            )

            idx = np.where(cls.train.y != (cls.train.y_hat > 0.1))
            ax[jj].scatter(
                cls.train.times[idx],
                np.ones_like(idx) * 3,
                marker="*",
                c="r",
                alpha=0.2,
                s=5,
            )

            idx = np.where(cls.valid.y != (cls.valid.y_hat > 0.1))
            ax[jj].scatter(
                cls.valid.times[idx],
                np.ones_like(idx) * 3,
                marker="*",
                c="r",
                alpha=0.2,
                s=5,
            )

        plt.subplots_adjust(hspace=1)

        return fig, ax


def plot_losses(dataset, cls_scores, data_df, data_decim_df):

    plot_yhats = True

    fig, ax = plt.subplots(len(cls_scores) + 1, 1, sharex=True)
    t = dataset.t()

    t_train = data_df[data_df["is_valid"] == False]["t"].to_numpy()
    t_valid = data_df[data_df["is_valid"] == True]["t"].to_numpy()

    for jj, scores in enumerate(cls_scores):
        ax[jj].plot(t, np.ones_like(t) * 0.6931, linewidth=0.4, c="k")
        ax[jj].plot(
            scores.train.times,
            (
                scores.train.losses["mean"]
                if isinstance(scores.train.losses, dict)
                else scores.train.losses
            ),
        )
        if plot_yhats:
            ax[jj].plot(
                scores.train.times,
                (
                    scores.train.y_hat["mean"]
                    if isinstance(scores.train.y_hat, dict)
                    else scores.train.y_hat
                ),
                linewidth=0.5,
            )
        ax[jj].plot(
            scores.valid.times,
            (
                scores.valid.losses["mean"]
                if isinstance(scores.valid.losses, dict)
                else scores.valid.losses
            ),
        )
        if plot_yhats:
            ax[jj].plot(
                scores.valid.times,
                (
                    scores.valid.y_hat["mean"]
                    if isinstance(scores.valid.y_hat, dict)
                    else scores.valid.y_hat
                ),
                linewidth=0.5,
            )

        ax[jj].add_patch(
            patches.Rectangle(
                (scores.valid.times[0], 0),
                width=(scores.valid.times[-1] - scores.valid.times[0]),
                height=(2),
                alpha=0.3,
                color="C6",
            )
        )
        ax[jj].set_ylim((0, 2.5))

    for i in range(dataset.n_chan):
        ax[-1].plot(dataset.t_decim(), dataset.LFP_decim[i], alpha=0.3)

    ax[-1].plot(dataset.t(), dataset.label * 0.7 * np.max(dataset.LFP))
    ax[-1].add_patch(
        patches.Rectangle(
            (t_valid[0], np.min(dataset.LFP)),
            width=(t_valid[-1] - t_valid[0]),
            height=(np.max(dataset.LFP) - np.min(dataset.LFP)),
            alpha=0.3,
            color="C6",
        )
    )
    return fig, ax
