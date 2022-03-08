from fastai.callback.wandb import *
from fastai.vision.all import *
from torchvision.transforms import ToPILImage, ToTensor

from .resnet1d import Trainer

DATA_PATH = Path("./../data")


def _conv_block(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf // 4, 1),
        ConvLayer(nf // 4, nf // 4, stride=stride),
        ConvLayer(nf // 4, nf, 1, act_cls=None, norm_type=NormType.BatchZero),
    )


class ResBlock(Module):
    def __init__(self, ni, nf, stride=1):
        self.convs = _conv_block(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


def _resnet_stem(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i + 1], 3, stride=2 if i == 0 else 1)
        for i in range(len(sizes) - 1)
    ] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]


class ResNet2d(nn.Sequential):
    def __init__(self, n_in, n_out, layers, expansion=1):
        stem = _resnet_stem(n_in, 32, 32, 64)
        self.block_szs = [64, 64, 128, 256, 512]
        for i in range(1, 5):
            self.block_szs[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(
            *stem,
            *blocks,
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(self.block_szs[len(layers)], n_out),
            Flatten(),
        )

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_szs[idx : idx + 2]
        return nn.Sequential(
            *[
                ResBlock(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
                for i in range(n_layers)
            ]
        )


def resnet18(n_in, n_out):
    return ResNet2d(n_in, n_out, [2, 2, 2, 2], 4)


###############################################################################
# FastAI pipeline:
##############################################################################


class LFPNormalizer2d(Transform):
    def __init__(self, stats):
        self.means, self.stds = torch.tensor(stats[0]), torch.tensor(stats[1])

        if torch.cuda.is_available():
            self.means = self.means.cuda()
            self.stds = self.stds.cuda()

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        xs = torch.unbind(X, 1)
        return torch.stack(
            [(x - self.means[ii]) / self.stds[ii] for ii, x in enumerate(xs)], 1
        )

    def decodes(self, X):
        if isinstance(X, TensorCategory):
            return X
        xs = torch.unbind(X, 1)

        return torch.stack(
            [(x * self.stds[ii]) + self.means[ii] for ii, x in enumerate(xs)], 1
        )


class Resizer(Transform):
    def __init__(self, target_size):
        self.target_size = target_size

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X
        return torch.stack(
            [ToTensor()(ToPILImage()(x).resize(self.target_size)) for x in X]
        ).squeeze()


class Trainer2d(Trainer):
    def __init__(self, layers=[2, 2, 1], wd=0.5, log_wandb=True, experiment=None):

        self.layers, self.wd = layers, wd

        super().__init__(log_wandb, experiment)

    def prepare_dls(self, dataset, windower, bs=128):

        if self.log_wandb:
            self.run.name = f"{dataset.pat_id}/{dataset.task}_{dataset.stim}_2d"

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

        self.data_df = self.windower.data_df

        def get_x(row):
            return torch.tensor(
                dataset.LFP[:, :, int(row["id_start"]) : int(row["id_end"])].copy()
            ).float()

        def get_y(row):
            return row["label"]

        def splitter(df):
            train = df.index[df["is_valid"] == 0].tolist()
            valid = df.index[df["is_valid"] == 1].tolist()
            return train, valid

        def LFP_block2d():
            return TransformBlock(
                item_tfms=[Resizer((160, 160))],
                batch_tfms=LFPNormalizer2d(
                    (
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.098, 0.098, 0.098, 0.098, 0.098, 0.098],
                    )
                ),
            )

        dblock = DataBlock(
            blocks=(LFP_block2d, CategoryBlock),
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        self.dls = dblock.dataloaders(self.data_df, bs=bs, num_workers=0)

        return self

    def prepare_learner(self, dls=None, wd=None):

        cbs = [WandbCallback()] if self.log_wandb else []

        dls = self.dls if dls is None else dls
        wd = self.wd if wd is None else wd

        self.resnet = ResNet2d(self.dataset.LFP.shape[0], 2, self.layers).cuda()
        self.learn = Learner(
            dls.cuda(),
            self.resnet.cuda(),
            metrics=[
                accuracy,
            ],
            loss_func=F.cross_entropy,
            cbs=cbs,
            wd=float(wd),
        )
        self.learn.recorder.train_metrics = True

        return self

    def train(self, n_epochs=45, lr_div=1):

        self.learn.fit_one_cycle(n_epochs, lr_div)
        self.learn.add_cb(EarlyStoppingCallback(min_delta=0.001, patience=3))

        # self.learn.fit_one_cycle(14, 10e-4)
        # self.learn.fit_one_cycle(25, 5 * 10e-5)
        self.learn.fit_one_cycle(35, 10e-5)
        self.learn.fit_one_cycle(35, 3 * 10e-6)
        self.learn.fit_one_cycle(35, 10e-6)
        self.learn.fit_one_cycle(35, 10e-7)

        [self.learn.remove_cb(cb) for cb in self.learn.cbs[3:]]
