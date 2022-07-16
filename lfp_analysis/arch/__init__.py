from .resnet1d import ResNet, ResNetMini
from .dense import DenseNetFull
from .lstm import FeatureCombiner
import torch
from torch import nn
from ..feature_extractor import RealTimePeriodogram




class Reducer(torch.nn.Module):
    def __init__(
        self,
        model_name="ResNetMini",
        model_hparams={"n_in": 4, "ks": 257, "num_blocks": [2]},
        n_in=4,
        sig_len=1535,
    ):

        super().__init__()

        self.net = create_model(model_name, model_hparams)
        self.feat_extractor = RealTimePeriodogram(1535)
        self.convs = FeatureCombiner(self.net, self.feat_extractor, n_in)

        n_feats = self.convs(torch.randn(2, n_in, sig_len)).shape[-1]

        self.cls = torch.nn.Linear(n_feats, 2)

    def forward(self, x):
        return self.cls(self.convs(x))


model_dict = {
    "ResNet": ResNet,
    "ResNetMini": ResNetMini,
    "DenseNet": DenseNetFull,
    "reducer": Reducer,
}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        import pdb

        # pdb.set_trace()
        return model_dict[model_name](**model_hparams)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are {model_dict.keys()}'


class Decoder(nn.Module):
    def __init__(self, n_ins: list[int], n_outs: list[int], p_drop=0.3):

        super().__init__()

        self.d1 = nn.Sequential(
            nn.Linear(n_ins[0], n_ins[0] * 3),
            nn.BatchNorm1d(n_ins[0] * 3),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(n_ins[0] * 3, n_outs[0]),
        )

        self.d2 = nn.Sequential(
            nn.Linear(n_ins[1], n_ins[1] * 2),
            nn.BatchNorm1d(n_ins[1] * 2),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(n_ins[1] * 2, n_outs[1]),
        )

    def forward(self, x, x_feats, regress_feats=True):

        x_ = x.view(x.shape[0] * x.shape[1], -1)

        if regress_feats:
            x_feats_ = x_feats.view(x_feats.shape[0] * x_feats.shape[1], -1)

            return self.d1(x_), self.d2(x_feats_)
        else:
            return self.d1(x_)


class ARConvs(nn.Module):
    def __init__(
        self, n_feats, hidden_size, convs_name, convs_hparams, decoder_hparams
    ):
        super().__init__()

        self.LSTM = nn.LSTM(n_feats, hidden_size)

        self.convs = create_model(convs_name, convs_hparams)
        self.decoder = Decoder(**decoder_hparams)

    def forward(self, x, decode_feats=True):

        # Encode:
        ## Encode convs:
        l_ = []
        for x_ in torch.unbind(x, 1):
            l_.append(self.convs(x_))

        x_conved = torch.stack(l_)

        ## Encode Auto-Regressive
        lstm_out = self.LSTM(x_conved)[0]

        # Decode:
        if decode_feats:
            out_class, out_regress = self.decoder(lstm_out, x_conved)
        else:
            out_class = self.decoder(lstm_out, [], regress_feats=False)
            out_regress = None
        out_class = out_class.squeeze()

        return out_class, out_regress


model_dict["ARConvs"] = ARConvs
