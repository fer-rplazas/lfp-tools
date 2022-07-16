import torch
#from .__init__ import create_model

class FeatureCombiner(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, feature_extractor, n_chan):

        super().__init__()

        self.feat_extractor = feature_extractor
        self.embeder = net

        out_net = net.hparams.c_hidden[len(net.hparams.num_blocks) - 1]
        out_feats = self.feat_extractor.freq_idx.shape[0] * n_chan

        self.bn = torch.nn.BatchNorm1d(out_net + out_feats)
        self.project = torch.nn.Linear(out_net + out_feats, int((out_net + out_feats)))

    def forward(self, x):
        x_spectral = torch.tensor(
            self.feat_extractor(x.detach().cpu().numpy()), dtype=torch.float32
        )
        x_spectral = x_spectral.to(x.device)
        x_embed = self.embeder.embed(x)

        x = torch.cat((x_spectral, x_embed), dim=1)
        return self.project(self.bn(x))
