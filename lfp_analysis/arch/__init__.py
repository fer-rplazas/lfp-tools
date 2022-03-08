from .resnet1d import ResNet, ResNetMini
from .dense import DenseNetFull

model_dict = {
    "ResNet": ResNet,
    "ResNetMini": ResNetMini,
    "DenseNet": DenseNetFull,
}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert (
            False
        ), f'Unknown model name "{model_name}". Available models are {model_dict.keys()}'
