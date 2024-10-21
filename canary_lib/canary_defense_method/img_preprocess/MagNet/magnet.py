import torch.nn as nn
import os
import torch


def get_defensive_model(defensive_models_path, defensive_model_name, dataset_name, device):
    if defensive_model_name == 'defensive-model-2':
        model = DefensiveModel2()
    else:
        raise ValueError("Undefined defensive model")
    model_state = torch.load(os.path.join(defensive_models_path,
                                          f'{dataset_name}_{defensive_model_name}_checkpoint.pth'))['state_dict']
    model.load_state_dict(model_state)
    model.eval().to(device)

    return model


def magnet_purify(img, model):
    purified_img = model(img)
    return purified_img


class DefensiveModel2(nn.Module):
    """Defensive model used for CIFAR-10 in MagNet paper
    """

    def __init__(self, in_channels=3):
        super(DefensiveModel2, self).__init__()
        self.conv_11 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv_31 = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        """Forward propagation

        :param X: Mini-batch of shape [-1, 1, H, W]
        :return: Mini-batch of shape [-1, 1, H, W]
        """
        X = torch.sigmoid(self.conv_11(X))
        X = torch.sigmoid(self.conv_21(X))
        X = torch.sigmoid(self.conv_31(X))

        return X
