import torch.nn as nn
import torch
from torchvision import models
from .loss import CCCLoss
from torch.functional import F


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class ImageModel(nn.Module):
    def __init__(self, task='EX'):
        super(ImageModel, self).__init__()

        self.base_model = models.resnet50(pretrained=True)
        self.task = task
        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=self.base_model.fc.in_features,
                                          out_features=12 + 8 + 2))

        self.modes = ['clip']
        self.base_model.fc = Dummy()
        self.loss_EX = nn.CrossEntropyLoss(ignore_index=7)
        self.loss_AU = nn.BCELoss()
        self.loss_VA = CCCLoss()

    def forward(self, x):
        clip = x['clip']  # bx3x1x112x112
        assert clip.size(2) == 1
        clip = clip.squeeze(2)

        features = self.base_model(clip)
        out = self.fc(features)

        return out

    def get_ex_loss(self, y_pred, y_true):
        y_pred = y_pred[:, 12:20]
        y_true = y_true.view(-1)
        return self.loss_EX(y_pred, y_true)

    def get_au_loss(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred[:, :12])
        return self.loss_AU(y_pred, y_true)

    def get_va_loss(self, y_pred, y_true):
        y_pred = torch.tanh(y_pred[:, 20:22])
        return self.loss_VA(y_pred, y_true)
