import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torch.functional import F

class ArcFace(nn.Module):
    
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s

class AULoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-1):
        super(AULoss, self).__init__()
        self.ignore = ignore
        self.loss_fn = nn.BCELoss()

    def forward(self, y_pred, y_true):
        """

        Args:
            y_pred: Nx12
            y_true: Nx12

        Returns:

        """
        index = y_true != self.ignore
        device = y_true.device
        loss = 0
        for i in range(y_true.shape[1]):
            index_i = index[:, i]
            y_true_i = y_true[:, i][index_i]
            y_pred_i = y_pred[:, i][index_i]
            if y_true_i.size(0) == 0:
                loss += torch.tensor(0.0, requires_grad=True).to(device)
                continue
            loss += self.loss_fn(y_pred_i, y_true_i)
        
        return loss

class CCCLoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-5.0):
        super(CCCLoss, self).__init__()
        self.ignore = ignore

    def forward(self, y_pred, y_true):
        """
        y_true: shape of (N, )
        y_pred: shape of (N, )
        """
        batch_size = y_pred.size(0)
        device = y_true.device
        index = y_true != self.ignore
        index.requires_grad = False

        y_true = y_true[index]
        y_pred = y_pred[index]
        if y_true.size(0) <= 1:
            loss = torch.tensor(0.0, requires_grad=True).to(device)
            return loss
        x_m = torch.mean(y_pred)
        y_m = torch.mean(y_true)

        x_std = torch.std(y_true)
        y_std = torch.std(y_pred)

        v_true = y_true - y_m
        v_pred = y_pred - x_m

        s_xy = torch.sum(v_pred * v_true)

        numerator = 2 * s_xy
        denominator = x_std ** 2 + y_std ** 2 + (x_m - y_m) ** 2 + 1e-8

        ccc = numerator / (denominator * batch_size)

        loss = torch.mean(1 - ccc)

        return loss
        
if __name__ == '__main__':
    logit = torch.randn(size=(1, 3))
    labels = -torch.randn(size=(1, 3))
    crit = CCCLoss()
    loss = crit(logit,logit)
    print(loss)
    print(crit.ccc)
