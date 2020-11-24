from typing import Any

import math
import torch

from torch.nn import CrossEntropyLoss, Module, functional as F


class L2Norm(Module):
    def __init__(self, p=2, dim=1):
        super(L2Norm, self).__init__()

        self.p = p
        self.dim = dim

    def forward(self, x: torch.Tensor, **kwargs: Any):
        return F.normalize(x)


class ArcFaceLinear(Module):
    def __init__(self, embedding_size, num_classes):
        super(ArcFaceLinear, self).__init__()

        self.weight = torch.nn.Parameter(
            data=torch.FloatTensor(num_classes, embedding_size),
            requires_grad=True
        )

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(features, F.normalize(self.weight))
        return cosine


class ArcFaceLoss(Module):
    def __init__(self, num_classes: int, s: float = 64.0, m: float = 0.5):
        super(ArcFaceLoss, self).__init__()
        assert 0 <= m <= math.pi / 2

        self.num_classes = num_classes

        self.crit = CrossEntropyLoss()
        # self.crit = FocalLossMultiClassFixed()

        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.easy_margin = False

    def forward(self, logits, targets):

        cos_phi = logits.float()
        sin_phi = torch.sqrt(1.0 - torch.pow(cos_phi, 2))

        # cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
        cos_phi_plus_m = cos_phi * self.cos_m - sin_phi * self.sin_m

        if self.easy_margin:
            cos_phi_plus_m = torch.where(cos_phi > 0,
                                         cos_phi_plus_m,
                                         cos_phi)
        else:
            # when phi not in [0, pi], use cosface instead
            cos_phi_plus_m = torch.where(cos_phi > self.th,
                                         cos_phi_plus_m,
                                         cos_phi - self.mm)

        one_hot = F.one_hot(targets, self.num_classes)
        output = one_hot * cos_phi_plus_m + (1.0 - one_hot) * cos_phi
        output *= self.s

        loss = self.crit(output, targets)

        return loss
