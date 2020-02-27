import math
import torch

from torch.nn import functional as F

from catalyst.contrib.nn import FocalLossMultiClass
from .focal import FocalLossMultiClassFixed
from .dense import DenseCrossEntropy


# class ArcFaceProduct(torch.nn.Module):
#     def __init__(self, embedding_size, num_classes):
#         super(ArcFaceProduct, self).__init__()
#         self.weight = torch.nn.Parameter(
#             torch.FloatTensor(num_classes, embedding_size), requires_grad=True)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         # self.weight.data.xavier_uniform_(-stdv, stdv)
#
#     def forward(self, features):
#         features = F.normalize(features)
#         weight = F.normalize(self.weight)
#         cos_phi = F.linear(features, weight)
#         cos_phi = cos_phi.clamp(-1, 1)
#         return cos_phi


class ArcFaceProduct(torch.nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(ArcFaceProduct, self).__init__()
        # self.linear = torch.nn.Linear(embedding_size, num_classes, bias=False)
        # stdv = 1. / math.sqrt(self.linear.weight.data.size(1))
        # self.linear.weight.data.uniform_(-stdv, stdv)

        self.weight = torch.nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        # features = F.normalize(features)
        # self.linear.weight.data = F.normalize(self.linear.weight)
        # return self.linear(features)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(torch.nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        assert 0 <= m <= math.pi / 2
        self.crit = FocalLossMultiClassFixed()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, cos_phi, labels):
        cos_phi = cos_phi.float()
        sin_phi = torch.sqrt(1.0 - torch.pow(cos_phi, 2))
        # cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
        cos_phi_plus_m = cos_phi * self.cos_m - sin_phi * self.sin_m
        # when phi not in [0, pi], use cosface instead
        cos_phi_plus_m = torch.where(cos_phi > self.th,
                                     cos_phi_plus_m,
                                     cos_phi - self.mm)

        output = labels * cos_phi_plus_m + (1.0 - labels) * cos_phi
        output *= self.s
        loss = self.crit(output, labels)
        return loss
