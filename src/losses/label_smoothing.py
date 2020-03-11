import torch
from torch.nn.functional import log_softmax#, one_hot


def one_hot(x, num_classes):
    return torch.eye(num_classes, device=x.device)[x, :]


class CrossEntropyLossSmoothed(torch.nn.Module):

    def __init__(self, eps=0.1):
        super(CrossEntropyLossSmoothed, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        num_classes = output.size(1)

        logprobs = log_softmax(output, dim=1)
        target = one_hot(target, num_classes=num_classes)

        target = (1 - self.eps) * target + \
                 self.eps * torch.zeros_like(target) / (num_classes - 1)

        loss = torch.mean(-logprobs * target, dim=1)

        return loss.mean()
