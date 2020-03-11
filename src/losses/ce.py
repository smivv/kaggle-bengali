import torch


class CrossEntropyLossOneHot(torch.nn.Module):
    def forward(self, input, target):
        return ((-target * torch.nn.LogSoftmax(dim=1)(input)).sum(dim=1)).mean()
