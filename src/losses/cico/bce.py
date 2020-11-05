import torch


class PANNsLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = torch.nn.BCELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits = torch.where(torch.isnan(logits),
        #                      torch.zeros_like(logits),
        #                      logits)
        # logits = torch.where(torch.isinf(logits),
        #                      torch.zeros_like(logits),
        #                      logits)
        #
        # target = target.float()

        return self.bce(logits, target)
