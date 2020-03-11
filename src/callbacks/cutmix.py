import torch
from catalyst.dl import CriterionCallback, State

from src.datasets.bengali import CUTMIX_KEYS, CUTMIX_LAMBDA_KEY


class CutMixCriterionCallback(CriterionCallback):

    def __init__(
            self,
            weights=None,
            **kwargs
    ):
        """
        Args:
        """
        super(CutMixCriterionCallback, self).__init__(**kwargs)

        self.weights = weights

        self.batch_size = None
        self.width = None
        self.height = None

    def _compute_loss(self, state: State, criterion):
        losses = []
        if CUTMIX_LAMBDA_KEY in state.input:

            lam = state.input[CUTMIX_LAMBDA_KEY].float()

            for input_key, output_key, cutmix_key in zip(
                    self.input_key, self.output_key, CUTMIX_KEYS):
                pred = state.output[output_key]
                y_a = state.input[input_key]
                y_b = state.input[cutmix_key]
                losses.append(((1 - lam) * criterion(pred, y_a) +
                               lam * criterion(pred, y_b)).mean())
        else:
            for input_key, output_key in zip(self.input_key, self.output_key):
                pred = state.output[output_key]
                target = state.input[input_key]
                losses.append(criterion(pred, target))

        if self.weights is not None:
            s = sum([l * w for l, w in zip(losses, self.weights)])
        else:
            s = sum(losses)
        return s
