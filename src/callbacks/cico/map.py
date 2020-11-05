import warnings
import numpy as np
from typing import Optional, List

from catalyst.dl import State, CallbackOrder, Callback
from sklearn.metrics import average_precision_score


class mAPCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "mAP"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        target = state.input[self.input_key].detach().cpu().numpy()
        clipwise_output = state.output[self.output_key].detach().cpu().numpy()

        assert not np.isnan(np.sum(target)) and not np.isnan(np.sum(clipwise_output))

        self.prediction.append(clipwise_output)
        self.target.append(target)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            score = average_precision_score(target, clipwise_output, average=None)
            score = np.nan_to_num(score).mean()

        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            score = average_precision_score(y_true, y_pred, average=None)
            score = np.nan_to_num(score).mean()

        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score