import numpy as np
from typing import Optional, List

from catalyst.dl import Runner, CallbackOrder, Callback, \
    CheckpointCallback
from sklearn.metrics import f1_score, average_precision_score


class F1Callback(Callback):
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "f1"
    ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, runner: Runner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, runner: Runner):
        target = runner.input[self.input_key].detach().cpu().numpy()
        clipwise_output = runner.output[self.output_key].detach().cpu().numpy()

        assert not np.isnan(np.sum(target)) and not np.isnan(
            np.sum(clipwise_output))

        self.prediction.append(clipwise_output)
        self.target.append(target)

        y_pred = clipwise_output.argmax(axis=1)
        y_true = target.argmax(axis=1)

        score = f1_score(y_true, y_pred, average="macro")

        runner.batch_metrics[self.prefix] = score

    def on_loader_end(self, runner: Runner):
        y_pred = np.concatenate(self.prediction, axis=0).argmax(axis=1)
        y_true = np.concatenate(self.target, axis=0).argmax(axis=1)

        score = f1_score(y_true, y_pred, average="macro")

        runner.loader_metrics[self.prefix] = score
        if runner.is_valid_loader:
            runner.epoch_metrics[runner.valid_loader + "_epoch_" +
                                 self.prefix] = score
        else:
            runner.epoch_metrics["train_epoch_" + self.prefix] = score


class mAPCallback(Callback):
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "mAP"
    ):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, runner: Runner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, runner: Runner):
        targ = runner.input[self.input_key].detach().cpu().numpy()
        clipwise_output = runner.output[self.output_key]

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        score = average_precision_score(targ, clipwise_output, average=None)
        score = np.nan_to_num(score).mean()
        runner.batch_metrics[self.prefix] = score

    def on_loader_end(self, runner: Runner):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = average_precision_score(y_true, y_pred, average=None)
        score = np.nan_to_num(score).mean()
        runner.loader_metrics[self.prefix] = score
        if runner.is_valid_loader:
            runner.epoch_metrics[runner.valid_loader + "_epoch_" +
                                 self.prefix] = score
        else:
            runner.epoch_metrics["train_epoch_" + self.prefix] = score
