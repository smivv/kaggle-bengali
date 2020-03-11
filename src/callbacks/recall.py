from typing import Any, List, Optional, Union  # isort:skip
# import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from catalyst.dl.core import Callback, CallbackOrder, State
from sklearn.metrics import recall_score

from src.datasets.bengali import CUTMIX_LAMBDA_KEY, CUTMIX_KEYS


class RecallCallback(Callback):
    def __init__(
            self,
            input_keys=None,
            output_keys=None,
            loss_keys=None,
            weights=None,
            prefix: str = "recall",
    ):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.loss_keys = loss_keys
        self.prefix = prefix
        self.weights = weights

        super(RecallCallback, self).__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: State):
        scores = []
        for loss_key, input_key, output_key, cutmix_key in zip(
                self.loss_keys, self.input_keys, self.output_keys, CUTMIX_KEYS):

            target = state.input[input_key].detach().cpu().numpy()
            predicted = state.output[output_key]
            predicted = F.softmax(predicted, 1)
            _, predicted = torch.max(predicted, 1)
            predicted = predicted.detach().cpu().numpy()

            if CUTMIX_LAMBDA_KEY in state.input:
                lam = state.input[CUTMIX_LAMBDA_KEY].detach().cpu().numpy()
                target2 = state.input[cutmix_key].detach().cpu().numpy()
                score = 100 * np.mean(lam * recall_score(target,
                                                         predicted,
                                                         average='macro',
                                                         zero_division=0) +
                                      (1 - lam) * recall_score(target2,
                                                               predicted,
                                                               average='macro',
                                                               zero_division=0))
            else:
                score = 100 * recall_score(target, predicted,
                                           average='macro', zero_division=0)
            state.metric_manager.add_batch_value(name=loss_key, value=score)
            scores.append(score)

        final_score = np.average(scores, weights=self.weights)
        state.metric_manager.add_batch_value(name=self.prefix,
                                             value=final_score)
