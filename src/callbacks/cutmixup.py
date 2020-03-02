from typing import List

import numpy as np

import torch

from catalyst.dl import CriterionCallback, State


class CutMixUpCallback(CriterionCallback):

    LAMBDA_INPUT_KEY = "CutMixUpCallback_lambda"

    def __init__(
            self,
            fields: List[str] = ("image",),
            alpha=1.0,
            prob=0.5,
            on_train_only=True,
            weights=None,
            method="both",
            method_prob=0.5,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for CutMixUpCallback is required"
        assert alpha >= 0, "alpha must be >= 0"
        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.prob = prob
        self.lam = None
        self.is_needed = True
        self.method_prob = method_prob
        self.weights = weights

        if method == "mixup":
            self.augment_fn = self.do_mixup
        elif method == "cutmix":
            self.augment_fn = self.do_cutmix
        elif method == "both":
            self.augment_fn = self.do_both

        self.batch_size = None
        self.width = None
        self.height = None

        self.probs = None
        self.perm = None

    def on_loader_start(self, state: State):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def _rand_bbox(self, lam, width, height):
        cut_rat = np.sqrt(1. - lam)
        cut_w = (width * cut_rat).astype(np.int)
        cut_h = (height * cut_rat).astype(np.int)
        # uniform
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        left = np.clip(cx - cut_w // 2, 0, width)
        right = np.clip(cx + cut_w // 2, 0, width)
        top = np.clip(cy - cut_h // 2, 0, height)
        bottom = np.clip(cy + cut_h // 2, 0, height)
        return left, top, right, bottom

    def do_mixup(self, state: State):
        for f in self.fields:
            for i in range(self.batch_size):
                if self.probs[i] < self.prob:
                    state.input[f][i] = self.lam[i] * state.input[f][i] + \
                                        (1 - self.lam[i]) * state.input[f][self.perm[i]]
                else:
                    self.lam[i] = 1

    def do_cutmix(self, state: State):
        for f in self.fields:
            for i in range(self.batch_size):
                if self.probs[i] < self.prob:
                        left, top, right, bottom = self._rand_bbox(self.lam[i], self.width, self.height)
                        state.input[f][i, :, top:bottom, left:right] = state.input[f][self.perm[i], :, top:bottom, left:right]
                        self.lam[i] = 1 - float((right - left) * (bottom - top)) / (self.height * self.width)
                else:
                    self.lam[i] = 1

    def do_both(self, state: State):
        for f in self.fields:
            for i in range(self.batch_size):
                if self.probs[i] < self.prob:
                    if np.random.rand() < self.method_prob:
                        left, top, right, bottom = self._rand_bbox(self.lam[i], self.width, self.height)
                        state.input[f][i, :, top:bottom, left:right] = state.input[f][self.perm[i], :, top:bottom, left:right]
                        self.lam[i] = 1 - float((right - left) * (bottom - top)) / (self.height * self.width)
                    else:
                        state.input[f][i] = self.lam[i] * state.input[f][i] + (1 - self.lam[i]) * state.input[f][self.perm[i]]
                else:
                    self.lam[i] = 1

    def on_batch_start(self, state: State):

        if not self.is_needed:
            return

        shape = state.input[self.fields[0]].shape
        self.batch_size = shape[0]
        self.height = shape[-2]
        self.width = shape[-1]

        if self.alpha > 0:
            self.lam = np.random.beta(
                    a=self.alpha,
                    b=self.alpha,
                    size=self.batch_size)
        else:
            self.lam = np.ones(size=self.batch_size)

        self.perm = np.random.permutation(self.batch_size)
        self.probs = np.random.rand(self.batch_size)

        if CutMixUpCallback.LAMBDA_INPUT_KEY not in state.input:
            self.augment_fn(state)
            state.input[CutMixUpCallback.LAMBDA_INPUT_KEY] = self.lam

    def _compute_loss(self, state: State, criterion):
        loss_arr = []
        if not self.is_needed:
            for input_key, output_key in zip(
                    self.input_key, self.output_key):
                pred = state.output[output_key]
                y = state.input[input_key]
                loss_arr.append(criterion(pred, y))
        else:
            if CutMixUpCallback.LAMBDA_INPUT_KEY not in state.input:
                raise ValueError("Lambda not found in state input!")

            self.lam = state.input[CutMixUpCallback.LAMBDA_INPUT_KEY]
            self.lam = torch.tensor(self.lam, requires_grad=False,
                                    dtype=torch.float, device=state.device)
            for input_key, output_key in zip(self.input_key, self.output_key):
                pred = state.output[output_key]
                y_a = state.input[input_key]
                y_b = state.input[input_key][self.perm]
                loss_arr.append((
                    (self.lam * criterion(pred, y_a) +
                     (1 - self.lam) * criterion(pred, y_b))).mean())

        if self.weights is not None:
            s = sum([l * w for l, w in zip(loss_arr, self.weights)])
        else:
            s = sum(loss_arr)
        return s
