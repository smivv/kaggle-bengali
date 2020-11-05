from typing import List

import numpy as np

import torch

from catalyst.dl import CriterionCallback, State


class CutMixUpCallback(CriterionCallback):
    LAMBDA_INPUT_KEY = "CutMixUpCallback_lambda"
    PERMUT_INPUT_KEY = "CutMixUpCallback_permut"

    def __init__(
            self,
            fields: List[str] = ("image",),
            alpha: float = 1.0,
            prob: float = 0.5,
            on_train_only: bool = True,
            weights: List[float] = None,
            method: str = "both",
            method_prob: float = 0.5,
            **kwargs
    ):
        """
        Cutmix & Mixup callback.

        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            prob (float): Probability to call one of augmentations.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
            weights (List[float]): Loss weights.
            method: (str): One of `cutmix`, `mixup` or `both`.
            method_prob (float): Probability to choose one method or another.
            **kwargs:
        """
        super(CutMixUpCallback, self).__init__(**kwargs)

        assert len(fields) > 0, \
            "At least one field for CutMixUpCallback is required"
        assert alpha >= 0, "alpha must be >= 0"

        self.fields: List[str] = fields
        self.alpha: float = alpha
        self.prob: float = prob
        self.on_train_only: bool = on_train_only

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
        self.permutations = None

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
            obj = state.input[f]
            for i in range(self.batch_size):
                if self.probs[i] < self.prob:
                    obj[i] = self.lam[i] * obj[i] + \
                             (1 - self.lam[i]) * obj[self.permutations[i]]
                else:
                    self.lam[i] = 1

    def do_cutmix(self, state: State):
        for f in self.fields:
            obj = state.input[f]
            for i in range(self.batch_size):
                if self.probs[i] < self.prob:
                    l, t, r, b = self._rand_bbox(
                        self.lam[i], self.width, self.height)
                    obj[i, :, t:b, l:r] = obj[self.permutations[i], :, t:b, l:r]
                    self.lam[i] = 1 - float((r - l) * (b - t)) / (
                            self.height * self.width)
                else:
                    self.lam[i] = 1

    def do_both(self, state: State):
        for f in self.fields:
            obj = state.input[f]
            for i in range(self.batch_size):
                if self.probs[i] < self.prob:
                    if np.random.rand() < self.method_prob:
                        l, t, r, b = self._rand_bbox(
                            self.lam[i], self.width, self.height)
                        obj[i, :, t:b, l:r] = obj[self.permutations[i], :, t:b, l:r]
                        self.lam[i] = 1 - float((r - l) * (b - t)) / \
                                      (self.height * self.width)
                    else:
                        obj[i] = self.lam[i] * obj[i] + \
                                 (1 - self.lam[i]) * obj[self.permutations[i]]
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
                size=self.batch_size
            )
        else:
            self.lam = np.ones(size=self.batch_size)

        self.permutations = np.random.permutation(self.batch_size)
        self.probs = np.random.rand(self.batch_size)

        if CutMixUpCallback.LAMBDA_INPUT_KEY not in state.input or \
                CutMixUpCallback.PERMUT_INPUT_KEY not in state.input:
            self.augment_fn(state)
            state.input[CutMixUpCallback.LAMBDA_INPUT_KEY] = self.lam
            state.input[CutMixUpCallback.PERMUT_INPUT_KEY] = self.permutations

    def _compute_loss(self, state: State, criterion):
        losses = []
        if self.is_needed:
            if CutMixUpCallback.LAMBDA_INPUT_KEY not in state.input or \
                    CutMixUpCallback.PERMUT_INPUT_KEY not in state.input:
                raise ValueError("Lambda and permuts not found in state input!")

            self.lam = state.input[CutMixUpCallback.LAMBDA_INPUT_KEY]
            self.lam = torch.tensor(self.lam, requires_grad=False,
                                    dtype=torch.float, device=state.device)

            self.permutations = state.input[CutMixUpCallback.PERMUT_INPUT_KEY]
            # self.perm = torch.tensor(self.perm, requires_grad=False,
            #                          dtype=torch.float, device=state.device)

            for input_key, output_key in zip(self.input_key, self.output_key):
                pred = state.output[output_key]
                y_a = state.input[input_key]
                y_b = state.input[input_key][self.permutations]
                # losses.append((self.lam * criterion(pred, y_a) +
                #                (1 - self.lam) * criterion(pred, y_b)).mean())
                losses.append(((1 - self.lam) * criterion(pred, y_a) +
                                self.lam * criterion(pred, y_b)).mean())
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
