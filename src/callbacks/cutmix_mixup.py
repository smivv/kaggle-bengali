from typing import List

import numpy as np

import torch

from catalyst.dl import CriterionCallback, State


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class MixupCutmixCallback(CriterionCallback):
    def __init__(
            self,
            fields: List[str] = ("image",),
            alpha=1.0,
            on_train_only=True,
            weight_grapheme_root=.5,
            weight_vowel_diacritic=.25,
            weight_consonant_diacritic=.25,
            method="cutmix",
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
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"
        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True
        self.weight_grapheme_root = weight_grapheme_root
        self.weight_vowel_diacritic = weight_vowel_diacritic
        self.weight_consonant_diacritic = weight_consonant_diacritic
        self.apply_mixup = True
        self.method = method

    def on_loader_start(self, state: State):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def do_mixup(self, state: State):
        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                             (1 - self.lam) * state.input[f][self.index]

    def do_cutmix(self, state: State):
        bbx1, bby1, bbx2, bby2 = \
            rand_bbox(state.input[self.fields[0]].shape, self.lam)
        for f in self.fields:
            state.input[f][:, :, bbx1:bbx2, bby1:bby2] = \
                state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]
        self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)
                        / (state.input[self.fields[0]].shape[-1]
                           * state.input[self.fields[0]].shape[-2]))

    def on_batch_start(self, state: State):
        if not self.is_needed:
            return
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1
        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        if np.random.rand() < 0.5:
            if self.method == "mixup":
                self.do_mixup(state)
            elif self.method == "cutmix":
                self.do_cutmix(state)

    def _compute_loss(self, state: State, criterion):
        loss_arr = [0, 0, 0]
        if not self.is_needed:
            for i, (input_key, output_key) in enumerate(
                    list(zip(self.input_key, self.output_key))):
                pred = state.output[output_key]
                y = state.input[input_key]
                loss_arr[i] = criterion(pred, y)
        else:
            for i, (input_key, output_key) in enumerate(
                    list(zip(self.input_key, self.output_key))):
                pred = state.output[output_key]
                y_a = state.input[input_key]
                y_b = state.input[input_key][self.index]
                loss_arr[i] = self.lam * criterion(pred, y_a) + \
                              (1 - self.lam) * criterion(pred, y_b)
        loss = loss_arr[0] * self.weight_grapheme_root + \
               loss_arr[1] * self.weight_vowel_diacritic + \
               loss_arr[2] * self.weight_consonant_diacritic
        return loss
