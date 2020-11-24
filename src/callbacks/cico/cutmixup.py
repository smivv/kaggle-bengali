from typing import List, Dict, Tuple, Optional, Union

import torch

import numpy as np

from catalyst.dl import CriterionCallback, Runner


class CutMixUpCallback(CriterionCallback):
    LAMBDA_INPUT_KEY = "CutMixUpCallback_lambda"
    PERMUT_INPUT_KEY = "CutMixUpCallback_permut"

    def __init__(
            self,
            fields: List[str] = ("image",),
            alpha: float = 1.0,
            prob: float = 0.5,
            on_train_only: bool = True,
            weights: Optional[List[float]] = None,
            method: str = "both",
            method_prob: float = 0.5,
            **kwargs
    ):
        """
        Cutmix & Mixup callback.

        Args:
            fields (List[str]): List of features which must be affected.
            alpha (float): Beta distribution a=b parameters.
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

        assert alpha >= 0, \
            "alpha must be >= 0"

        assert method in ["mixup", "cutmix", "both"], \
            "method should be one of mixup, cutmix or both"

        self._fields: List[str] = fields
        self._alpha: float = alpha
        self._prob: float = prob
        self._on_train_only: bool = on_train_only

        self._lambda: Optional[np.ndarray] = None
        self._to_augment: bool = True
        self._method_prob = method_prob
        self._weights: List[float] = weights

        if method == "mixup":
            self._augment_fn = self._do_mixup
        elif method == "cutmix":
            self._augment_fn = self._do_cutmix
        elif method == "both":
            self._augment_fn = self._do_both

        self._batch_size = None
        self._height = None
        self._width = None

        self._permuts = None
        self._probs = None

    def _random_boxes(
            self,
            lam: Union[np.ndarray, int],
            width: int,
            height: int
    ) -> Tuple[Union[np.ndarray, int], Union[np.ndarray, int],
               Union[np.ndarray, int], Union[np.ndarray, int]]:
        """
        Generates random baounding boxes.

        Args:
            lam (np.ndarray): Array of lambdas.
            width (int): Width of image.
            height (int): Height of image.

        Returns (Tuple[Union[np.ndarray, int], Union[np.ndarray, int],
                  Union[np.ndarray, int], Union[np.ndarray, int]]):
            Tuple of Left, Top, Right and Bottom coordinates.

        """

        cut_rat: np.ndarray = np.sqrt(1. - lam)

        cut_w: np.ndarray = (width * cut_rat).astype(np.int)
        cut_h: np.ndarray = (height * cut_rat).astype(np.int)

        # uniformly distributed
        cx: np.ndarray = np.random.randint(width)
        cy: np.ndarray = np.random.randint(height)

        left: np.ndarray = np.clip(cx - cut_w // 2, 0, width)
        top: np.ndarray = np.clip(cy - cut_h // 2, 0, height)
        right: np.ndarray = np.clip(cx + cut_w // 2, 0, width)
        bottom: np.ndarray = np.clip(cy + cut_h // 2, 0, height)

        return left, top, right, bottom

    def _do_mixup(self, runner: Runner):
        """
        Do mixup augmentation.

        Args:
            runner (Runner): Runner class.

        """
        for f in self._fields:
            batch = runner.input[f]
            for i in range(self._batch_size):
                if self._probs[i] < self._prob:
                    batch[i] = self._lambda[i] * batch[i] + \
                               (1 - self._lambda[i]) * batch[self._permuts[i]]
                else:
                    self._lambda[i] = 1

    def _do_cutmix(self, runner: Runner):
        """
        Do cutmix augmentation.

        Args:
            runner (Runner): Runner class.

        """
        for f in self._fields:
            batch = runner.input[f]

            for i in range(self._batch_size):
                if self._probs[i] < self._prob:
                    # generate random bbox
                    l, t, r, b = self._random_boxes(
                        lam=self._lambda[i],
                        width=self._width,
                        height=self._height
                    )
                    # erase it and put in current
                    batch[i, :, t:b, l:r] = \
                        batch[self._permuts[i], :, t:b, l:r]

                    self._lambda[i] = 1 - float((r - l) * (b - t)) / \
                                      (self._height * self._width)
                else:
                    self._lambda[i] = 1

    def _do_both(self, runner: Runner):
        """
        Do both augmentations together.

        Args:
            runner (Runner): Runner class.

        """
        for f in self._fields:
            batch = runner.input[f]
            for i in range(self._batch_size):
                if self._probs[i] < self._prob:
                    if np.random.rand() < self._method_prob:
                        # generate random bbox
                        l, t, r, b = self._random_boxes(
                            lam=self._lambda[i],
                            width=self._width,
                            height=self._height
                        )
                        # erase it and put in current
                        batch[i, :, t:b, l:r] = \
                            batch[self._permuts[i], :, t:b, l:r]

                        self._lambda[i] = 1 - float((r - l) * (b - t)) / \
                                          (self._height * self._width)
                    else:
                        batch[i] = self._lambda[i] * batch[i] + \
                                   (1 - self._lambda[i]) * batch[
                                       self._permuts[i]]
                else:
                    self._lambda[i] = 1

    def _compute_loss(self, runner: Runner, criterion) -> torch.Tensor:
        """
        Overloaded _compute_loss function for parent class.

        Args:
            runner (Runner): Runner class.
            criterion (Criterion): Criterion class.

        Returns (torch.Tensor): Loss value.

        """
        losses = []
        if self._to_augment:
            if CutMixUpCallback.LAMBDA_INPUT_KEY not in runner.input or \
                    CutMixUpCallback.PERMUT_INPUT_KEY not in runner.input:
                raise ValueError(
                    "Lambda and permuts not found in runner input!")

            self._permuts = runner.input[CutMixUpCallback.PERMUT_INPUT_KEY]
            self._lambda = runner.input[CutMixUpCallback.LAMBDA_INPUT_KEY]
            self._lambda = torch.tensor(
                data=self._lambda,
                requires_grad=False,
                dtype=torch.float,
                device=runner.device
            )

            for input_key, output_key in zip(self.input_key, self.output_key):
                pred = runner.output[output_key]
                y_a = runner.input[input_key]
                y_b = runner.input[input_key][self._permuts]
                losses.append(
                    ((1 - self._lambda) * criterion(pred, y_a) +
                     self._lambda * criterion(pred, y_b)).mean()
                )
        else:
            for input_key, output_key in zip(self.input_key, self.output_key):
                pred = runner.output[output_key]
                target = runner.input[input_key]
                losses.append(criterion(pred, target))

        if self._weights is not None:
            s = sum([l * w for l, w in zip(losses, self._weights)])
        else:
            s = sum(losses)

        return s

    def on_loader_start(self, runner: Runner):
        """
        Perform action on loader start event.

        Args:
            runner (Runner): Runner class.

        """
        self._to_augment = not self._on_train_only or \
                           runner.loader_name.startswith("train")

    def on_batch_start(self, runner: Runner):
        """
        Perform action on batch start event.

        Args:
            runner (Runner): Runner class.

        """
        if not self._to_augment:
            return

        shape = runner.input[self._fields[0]].shape

        self._batch_size = shape[0]
        self._height = shape[-2]
        self._width = shape[-1]

        if self._alpha > 0:
            self._lambda = np.random.beta(
                a=self._alpha,
                b=self._alpha,
                size=self._batch_size
            )
        else:
            self._lambda = np.ones(size=self._batch_size)

        self._permuts = np.random.permutation(self._batch_size)
        self._probs = np.random.rand(self._batch_size)

        if CutMixUpCallback.LAMBDA_INPUT_KEY not in runner.input or \
                CutMixUpCallback.PERMUT_INPUT_KEY not in runner.input:
            self._augment_fn(runner)

            runner.input[CutMixUpCallback.LAMBDA_INPUT_KEY] = self._lambda
            runner.input[CutMixUpCallback.PERMUT_INPUT_KEY] = self._permuts
