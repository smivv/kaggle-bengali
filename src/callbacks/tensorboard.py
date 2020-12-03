# flake8: noqa
# isort: skip_file
from typing import Tuple, List, Dict, Any, Union, Callable, Optional
from functools import partial

import torch
import torchvision.utils

import numpy as np

from pathlib import Path
from torch import Tensor
from numpy import ndarray

from skimage.color import label2rgb

from catalyst.dl import Callback, CallbackOrder, CallbackNode, Runner
from catalyst.contrib.tools.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def encode_mask_with_color(
        masks: ndarray,
) -> ndarray:
    """

    Args:
        masks (ndarray): semantic mask batch tensor
    Returns:
        ndarray: list of semantic masks

    """
    if masks.ndim == 3:
        masks = np.concatenate([label2rgb(mask, bg_label=0) for mask in masks])
    else:
        masks = label2rgb(masks, bg_label=0)
    return masks


def mask_to_overlay_image(
        image: ndarray,
        mask: ndarray,
        mask_strength: float
) -> ndarray:
    """

    Args:
        image (ndarray):
        mask (ndarray):
        mask_strength (float):

    Returns (ndarray):

    """
    # mask = label2rgb(mask, bg_label=0)
    # image_with_overlay = image * (1 - mask_strength) + mask * mask_strength
    # image_with_overlay = (
    #     (image_with_overlay * 255).clip(0, 255).round().astype(np.uint8)
    # )
    # color = [0, 255, 0]
    color = np.asarray([0, 0, 255])

    image_with_overlay = image.copy()
    # image_with_overlay[mask == 0] = 0
    image_with_overlay[mask != 0] = \
        mask_strength * image[mask != 0] + (1 - mask_strength) * color

    return image_with_overlay


class TensorboardCallback(Callback):
    TENSORBOARD_LOGGER_KEY = "_tensorboard"

    def __init__(
            self,
            input_keys: Optional[Union[str, List[str]]] = None,
            output_keys: Optional[Union[str, List[str]]] = None,
            batch_frequency: int = 40,
    ):
        """

        Args:
            input_keys (Optional[Union[str, List[str]]]): Input keys.
            output_keys (Optional[Union[str, List[str]]]): Output keys.
            batch_frequency (int):
                Frequency of process to be called (default=None).
        """
        super(TensorboardCallback, self).__init__(
            order=CallbackOrder.logging + 1#, node=CallbackNode.master
        )

        assert batch_frequency is None or batch_frequency > 0

        self.input_keys = TensorboardCallback._check_keys(input_keys)
        self.output_keys = TensorboardCallback._check_keys(output_keys)

        if len(self.input_keys) + len(self.output_keys) == 0:
            raise ValueError("Useless visualizer: pass at least one image key")

        self._batch_frequency = batch_frequency

        self._loader_batch_count = 0
        self._loader_processed_in_current_epoch = False
        self._reset()

        self.logger = None

    @staticmethod
    def _check_keys(keys) -> List[str]:
        if keys is None:
            return []
        elif isinstance(keys, str):
            return [keys]
        elif isinstance(keys, (tuple, list)):
            assert all(isinstance(k, str) for k in keys)
            return list(keys)
        else:
            raise ValueError(
                f"Unexpected format of keys' "
                f"argument: must be string or list/tuple"
            )

    def _reset(self):
        self._loader_batch_count = 0
        self._loader_processed_in_current_epoch = False

    @staticmethod
    def _detach(
            tensors: Union[Tensor, List[Tensor]]
    ) -> Union[ndarray, List[ndarray]]:
        """
        Detaches tensor from device.
        
        Args:
            tensors (Union[Tensor, List[Tensor]]): Tensor to detach

        Returns (Union[ndarray, List[ndarray]]): Data from detached tensor

        """
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()

    @staticmethod
    def _get_tensorboard_logger(runner: Runner) -> SummaryWriter:
        """

        Args:
            runner:

        Returns:

        """
        tb_key = VisualizationCallback.TENSORBOARD_LOGGER_KEY
        if (
                tb_key in runner.callbacks
                and runner.loader_name in runner.callbacks[tb_key].loggers
        ):
            return runner.callbacks[tb_key].loggers[runner.loader_name]
        # elif runner.stage_name.startswith("infer"):
        #     return SummaryWriter(str(Path(runner.logdir) / f"infer_log"))
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {runner.loader_name}"
        )

    def get_tensors(
            self,
            runner: Runner,
            input_keys: List[str] = None,
            output_keys: List[str] = None,
    ) -> List[Tensor]:
        """
        Get tensors by their keys.
        
        Args:
            runner (Runner): Runner class
            input_keys (List[str]): Input keys strings
            output_keys (List[str]): Output keys strings

        Returns (List[Tensor]): List of tensors.

        """
        assert input_keys is not None or output_keys is not None
        if input_keys:
            return [runner.input[input_key] for input_key in input_keys]
        else:
            return [runner.output[output_key] for output_key in output_keys]

    def preprocess(
            self,
            tensors: Union[Tensor, List[Tensor]]
    ) -> Union[ndarray, List[ndarray]]:
        """
        Preprocess tensors.
        
        Args:
            tensors (Union[Tensor, List[Tensor]]): Tensors to preprocess

        Returns (Union[ndarray, List[ndarray]]): Data from tensors

        """
        return TensorboardCallback._detach(tensors)

    def process(
            self,
            data: Dict[str, ndarray]
    ) -> Dict[str, ndarray]:
        """
        Processes data.

        Args:
            data (Dict[str, ndarray]): Data to process

        Returns (Dict[str, ndarray]): Processed data

        """
        return data

    def postprocess(
            self,
            runner: Runner,
            data: Dict[str, ndarray]
    ):
        """
        Postprocessing function.

        Args:
            runner (Runner): Runner class
            data (Dict[str, ndarray]): Data to postprocess

        """
        return data

    def run(self, runner: Runner):
        """
        Run full pipeline of callback.

        Args:
            runner (Runner): Runner class

        """
        for key_set, key_values in {
            "input_keys": self.input_keys,
            "output_keys": self.output_keys
        }.items():
            if len(key_values) == 0:
                continue

            tensors: List[Tensor] = self.get_tensors(
                runner=runner,
                **{key_set: key_values}
            )

            tensors: List[ndarray] = self.preprocess(tensors=tensors)

            data: Dict[str, ndarray] = self.process(
                data={k: v for k, v in zip(key_values, tensors)}
            )

            self.postprocess(runner=runner, data=data)

        self._loader_processed_in_current_epoch = True

    def on_loader_start(self, runner: Runner):
        """
        Performs action on loader start.

        Args:
            runner (Runner): Runner class

        """
        self._reset()

    def on_batch_end(self, runner: Runner):
        """
        Performs action on batch end.

        Args:
            runner (Runner): Runner class

        """
        if self._batch_frequency is None:
            self.run(runner=runner)
        elif not (self._loader_batch_count % self._batch_frequency):
            self.run(runner=runner)

        self._loader_batch_count += 1

    def on_loader_end(self, runner: Runner):
        """
        Performs action on loader end.

        Args:
            runner (Runner): Runner class

        """
        if not self._loader_processed_in_current_epoch:
            self.run(runner)


class VisualizationCallback(TensorboardCallback):

    def __init__(
            self,
            concat_images: bool = True,
            max_images: int = 20,
            num_rows: int = 5,
            denorm_fn: str = "default",
            *args,
            **kwargs
    ):
        super(VisualizationCallback, self).__init__(*args, **kwargs)

        self._concat_images = concat_images
        self._max_images = max_images
        self._num_rows = num_rows

        self._denorm_fn = VisualizationCallback._get_denorm_fn(denorm_fn)

    @staticmethod
    def _get_denorm_fn(fn_name) -> Callable:
        # y = (x - mean) / std => x = y * std + mean
        if fn_name.lower() == "default":
            # normalization from [-1, 1] to [0, 1]
            return lambda x: x * 2 + .5
        elif fn_name.lower() == "imagenet":
            return lambda x: x * 0.225 + 0.449
        elif fn_name is None or fn_name.lower() == "none":
            return lambda x: x
        else:
            raise ValueError("Unknown `denorm_fn`")

    def preprocess(
            self,
            tensors: Union[Tensor, List[Tensor]]
    ) -> Union[ndarray, List[ndarray]]:
        """
        Preprocess tensors.

        Args:
            tensors (Union[Tensor, List[Tensor]]): Tensors to preprocess

        Returns (Union[Tensor, List[Tensor]]): Data from tensors

        """
        if isinstance(tensors, list):
            return [
                self._denorm_fn(tensor)
                if isinstance(tensor, ndarray) or isinstance(tensor, Tensor)
                else tensor for tensor in tensors
            ]
        else:
            return self._denorm_fn(tensors) \
                if isinstance(tensors, ndarray) or isinstance(tensors, Tensor) \
                else tensors

    def postprocess(
            self,
            runner: Runner,
            data: Dict[str, ndarray]
    ):
        """
        Postprocessing function.

        Args:
            runner (Runner): Runner class
            data (Dict[str, ndarray]): Data to postprocess

        """
        tb_logger = self._get_tensorboard_logger(runner)

        for key, images in data.items():
            images = images[:self._max_images]
            image = torchvision.utils.make_grid(images, nrow=self._num_rows)
            tb_logger.add_image(key, image, runner.global_sample_step)

    def run(self, runner: Runner):
        """
        Run full pipeline of callback.

        Args:
            runner (Runner): Runner class

        """
        for key_set, key_values in {
            "input_keys": self.input_keys,
            "output_keys": self.output_keys
        }.items():
            if len(key_values) == 0:
                continue

            tensors: List[Tensor] = self.get_tensors(
                runner=runner,
                **{key_set: key_values}
            )

            if self._concat_images:
                tensors: Tensor = torch.cat(tensors, dim=3)

            tensors: Union[ndarray, List[ndarray]] = self.preprocess(tensors)

            if self._concat_images:
                data: Dict[str, ndarray] = self.process(
                    data={f"{'|'.join(key_values)}": tensors}
                )
            else:
                data: Dict[str, ndarray] = self.process(
                    data={k: v for k, v in zip(key_values, tensors)}
                )

            self.postprocess(runner=runner, data=data)

        self._loader_processed_in_current_epoch = True


class MaskVisualizationCallback(VisualizationCallback):
    def __init__(
            self,
            mask_key: str,
            image_key: str = None,
            threshold: float = 0.5,
            mode: str = "image_with_mask",
            mask_strength: float = 0.5,
            *args,
            **kwargs
    ):
        super(MaskVisualizationCallback, self).__init__(
            input_keys=image_key,
            output_keys=mask_key,
            *args,
            **kwargs
        )

        assert mode in ["mask", "image_with_mask"]

        if mode == "image_with_mask" and image_key is None:
            raise ValueError("Image key should be passed if 'image_with_mask' "
                             "was chosen")

        self.mask_strength = mask_strength
        self.mode = mode

        self.mask_denorm_fn = partial(encode_mask_with_color,
                                      threshold=threshold)

    def preprocess_mask(
            self,
            tensors: Union[Tensor, List[Tensor]]
    ) -> ndarray:
        """

        Args:
            tensors:

        Returns:

        """
        return VisualizationCallback._detach(self.mask_denorm_fn(tensors))

    def process(
            self,
            tensors: Dict[str, ndarray]
    ) -> Dict[str, ndarray]:
        """

        Args:
            tensors:

        Returns:

        """
        return super(MaskVisualizationCallback, self).process({
            "mask_overlay_to_image": mask_to_overlay_image(
                image=tensors[self.input_keys[0]],
                mask=tensors[self.output_keys[0]],
                mask_strength=self.mask_strength
            )
        })

    def visualize(self, runner: Runner):
        """

        Args:
            runner:

        Returns:

        """
        if self.mode == "image_with_mask":
            image_tensor: List[Tensor] = \
                self.get_tensors(runner, self.input_keys)
            mask_tensor: List[Tensor] = \
                self.get_tensors(runner, self.output_keys)

            if self._concat_images:
                image_tensor: Tensor = torch.cat(image_tensor, dim=3)
                mask_tensor: Tensor = torch.cat(mask_tensor, dim=3)

            image_tensor: ndarray = self.preprocess(image_tensor)
            mask_tensor: ndarray = self.preprocess_mask(mask_tensor)

            visualizations = self.process({
                k: v for k, v in zip(self.input_keys + self.output_keys,
                                     image_tensor + mask_tensor)}
            )

            self.postprocess(runner, visualizations)

            self._loader_visualized_in_current_epoch = True
        else:
            super(MaskVisualizationCallback, self).run(runner)


class ProjectorCallback(VisualizationCallback):

    def __init__(
            self,
            image_key: str,
            labels_key: str,
            embeddings_key: str,
            tag: str,
            *args,
            **kwargs,
    ):
        """

        Args:
            image_key (str): Image key to build sprite.
            embeddings_key (str): Embeddings key to save to projector.
        """
        super(ProjectorCallback, self).__init__(
            input_keys=[image_key, labels_key],
            output_keys=embeddings_key,
            *args,
            **kwargs
        )

        self.tag = tag

        self.image_key = image_key
        self.labels_key = labels_key
        self.embeddings_key = embeddings_key

    def postprocess(
            self,
            runner: Runner,
            data: Dict[str, ndarray]
    ):
        """
        Postprocessing function.

        Args:
            runner (Runner): Runner class
            data (Dict[str, ndarray]): Data to postprocess

        """
        tb_logger: SummaryWriter = self._get_tensorboard_logger(runner)

        images = data[self.image_key]
        labels: List = data[self.labels_key]
        embeddings = data[self.embeddings_key]

        tb_logger.add_embedding(
            tag=self.tag,
            mat=embeddings,
            metadata=labels,
            label_img=images,
            global_step=runner.global_batch_step,
        )

    def run(self, runner: Runner):
        """
        Run full pipeline of callback.

        Args:
            runner (Runner): Runner class

        """
        data: Dict[str, ndarray] = {}
        for key_set, key_values in {
            "input_keys": self.input_keys,
            "output_keys": self.output_keys
        }.items():
            if len(key_values) == 0:
                continue

            tensors: List[Tensor] = self.get_tensors(
                runner=runner, **{key_set: key_values}
            )

            tensors: List[ndarray] = self.preprocess(tensors=tensors)

            data.update(
                self.process(
                    data={k: v for k, v in zip(key_values, tensors)}
                )
            )

        self.postprocess(runner=runner, data=data)

        self._loader_processed_in_current_epoch = True


__all__ = [
    "ProjectorCallback",
    "VisualizationCallback",
    "MaskVisualizationCallback"
]
