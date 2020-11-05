# flake8: noqa
# isort: skip_file
import cv2

import numpy as np

import torch
import torchvision.utils

from catalyst.dl import Callback, CallbackOrder, State
from catalyst.contrib.tools.tensorboard import SummaryWriter


class VisualizationCallback2(Callback):
    TENSORBOARD_LOGGER_KEY = "_tensorboard"

    def __init__(
        self,
        image_and_label_keys=None,
        batch_frequency=25,
        concat_images=True,
        max_images=20,
        num_rows=5,
        denorm="default"
    ):
        super(VisualizationCallback2, self).__init__(CallbackOrder.External)

        if isinstance(image_and_label_keys, str):
            self.image_and_label_keys = {image_and_label_keys: None}
        elif isinstance(image_and_label_keys, (tuple, list)):
            assert all(isinstance(k, str) for k in image_and_label_keys)
            self.image_and_label_keys = {k: None for k in image_and_label_keys}
        elif isinstance(image_and_label_keys, dict):
            assert all([isinstance(k, (str, tuple, list))
                        for k in image_and_label_keys.values()])
            self.image_and_label_keys = {
                k: list(v) for k, v in image_and_label_keys.items()}
        else:
            raise ValueError(
                f"Unexpected format of 'image_and_label_keys' "
                f"argument: must be string, list or dict"
            )

        self.batch_frequency = int(batch_frequency)
        assert self.batch_frequency > 0

        self.concat_images = concat_images
        self.max_images = max_images

        # y = (x - mean) / std => x = y * std + mean
        if denorm.lower() == "default":
            # normalization from [-1, 1] to [0, 1] (the latter is valid for tb)
            self.denorm = lambda x: x * 2 + .5
        elif denorm.lower() == "imagenet":
            # normalization from [-1, 1] to [0, 1] (the latter is valid for tb)
            self.denorm = lambda x: x * 0.225 + 0.449
        elif denorm is None or denorm.lower() == "none":
            self.denorm = lambda x: x
        else:
            raise ValueError("unknown denorm fn")
        self._num_rows = num_rows
        self._reset()

    def _reset(self):
        self._loader_batch_count = 0
        self._loader_visualized_in_current_epoch = False

    @staticmethod
    def _get_tensorboard_logger(state: State) -> SummaryWriter:
        tb_key = VisualizationCallback2.TENSORBOARD_LOGGER_KEY
        if (
            tb_key in state.callbacks
            and state.loader_name in state.callbacks[tb_key].loggers
        ):
            return state.callbacks[tb_key].loggers[state.loader_name]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}"
        )

    def _put_text(self, img, text):

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 10)
        font_color = (255, 255, 255)
        line_type = 2

        cv2.putText(img=img, text=text,
                    org=position, fontFace=font_face, fontScale=1,
                    color=font_color, lineType=line_type)

        return img

    def _detach(self, tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = np.asarray(tensor)
        return tensor

    def _chw2hwc(self, data: np.ndarray):
        return data.transpose((1, 2, 0))

    def _hwc2chw(self, data: np.ndarray):
        return data.transpose((2, 0, 1))

    def compute_visualizations(self, state):

        visualizations = dict()

        state_values = {**state.input, **state.output}

        for image_key, label_keys in self.image_and_label_keys.items():

            if image_key not in state_values:
                print(f"`{image_key}` not found!")

            tensors = self._detach(state_values[image_key])  # B x 1
            labels = [self._detach(state_values[k]) for k in label_keys]  # B x num(label_keys)
            name = "|".join(label_keys)

            for i in range(len(tensors)):
                label = "|".join(str(label[i]) for label in labels)
                tensor = self._chw2hwc(tensors[i])
                tensor = self.denorm(tensor)
                self._put_text(tensor, label)
                tensors[i] = self._hwc2chw(tensor)

            if self.concat_images:
                visualizations[name] = np.concatenate(tensors, axis=3)
            else:
                for i, (k, v) in enumerate(zip(label_keys, tensors)):
                    visualizations[k] = v

        return visualizations

    def save_visualizations(self, state, visualizations):
        tb_logger = self._get_tensorboard_logger(state)
        for key, batch_images in visualizations.items():
            batch_images = batch_images[:self.max_images]
            image = torchvision.utils.make_grid(
                batch_images, nrow=self._num_rows
            )
            tb_logger.add_image(key, image, global_step=state.global_sample_step)

    def visualize(self, state):
        visualizations = self.compute_visualizations(state)
        self.save_visualizations(state, visualizations)
        self._loader_visualized_in_current_epoch = True

    def on_loader_start(self, state: State):
        self._reset()

    def on_loader_end(self, state: State):
        if not self._loader_visualized_in_current_epoch:
            self.visualize(state)

    def on_batch_end(self, state: State):
        self._loader_batch_count += 1
        if self._loader_batch_count % self.batch_frequency:
            self.visualize(state)


__all__ = ["VisualizationCallback2"]