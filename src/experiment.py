import torch
import torch.nn as nn

from typing import Union, List, Dict

from catalyst.data.augmentor import Augmentor
from catalyst.dl import ConfigExperiment

from .datasets.bengali import get_datasets
from .transforms.bengali import get_transforms


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            for param in model_.backbone.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model_.backbone.parameters():
                param.requires_grad = True

        return model_

    @staticmethod
    def get_transforms(
        stage: str = None,
        mode: str = None,
        image_height: int = 224,
        image_width: int = 224,
        one_hot_classes: int = None
    ):
        result_fn = get_transforms(image_height=image_height,
                                   image_width=image_width)[mode]
        return Augmentor(
            dict_key="image", augment_fn=lambda x: result_fn(image=x)["image"]
        )

    def get_datasets(
        self,
        stage: str,
        dataset_path: str = None,
        target_to_use: List = None,
        image_height: int = 224,
        image_width: int = 224,
        test_size: float = 0.2,
        test_only: bool = False,
        use_original: bool = False,
        load_from: str = None,
        files_to_load: List = None,
        to_one_hot: bool = False,
        num_folds: int = None,
        fold: int = None,
        stratification: str = "sklearn_stratified",
    ):
        return get_datasets(
            dataset_path=dataset_path,
            transforms=get_transforms(image_height=image_height,
                                      image_width=image_width),
            target_to_use=target_to_use,
            test_size=test_size,
            test_only=test_only,
            use_original=use_original,
            load_from=load_from,
            files_to_load=files_to_load,
            to_one_hot=to_one_hot,
            num_folds=num_folds,
            fold=fold,
            stratification=stratification,
        )
