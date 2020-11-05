import torch
import torch.nn as nn

from catalyst.dl import ConfigExperiment
from catalyst.data.sampler import BalanceClassSampler

from src.datasets.cico import get_datasets
from .transforms.cico import get_transforms


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

    def get_datasets(
            self,
            stage: str,
            *args,
            **kwargs
    ):
        kwargs["transforms"] = get_transforms(kwargs["image_size"])

        datasets = get_datasets(*args, **kwargs)

        datasets["train"] = {
            "dataset": datasets["train"],
            "sampler": BalanceClassSampler(
                labels=[datasets["train"].get_label(i)
                        for i in range(len(datasets["train"]))],
                mode="upsampling",
            )
        }

        if stage.startswith("infer"):
            datasets["infer_train"] = datasets["train"]
            del datasets["train"]

            if "doe" in stage:
                datasets["infer_valid"] = datasets["valid"]
                del datasets["valid"]

        return datasets
