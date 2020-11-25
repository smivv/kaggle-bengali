from typing import Tuple, List, Dict, Optional, Union, Any, Callable

import os
import cv2
import random
import collections
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE = 224

SEED = 69
TEST_SIZE = 0.1

INPUT_FILENAME_KEY = "filename"
INPUT_IMAGE_KEY = "image"
INPUT_TARGET_KEY = "target"
INPUT_LABEL_KEY = "label"

OUTPUT_EMBEDDINGS_KEY = "embeddings"
OUTPUT_TARGET_KEY = "logits"

INPUT_KEYS = [INPUT_IMAGE_KEY, INPUT_TARGET_KEY]
OUTPUT_KEYS = [OUTPUT_TARGET_KEY]


def _read_img(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR)
    return img


def to_one_hot(label, num_classes):
    b = np.zeros(num_classes, dtype=np.long)
    b[label] = 1
    return b


class CICODataset(Dataset):

    def __init__(
            self,
            images: List[Union[str, Path]],
            label_ids: Dict[str, int],
            class_names: List[str],
            transforms: Callable = None,
            use_one_hot: bool = False,
    ):
        self.images = images
        self.label_ids = label_ids

        self.transforms = transforms
        self.use_one_hot = use_one_hot
        self.class_names = class_names
        self.num_classes = len(class_names)

    def __len__(self):
        # if len(self.images) > 1000:
        #     return 1000
        return len(self.images)

    def get_item(self, idx):
        path: Path = self.images[idx]

        image = _read_img(path.as_posix())

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = path.parent.name

        if label not in self.label_ids:
            target = 0
        else:
            target = self.label_ids[label]

        if self.use_one_hot:
            target = to_one_hot(target, self.num_classes)

        return {
            INPUT_FILENAME_KEY: path.stem,
            INPUT_IMAGE_KEY: image,
            INPUT_TARGET_KEY: target,
            INPUT_LABEL_KEY: label,
        }

    def get_label(self, idx):
        path: Path = self.images[idx]
        label = path.parent.name
        target = self.label_ids[label]
        return target

    def __getitem__(self, idx):
        return self.get_item(idx)


def get_data(data_path: Path) -> List[Path]:
    return list(data_path.glob("**/*.jpg"))


def get_df(df_path: Path):
    if df_path.suffix == ".csv":
        return pd.read_csv(df_path)
    else:
        return pd.read_excel(df_path)


def get_datasets(
        dataset_path: Union[str, Path],
        class_names: List[str],
        additional_paths: List[Union[str, Path]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        use_one_hot: bool = False,
        *args,
        **kwargs,
):
    if transforms is None:
        transforms = {"train": None, "valid": None, "test": None, "infer": None}

    dataset_path: Path = Path(dataset_path)
    train, valid = None, None

    if (dataset_path / "train").exists():
        train = get_data(dataset_path / "train")

    if (dataset_path / "test").exists():
        valid = get_data(dataset_path / "test")

    if train is not None and valid is not None:
        datasets = {
            "train": train,
            "valid": valid,
        }
    else:
        datasets = {
            "train": get_data(dataset_path)
        }

    label_ids = {name: i for i, name in enumerate(class_names)}

    for key, images in datasets.items():
        datasets[key] = CICODataset(
            images=images,
            label_ids=label_ids,
            transforms=transforms[key],
            use_one_hot=use_one_hot,
            class_names=class_names,
        )

    if additional_paths is not None:
        for path in additional_paths:
            path = Path(path)
            datasets["extra_" + path.name] = CICODataset(
                images=get_data(path),
                label_ids=label_ids,
                transforms=transforms["infer"],
                use_one_hot=use_one_hot,
                class_names=class_names,
            )

    return datasets


def get_loaders(
        dataset_path: Union[str, Path],
        num_classes: int,
        transforms: Dict[str, Callable] = None,
        use_one_hot: bool = False,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
):
    loaders = collections.OrderedDict()

    datasets = get_datasets(
        dataset_path=dataset_path,
        num_classes=num_classes,
        transforms=transforms,
        use_one_hot=use_one_hot,
    )

    for key, dataset in datasets.items():
        loaders[key] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False if key == "train" else shuffle,
        )

    return loaders


def get_labels(
        dataset_path: Union[str, Path],
        class_names: List[str],
        subset: str = "train"
):
    dataset_path: Path = Path(dataset_path)

    images = get_data(dataset_path / subset)

    label_ids = {name: i for i, name in enumerate(class_names)}

    labels = [label_ids[img.parent.name] for img in images]

    return labels
