import os
import cv2
import random
import collections
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

IMAGE_HEIGHT = 137
IMAGE_WIDTH = 236
IMAGE_SIZE = 128

SEED = 69
TEST_SIZE = 0.2

W1, W2, W3 = .5, .25, .25
METRIC_WEIGHTS = [W1, W2, W3]

LOAD_FROM_PARQUETS = True

BATCH_SIZE = 16
NUM_WORKERS = 4

IMAGE_KEY = "image"

GRAPHEME_KEY = "grapheme_root"
VOWEL_KEY = "vowel_diacritic"
CONSONANT_KEY = 'consonant_diacritic'

INPUT_KEYS = [GRAPHEME_KEY, VOWEL_KEY, CONSONANT_KEY]
OUTPUT_KEYS = [f"{k}_pred" for k in INPUT_KEYS]

NUM_CLASSES = [168, 11, 7]


class BengaliDataset(Dataset):

    def __init__(self,
                 images,
                 labels=None,
                 transforms=None,
                 target_to_use=None,
                 use_parquet=False,
                 to_one_hot=False,
                 mix_to_use=None):

        self.images = images

        if use_parquet:
            self.image_ids = self.images.iloc[:, 0].values
            self.images = self.images.iloc[:, 1:].values

        self.labels = labels
        self.transforms = transforms
        self.use_parquet = use_parquet
        self.target_to_use = target_to_use
        self.to_one_hot = to_one_hot
        self.mix_to_use = mix_to_use

    def __len__(self):
        return len(self.images)

    def _readimg(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def to_one_hot(label, num_classes):
        b = np.zeros(num_classes, dtype=np.long)
        b[label] = 1
        return b

    @staticmethod
    def _bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    @staticmethod
    def _crop_resize(img, size=IMAGE_SIZE, pad=16):
        # crop a box around pixels large than the threshold
        # some images contain line at the sides
        ymin, ymax, xmin, xmax = BengaliDataset._bbox(img[5:-5, 5:-5] > 80)
        # cropping may cut too much, so we need to add it back
        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0
        xmax = xmax + 13 if (xmax < IMAGE_WIDTH - 13) else IMAGE_WIDTH
        ymax = ymax + 10 if (ymax < IMAGE_HEIGHT - 10) else IMAGE_HEIGHT
        img = img[ymin:ymax, xmin:xmax]
        # remove lo intensity pixels as noise
        img[img < 28] = 0
        lx, ly = xmax - xmin, ymax - ymin
        l = max(lx, ly) + pad
        # make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode="constant")
        return cv2.resize(img, (size, size))

    @staticmethod
    def rand_bbox(lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(IMAGE_WIDTH * cut_rat)
        cut_h = np.int(IMAGE_HEIGHT * cut_rat)

        # uniform
        cx = np.random.randint(IMAGE_WIDTH)
        cy = np.random.randint(IMAGE_HEIGHT)

        left = np.clip(cx - cut_w // 2, 0, IMAGE_WIDTH)
        top = np.clip(cy - cut_h // 2, 0, IMAGE_HEIGHT)
        right = np.clip(cx + cut_w // 2, 0, IMAGE_WIDTH)
        bottom = np.clip(cy + cut_h // 2, 0, IMAGE_HEIGHT)

        return left, top, right, bottom

    def cutmix(self, item, beta=1.0, p=0.2):
        if beta <= 0 or np.random.rand(1) > p:
            return item

        item2 = self.get_item(random.choice(range(len(self))))

        lam = np.random.beta(beta, beta)
        l, t, r, b = BengaliDataset.rand_bbox(lam)
        item[IMAGE_KEY][l:r, t:b] = item2[IMAGE_KEY][l:r, t:b]

        lam = 1 - ((r - l) * (b - t) / (IMAGE_HEIGHT * IMAGE_WIDTH))

        for i in self.target_to_use:
            item[INPUT_KEYS[i]] = item[INPUT_KEYS[i]] * lam + \
                                  item2[INPUT_KEYS[i]] * (1.0 - lam)

        return item

    def mixup(self, item, beta=1.0, p=0.2):
        if beta <= 0 or np.random.rand(1) > p:
            return item

        item2 = self.get_item(random.choice(range(len(self))))

        lam = np.random.beta(beta, beta)
        item[IMAGE_KEY] = item[IMAGE_KEY] * lam + item2[IMAGE_KEY] * (1. - lam)

        for i in self.target_to_use:
            item[INPUT_KEYS[i]] = item[INPUT_KEYS[i]] * lam + \
                                  item2[INPUT_KEYS[i]] * (1. - lam)

        return item

    def get_item(self, idx):
        if self.use_parquet:
            image_id = self.image_ids[idx]
            image = 255 - self.images[idx].reshape(IMAGE_HEIGHT,
                                                   IMAGE_WIDTH).astype(np.uint8)
            image = (image * (255.0 / image.max())).astype(np.uint8)
            image = BengaliDataset._crop_resize(image)
        else:
            path = self.images[idx]
            image_id = os.path.basename(path).split('.')[0]
            image = self._readimg(path)
        image = image[..., np.newaxis]

        result = {
            IMAGE_KEY: image,
        }

        if self.transforms:
            transformed = self.transforms(**result)
            result[IMAGE_KEY] = transformed[IMAGE_KEY]

        result["image_id"] = image_id

        if self.labels is None:
            return result

        labels = self.labels.loc[self.labels['image_id'] == image_id][
            INPUT_KEYS].values
        labels = labels.squeeze().tolist()

        for i in self.target_to_use:
            result[INPUT_KEYS[i]] = BengaliDataset.to_one_hot(
                labels[i], NUM_CLASSES[i]) if self.to_one_hot else labels[i]
        return result

    def __getitem__(self, idx):
        item = self.get_item(idx)
        if self.labels is not None and self.transforms is not None:
            if self.mix_to_use == "cutmix":
                item = self.cutmix(item)
            if self.mix_to_use == "mixup":
                item = self.mixup(item)
        return item


def get_parquet_paths(dataset_path, name, files_to_load=None):
    if files_to_load is None:
        files_to_load = [0, 1, 2, 3]
    return [os.path.join(dataset_path, f"{name}_image_data_{i}.parquet")
            for i in files_to_load]


def get_feather_paths(dataset_path, name, files_to_load=None):
    if files_to_load is None:
        files_to_load = [0, 1, 2, 3]
    return [os.path.join(dataset_path, f"{name}_image_data_{i}.feather")
            for i in files_to_load]


def get_data(dataset_path, name, files_to_load=None):
    if LOAD_FROM_PARQUETS:
        paths = get_parquet_paths(dataset_path, name, files_to_load)
        if len(paths) == 1:
            return pd.read_parquet(paths[0])
        else:
            return pd.concat([pd.read_parquet(p) for p in paths],
                             ignore_index=True)
    else:
        paths = get_feather_paths(dataset_path, name, files_to_load)
        if len(paths) == 1:
            return pd.read_feather(paths[0])
        else:
            return pd.concat([pd.read_feather(p) for p in paths],
                             ignore_index=True)


def get_csv(dataset_path, name):
    return pd.read_csv(os.path.join(dataset_path, f"{name}.csv"))


def get_datasets(
        dataset_path,
        transforms=None,
        target_to_use=None,
        test_size=None,
        test_only=False,
        use_parquet=False,
        files_to_load=None,
):
    datasets = collections.OrderedDict()

    if transforms is None:
        transforms = {"train": None, "valid": None, "test": None}

    if test_only:
        if use_parquet:
            test_images = get_data(dataset_path, "test", files_to_load)
        else:
            img_dir = os.path.join(dataset_path, "images", "test")
            test_images = [os.path.join(img_dir, p)
                           for p in os.listdir(img_dir)]

        datasets["test"] = BengaliDataset(test_images,
                                          transforms=transforms["test"],
                                          target_to_use=target_to_use,
                                          use_parquet=use_parquet)
        del test_images, transforms
        return datasets

    if use_parquet:
        train_images = get_data(dataset_path, "train", files_to_load)
    else:
        img_dir = os.path.join(dataset_path, "images", "train")
        train_images = [os.path.join(img_dir, p) for p in os.listdir(img_dir)]

    if test_size is not None:
        datasets["train"], datasets["valid"] = \
            train_test_split(train_images, test_size=test_size,
                             random_state=SEED, shuffle=True)
    else:
        datasets["train"] = train_images

    labels = get_csv(dataset_path, name="train")

    for key, dataset in datasets.items():
        datasets[key] = BengaliDataset(dataset, labels,
                                       transforms=transforms[key],
                                       target_to_use=target_to_use,
                                       use_parquet=use_parquet)
    del train_images, labels, transforms
    return datasets


def get_loaders(
        dataset_path,
        batch_size: int = 64,
        num_workers: int = 4,
        transforms=None,
        shuffle=False,
        test_size=None,
        target_to_use=None,
        test_only=False,
        use_parquet=False,
        files_to_load=None,
):
    datasets = get_datasets(dataset_path=dataset_path,
                            transforms=transforms,
                            target_to_use=target_to_use,
                            test_size=test_size,
                            test_only=test_only,
                            use_parquet=use_parquet,
                            files_to_load=files_to_load)

    loaders = collections.OrderedDict()

    for key, dataset in datasets.items():
        loaders[key] = DataLoader(dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle)
    del datasets
    return loaders


def get_num_classes(dataset_path):
    if NUM_CLASSES is not None:
        return NUM_CLASSES
    df = get_csv(dataset_path, "class_map")
    return [len(df.loc[df['component_type'] == k]) for k in INPUT_KEYS]


def get_filter(dataset_path):
    return get_csv(dataset_path, name="train") \
        .groupby(INPUT_KEYS).size().reset_index().rename(columns={0: 'size'})
