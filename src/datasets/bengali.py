import os
import cv2
import random
import collections
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

IMAGE_HEIGHT = 137
IMAGE_WIDTH = 236
IMAGE_SIZE = 128

SEED = 69
TEST_SIZE = 0.1

W1, W2, W3 = .5, .25, .25
METRIC_WEIGHTS = [W1, W2, W3]

LOAD_FROM_PARQUETS = True

BATCH_SIZE = 16
NUM_WORKERS = 4

IMAGE_KEY = "image"

GRAPHEME_INPUT_KEY = "grapheme_root"
VOWEL_INPUT_KEY = "vowel_diacritic"
CONSONANT_INPUT_KEY = 'consonant_diacritic'

GRAPHEME_KEY = GRAPHEME_INPUT_KEY
VOWEL_KEY = VOWEL_INPUT_KEY
CONSONANT_KEY = CONSONANT_INPUT_KEY

GRAPHEME_OUTPUT_KEY = "grapheme_root_pred"
VOWEL_OUTPUT_KEY = "vowel_diacritic_pred"
CONSONANT_OUTPUT_KEY = 'consonant_diacritic_pred'

INPUT_KEYS = (GRAPHEME_INPUT_KEY, VOWEL_INPUT_KEY, CONSONANT_INPUT_KEY)
OUTPUT_KEYS = (GRAPHEME_OUTPUT_KEY, VOWEL_OUTPUT_KEY, CONSONANT_OUTPUT_KEY)

NUM_CLASSES = [168, 11, 7]


class BengaliDataset(Dataset):

    def __init__(self,
                 images,
                 labels=None,
                 transforms=None,
                 target_to_use=None,
                 use_original=False,
                 use_parquet=False,
                 to_one_hot=False):

        self.images = images

        if use_parquet:
            self.image_ids = self.images.iloc[:, 0].values
            self.images = self.images.iloc[:, 1:].values

        self.labels = labels
        self.transforms = transforms
        self.use_original = use_original
        self.use_parquet = use_parquet
        self.target_to_use = target_to_use
        self.to_one_hot = to_one_hot

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

    def get_item(self, idx):
        if self.use_parquet:
            image_id = self.image_ids[idx]
            image = 255 - self.images[idx].reshape(IMAGE_HEIGHT,
                                                   IMAGE_WIDTH).astype(np.uint8)
            image = (image * (255.0 / image.max())).astype(np.uint8)
            if not self.use_original:
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
            result[IMAGE_KEY] = self.transforms(**result)[IMAGE_KEY]

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
        return self.get_item(idx)


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
        use_original=False,
        use_parquet=False,
        files_to_load=None,
        to_one_hot=False,
        stratified=False
):
    datasets = collections.OrderedDict()

    if transforms is None:
        transforms = {"train": None, "valid": None, "test": None}

    if test_only:
        if use_parquet:
            test_images = get_data(dataset_path, "test", files_to_load)
        else:
            img_dir = os.path.join(dataset_path,
                                   "original" if use_original else "images",
                                   "test")
            test_images = [os.path.join(img_dir, p)
                           for p in os.listdir(img_dir)]

        datasets["test"] = BengaliDataset(test_images,
                                          transforms=transforms["test"],
                                          target_to_use=target_to_use,
                                          use_original=use_original,
                                          use_parquet=use_parquet,
                                          to_one_hot=to_one_hot)
        del test_images, transforms
        return datasets

    if use_parquet:
        train_images = get_data(dataset_path, "train", files_to_load)
    else:
        img_dir = os.path.join(dataset_path,
                               "original" if use_original else "images",
                               "train")
        train_images = [os.path.join(img_dir, p) for p in os.listdir(img_dir)]

    labels = get_csv(dataset_path, name="train")

    if test_size is not None:
        if stratified:
            subsets = get_csv(dataset_path, name="train_val")

            if use_parquet:
                merged = pd.merge(labels, subsets, on="image_id")
                datasets["train"] = merged.loc[merged['subset'] == "train"]
                datasets["valid"] = merged.loc[merged['subset'] == "valid"]
            else:
                img_dir = os.path.join(dataset_path,
                                       "original" if use_original else "images",
                                       "train")
                train = subsets.loc[subsets['subset'] == "train"]
                valid = subsets.loc[subsets['subset'] == "valid"]

                datasets["train"] = [os.path.join(img_dir, p["image_id"] + ".jpeg")
                                     for i, p in train.iterrows()]
                datasets["valid"] = [os.path.join(img_dir, p["image_id"] + ".jpeg")
                                     for i, p in valid.iterrows()]
            # splitter = MultilabelStratifiedShuffleSplit(n_splits=1,
            #                                             test_size=test_size,
            #                                             random_state=SEED)
            # for train_indcs, valid_indcs in splitter.split(
            #         X=train_images, y=labels[INPUT_KEYS].values):
            #     datasets["train"], datasets["valid"] = \
            #         [train_images[i] for i in train_indcs], \
            #         [train_images[i] for i in valid_indcs]
            #
            # del splitter
        else:
            datasets["train"], datasets["valid"] = \
                train_test_split(train_images, test_size=test_size,
                                 random_state=SEED, shuffle=True)

    else:
        datasets["train"] = train_images

    for key, dataset in datasets.items():
        datasets[key] = BengaliDataset(dataset, labels,
                                       transforms=transforms[key],
                                       target_to_use=target_to_use,
                                       use_original=use_original,
                                       use_parquet=use_parquet,
                                       to_one_hot=to_one_hot)
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
        use_original=False,
        use_parquet=False,
        files_to_load=None,
        to_one_hot=False,
):
    if target_to_use is None:
        target_to_use = [0, 1, 2]

    datasets = get_datasets(dataset_path=dataset_path,
                            transforms=transforms,
                            target_to_use=target_to_use,
                            test_size=test_size,
                            test_only=test_only,
                            use_original=use_original,
                            use_parquet=use_parquet,
                            files_to_load=files_to_load,
                            to_one_hot=to_one_hot)

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
