import os
import cv2
import random
import collections
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, \
    MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, \
    StratifiedKFold

# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

IMAGE_HEIGHT = 137
IMAGE_WIDTH = 236

IMAGE_SIZE = 128

SEED = 69
TEST_SIZE = 0.1

W1, W2, W3 = .5, .25, .25
METRIC_WEIGHTS = [W1, W2, W3]

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
CONSONANT_OUTPUT_KEY = "consonant_diacritic_pred"

CUTMIX_LAMBDA_KEY = "cutmix_lambda"
CUTMIX_GRAPHEME_KEY = "cutmix_grapheme_root"
CUTMIX_VOWEL_KEY = "cutmix_vowel_diacritic"
CUTMIX_CONSONANT_KEY = "cutmix_grapheme_root"

INPUT_KEYS = [GRAPHEME_INPUT_KEY, VOWEL_INPUT_KEY, CONSONANT_INPUT_KEY]
OUTPUT_KEYS = [GRAPHEME_OUTPUT_KEY, VOWEL_OUTPUT_KEY, CONSONANT_OUTPUT_KEY]
CUTMIX_KEYS = [CUTMIX_GRAPHEME_KEY, CUTMIX_VOWEL_KEY, CUTMIX_CONSONANT_KEY]

NUM_CLASSES = [168, 11, 7]


class BengaliDataset(Dataset):
    PARQUET, NPY, JPEG = "parquet", "npy", "jpeg"

    def __init__(self,
                 images,
                 labels=None,
                 transforms=None,
                 target_to_use=None,
                 use_original=False,
                 to_one_hot=False):

        self.images = images

        if isinstance(images, pd.DataFrame):
            self.load_from = BengaliDataset.PARQUET
            self.image_ids = self.images.iloc[:, 0].values
            self.images = self.images.iloc[:, 1:].values
        elif len(images) > 0 and images[0].endswith(BengaliDataset.NPY):
            self.load_from = BengaliDataset.NPY
        elif len(images) > 0 and images[0].endswith(BengaliDataset.JPEG):
            self.load_from = BengaliDataset.JPEG
        else:
            raise NotImplementedError("Something wrong with data format")

        self.labels = labels
        self.transforms = transforms
        self.use_original = use_original
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
    def rand_bbox(lam, height, width):
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(width * cut_rat)
        cut_h = np.int(height * cut_rat)

        # uniform
        cx = np.random.randint(width)
        l = np.clip(cx - cut_w // 2, 0, width)
        r = np.clip(cx + cut_w // 2, 0, width)

        cy = np.random.randint(height)
        t = np.clip(cy - cut_h // 2, 0, height)
        b = np.clip(cy + cut_h // 2, 0, height)

        return l, t, r, b

    def get_item(self, idx):
        if self.load_from == BengaliDataset.PARQUET:
            image_id = self.image_ids[idx]
            image = 255 - self.images[idx].reshape(IMAGE_HEIGHT,
                                                   IMAGE_WIDTH).astype(np.uint8)
            image = (image * (255.0 / image.max())).astype(np.uint8)
            if not self.use_original:
                image = BengaliDataset._crop_resize(image)

        elif self.load_from == BengaliDataset.NPY:
            path = self.images[idx]
            image_id = os.path.basename(path).split('.')[0]
            image = np.load(path)

        elif self.load_from == BengaliDataset.JPEG:
            path = self.images[idx]
            image_id = os.path.basename(path).split('.')[0]
            image = self._readimg(path)
        else:
            raise ValueError("Unknown type of data. "
                             "`parquet`, `npy` or `jpeg` formats only!")

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


class CutMixDataset(Dataset):
    def __init__(self,
                 dataset,
                 image_key,
                 input_keys,
                 num_classes,
                 num_mix=1,
                 beta=1.,
                 prob=1.0):

        self.dataset = dataset
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

        self.image_key = image_key
        self.input_keys = input_keys
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample = self.dataset[index]
        size = sample[self.image_key].size()
        height, width = size[-2], size[-1]

        for _ in range(self.num_mix):
            if self.beta <= 0 or np.random.rand(1) > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.randint(a=0, b=len(self) - 1)

            sample_ = self.dataset[rand_index]

            l, t, r, b = BengaliDataset.rand_bbox(lam, height, width)

            sample[self.image_key][:, t:b, l:r] = \
                sample_[self.image_key][:, t:b, l:r]

            sample[CUTMIX_LAMBDA_KEY] = float((r - l) * (b - t)) / (
                        height * width)

            for cutmix_key, input_key in zip(CUTMIX_KEYS, INPUT_KEYS):
                sample[cutmix_key] = sample_[input_key]

            # for key, num_classes in zip(self.input_keys, self.num_classes):
            #     sample[key] = lam * BengaliDataset.to_one_hot(sample[key], num_classes) + \
            #                 (1. - lam) * BengaliDataset.to_one_hot(sample_[key], num_classes)
        return sample

    def __len__(self):
        return len(self.dataset)


def get_parquet_paths(dataset_path, name, files_to_load=None):
    if files_to_load is None:
        files_to_load = [0, 1, 2, 3]
    return [os.path.join(dataset_path, f"{name}_image_data_{i}.parquet")
            for i in files_to_load]


def get_dataframe(dataset_path, name, files_to_load=None):
    paths = get_parquet_paths(dataset_path, name, files_to_load)
    if len(paths) == 1:
        return pd.read_parquet(paths[0])
    else:
        return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def get_data(dataset_path,
             labels,
             subset,
             load_from,
             files_to_load):
    if load_from == BengaliDataset.PARQUET:
        data = get_dataframe(dataset_path, subset, files_to_load)
    elif load_from == BengaliDataset.NPY or load_from == BengaliDataset.JPEG:
        data = labels["image_id"].values
    else:
        raise ValueError("Unknown type of data. "
                         "`parquet`, `npy` or `jpeg` formats only!")
    return data


def get_csv(dataset_path, name):
    return pd.read_csv(os.path.join(dataset_path, f"{name}.csv"))


def get_datasets(
        dataset_path,
        transforms=None,
        target_to_use=None,
        test_size=None,
        train_only=False,
        test_only=False,
        use_original=False,
        load_from=None,
        files_to_load=None,
        to_one_hot=False,
        stratification="sklearn_stratified",
        num_folds=None,
        fold=None,
        cutmix=False
):
    datasets = collections.OrderedDict()

    labels = get_csv(dataset_path, name="test" if test_only else "train")

    if transforms is None:
        transforms = {"train": None, "valid": None, "test": None}

    if train_only and test_only:
        raise NotImplementedError("train_only can't be used together with "
                                  "test_only")

    if test_only:
        datasets["test"] = get_data(dataset_path=dataset_path,
                                    labels=labels,
                                    subset="test",
                                    load_from=load_from,
                                    files_to_load=files_to_load)
    if train_only:
        datasets["train"] = get_data(dataset_path=dataset_path,
                                     labels=labels,
                                     subset="train",
                                     load_from=load_from,
                                     files_to_load=files_to_load)

        datasets["valid"] = get_data(dataset_path=dataset_path,
                                     labels=labels,
                                     subset="train",
                                     load_from=load_from,
                                     files_to_load=files_to_load)
    else:
        labels['label'] = labels[INPUT_KEYS].apply(tuple, axis=1)
        labels['label'] = pd.factorize(labels['label'])[0] + 1

        images = get_data(dataset_path=dataset_path,
                          labels=labels,
                          subset="train",
                          load_from=load_from,
                          files_to_load=files_to_load)

        if test_size is not None:
            if stratification == "sklearn_random":
                print("Using train_test_split...")

                datasets["train"], datasets["valid"] = \
                    train_test_split(images,
                                     test_size=test_size,
                                     random_state=SEED,
                                     shuffle=True)

            elif stratification == "sklearn_stratified":
                if fold is None:
                    print("Using StratifiedShuffleSplit...")

                    splitter = StratifiedShuffleSplit(n_splits=1,
                                                      test_size=test_size,
                                                      random_state=SEED)
                    train_indcs, valid_indcs = next(splitter.split(
                        X=images, y=labels[INPUT_KEYS].values))

                    datasets["train"], datasets["valid"] = \
                        images[train_indcs], images[valid_indcs]
                elif isinstance(num_folds, int) and isinstance(fold, int):
                    print(f"StratifiedKFold with {fold} of {num_folds} folds")

                    splitter = StratifiedKFold(n_splits=num_folds,
                                               shuffle=True,
                                               random_state=SEED)
                    for i, (train_indcs, valid_indcs) in enumerate(
                            splitter.split(X=images, y=labels['label'].values)):
                        if i == fold:
                            datasets["train"], datasets["valid"] = \
                                images[train_indcs], images[valid_indcs]

                            check_stratification(labels.loc[train_indcs],
                                                 labels.loc[valid_indcs])
                            break
                else:
                    raise NotImplementedError("Folding not implemented yet.")
            else:
                raise NotImplementedError(
                    f"{stratification} method not implemented")
        else:
            datasets["train"] = images

    if load_from != BengaliDataset.PARQUET and use_original:
        img_dir = "original_npy" \
            if load_from == BengaliDataset.NPY else "original"
    else:
        if load_from == BengaliDataset.NPY:
            raise NotImplementedError("Combination of not original images "
                                      "and npy format is not implemented yet.")
        else:
            img_dir = "images"

    for key, dataset in datasets.items():
        if load_from != BengaliDataset.PARQUET:
            dataset = [os.path.join(dataset_path, img_dir, "train",
                                    f"{i}.{load_from}") for i in dataset]

        datasets[key] = BengaliDataset(dataset,
                                       None if key == "test" else labels,
                                       transforms=transforms[key],
                                       target_to_use=target_to_use,
                                       use_original=use_original,
                                       to_one_hot=to_one_hot)
        if cutmix and key == "train":
            datasets[key] = CutMixDataset(
                dataset=datasets[key],
                image_key=IMAGE_KEY,
                input_keys=INPUT_KEYS,
                num_classes=NUM_CLASSES
            )
    return datasets


def get_loaders(
        dataset_path,
        batch_size: int = 64,
        num_workers: int = 4,
        transforms=None,
        shuffle=False,
        test_size=None,
        target_to_use=None,
        train_only=False,
        test_only=False,
        use_original=False,
        load_from=None,
        files_to_load=None,
        to_one_hot=False,
        stratification="sklearn_stratified",
        num_folds=None,
        fold=None
):
    if target_to_use is None:
        target_to_use = [0, 1, 2]

    datasets = get_datasets(dataset_path=dataset_path,
                            transforms=transforms,
                            target_to_use=target_to_use,
                            test_size=test_size,
                            train_only=train_only,
                            test_only=test_only,
                            use_original=use_original,
                            load_from=load_from,
                            files_to_load=files_to_load,
                            to_one_hot=to_one_hot,
                            stratification=stratification,
                            num_folds=num_folds,
                            fold=fold)

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


def count(df):
    return df.groupby(INPUT_KEYS).size().reset_index().rename(
        columns={0: "size"})


def get_filter(dataset_path):
    return count(get_csv(dataset_path, name="train"))


def check_stratification(train, valid):
    train_count, val_count = count(train), count(valid)

    total = train_count["size"] + val_count["size"]
    train_part = train_count["size"] / total
    val_part = val_count["size"] / total
    relative = val_part / train_part

    for k, v in {"Train": train_part, "Valid": val_part,
                 "Valid relative to train": relative}.items():
        print("---------------------------------------------------------------")
        print(k)
        print(v)
        print(",".join([f"{m}: {f(v):.2}"
                        for m, f in {"min": np.min, "max": np.max,
                                     "mean": np.mean, "std": np.std}.items()]))
        print("---------------------------------------------------------------")
