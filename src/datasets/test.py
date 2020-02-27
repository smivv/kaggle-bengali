import os
import cv2
import time
import tqdm
import random
import collections
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification

DATASET_PATH = "/workspace/Datasets/BENGALI/"
KEYS = ("grapheme_root", "vovew_diacritic", "consonant_diacritic")
SEED = 69


def get_csv(dataset_path, name):
    return pd.read_csv(os.path.join(dataset_path, f"{name}.csv"))


def count(df):
    return df.groupby(KEYS).size().reset_index().rename(columns={0: "size"})


def test(dataset_path, test_size, stratification):
    df = get_csv(dataset_path, name="train")
    img_ids = df["image_id"]

    if stratification == 1:

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1,
                                                    test_size=test_size,
                                                    random_state=SEED)

        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()

    elif stratification == 2:
        df["subset"] = np.nan
        stratifier = IterativeStratification(n_splits=2, order=1,
                                             sample_distribution_per_fold=[
                                                 1.0 - test_size, test_size])

        for train_indcs, valid_indcs in stratifier.split(X=img_ids, y=df[KEYS]):
            train_set = df.loc[df.index.intersection(train_indcs)].copy()
            valid_set = df.loc[df.index.intersection(valid_indcs)].copy()

    elif stratification == 0:
        train_set, valid_set = train_test_split(df[KEYS], test_size=test_size,
                                                random_state=SEED, shuffle=True)
    else:
        raise ValueError("Try something else :)")

    return train_set, valid_set


for s in [0, 1, 2]:
    start = time.time()
    df_t, df_v = test(stratified=s)
    print(f"Done for {time.time() - start} seconds")
    cdf_t, cdf_v = count(df_t), count(df_v)
    cdf_s = cdf_t.copy()
    cdf_s["size"] = (cdf_t["size"] + cdf_v["size"]) / cdf_v["size"]
    print(cdf_t)
    print(cdf_v)
    print(cdf_s)
    print(np.min(cdf_s["size"].values), np.max(cdf_s["size"].values))

df = pd.read_csv(os.path.join(DATASET_PATH, 'train_val.csv'))
print(len(df.loc[df['subset'] == "train"]),
      len(df.loc[df['subset'] == "valid"]), )
# df.loc[df.subset == 'train', 'subset'] = "tmp"
# df.loc[df.subset == 'valid', 'subset'] = "train"
# df.loc[df.subset == 'tmp', 'subset'] = "valid"
# df.to_csv(os.path.join(DATASET_PATH, "train_val.csv"), index=False)
