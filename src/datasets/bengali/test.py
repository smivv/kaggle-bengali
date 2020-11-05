import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification

DATASET_PATH = "/workspace/Datasets/BENGALI/"
KEYS = ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]
SEED = 69


def get_csv(dataset_path, name):
    return pd.read_csv(os.path.join(dataset_path, f"{name}.csv"))


def count(df):
    return df.groupby(KEYS).size().reset_index().rename(columns={0: "size"})


def split(dataset_path, test_size, stratification):
    df = get_csv(dataset_path, name="train")
    img_ids = df["image_id"]

    if stratification == "sklearn":
        train_set, valid_set = train_test_split(df[KEYS], test_size=test_size,
                                                random_state=SEED, shuffle=True)
    elif stratification == "sklearn_stratified":

        df['subset'] = np.nan
        splitter = StratifiedShuffleSplit(n_splits=1,
                                          test_size=test_size,
                                          random_state=SEED)

        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()

        df.iloc[train_indcs, -1] = 'train'
        df.iloc[valid_indcs, -1] = 'valid'

        df.to_csv(os.path.join(dataset_path, 'train_stratified.csv'), index=None)

    elif stratification == "iterstrat":

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1,
                                                    test_size=test_size,
                                                    random_state=SEED)

        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()

    elif stratification == "skmultilearn":

        splitter = IterativeStratification(n_splits=2, order=2,
                                           sample_distribution_per_fold=[
                                               test_size, 1.0 - test_size])

        train_indcs, valid_indcs = next(splitter.split(X=img_ids, y=df[KEYS]))
        train_set = df.loc[df.index.intersection(train_indcs)].copy()
        valid_set = df.loc[df.index.intersection(valid_indcs)].copy()

    else:
        raise ValueError("Try something else :)")

    return train_set, valid_set


method = "sklearn_stratified"

start = time.time()

train, valid = split(dataset_path=DATASET_PATH,
                     test_size=0.1,
                     stratification=method)

print(f"Dataset split done for {time.time() - start} seconds")

train_count, val_count = count(train), count(valid)

print(train_count, val_count)

total = train_count["size"] + val_count["size"]
train_part = train_count["size"] / total
val_part = val_count["size"] / total
relative = val_part / train_part


print(relative)
print(f"Relative min: {np.min(relative)}, Relative max: {np.max(relative)}, Relative mean: {np.mean(relative)}, Relative std: {np.std(relative)}")
print(f"Train min: {np.min(train_part)}, Train max: {np.max(train_part)}, Train mean: {np.mean(train_part)}, Train std: {np.std(train_part)}")
print(f"Valid min: {np.min(val_part)}, Valid max: {np.max(val_part)}, Valid mean: {np.mean(val_part)}, Valid std: {np.std(val_part)}")
