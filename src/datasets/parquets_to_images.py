import os
import cv2
import tqdm

import numpy as np
import pandas as pd

from multiprocessing.pool import ThreadPool


join = os.path.join

TEST_SIZE = 0.25

IMAGE_HEIGHT = 137
IMAGE_WIDTH = 236

DATASET_PATH = "/workspace/Datasets/BENGALI"
OUTPUT_PATH = join(DATASET_PATH, "original")


verbose = False
num_threads = 1

bar = tqdm.tqdm(total=200840)
files = [0, 1, 2, 3]


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
        return pd.concat([pd.read_parquet(p) for p in paths],
                         ignore_index=True)


def extract(fileindex):

    df = get_dataframe(DATASET_PATH, "train", [fileindex])

    images = df.iloc[:, 1:].values
    image_ids = df.iloc[:, 0].values

    print(f"Processing {fileindex}-th file for {subset}..")

    for idx, row in df.iterrows():

        filename = image_ids[idx]
        image = 255 - images[idx].reshape(IMAGE_HEIGHT,
                                          IMAGE_WIDTH).astype(np.uint8)
        image = (image * (255.0 / image.max())).astype(np.uint8)

        cv2.imwrite(join(OUTPUT_PATH, subset, f"{filename}.jpeg"), image)

        bar.update(1)

    del df, images, image_ids


for subset in ["train"]:

    if not os.path.exists(join(OUTPUT_PATH, subset)):
        os.makedirs(join(OUTPUT_PATH, subset), exist_ok=True)

    for fileindex in files:

        extract(fileindex)
