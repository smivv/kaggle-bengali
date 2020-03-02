import os
import cv2
import tqdm

import numpy as np
import pandas as pd

from .bengali import get_parquet_paths, BengaliDataset

join = os.path.join

TEST_SIZE = 0.25

DATAPATH = "/workspace/Datasets/BENGALI"



for fileindex in [0, 1, 2, 3]:
    for subset in ["train"]:
        df = pd.concat([
            pd.read_parquet(p) for p in get_parquet_paths(subset, fileindex)
        ], ignore_index=True)

        print(f"Processing {fileindex}-th file for {subset}..")
        bar = tqdm.tqdm(total=len(df))

        for index, row in df.iterrows():
            image = BengaliDataset.get_item(df, index)

            filename = row["image_id"]

            if not os.path.exists(join(DATAPATH, "original", subset)):
                os.makedirs(join(DATAPATH, "original", subset), exist_ok=True)

            cv2.imwrite(join(DATAPATH, "original", subset, f"{filename}.jpeg"), image)
            bar.update(1)
