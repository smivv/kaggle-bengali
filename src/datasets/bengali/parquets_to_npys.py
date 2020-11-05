import os
import tqdm

import numpy as np

from src.datasets.bengali import get_dataframe, IMAGE_HEIGHT, IMAGE_WIDTH

join = os.path.join

TEST_SIZE = 0.25

DATASET_PATH = "/workspace/Datasets/BENGALI"
OUTPUT_PATH = join(DATASET_PATH, "original_npy")

for fileindex in [0, 1, 2, 3]:
    for subset in ["train"]:
        if not os.path.exists(join(OUTPUT_PATH, subset)):
            os.makedirs(join(OUTPUT_PATH, subset), exist_ok=True)

        df = get_dataframe(DATASET_PATH, "train", [fileindex])

        images = df.iloc[:, 1:].values
        image_ids = df.iloc[:, 0].values

        print(f"Processing {fileindex}-th file for {subset}..")
        bar = tqdm.tqdm(total=len(df))

        for idx, row in df.iterrows():

            filename = image_ids[idx]
            image = 255 - images[idx].reshape(IMAGE_HEIGHT,
                                              IMAGE_WIDTH).astype(np.uint8)
            image = (image * (255.0 / image.max())).astype(np.uint8)

            np.save(join(OUTPUT_PATH, subset, f"{filename}.npy"), image)
            bar.update(1)
