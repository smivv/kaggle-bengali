import pandas as pd

from .bengali import get_parquet_paths, get_feather_paths


DATAPATH = "/workspace/Datasets/BENGALI"
FILES_TO_LOAD = [0, 1, 2, 3]


for fileindex in FILES_TO_LOAD:
    for subset in ["train", "test"]:

        from_path = get_parquet_paths(DATAPATH, subset, [fileindex])[0]
        to_path = get_feather_paths(DATAPATH, subset, [fileindex])[0]

        print(f"Converting from {from_path} to {to_path}")

        pd.read_parquet(from_path).to_feather(to_path)
