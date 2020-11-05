import numpy as np
import pandas as pd

from ast import literal_eval


def rand_bbox(lam, width, height):
    cut_rat = np.sqrt(1. - lam)
    cut_w = (width * cut_rat).astype(np.int)
    cut_h = (height * cut_rat).astype(np.int)
    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    left = np.clip(cx - cut_w // 2, 0, width)
    right = np.clip(cx + cut_w // 2, 0, width)
    top = np.clip(cy - cut_h // 2, 0, height)
    bottom = np.clip(cy + cut_h // 2, 0, height)
    return left, top, right, bottom


def to_one_hot(label, num_classes):
    oh = np.zeros(num_classes, dtype=np.long)
    oh[label] = 1
    return oh


def build_index(path):
    df = pd.read_excel(path, sheet_name='Benchmarking')
    df = df[df['Utilized'] == 1]
    df = df.loc[df['Test case'].notnull()]
    df['Test Set'] = df['Test Set'].apply(lambda x: literal_eval(x))

    index = {}
    for i, row in df.iterrows():
        cat, subcat, var = row['Category'], row['Subcategory'], row['Variant']
        for cam in row['Test Set']:
            index[f"{cat}_{subcat}_{var}_{cam}"] = row['Test case']

    return index
