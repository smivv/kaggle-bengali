import numpy as np


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
