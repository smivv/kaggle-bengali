import random
import numpy as np

from torch.utils.data.dataset import Dataset

from src.datasets.bengali import IMAGE_KEY, INPUT_KEYS
from src.utils.utils import rand_bbox, to_one_hot


class CutMixUpDataset(Dataset):
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
        img = sample[self.image_key]

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.randint(a=0, b=len(self))

            sample_ = self.dataset[rand_index]
            img_ = sample_[self.image_key]

            height, width = img_.size()[-2:-1]

            l, t, r, b = rand_bbox(lam, height, width)

            sample[self.image_key][:, t:b, l:r] = \
                img_[self.image_key][:, t:b, l:r]

            lam = 1 - float((r - l) * (b - t)) / (height * width)

            for input_key, num_classes in zip(self.input_keys, self.num_classes):
                sample[input_key] = lam * to_one_hot(sample[input_key], num_classes) + \
                            (1. - lam) * to_one_hot(sample_[input_key], num_classes)
        return img

    def __len__(self):
        return len(self.dataset)
