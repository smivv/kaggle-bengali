import albumentations as A

from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (Resize, ShiftScaleRotate, IAAPerspective,
                            RandomBrightnessContrast, RandomGamma,
                            ImageCompression, Normalize, Compose)


def pre_transforms(image_size):
    return [
        Resize(image_size, image_size, p=1),
    ]


def hard_transforms():
    return [
        # Random shifts, stretches and turns with a 50% probability
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=2,
            p=0.5
        ),
        IAAPerspective(scale=(0.02, 0.05), p=0.3),
        # Random brightness / contrast with a 30% probability
        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        # Random gamma changes with a 30% probability
        RandomGamma(gamma_limit=(85, 115), p=0.3),
        ImageCompression(quality_lower=80),
    ]


def post_transforms():
    return [
        Normalize(mean=0.06922848809290576, std=0.20515700083327537),
        ToTensor(),
        # ToTensorV2(),
    ]


def get_transforms(image_size=224):
    return {
        "train": Compose(pre_transforms(image_size=image_size) +
                         hard_transforms() +
                         post_transforms()),
        "valid": Compose(pre_transforms(image_size=image_size) +
                         post_transforms()),
        "test": Compose(pre_transforms(image_size=image_size) +
                        post_transforms())
    }
