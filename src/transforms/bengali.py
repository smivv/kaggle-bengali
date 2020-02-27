import albumentations as A

from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (Resize, ShiftScaleRotate, IAAPerspective,
                            GaussNoise, Blur, GaussianBlur,
                            RandomBrightnessContrast, RandomGamma,
                            ImageCompression, CoarseDropout, Normalize, Compose)


def pre_transforms(image_height, image_width):
    return [
        Resize(image_height, image_width, p=1),
    ]


def hard_transforms(image_height, image_width):
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
        # RandomBrightnessContrast(
        #     brightness_limit=0.2, contrast_limit=0.2, p=0.3
        # ),
        # GaussNoise(var_limit=1.0),
        # Blur(),
        # GaussianBlur(blur_limit=3),
        CoarseDropout(
            max_holes=1,
            max_height=int(image_height/4),
            max_width=int(image_width/4),
            min_holes=1,
            min_height=int(image_height/4),
            min_width=int(image_width/4),
            fill_value=0,
            p=0.5
        )
        # Random gamma changes with a 30% probability
        # RandomGamma(gamma_limit=(85, 115), p=0.3),
        # ImageCompression(quality_lower=80),
    ]


def post_transforms():
    return [
        Normalize(mean=0.0692, std=0.2051),
        # ToTensor(
        #     normalize=None
        # normalize=dict(
        #     mean=[0.06922848809290576],
        #     std=[0.20515700083327537]
        # )
        # ),
        # ToTensor()
        ToTensorV2(),
    ]


def get_transforms(image_height=224, image_width=224):
    return {
        "train": Compose(pre_transforms(image_height, image_width) +
                         hard_transforms(image_height, image_width) +
                         post_transforms()),
        "valid": Compose(pre_transforms(image_height, image_width) +
                         post_transforms()),
        "test": Compose(pre_transforms(image_height, image_width) +
                        post_transforms())
    }
