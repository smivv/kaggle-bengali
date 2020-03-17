import albumentations as A

from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (Resize, ShiftScaleRotate, IAAPerspective,
                            MultiplicativeNoise, GaussNoise,
                            Blur, GaussianBlur,
                            RandomBrightnessContrast, RandomGamma,
                            ImageCompression, CoarseDropout, Cutout,
                            Normalize, Compose, OneOf)


def pre_transforms(image_height, image_width):
    return [
        Resize(height=image_height, width=image_width, p=1),
    ]


def hard_transforms(image_height, image_width):
    num = 6
    size = 50
    return [
        # Random shifts, stretches and turns with a 50% probability
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            border_mode=2,
            # border_mode=0,
            # value=0,
            p=0.5
        ),
        # IAAPerspective(scale=(0.02, 0.05), p=0.3),
        # Random brightness / contrast with a 30% probability
        # RandomBrightnessContrast(
        #     brightness_limit=0.2, contrast_limit=0.2, p=0.3
        # ),
        # OneOf([
        #     GaussNoise(var_limit=1.0, p=0.2),
        #     MultiplicativeNoise(multiplier=(0.9, 1), p=0.2)
        # ], p=1.0),
        #
        # OneOf([
        #     GaussianBlur(blur_limit=3, p=0.2),
        #     Blur(p=0.2),
        # ], p=1.0),

        CoarseDropout(
            min_holes=num,
            max_holes=num,
            # min_height=image_height // 4,
            # max_height=image_height // 4,
            # min_width=image_width // 4,
            # max_width=image_width // 4,
            min_height=size,
            max_height=size,
            min_width=size,
            max_width=size,
            fill_value=0,
            p=1.0
        )
        # Random gamma changes with a 30% probability
        # RandomGamma(gamma_limit=(85, 115), p=0.3),
        # ImageCompression(quality_lower=80),
    ]


def tta_transforms():
    return [
        # Random shifts, stretches and turns with a 50% probability
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            border_mode=2,
            # border_mode=0,
            # value=0,
            p=0.5
        ),
    ]


def post_transforms():
    return [
        Normalize(mean=0.0692, std=0.2051),
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
