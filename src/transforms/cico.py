import albumentations as A

from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (
    SmallestMaxSize, LongestMaxSize, PadIfNeeded,
    Resize, ShiftScaleRotate, IAAPerspective,
    MultiplicativeNoise, GaussNoise,
    Blur, GaussianBlur,
    RandomBrightnessContrast, RandomGamma,
    ImageCompression,
    RandomCrop, CoarseDropout, Cutout,
    Normalize, Compose, OneOf
)

BORDER_CONSTANT = 2


def resize_transforms(image_size=224):
    pre_size = int(image_size * 1.5)

    random_crop = Compose([
      SmallestMaxSize(pre_size, p=1),
      RandomCrop(
          image_size, image_size, p=1
      )

    ])

    resize = Compose([
        Resize(image_size, image_size, p=1)
    ])

    random_crop_big = Compose([
      LongestMaxSize(pre_size, p=1),
      RandomCrop(
          image_size, image_size, p=1
      )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
      OneOf([
          random_crop,
          resize,
          random_crop_big
      ], p=1)
    ]

    return result


def pre_transforms(image_size):
    return [
        Resize(height=image_size, width=image_size, p=1),
    ]


def hard_transforms(image_size):
    min_holes, max_holes = 1, 2
    size = 30

    return [
        # Random shifts, stretches and turns with a 50% probability
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=180,
            border_mode=BORDER_CONSTANT,
            p=0.1
        ),

        IAAPerspective(scale=(0.02, 0.05), p=0.1),

        # Random brightness / contrast with a 30% probability
        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.1
        ),

        OneOf([
            GaussNoise(var_limit=1.0, p=1.0),
            MultiplicativeNoise(multiplier=(0.9, 1), p=1.0)
        ], p=0.1),

        OneOf([
            GaussianBlur(blur_limit=3, p=1.0),
            Blur(p=1.0),
        ], p=0.1),

        # CoarseDropout(
        #     min_holes=min_holes,
        #     max_holes=max_holes,
        #     # min_height=image_height // 4,
        #     # max_height=image_height // 4,
        #     # min_width=image_width // 4,
        #     # max_width=image_width // 4,
        #     min_height=size,
        #     max_height=size,
        #     min_width=size,
        #     max_width=size,
        #     fill_value=0,
        #     p=1.0
        # ),

        # Random gamma changes with a 30% probability
        RandomGamma(gamma_limit=(85, 115), p=0.1),
        ImageCompression(
            quality_lower=70,
            quality_upper=100,
            p=0.1
        ),
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
        Normalize(mean=0.449, std=0.226),
        # Normalize(mean=0.0692, std=0.2051),
        ToTensorV2(),
    ]


def get_transforms(image_size=224):
    return {
        "train": Compose(
            resize_transforms(image_size) +
            hard_transforms(image_size) +
            post_transforms())
        ,
        "valid": Compose(
            pre_transforms(image_size) +
            post_transforms()
        ),
        "test": Compose(
            pre_transforms(image_size) +
            post_transforms()
        ),
        "infer": Compose(
            pre_transforms(image_size) +
            post_transforms()
        )
    }
