import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        # A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        # A.CropNonEmptyMaskIfExists(height=config.IMAGE_HEIGHT_TRAIN, width=config.IMAGE_WIDTH_TRAIN),
        A.Resize(height=config.IMAGE_HEIGHT_TRAIN, width=config.IMAGE_WIDTH_TRAIN),
        A.PadIfNeeded(min_height=config.IMAGE_HEIGHT_TRAIN_PADDED, min_width=config.IMAGE_WIDTH_TRAIN_PADDED,
                      always_apply=True,
                      border_mode=0),
    ]
    return A.Compose(train_transform)


def get_validation_transformation():
    """Add paddings to make image shape divisible by 32"""
    valid_transforms = [
        A.Resize(height=config.IMAGE_HEIGHT_TRAIN, width=config.IMAGE_WIDTH_TRAIN),
        A.PadIfNeeded(min_height=config.IMAGE_HEIGHT_TRAIN_PADDED, min_width=config.IMAGE_WIDTH_TRAIN_PADDED,
                      always_apply=True,
                      border_mode=0),


        # A.Resize(height=config.IMAGE_HEIGHT_TRAIN, width=config.IMAGE_WIDTH_TRAIN),
        # A.Resize(height=config.IMAGE_HEIGHT_ORIGINAL, width=config.IMAGE_WIDTH_ORIGINAL),
        # A.PadIfNeeded(min_height=1088, min_width=1920)
    ]
    return A.Compose(valid_transforms)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)
