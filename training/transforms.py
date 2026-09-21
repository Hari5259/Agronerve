"""Image transformations and realistic agricultural data augmentations for AgroNerve."""

import torchvision.transforms as transforms
from config import settings

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = settings.VISION_IMAGE_SIZE) -> transforms.Compose:
    """Returns realistic mobile-camera data augmentation pipeline for training.

    Simulates:
    - Distance / handheld framing variations (RandomResizedCrop)
    - Leaf orientation / phone angle (RandomRotation, RandomHorizontalFlip)
    - Sunlight, cloud shadow, and indoor lighting (ColorJitter)
    - Slight lens defocus / motion blur (GaussianBlur)
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.03,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = settings.VISION_IMAGE_SIZE) -> transforms.Compose:
    """Returns deterministic, un-augmented transform pipeline for validation, testing, and live inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
