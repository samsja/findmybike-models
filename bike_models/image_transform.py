from typing import Callable, Tuple

import torch
import torchvision
from PIL import Image

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMAGE_SHAPE = (158, 258)


def get_transforms(
    image_shape: Tuple[int, int], rect_size: torch.Tensor, loc: torch.Tensor
) -> Callable:

    all_transform = [
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        DrawPattern(rect_size, loc),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
    ]

    return torchvision.transforms.Compose(all_transform)
