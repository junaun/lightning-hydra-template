from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB `uint8` numpy array of shape (H, W, 3).

    :param path: Path to the image.
    :return: The image.
    """
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


class ImageFolderDataset(Dataset):
    """Image classification dataset with one subfolder per class: `root/<class_name>/<image>`.

    Images are loaded as numpy arrays and passed to an albumentations `transform`.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., dict[str, Any]] | None = None,
        classes: Sequence[str] | None = None,
    ) -> None:
        """Initialize an `ImageFolderDataset`.

        :param root: The root directory containing one subfolder per class.
        :param transform: An albumentations transform called as `transform(image=...)`.
        :param classes: Class names (subfolder names) in label order. Defaults to the sorted
            subfolders of `root`. Pass the train classes when building val/test datasets.
        """
        self.root = Path(root)
        self.transform = transform
        self.classes = list(classes or sorted(d.name for d in self.root.iterdir() if d.is_dir()))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = [
            (path, idx)
            for name, idx in self.class_to_idx.items()
            for path in sorted((self.root / name).rglob("*"))
            if path.suffix.lower() in IMG_EXTENSIONS
        ]
        if not self.samples:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the transformed image and its class index.

        :param index: The sample index.
        :return: A tuple of (image, label).
        """
        path, label = self.samples[index]
        image = load_rgb(path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class SegmentationFolderDataset(Dataset):
    """Semantic segmentation dataset with images and masks in separate folders, paired by filename
    stem: `image_dir/<stem><image ext>` <-> `mask_dir/<stem><mask_suffix>`.

    Masks are single-channel images whose pixel values are class indices.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_glob: str = "*.jpg",
        mask_suffix: str = ".png",
        mask_offset: int = 0,
        transform: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a `SegmentationFolderDataset`.

        :param image_dir: The directory containing the images.
        :param mask_dir: The directory containing the masks.
        :param image_glob: Glob pattern used to find images. Defaults to `"*.jpg"`.
        :param mask_suffix: Suffix (incl. extension) of the mask files. Defaults to `".png"`.
        :param mask_offset: Value added to every mask pixel, e.g. `-1` for 1-based masks.
        :param transform: An albumentations transform called as `transform(image=..., mask=...)`.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_offset = mask_offset
        self.transform = transform
        self.samples = [
            (path, self.mask_dir / f"{path.stem}{mask_suffix}")
            for path in sorted(self.image_dir.glob(image_glob))
            if (self.mask_dir / f"{path.stem}{mask_suffix}").exists()
        ]
        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found in {self.image_dir} and {self.mask_dir}"
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the transformed image and its mask.

        :param index: The sample index.
        :return: A tuple of (image, mask), where mask is a `long` tensor of shape (H, W).
        """
        image_path, mask_path = self.samples[index]
        image = load_rgb(image_path)
        with Image.open(mask_path) as mask_img:
            mask = np.asarray(mask_img)
        if self.mask_offset:
            mask = (mask.astype(np.int16) + self.mask_offset).clip(0, 255).astype(np.uint8)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, torch.as_tensor(mask).long()
