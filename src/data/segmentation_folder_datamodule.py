from pathlib import Path
from typing import Any

from torch.utils.data import Subset

from src.data.base_datamodule import BaseDataModule, split_indices
from src.data.components.datasets import SegmentationFolderDataset
from src.data.components.transforms import build_transforms
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SegmentationFolderDataModule(BaseDataModule):
    """`LightningDataModule` for semantic segmentation from an image folder and a mask folder.

    Expected layout (images and masks are paired by filename stem):

    ```
    data_dir/
    ├── images/<stem>.jpg
    └── masks/<stem>.png    # single channel, pixel value = class index
    ```

    The pairs are randomly split into train/val/test using `val_split` and `test_split`.

    Augmentations are fully defined in the `transforms` config and applied jointly to the image
    and the mask, see `src/data/components/transforms.py` and `configs/data/transforms/`.
    """

    def __init__(
        self,
        data_dir: str,
        num_classes: int,
        transforms: dict[str, Any] | None = None,
        image_subdir: str = "images",
        mask_subdir: str = "masks",
        image_glob: str = "*.jpg",
        mask_suffix: str = ".png",
        mask_offset: int = 0,
        ignore_index: int = 255,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initialize a `SegmentationFolderDataModule`.

        :param data_dir: The dataset root directory.
        :param num_classes: The number of classes (excluding `ignore_index`).
        :param transforms: The transforms config with `train`, `eval` and `post` sections.
        :param image_subdir: Images subfolder of `data_dir`.
        :param mask_subdir: Masks subfolder of `data_dir`.
        :param image_glob: Glob pattern used to find images.
        :param mask_suffix: Suffix (incl. extension) of the mask files.
        :param mask_offset: Value added to every mask pixel, e.g. `-1` for 1-based masks.
        :param ignore_index: Mask value ignored by the loss and metrics.
        :param val_split: Fraction of samples used for validation.
        :param test_split: Fraction of samples used for testing.
        :param seed: The seed for the train/val/test split.
        :param batch_size: The global batch size (divided by the number of devices).
        :param num_workers: The number of dataloader workers.
        :param pin_memory: Whether to pin memory.
        :param persistent_workers: Whether to keep dataloader workers alive between epochs.
        """
        super().__init__(batch_size, num_workers, pin_memory, persistent_workers)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = build_transforms(transforms, "train")
        self.eval_transforms = build_transforms(transforms, "eval")

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.hparams.num_classes

    def setup_datasets(self) -> None:
        """Create the train, validation and test datasets."""
        root = Path(self.hparams.data_dir)

        def make_dataset(transform: Any) -> SegmentationFolderDataset:
            return SegmentationFolderDataset(
                image_dir=root / self.hparams.image_subdir,
                mask_dir=root / self.hparams.mask_subdir,
                image_glob=self.hparams.image_glob,
                mask_suffix=self.hparams.mask_suffix,
                mask_offset=self.hparams.mask_offset,
                transform=transform,
            )

        # same samples with different transforms, so val/test samples are not augmented
        train_ds, eval_ds = make_dataset(self.train_transforms), make_dataset(self.eval_transforms)
        train_idx, val_idx, test_idx = split_indices(
            len(train_ds), (self.hparams.val_split, self.hparams.test_split), self.hparams.seed
        )
        self.data_train = Subset(train_ds, train_idx)
        self.data_val = Subset(eval_ds, val_idx)
        self.data_test = Subset(eval_ds, test_idx)

        log.info(
            f"train: {len(self.data_train)} | val: {len(self.data_val)} | "
            f"test: {len(self.data_test)}"
        )
        log.info(f"Train transforms: {self.train_transforms}")


if __name__ == "__main__":
    _ = SegmentationFolderDataModule(data_dir="data/", num_classes=2)
