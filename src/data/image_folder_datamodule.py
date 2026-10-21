from pathlib import Path
from typing import Any

from torch.utils.data import Subset

from src.data.base_datamodule import BaseDataModule, split_indices
from src.data.components.datasets import ImageFolderDataset
from src.data.components.transforms import build_transforms
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ImageFolderDataModule(BaseDataModule):
    """`LightningDataModule` for image classification from class-per-subfolder directories.

    Expected layout (val/test folders are optional):

    ```
    data_dir/
    ├── train/<class_name>/*.jpg
    ├── val/<class_name>/*.jpg
    └── test/<class_name>/*.jpg
    ```

    - if `val_subdir` is not given, `val_split` of the train images are used for validation
    - if `test_subdir` is not given, the validation set is also used for testing

    Augmentations are fully defined in the `transforms` config, see
    `src/data/components/transforms.py` and `configs/data/transforms/`.
    """

    def __init__(
        self,
        data_dir: str,
        num_classes: int,
        transforms: dict[str, Any] | None = None,
        train_subdir: str = "train",
        val_subdir: str | None = "val",
        test_subdir: str | None = None,
        val_split: float = 0.1,
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initialize an `ImageFolderDataModule`.

        :param data_dir: The dataset root directory.
        :param num_classes: The expected number of classes (checked against the train folder).
        :param transforms: The transforms config with `train`, `eval` and `post` sections.
        :param train_subdir: Train subfolder of `data_dir`.
        :param val_subdir: Validation subfolder of `data_dir`, or `None` to split it from train.
        :param test_subdir: Test subfolder of `data_dir`, or `None` to test on the validation set.
        :param val_split: Fraction of train images used for validation if `val_subdir` is `None`.
        :param seed: The seed for the train/val split.
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

        self.class_names: list[str] = []

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.hparams.num_classes

    def setup_datasets(self) -> None:
        """Create the train, validation and test datasets."""
        root = Path(self.hparams.data_dir)
        train_dir = root / self.hparams.train_subdir

        train_full = ImageFolderDataset(train_dir, transform=self.train_transforms)
        self.class_names = train_full.classes
        if len(self.class_names) != self.hparams.num_classes:
            raise ValueError(
                f"Found {len(self.class_names)} classes in {train_dir}, "
                f"but `num_classes={self.hparams.num_classes}`."
            )

        if self.hparams.val_subdir:
            self.data_train = train_full
            self.data_val = ImageFolderDataset(
                root / self.hparams.val_subdir, self.eval_transforms, self.class_names
            )
        else:
            # same images with eval transforms, so validation samples are not augmented
            train_eval = ImageFolderDataset(train_dir, self.eval_transforms, self.class_names)
            train_idx, val_idx = split_indices(
                len(train_full), (self.hparams.val_split,), self.hparams.seed
            )
            self.data_train = Subset(train_full, train_idx)
            self.data_val = Subset(train_eval, val_idx)

        if self.hparams.test_subdir:
            self.data_test = ImageFolderDataset(
                root / self.hparams.test_subdir, self.eval_transforms, self.class_names
            )
        else:
            log.info("No `test_subdir` given, using the validation set for testing.")
            self.data_test = self.data_val

        log.info(
            f"Classes: {len(self.class_names)} | train: {len(self.data_train)} | "
            f"val: {len(self.data_val)} | test: {len(self.data_test)}"
        )
        log.info(f"Train transforms: {self.train_transforms}")


if __name__ == "__main__":
    _ = ImageFolderDataModule(data_dir="data/", num_classes=2)
