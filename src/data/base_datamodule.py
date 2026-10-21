from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def split_indices(
    num_samples: int, fractions: tuple[float, ...], seed: int = 42
) -> list[list[int]]:
    """Randomly split `range(num_samples)` into a remainder part and parts of given fractions.

    :param num_samples: The number of samples to split.
    :param fractions: The fraction of samples for each extra part, e.g. `(0.1, 0.1)`.
    :param seed: The seed for the random permutation.
    :return: A list of index lists: `[remainder, part_1, part_2, ...]`.
    """
    perm = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed)).tolist()
    sizes = [int(round(num_samples * f)) for f in fractions]
    parts, start = [], 0
    for size in sizes:
        parts.append(perm[start : start + size])
        start += size
    return [perm[start:], *parts]


class BaseDataModule(LightningDataModule):
    """Shared `LightningDataModule` logic: per-device batch size and the dataloaders.

    Subclasses implement `setup_datasets()`, which sets `self.data_train`, `self.data_val` and
    `self.data_test`.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initialize a `BaseDataModule`.

        :param batch_size: The global batch size (divided by the number of devices).
        :param num_workers: The number of dataloader workers.
        :param pin_memory: Whether to pin memory.
        :param persistent_workers: Whether to keep dataloader workers alive between epochs.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def setup_datasets(self) -> None:
        """Create `self.data_train`, `self.data_val` and `self.data_test`."""
        raise NotImplementedError

    def setup(self, stage: str | None = None) -> None:
        """Divide the batch size by the number of devices and create the datasets once.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()` on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices "
                    f"({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.setup_datasets()

    def _dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
            drop_last=shuffle,  # avoid single-sample batches with batchnorm during training
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return self._dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return self._dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return self._dataloader(self.data_test, shuffle=False)
