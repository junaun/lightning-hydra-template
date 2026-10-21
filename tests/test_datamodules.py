from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import DictConfig

from tests.conftest import NUM_CLASSES


@pytest.mark.parametrize("batch_size", [2, 4])
def test_image_folder_datamodule(cfg_train: DictConfig, batch_size: int) -> None:
    """Tests `ImageFolderDataModule`: splits, dataloaders, and dtypes/shapes of a batch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    cfg_train.data.batch_size = batch_size
    dm = hydra.utils.instantiate(cfg_train.data)
    dm.setup()

    assert dm.class_names == [f"class_{i}" for i in range(NUM_CLASSES)]
    assert len(dm.data_train) == 8 * NUM_CLASSES
    assert len(dm.data_val) == 4 * NUM_CLASSES
    assert dm.data_test is dm.data_val  # no test_subdir

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (batch_size, 3, 32, 32)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_image_folder_datamodule_val_split(cfg_train: DictConfig) -> None:
    """Tests that the validation set is split from the train set if `val_subdir` is None."""
    cfg_train.data.val_subdir = None
    cfg_train.data.val_split = 0.25
    dm = hydra.utils.instantiate(cfg_train.data)
    dm.setup()

    assert len(dm.data_train) + len(dm.data_val) == 8 * NUM_CLASSES
    assert len(dm.data_val) == 6
    assert not set(dm.data_train.indices) & set(dm.data_val.indices)


def test_image_folder_datamodule_num_classes(cfg_train: DictConfig) -> None:
    """Tests that a wrong `num_classes` is reported."""
    cfg_train.data.num_classes = NUM_CLASSES + 1
    dm = hydra.utils.instantiate(cfg_train.data)
    with pytest.raises(ValueError, match="classes"):
        dm.setup()


def test_segmentation_folder_datamodule(cfg_train_seg: DictConfig, seg_data_dir: Path) -> None:
    """Tests `SegmentationFolderDataModule`: splits, and dtypes/shapes/values of a batch.

    :param cfg_train_seg: A DictConfig containing a valid segmentation training configuration.
    :param seg_data_dir: The synthetic segmentation dataset.
    """
    dm = hydra.utils.instantiate(cfg_train_seg.data)
    dm.setup()

    num_samples = len(list((seg_data_dir / "images").glob("*.jpg")))
    assert len(dm.data_train) + len(dm.data_val) + len(dm.data_test) == num_samples
    assert len(dm.data_val) == len(dm.data_test) == 2

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 3, 32, 32)
    assert x.dtype == torch.float32
    assert y.shape == (4, 32, 32)
    assert y.dtype == torch.int64
    assert set(y.unique().tolist()) <= set(range(NUM_CLASSES))
