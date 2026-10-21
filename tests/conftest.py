"""This file prepares synthetic datasets and config fixtures for other tests."""

from pathlib import Path

import numpy as np
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
from PIL import Image

NUM_CLASSES = 3


@pytest.fixture(scope="package")
def cls_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic image classification dataset: `train/` and `val/` with one subfolder per
    class. Each class has its own base color, so it's learnable.

    :return: The dataset root directory.
    """
    root = tmp_path_factory.mktemp("cls_data")
    rng = np.random.default_rng(0)
    for split, num_images in (("train", 8), ("val", 4)):
        for label in range(NUM_CLASSES):
            class_dir = root / split / f"class_{label}"
            class_dir.mkdir(parents=True)
            for i in range(num_images):
                image = np.full((40, 48, 3), 60 * label + 40, dtype=np.int16)
                image += rng.integers(-30, 30, image.shape, dtype=np.int16)
                Image.fromarray(image.clip(0, 255).astype(np.uint8)).save(class_dir / f"{i}.jpg")
    return root


@pytest.fixture(scope="package")
def seg_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic segmentation dataset: `images/<i>.jpg` and `masks/<i>.png`, where every
    mask is split into vertical stripes with class indices `0..NUM_CLASSES-1`.

    :return: The dataset root directory.
    """
    root = tmp_path_factory.mktemp("seg_data")
    (root / "images").mkdir()
    (root / "masks").mkdir()
    rng = np.random.default_rng(0)
    for i in range(20):
        mask = np.zeros((40, 48), dtype=np.uint8)
        mask[:, 16:32] = 1
        mask[:, 32:] = 2
        image = (mask[..., None] * 80 + rng.integers(0, 40, (40, 48, 3))).astype(np.uint8)
        Image.fromarray(image).save(root / "images" / f"{i}.jpg")
        Image.fromarray(mask).save(root / "masks" / f"{i}.png")
    return root


def cls_overrides(data_dir: Path) -> list[str]:
    """Config overrides to run the classification configs offline on the synthetic dataset."""
    return [
        f"data.data_dir={data_dir}",
        f"data.num_classes={NUM_CLASSES}",
        "data.batch_size=4",
        "data.num_workers=0",
        "data.transforms.image_size=32",
        "model.net.pretrained=false",
    ]


def seg_overrides(data_dir: Path) -> list[str]:
    """Config overrides to run the segmentation configs offline on the synthetic dataset."""
    return [
        f"data.data_dir={data_dir}",
        f"data.num_classes={NUM_CLASSES}",
        "data.image_subdir=images",
        "data.mask_subdir=masks",
        "data.mask_offset=0",
        "data.batch_size=4",
        "data.num_workers=0",
        "data.transforms.image_size=32",
        "model.net.encoder_weights=null",
    ]


def compose_test_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    """Compose a config and apply the default settings for all tests.

    :param config_name: The name of the main config, e.g. `"train.yaml"`.
    :param overrides: Hydra overrides.
    :return: The composed config.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_train_global(cls_data_dir: Path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training (classification).

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    return compose_test_cfg("train.yaml", cls_overrides(cls_data_dir))


@pytest.fixture(scope="package")
def cfg_train_seg_global(seg_data_dir: Path) -> DictConfig:
    """A pytest fixture for setting up a Hydra DictConfig for training a segmentation model.

    :return: A DictConfig object containing a Hydra configuration for segmentation training.
    """
    overrides = ["experiment=segmentation_example", *seg_overrides(seg_data_dir)]
    return compose_test_cfg("train.yaml", overrides)


@pytest.fixture(scope="package")
def cfg_eval_global(cls_data_dir: Path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    return compose_test_cfg("eval.yaml", ["ckpt_path=.", *cls_overrides(cls_data_dir)])


def _with_tmp_paths(cfg: DictConfig, tmp_path: Path) -> DictConfig:
    cfg = cfg.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which uses a temporary
    logging path `tmp_path`. Each test generates its own temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    yield _with_tmp_paths(cfg_train_global, tmp_path)

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_train_seg(cfg_train_seg_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_seg_global()` fixture, which uses a
    temporary logging path `tmp_path`.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    yield _with_tmp_paths(cfg_train_seg_global, tmp_path)

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which uses a temporary
    logging path `tmp_path`.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    yield _with_tmp_paths(cfg_eval_global, tmp_path)

    GlobalHydra.instance().clear()
