import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf


@pytest.mark.parametrize("cfg_name", ["cfg_train", "cfg_train_seg"])
def test_train_fast_dev_run(cfg_name: str, request: pytest.FixtureRequest) -> None:
    """Run for 1 train, val and test step (classification and segmentation).

    :param cfg_name: The name of the config fixture.
    :param request: The pytest request object, used to get the config fixture.
    """
    cfg: DictConfig = request.getfixturevalue(cfg_name)
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "cpu"
    train(cfg)


@RunIf(min_gpus=1)
@pytest.mark.parametrize("cfg_name", ["cfg_train", "cfg_train_seg"])
def test_train_fast_dev_run_gpu(cfg_name: str, request: pytest.FixtureRequest) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_name: The name of the config fixture.
    :param request: The pytest request object, used to get the config fixture.
    """
    cfg: DictConfig = request.getfixturevalue(cfg_name)
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.accelerator = "gpu"
    train(cfg)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = "bf16-mixed"
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.callbacks.model_checkpoint.save_top_k = -1  # keep every epoch

    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files
    assert "val/acc" in metric_dict
