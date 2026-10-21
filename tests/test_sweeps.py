from pathlib import Path

import pytest

from tests.conftest import cls_overrides, seg_overrides
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]", "trainer.accelerator=cpu", "extras.enforce_tags=false"]


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.parametrize("experiment", ["classification_example", "segmentation_example"])
def test_experiments(
    tmp_path: Path, cls_data_dir: Path, seg_data_dir: Path, experiment: str
) -> None:
    """Test running the example experiment configs with `fast_dev_run=True` on synthetic data.

    :param tmp_path: The temporary logging path.
    :param experiment: The experiment config name.
    """
    data_overrides = (
        cls_overrides(cls_data_dir)
        if experiment.startswith("classification")
        else seg_overrides(seg_data_dir)
    )
    command = [
        startfile,
        f"experiment={experiment}",
        "hydra.run.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
        *data_overrides,
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path, cls_data_dir: Path) -> None:
    """Test default hydra sweep.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
        *cls_overrides(cls_data_dir),
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path, cls_data_dir: Path) -> None:
    """Test default hydra sweep with ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=2",
        "model.optimizer.lr=0.005,0.01",
        *cls_overrides(cls_data_dir),
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_tune(tmp_path: Path, cls_data_dir: Path) -> None:
    """Test Optuna hyperparameter search with `src/tune.py`.

    :param tmp_path: The temporary logging path.
    """
    command = [
        "src/tune.py",
        "hydra.run.dir=" + str(tmp_path),
        "optuna.n_trials=3",
        "optuna.sampler.n_startup_trials=2",
        "optuna.params.batch_size.choices=[2,4]",
        "trainer.max_epochs=1",
        *cls_overrides(cls_data_dir),
    ] + overrides
    run_sh_command(command)

    assert (tmp_path / "best_params.yaml").exists()
    assert len(list((tmp_path / "trials").iterdir())) == 3


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_tune_wandb(tmp_path: Path, cls_data_dir: Path) -> None:
    """Test Optuna hyperparameter search with wandb logging (offline).

    :param tmp_path: The temporary logging path.
    """
    command = [
        "src/tune.py",
        "hydra.run.dir=" + str(tmp_path),
        "optuna.n_trials=2",
        "trainer.max_epochs=1",
        "logger=wandb",
        "logger.wandb.offline=true",
        "trainer.accelerator=cpu",
        "extras.enforce_tags=false",
        *cls_overrides(cls_data_dir),
    ]
    run_sh_command(command)
