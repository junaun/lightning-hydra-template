import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@pytest.mark.parametrize("cfg_name", ["cfg_train", "cfg_train_seg", "cfg_eval"])
def test_configs(cfg_name: str, request: pytest.FixtureRequest) -> None:
    """Tests that the training (classification, segmentation) and evaluation configurations can
    be instantiated.

    :param cfg_name: The name of the config fixture.
    :param request: The pytest request object, used to get the config fixture.
    """
    cfg: DictConfig = request.getfixturevalue(cfg_name)
    assert cfg
    assert cfg.data
    assert cfg.model
    assert cfg.trainer

    HydraConfig().set_config(cfg)

    hydra.utils.instantiate(cfg.data)
    hydra.utils.instantiate(cfg.model)
    hydra.utils.instantiate(cfg.trainer)
