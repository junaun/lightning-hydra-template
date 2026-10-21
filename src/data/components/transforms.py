import os
from collections.abc import Mapping
from typing import Any

# `albumentations==2.0.8` is the last release of the archived MIT-licensed repo,
# so skip its update check (it would always suggest upgrading)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A  # noqa: E402
import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def _to_container(cfg: Any) -> Any:
    """Convert a `DictConfig`/`ListConfig` into a plain python container."""
    if isinstance(cfg, DictConfig) or OmegaConf.is_list(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def build_transforms(transforms_cfg: Mapping[str, Any] | None, stage: str) -> A.Compose:
    """Build an albumentations pipeline for a given stage from the transforms config.

    The transforms config (see `configs/data/transforms/`) looks like:

    ```yaml
    train:
      ops:            # name -> albumentations op, applied in order
        hflip:
          _target_: albumentations.HorizontalFlip
          p: 0.5
    eval:
      ops: {...}
    post: {...}       # name -> op, always appended after `ops` (e.g. Normalize + ToTensorV2)
    compose: {...}    # optional kwargs for `A.Compose` (e.g. `additional_targets`, `bbox_params`)
    ```

    Ops are kept in a mapping (instead of a list) so they can be added, removed or tweaked from
    the command line by name. An op set to `null` is skipped.

    :param transforms_cfg: The transforms config.
    :param stage: Which pipeline to build, e.g. `"train"` or `"eval"`.
    :return: The composed albumentations pipeline.
    """
    if not transforms_cfg:
        return A.Compose([])

    stage_cfg = transforms_cfg.get(stage) or {}
    ops = list((stage_cfg.get("ops") or {}).values())
    ops += list((transforms_cfg.get("post") or {}).values())

    compose_kwargs = _to_container(transforms_cfg.get("compose") or {})
    return A.Compose(
        [hydra.utils.instantiate(op) for op in ops if op is not None], **compose_kwargs
    )
