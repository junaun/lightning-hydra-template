from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from hydra import compose, initialize

from src.data.components.transforms import build_transforms

TRANSFORMS_DIR = Path(__file__).parents[1] / "configs" / "data" / "transforms"
PRESETS = sorted(p.stem for p in TRANSFORMS_DIR.glob("*.yaml"))


def compose_transforms(overrides: list[str]) -> dict:
    """Compose the train config and return its `data.transforms` node."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    return cfg.data.transforms


@pytest.mark.parametrize("preset", PRESETS)
def test_presets(preset: str) -> None:
    """Every transforms preset builds into train/eval pipelines returning a normalized tensor."""
    cfg = compose_transforms([f"data/transforms={preset}", "data.transforms.image_size=32"])
    image = np.random.default_rng(0).integers(0, 255, (40, 48, 3), dtype=np.uint8)

    for stage in ("train", "eval"):
        pipeline = build_transforms(cfg, stage)
        assert isinstance(pipeline, A.Compose)
        out = pipeline(image=image)["image"]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 32, 32)
        assert out.dtype == torch.float32


def test_override_ops() -> None:
    """Ops can be removed, added, disabled and tweaked by name from the command line."""
    cfg = compose_transforms(
        [
            "~data.transforms.train.ops.color",
            "+data.transforms.train.ops.blur={_target_: albumentations.GaussianBlur, p: 0.2}",
            "data.transforms.train.ops.hflip=null",
            "data.transforms.train.ops.crop.scale=[0.5,1.0]",
        ]
    )
    pipeline = build_transforms(cfg, "train")
    names = [type(t).__name__ for t in pipeline.transforms]

    assert names == ["RandomResizedCrop", "GaussianBlur", "Normalize", "ToTensorV2"]
    assert pipeline.transforms[0].scale == (0.5, 1.0)


def test_image_mask_alignment() -> None:
    """Spatial ops are applied jointly to the image and the mask."""
    cfg = compose_transforms(
        [
            "data=segmentation_folder",
            "data.transforms.image_size=32",
            "~data.transforms.train.ops.brightness_contrast",
            "data.transforms.train.ops.crop.scale=[1.0,1.0]",
            "++data.transforms.train.ops.crop.ratio=[1.0,1.0]",
            "data.transforms.train.ops.hflip.p=1.0",
            "~data.transforms.post.normalize",
        ]
    )
    pipeline = build_transforms(cfg, "train")

    # the mask equals the binarized image, so they stay equal after geometric transforms
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :20] = 1
    image = np.repeat(mask[..., None] * 255, 3, axis=2)

    out = pipeline(image=image, mask=mask)
    # image is resized bilinearly and the mask with nearest neighbor, so allow edge differences
    agreement = ((out["image"][0] > 127) == (out["mask"] == 1)).float().mean()
    assert agreement > 0.95
    # horizontal flip always applied, so the foreground ends up on the right side
    assert out["mask"][:, -1].float().mean() > out["mask"][:, 0].float().mean()
