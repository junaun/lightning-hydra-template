"""Download the datasets used by the example experiments into `data/`.

- `classification_example`: Imagenette 160px (~100 MB), already in `ImageFolder` layout
  -> `data/imagenette2-160/{train,val}/<class>/*.JPEG`
- `segmentation_example`: Oxford-IIIT Pet (~800 MB), images + trimap masks
  -> `data/oxford-iiit-pet/{images,annotations/trimaps}`

Usage:
    python scripts/download_example_data.py [--data-dir data] [--only imagenette|oxford_pet]
"""

import argparse
from pathlib import Path

from torchvision.datasets import Imagenette, OxfordIIITPet

DATASETS = {
    "imagenette": lambda root: Imagenette(root, split="train", size="160px", download=True),
    "oxford_pet": lambda root: OxfordIIITPet(root, target_types="segmentation", download=True),
}


def main() -> None:
    """Download the example datasets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=Path(__file__).parents[1] / "data", type=Path)
    parser.add_argument("--only", choices=list(DATASETS), default=None)
    args = parser.parse_args()

    for name, download in DATASETS.items():
        if args.only in (None, name):
            print(f"Downloading {name} to {args.data_dir}...")
            download(str(args.data_dir))


if __name__ == "__main__":
    main()
