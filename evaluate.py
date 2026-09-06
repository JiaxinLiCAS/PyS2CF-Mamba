#!/usr/bin/env python3
"""Evaluate a saved PyS2CF-Mamba checkpoint on its paper test split."""

import argparse
from pathlib import Path

import numpy as np
import torch

from models import PyS2CFMamba
from pys2cf.data import (
    DATASETS,
    apply_preprocessing_state,
    load_dataset,
    make_label_mask,
)
from pys2cf.engine import make_tiles, predict_tiled
from pys2cf.metrics import classification_metrics
from pys2cf.reproducibility import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_json(args.config)
    dataset_name = config["dataset"]
    dataset_spec = DATASETS[dataset_name]
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    data_root = Path(config["data_root"]).expanduser()
    if not data_root.is_absolute():
        data_root = project_root / data_root
    cube, labels = load_dataset(dataset_name, data_root)
    preprocessing_path = checkpoint_path.parents[1] / "preprocessing.npz"
    if not preprocessing_path.is_file():
        raise FileNotFoundError(f"Saved preprocessing state not found: {preprocessing_path}")
    with np.load(preprocessing_path, allow_pickle=False) as payload:
        preprocessing_state = {key: payload[key] for key in payload.files}
    image_cpu = apply_preprocessing_state(cube, preprocessing_state)
    with np.load(checkpoint_path.with_name("split.npz"), allow_pickle=False) as split:
        test_idx = split["test_idx"]
    test_mask = make_label_mask(labels, test_idx)

    model = PyS2CFMamba(
        num_classes=dataset_spec.num_classes,
        in_channels=config["preprocessing"]["pca_components"],
        **config["model"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    tile_config = config["tiling"]
    tiles = make_tiles(
        labels.shape[0],
        labels.shape[1],
        tile_config["tile_size"],
        tile_config["overlap"],
    )
    prediction = predict_tiled(
        model=model,
        image_cpu=image_cpu,
        tiles=tiles,
        num_classes=dataset_spec.num_classes,
        device=device,
    )
    metrics = classification_metrics(prediction, test_mask.numpy(), dataset_spec.num_classes)
    print(
        "OA={:.2f} AA={:.2f} Kappa={:.2f}".format(
            metrics["oa"] * 100,
            metrics["aa"] * 100,
            metrics["kappa"] * 100,
        )
    )
    save_json(checkpoint_path.with_name("evaluation.json"), metrics)


if __name__ == "__main__":
    main()
