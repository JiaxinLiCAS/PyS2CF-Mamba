#!/usr/bin/env python3
"""Train and evaluate PyS2CF-Mamba with the paper protocol."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from models import PyS2CFMamba
from pys2cf.data import (
    DATASETS,
    load_dataset,
    make_paper_split,
    make_label_mask,
    preprocess_hsi,
)
from pys2cf.engine import make_tiles, predict_tiled, tiles_with_labels, train_one_epoch
from pys2cf.metrics import classification_metrics
from pys2cf.reproducibility import load_json, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment configuration.")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:1.")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_json(Path(args.config).expanduser().resolve())
    data_root = Path(config["data_root"]).expanduser()
    if not data_root.is_absolute():
        data_root = project_root / data_root
    config["data_root"] = str(data_root.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. mamba-ssm training requires CUDA.")

    dataset_name = config["dataset"]
    dataset_spec = DATASETS[dataset_name]
    model_parameters = {
        "in_channels": config["preprocessing"]["pca_components"],
        **config["model"],
    }
    output_root = Path(config["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_root = output_root.resolve() / dataset_name
    run_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(run_root / "train.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("PyS2CF-Mamba")
    save_json(run_root / "resolved_config.json", config)

    logger.info("Loading %s from %s", dataset_name, config["data_root"])
    cube, labels = load_dataset(dataset_name, config["data_root"])
    logger.info("Cube=%s labels=%s", cube.shape, labels.shape)
    image_cpu, preprocessing_state = preprocess_hsi(cube, **config["preprocessing"])
    np.savez(run_root / "preprocessing.npz", **preprocessing_state)
    del cube

    tile_config = config["tiling"]
    tiles = make_tiles(
        labels.shape[0],
        labels.shape[1],
        tile_config["tile_size"],
        tile_config["overlap"],
    )
    training_config = config["training"]
    results = []

    for run_index, seed in enumerate(config["seeds"]):
        set_seed(seed)
        seed_root = run_root / f"run{run_index}_seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        train_idx, val_idx, test_idx = make_paper_split(
            labels,
            seed,
            config["split"]["train_samples_per_class"],
            config["split"]["val_samples_per_class"],
        )
        np.savez_compressed(
            seed_root / "split.npz",
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        train_mask = make_label_mask(labels, train_idx)
        val_mask = make_label_mask(labels, val_idx)
        test_mask = make_label_mask(labels, test_idx)
        validation_tiles = tiles_with_labels(val_mask, tiles)

        model = PyS2CFMamba(num_classes=dataset_spec.num_classes, **model_parameters).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
        # Deliberately unweighted: the paper protocol samples the same K per class.
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=training_config["label_smoothing"],
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

        best_metrics = None
        best_epoch = None
        checkpoint_path = seed_root / "best.pt"
        started = time.perf_counter()

        for epoch in range(training_config["epochs"]):
            loss = train_one_epoch(
                model=model,
                image_cpu=image_cpu,
                train_mask_cpu=train_mask,
                tiles=tiles,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                num_update_groups=tile_config["num_update_groups"],
            )
            validation_prediction = predict_tiled(
                model=model,
                image_cpu=image_cpu,
                tiles=validation_tiles,
                num_classes=dataset_spec.num_classes,
                device=device,
            )
            validation_metrics = classification_metrics(
                validation_prediction,
                val_mask.numpy(),
                dataset_spec.num_classes,
            )
            if best_metrics is None or validation_metrics["oa"] > best_metrics["oa"]:
                best_metrics = validation_metrics
                best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                "seed=%d epoch=%03d loss=%.6f val_OA=%.6f val_AA=%.6f val_Kappa=%.6f",
                seed,
                epoch + 1,
                loss,
                validation_metrics["oa"],
                validation_metrics["aa"],
                validation_metrics["kappa"],
            )

        train_seconds = time.perf_counter() - started
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        test_started = time.perf_counter()
        test_prediction = predict_tiled(
            model=model,
            image_cpu=image_cpu,
            tiles=tiles,
            num_classes=dataset_spec.num_classes,
            device=device,
        )
        test_seconds = time.perf_counter() - test_started
        test_metrics = classification_metrics(
            test_prediction,
            test_mask.numpy(),
            dataset_spec.num_classes,
        )
        seed_result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation": best_metrics,
            "test": test_metrics,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        save_json(seed_root / "metrics.json", seed_result)
        np.save(seed_root / "prediction.npy", test_prediction.astype(np.int16))
        results.append(seed_result)
        logger.info(
            "seed=%d test_OA=%.6f test_AA=%.6f test_Kappa=%.6f best_epoch=%d",
            seed,
            test_metrics["oa"],
            test_metrics["aa"],
            test_metrics["kappa"],
            best_epoch,
        )
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "dataset": dataset_name,
        "num_runs": len(results),
        "oa": mean_std([item["test"]["oa"] for item in results]),
        "aa": mean_std([item["test"]["aa"] for item in results]),
        "kappa": mean_std([item["test"]["kappa"] for item in results]),
        "runs": results,
    }
    save_json(run_root / "summary.json", summary)
    logger.info(
        "Finished %s | OA %.2f +/- %.2f | AA %.2f +/- %.2f | Kappa %.2f +/- %.2f",
        dataset_name,
        summary["oa"]["mean"] * 100,
        summary["oa"]["std"] * 100,
        summary["aa"]["mean"] * 100,
        summary["aa"]["std"] * 100,
        summary["kappa"]["mean"] * 100,
        summary["kappa"]["std"] * 100,
    )


if __name__ == "__main__":
    main()
