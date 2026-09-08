"""Dataset loading, preprocessing, and paper split handling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class DatasetSpec:
    cube_path: str
    label_path: str
    cube_key: str
    label_key: str
    num_classes: int


DATASETS = {
    "LongKou": DatasetSpec(
        cube_path="LongKou/WHU_Hi_LongKou.mat",
        label_path="LongKou/WHU_Hi_LongKou_gt.mat",
        cube_key="WHU_Hi_LongKou",
        label_key="WHU_Hi_LongKou_gt",
        num_classes=9,
    ),
    "QUH-Qingyun": DatasetSpec(
        cube_path="QUH-Qingyun/QUH-Qingyun.mat",
        label_path="QUH-Qingyun/QUH-Qingyun_GT.mat",
        cube_key="Chengqu",
        label_key="ChengquGT",
        num_classes=6,
    ),
    "QUH-Tangdaowan": DatasetSpec(
        cube_path="QUH-Tangdaowan/QUH-Tangdaowan.mat",
        label_path="QUH-Tangdaowan/QUH-Tangdaowan_GT.mat",
        cube_key="Tangdaowan",
        label_key="TangdaowanGT",
        num_classes=18,
    ),
}


def _load_mat_variable(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    payload = sio.loadmat(path)
    if key not in payload:
        public_keys = sorted(item for item in payload if not item.startswith("__"))
        raise KeyError(f"{path} does not contain '{key}'. Available keys: {public_keys}")
    return np.asarray(payload[key])


def load_dataset(dataset_name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    spec = DATASETS[dataset_name]
    root = Path(data_root).expanduser().resolve()
    cube = _load_mat_variable(root / spec.cube_path, spec.cube_key)
    labels = _load_mat_variable(root / spec.label_path, spec.label_key)
    cube = np.asarray(cube, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int16).squeeze()
    if labels.ndim != 2 or int(labels.max()) != spec.num_classes:
        raise ValueError(f"Invalid labels for {dataset_name}: shape={labels.shape}.")
    if cube.ndim != 3 or cube.shape[:2] != labels.shape:
        raise ValueError(f"Invalid cube shape for {dataset_name}: {cube.shape}.")
    return cube, labels


def preprocess_hsi(
    cube: np.ndarray,
    pca_components: int = 30,
    gaussian_sigma: float = 1.0,
    stretch_low: float = 2.0,
    stretch_high: float = 98.0,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Apply the paper pipeline: Gaussian -> PCA-30 -> 2-98% stretch -> uint8."""
    if not 0.0 <= stretch_low < stretch_high <= 100.0:
        raise ValueError("Expected 0 <= stretch_low < stretch_high <= 100.")
    if pca_components > cube.shape[2]:
        raise ValueError("pca_components cannot exceed the input band count.")

    smoothed = gaussian_filter(cube, sigma=gaussian_sigma)
    height, width, bands = smoothed.shape
    pca = PCA(n_components=pca_components)
    features = pca.fit_transform(smoothed.reshape(-1, bands))
    features = features.reshape(height, width, pca_components).astype(np.float32)

    stretch_min = np.percentile(features, stretch_low, axis=(0, 1)).astype(np.float32)
    stretch_max = np.percentile(features, stretch_high, axis=(0, 1)).astype(np.float32)
    image_tensor = _stretch_to_tensor(features, stretch_min, stretch_max)
    preprocessing_state = {
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "stretch_min": stretch_min,
        "stretch_max": stretch_max,
        "gaussian_sigma": np.asarray(gaussian_sigma, dtype=np.float32),
    }
    return image_tensor, preprocessing_state


def _stretch_to_tensor(
    features: np.ndarray,
    stretch_min: np.ndarray,
    stretch_max: np.ndarray,
) -> torch.Tensor:
    """Apply saved percentile limits and convert HWC features to NCHW."""
    scale = stretch_max - stretch_min
    scale[scale == 0] = 1.0
    features = np.clip((features - stretch_min) / scale, 0.0, 1.0)
    image = (features * 255.0).astype(np.uint8).transpose(2, 0, 1).copy()
    return torch.from_numpy(image).float().unsqueeze(0) / 255.0


def apply_preprocessing_state(
    cube: np.ndarray,
    state: dict[str, np.ndarray],
) -> torch.Tensor:
    """Apply a saved Gaussian/PCA/stretch state without refitting PCA."""
    smoothed = gaussian_filter(cube, sigma=float(state["gaussian_sigma"]))
    height, width, bands = smoothed.shape
    pca_components = state["pca_components"]
    pca_mean = state["pca_mean"]
    if pca_mean.size != bands:
        raise ValueError(
            f"Saved PCA expects {pca_mean.size} bands, but the cube contains {bands}."
        )
    features = (smoothed.reshape(-1, bands) - pca_mean) @ pca_components.T
    features = features.reshape(height, width, pca_components.shape[0])

    return _stretch_to_tensor(features, state["stretch_min"], state["stretch_max"])


def make_paper_split(
    labels: np.ndarray,
    seed: int,
    train_samples_per_class: int,
    val_samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the strict per-class limited-label split used in the paper.

    Each class is independently shuffled with a deterministic NumPy RNG. The
    first K pixels are training samples, the next V are validation samples, and
    every remaining labelled pixel is assigned to testing. No rare-class
    fallback or proportional split is used.
    """
    if train_samples_per_class <= 0 or val_samples_per_class <= 0:
        raise ValueError("Per-class training and validation budgets must be positive.")

    flat_labels = labels.ravel()
    num_classes = int(flat_labels.max())
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    required = train_samples_per_class + val_samples_per_class
    for class_id in range(1, num_classes + 1):
        class_indices = np.flatnonzero(flat_labels == class_id).astype(np.int64)
        if class_indices.size < required:
            raise ValueError(
                f"Class {class_id} contains {class_indices.size} labelled pixels, "
                f"but the protocol requires at least {required}."
            )
        rng.shuffle(class_indices)
        train_indices.append(class_indices[:train_samples_per_class])
        val_indices.append(class_indices[train_samples_per_class:required])
        test_indices.append(class_indices[required:])

    return (
        np.concatenate(train_indices),
        np.concatenate(val_indices),
        np.concatenate(test_indices),
    )


def make_label_mask(labels: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    flat_labels = labels.ravel()
    mask = np.full(flat_labels.shape, -1, dtype=np.int64)
    mask[indices] = flat_labels[indices].astype(np.int64) - 1
    return torch.from_numpy(mask.reshape(labels.shape))
