"""Dense tiled training and inference used for the paper experiments."""

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch.nn import functional as F


Tile = tuple[int, int, int, int]


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_tiles(height: int, width: int, tile_size: int, overlap: int) -> list[Tile]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Expected tile_size > 0 and 0 <= overlap < tile_size.")
    return [
        (top, min(top + tile_size, height), left, min(left + tile_size, width))
        for top in _tile_starts(height, tile_size, overlap)
        for left in _tile_starts(width, tile_size, overlap)
    ]


def labelled_pixels(label_mask: torch.Tensor, tile: Tile) -> int:
    top, bottom, left, right = tile
    return int((label_mask[top:bottom, left:right] >= 0).sum().item())


def tiles_with_labels(label_mask: torch.Tensor, tiles: Iterable[Tile]) -> list[Tile]:
    return [tile for tile in tiles if labelled_pixels(label_mask, tile) > 0]


def _balance_tiles_by_label_count(
    tiles_with_counts: list[tuple[Tile, int]],
    num_groups: int,
) -> list[list[tuple[Tile, int]]]:
    """Balance labelled pixels across optimizer-update groups."""
    num_groups = max(1, min(num_groups, len(tiles_with_counts)))
    groups: list[list[tuple[Tile, int]]] = [[] for _ in range(num_groups)]
    label_counts = [0] * num_groups
    ordered_tiles = list(tiles_with_counts)
    random.shuffle(ordered_tiles)
    ordered_tiles.sort(key=lambda item: item[1], reverse=True)
    for tile, valid_pixels in ordered_tiles:
        group_index = min(range(num_groups), key=label_counts.__getitem__)
        groups[group_index].append((tile, valid_pixels))
        label_counts[group_index] += valid_pixels
    return [group for group in groups if group]


def train_one_epoch(
    model: torch.nn.Module,
    image_cpu: torch.Tensor,
    train_mask_cpu: torch.Tensor,
    tiles: list[Tile],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_update_groups: int = 2,
) -> float:
    """Train on labelled pixels, using one optimizer step per tile group."""
    model.train()
    training_tiles = []
    for tile in tiles:
        valid_pixels = labelled_pixels(train_mask_cpu, tile)
        if valid_pixels > 0:
            training_tiles.append((tile, valid_pixels))
    if not training_tiles:
        raise RuntimeError("No training tile contains labelled pixels.")

    tile_groups = _balance_tiles_by_label_count(training_tiles, num_update_groups)
    total_valid = sum(count for _, count in training_tiles)
    average_loss = 0.0
    use_amp = device.type == "cuda"
    train_mask_batched = train_mask_cpu.unsqueeze(0)

    for group in tile_groups:
        group_valid = sum(count for _, count in group)
        optimizer.zero_grad(set_to_none=True)
        for (top, bottom, left, right), valid_pixels in group:
            image = image_cpu[:, :, top:bottom, left:right].to(device)
            labels = train_mask_batched[:, top:bottom, left:right].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = F.interpolate(
                    model(image),
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(logits, labels.long())
                weighted_loss = loss * (valid_pixels / group_valid)
            scaler.scale(weighted_loss).backward()
            average_loss += float(loss.detach().cpu()) * (valid_pixels / total_valid)
        scaler.step(optimizer)
        scaler.update()
    return average_loss


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image_cpu: torch.Tensor,
    tiles: list[Tile],
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a full scene and average logits in overlapping regions."""
    model.eval()
    height, width = image_cpu.shape[-2:]
    logit_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    logit_count = np.zeros((height, width), dtype=np.float32)
    use_amp = device.type == "cuda"

    for top, bottom, left, right in tiles:
        image = image_cpu[:, :, top:bottom, left:right].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            logits = F.interpolate(
                logits,
                size=(bottom - top, right - left),
                mode="bilinear",
                align_corners=True,
            )
        logit_sum[:, top:bottom, left:right] += logits[0].float().cpu().numpy()
        logit_count[top:bottom, left:right] += 1.0

    logit_sum /= np.maximum(logit_count[None], 1.0)
    return np.argmax(logit_sum, axis=0)
