"""Dataset splits: photo_id-aware train/val/test splitting.

Ensures that all crops from the same photo_id go into the same split,
preventing data leakage between train and test sets.
"""

from __future__ import annotations

import random
from typing import List


def split_photo_ids(
    photo_ids: list[str],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split a list of photo IDs into train/val/test sets.

    Splits are by photo_id, so all crops from one photo go to the same
    split. This prevents data leakage.

    Args:
        photo_ids: list of unique photo identifiers.
        ratios: (train, val, test) ratios summing to 1.0.
        seed: random seed for reproducibility.

    Returns:
        dict with keys "train", "val", "test", each containing a list of photo IDs.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    rng = random.Random(seed)
    shuffled = list(photo_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return {"train": train, "val": val, "test": test}