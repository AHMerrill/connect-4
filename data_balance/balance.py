# connect-4/balancing/balance.py
"""
Dynamic move-depth reweighting for Connect-4 datasets.

Design principles:
- NO samples are dropped.
- NO resampling.
- Weighting is fully data-driven (based on actual move distribution).
- Overrepresented early-game boards contribute less to loss.
- Underrepresented late-game boards contribute more to loss.
- Safe to apply AFTER mirroring.

Intended usage:
    data = mirror_dataset(data)
    data, sample_weights = compute_move_balance_weights(data)
"""

import numpy as np
from typing import Dict, Tuple


def compute_move_balance_weights(
    data: Dict[str, np.ndarray],
    num_bins: int = 10,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute per-sample loss weights based on move count bins.

    Required keys in `data`:
      - "X" : (N, 6, 7, 2)

    Returns:
      - data (unchanged, passed through for convenience)
      - sample_weights : (N,) float array, mean-normalized
    """

    if "X" not in data:
        raise KeyError("Dataset must contain key 'X'")

    X = data["X"]
    N = X.shape[0]

    # --------------------------------------------------
    # 1. Compute move count from board occupancy
    # --------------------------------------------------
    # Each channel is a binary plane; sum gives stones on board
    move_count = np.count_nonzero(
        X[..., 0] + X[..., 1],
        axis=(1, 2),
    )

    # --------------------------------------------------
    # 2. Bin by move count (uniform-width bins)
    # --------------------------------------------------
    min_m, max_m = move_count.min(), move_count.max()
    bins = np.linspace(min_m, max_m + 1, num_bins + 1)

    bin_ids = np.digitize(move_count, bins) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    # --------------------------------------------------
    # 3. Inverse-frequency weighting (dynamic)
    # --------------------------------------------------
    bin_counts = np.bincount(bin_ids, minlength=num_bins)

    # Highest-frequency bin gets weight 1.0 (before normalization)
    bin_weights = bin_counts.max() / np.maximum(bin_counts, 1)

    sample_weights = bin_weights[bin_ids]

    # Normalize so mean weight == 1.0 (stable training)
    sample_weights /= sample_weights.mean()

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print("Move-count balancing:")
    print(f"  samples           : {N}")
    print(f"  bins              : {num_bins}")
    print(f"  move range        : [{min_m}, {max_m}]")
    print(f"  bin counts        : {bin_counts.tolist()}")
    print(f"  bin weight range  : [{bin_weights.min():.3f}, {bin_weights.max():.3f}]")
    print(f"  sample weight μ   : {sample_weights.mean():.3f}")

    return data, sample_weights
