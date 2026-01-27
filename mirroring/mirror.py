# connect-4/mirroring/mirror.py
"""
Mirror Connect-4 datasets left↔right.

Design choice (IMPORTANT, intentional):
- This file DOES NOT deduplicate or average boards.
- If a mirrored board already exists in the dataset, both copies are kept.
- This preserves MCTS visitation frequency as learning signal.
- Dataset balance is handled later via loss reweighting, not here.

This keeps mirroring simple, reproducible, and non-destructive.
"""

import numpy as np
from typing import Dict


def mirror_dataset(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Mirrors boards and all aligned targets horizontally.

    Expected keys in `data`:
      - "X"        : (N, 6, 7, 2)
      - "policy"  : (N, 7)
      - "value"   : (N, 1)
    Optional keys (mirrored if present):
      - "boards"  : (N, 6, 7)
      - "visits"  : (N, 7)
      - "scores"  : (N, 7)
      - "q"       : (N, 7)

    Returns:
      New dict with original + mirrored samples concatenated.
    """

    required = ["X", "policy", "value"]
    for k in required:
        if k not in data:
            raise KeyError(f"Required key '{k}' not found in dataset")

    X = data["X"]
    policy = data["policy"]
    value = data["value"]

    N = X.shape[0]

    # --- Mirror core tensors ---
    X_m = X[:, :, ::-1, :]           # mirror columns
    policy_m = policy[:, ::-1]       # mirror action probs
    value_m = value.copy()           # scalar, unchanged

    out = {
        "X": np.concatenate([X, X_m], axis=0),
        "policy": np.concatenate([policy, policy_m], axis=0),
        "value": np.concatenate([value, value_m], axis=0),
    }

    # --- Optional aligned tensors ---
    if "boards" in data:
        boards = data["boards"]
        boards_m = boards[:, :, ::-1]
        out["boards"] = np.concatenate([boards, boards_m], axis=0)

    if "visits" in data:
        visits = data["visits"]
        visits_m = visits[:, ::-1]
        out["visits"] = np.concatenate([visits, visits_m], axis=0)

    if "scores" in data:
        scores = data["scores"]
        scores_m = scores[:, ::-1]
        out["scores"] = np.concatenate([scores, scores_m], axis=0)

    if "q" in data:
        q = data["q"]
        q_m = q[:, ::-1]
        out["q"] = np.concatenate([q, q_m], axis=0)

    print(f"Mirroring complete: {N} → {out['X'].shape[0]} samples")

    return out
