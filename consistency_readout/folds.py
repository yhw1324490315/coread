"""Deterministic grouped fold splitting routines."""

from __future__ import annotations

import hashlib
import numpy as np


def inner_folds(groups: np.ndarray | list[str], n: int = 3, seed: int | str = 20260923) -> np.ndarray:
    """Create balanced grouped k-fold assignments.
    
    Args:
        groups: 1D array of group/sample IDs. All samples with identical group ID are kept in the same fold.
        n: Number of folds (default: 3).
        seed: Random seed or hash prefix for deterministic tie-breaking.
        
    Returns:
        1D integer array of fold assignments in [0, n-1].
    """
    groups_arr = np.asarray(groups, str)
    unique, counts = np.unique(groups_arr, return_counts=True)
    if len(unique) < n:
        raise ValueError(f"Insufficient independent groups ({len(unique)}) for {n} folds")
        
    # Sort groups descending by count, tie-break by sha256 hash
    order = sorted(
        zip(unique, counts),
        key=lambda z: (-z[1], hashlib.sha256(f"{seed}:{z[0]}".encode()).hexdigest())
    )
    
    sizes = [0] * n
    mapping = {}
    for group, count in order:
        # Place into fold with smallest current total size
        fold = min(range(n), key=lambda f: (sizes[f], f))
        mapping[group] = fold
        sizes[fold] += int(count)
        
    return np.array([mapping[g] for g in groups_arr], dtype=int)
