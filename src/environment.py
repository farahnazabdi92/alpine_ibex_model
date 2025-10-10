from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

def project_paths(project_root: Path) -> Tuple[Path, Path, Path]:
    data = project_root / "data"
    figures = project_root / "figures"
    utils = project_root / "utils"
    data.mkdir(exist_ok=True, parents=True)
    figures.mkdir(exist_ok=True, parents=True)
    return data, figures, utils

def load_terrain(data_dir: Path) -> np.ndarray:
    """Load normalized slope/terrain array stored in .npy (values ~ 0..1)."""
    path = data_dir / "terrain.npy"
    arr = np.load(path)
    # Normalize if not in 0..1
    arr = arr.astype(float)
    if arr.max() > 1.01:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def load_salt_points(data_dir: Path) -> np.ndarray:
    """Load salt points CSV with columns: x,y  (float)."""
    path = data_dir / "salt_points.csv"
    df = pd.read_csv(path)
    pts = df[["x", "y"]].to_numpy(dtype=float)
    return pts

def modify_terrain(terrain: np.ndarray, slope_modifier: float) -> np.ndarray:
    return np.clip(terrain * float(slope_modifier), 0.0, 1.0)

def modify_salt_points(salt_points: np.ndarray, salt_modifier: float) -> np.ndarray:
    if salt_modifier >= 1.0:
        return salt_points
    # randomly sample a subset
    n = len(salt_points)
    keep = int(max(1, np.floor(n * salt_modifier)))
    idx = np.random.choice(np.arange(n), size=keep, replace=False)
    return salt_points[idx]