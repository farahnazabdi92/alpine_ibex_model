from __future__ import annotations
import numpy as np

def sigmoid(z: float) -> float:
    z = float(z)
    return 1.0 / (1.0 + np.exp(-z))

def distance_clipped(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


