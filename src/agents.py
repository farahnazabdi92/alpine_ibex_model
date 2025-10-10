from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from utils.calculations import sigmoid, distance_clipped

@dataclass
class IbexAgent:
    id: int
    x: float
    y: float
    energy: float = 1.0           # 0..1
    salt_need: float = 0.3        # 0..1
    skill: float = 0.5            # 0..1 (climbing skill)
    risk_tolerance: float = 0.5   # 0..1 (higher => accepts higher risk)
    alive: bool = True
    path: List[Tuple[float, float]] = field(default_factory=list)
