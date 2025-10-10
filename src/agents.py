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

    # Simulation constants
    intake_radius: float = 2.5
    base_move_cost: float = 0.003
    slope_move_cost: float = 0.004
    salt_gain_per_intake: float = 0.25
    energy_gain_per_intake: float = 0.15
    salt_growth_per_step: float = 0.01
    salt_seek_threshold: float = 0.4
    max_step: float = 1.5

    def step(self, terrain: np.ndarray, salt_points: np.ndarray, slope_modifier: float = 1.0):
        """Advance the agent by one time step.
        - compute local slope and fall probability
        - optionally move (seek salt or roam)
        - update energy/salt
        - record path
        """
        if not self.alive:
            return

        h, w = terrain.shape
        # Clip to valid indices
        ix = int(np.clip(round(self.x), 0, w - 1))
        iy = int(np.clip(round(self.y), 0, h - 1))