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

        local_slope = float(terrain[iy, ix]) * slope_modifier  # assume terrain stores slope 0..1
        # Probability of fall grows with slope, decreases with skill & tolerance
        # a simple logistic: sigmoid( 8*slope - 5*(skill+risk_tol)/2 )
        p_fall = sigmoid(8.0 * local_slope - 5.0 * (self.skill + self.risk_tolerance) * 0.5)
        if np.random.rand() < p_fall * 0.05:  # falls are rare but possible
            self.alive = False
            self.path.append((self.x, self.y))
            return

        # Choose a movement direction
        target = None
        if self.salt_need > self.salt_seek_threshold and len(salt_points) > 0:
            # Seek nearest salt point
            dists = np.sqrt(((salt_points[:, 0] - self.x) ** 2) + ((salt_points[:, 1] - self.y) ** 2))
            nearest_idx = int(np.argmin(dists))
            target = (float(salt_points[nearest_idx, 0]), float(salt_points[nearest_idx, 1]))

        # Compute movement vector
        dx = dy = 0.0
        if target is not None:
            # Move towards salt, but avoid very steep slopes by shortening the step
            vec = np.array([target[0] - self.x, target[1] - self.y], dtype=float)
            norm = np.linalg.norm(vec) + 1e-8
            step_len = min(self.max_step, norm)
            # reduce step length when slope is high
            step_len *= float(np.clip(1.0 - local_slope, 0.2, 1.0))
            move = vec / norm * step_len
            dx, dy = float(move[0]), float(move[1])

        else:
            # Random roaming with slight downhill bias
            angle = np.random.rand() * 2 * np.pi
            step_len = self.max_step * (0.6 + 0.4 * (1.0 - local_slope))
            dx = float(np.cos(angle) * step_len)
            dy = float(np.sin(angle) * step_len)

        # Apply movement & keep within bounds
        self.x = float(np.clip(self.x + dx, 0, w - 1))
        self.y = float(np.clip(self.y + dy, 0, h - 1))

        # Energy & salt dynamics
        move_cost = self.base_move_cost + self.slope_move_cost * local_slope
        self.energy = float(np.clip(self.energy - move_cost, 0.0, 1.0))
        self.salt_need = float(np.clip(self.salt_need + self.salt_growth_per_step, 0.0, 1.0))

        # Intake if close to salt
        if len(salt_points) > 0:
            dist_to_salt = np.min(np.sqrt(((salt_points[:, 0] - self.x) ** 2) + ((salt_points[:, 1] - self.y) ** 2)))
            if dist_to_salt <= self.intake_radius:
                self.salt_need = float(np.clip(self.salt_need - self.salt_gain_per_intake, 0.0, 1.0))
                self.energy = float(np.clip(self.energy + self.energy_gain_per_intake, 0.0, 1.0))

