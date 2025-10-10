from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd

from src.agents import IbexAgent

@dataclass
class IbexModel:
    terrain: np.ndarray
    salt_points: np.ndarray
    n_agents: int = 60
    time_steps: int = 200
    slope_modifier: float = 1.0
    seed: int | None = None

    agents: List[IbexAgent] = field(default_factory=list)
    step_idx: int = 0

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        h, w = self.terrain.shape

        # Spawn agents avoiding exact salt point collision
        salt_set = set((float(x), float(y)) for x, y in self.salt_points.tolist())
        for i in range(self.n_agents):
            while True:
                x = float(np.random.uniform(0, w - 1))
                y = float(np.random.uniform(0, h - 1))
                if (round(x, 1), round(y, 1)) not in salt_set:
                    break
            a = IbexAgent(
                id=i,
                x=x, y=y,
                energy=float(np.random.uniform(0.6, 1.0)),
                salt_need=float(np.random.uniform(0.2, 0.6)),
                skill=float(np.random.uniform(0.3, 0.8)),
                risk_tolerance=float(np.random.uniform(0.3, 0.8)),
            )
            self.agents.append(a)

    def step(self):
        for a in self.agents:
            a.step(self.terrain, self.salt_points, self.slope_modifier)
        self.step_idx += 1

    def run(self) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for t in range(self.time_steps):
            self.step()
            for a in self.agents:
                records.append({
                    "step": t,
                    "id": a.id,
                    "x": a.x,
                    "y": a.y,
                    "energy": a.energy,
                    "salt_need": a.salt_need,
                    "alive": int(a.alive),
                })
        return pd.DataFrame.from_records(records)


