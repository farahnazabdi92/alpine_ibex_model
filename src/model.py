import pandas as pd
import numpy as np
from src.agents import IbexAgent
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

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


class IbexABM:
    def __init__(self, terrain, salt_points, n_agents=60, time_steps=200, seed=123):
        self.terrain = terrain
        self.salt_points = salt_points
        self.n_agents = n_agents
        self.time_steps = time_steps
        self.rng = None
        self.agents = [IbexAgent(i, terrain, salt_points) for i in range(n_agents)]
        self.history = []

    def run(self, log_every=20):
        for t in range(self.time_steps):
            alive_count = 0
            for agent in self.agents:
                agent.step()
                if agent.alive:
                    alive_count += 1
                    self.history.append({
                        "time": t, "id": agent.id, "x": agent.x, "y": agent.y,
                        "energy": agent.energy, "salt_need": agent.salt_need,
                        "alive": agent.alive, "consumed_salt_id": agent.consumed_salt_id
                    })
            if t % log_every == 0:
                print(f"Step {t}: {alive_count} ibex alive.")

    def to_dataframe(self):
        return pd.DataFrame(self.history)
