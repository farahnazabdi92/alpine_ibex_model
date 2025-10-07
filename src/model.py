import pandas as pd
from src.agents import IbexAgent

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
