import numpy as np
from utils.calculations import sigmoid, distance

class IbexAgent:
    """
    Alpine ibex individual agent.
    v2 changes:
    - Lower energy decay per step
    - Slower salt-need growth
    - Lower fall probability (more weight to skill)
    - Larger salt consumption radius
    """
    def __init__(self, idx, terrain, salt_points):
        self.id = idx
        n = terrain.shape[0]
        self.x, self.y = np.random.randint(0, n), np.random.randint(0, n)
        self.energy = 1.0
        self.salt_need = np.random.uniform(0.25, 0.5)
        self.climb_skill = np.random.uniform(0.45, 0.85)
        self.risk_tolerance = np.random.uniform(0.45, 0.75)
        self.alive = True

        self.terrain = terrain          # slope in [0, ~1.2]
        self.salt_points = salt_points  # array of [x,y]

        self.consumed_salt_id = None

    def step(self):
        if not self.alive:
            return

        slope_here = float(self.terrain[self.x, self.y])

        # --- v2: gentler fall probability
        p_fall = sigmoid(2.8 * slope_here - 3.5 * self.climb_skill)

        # stochastic falling (tempered by risk tolerance)
        if np.random.rand() < p_fall * (1 - self.risk_tolerance):
            self.alive = False
            return

        # --- v2: gentler energy and slower salt accumulation
        self.energy -= 0.004 * (1.0 + slope_here)
        self.salt_need += 0.02

        # nearest salt point
        nearest_idx, nearest_pt, min_d = None, None, float("inf")
        for i, s in enumerate(self.salt_points):
            d = distance((self.x, self.y), (s[0], s[1]))
            if d < min_d:
                min_d, nearest_idx, nearest_pt = d, i, s

        # consume salt if within radius (v2: radius 5)
        if nearest_pt is not None and min_d < 5.0:
            self.energy = min(1.0, self.energy + 0.25)
            self.salt_need = max(0.0, self.salt_need - 0.6)
            self.consumed_salt_id = int(nearest_idx)

        # movement: be exploratory if need salt
        n = self.terrain.shape[0]
        explore = (np.random.rand() < 0.6) or (self.salt_need > 0.55)
        if explore and nearest_pt is not None:
            # biased random step towards nearest salt
            dx = int(np.sign(nearest_pt[0] - self.x))
            dy = int(np.sign(nearest_pt[1] - self.y))
            # add jitter
            dx += np.random.choice([-1,0,1])
            dy += np.random.choice([-1,0,1])
        else:
            dx, dy = np.random.choice([-1,0,1]), np.random.choice([-1,0,1])

        self.x = int(np.clip(self.x + dx, 0, n-1))
        self.y = int(np.clip(self.y + dy, 0, n-1))

        if self.energy <= 0:
            self.alive = False
