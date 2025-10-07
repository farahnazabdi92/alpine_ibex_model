import os
from src.model import IbexABM

def run_scenario(project_root, terrain, salt_points, name, salt_modifier=1.0, slope_modifier=1.0,
                 n_agents=60, time_steps=200):
    print(f"\nRunning scenario: {name}")
    terrain_mod = terrain * slope_modifier
    n_salt = max(1, int(len(salt_points) * salt_modifier))
    salt_mod = salt_points[:n_salt]
    model = IbexABM(terrain_mod, salt_mod, n_agents=n_agents, time_steps=time_steps)
    model.run()
    df = model.to_dataframe()
    out = os.path.join(project_root, f"data/results_{name}.csv")
    df.to_csv(out, index=False)
    print(f"Saved {name} results to {out}")
    return df
