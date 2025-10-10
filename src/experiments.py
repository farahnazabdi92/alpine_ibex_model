from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.model import IbexModel
from src.environment import project_paths, load_terrain, load_salt_points, modify_terrain, modify_salt_points
from utils.visualization import plot_heatmap, plot_population_timeseries

def run_scenario(project_root: Path, name: str, salt_modifier: float = 1.0, slope_modifier: float = 1.0,
                 n_agents: int = 60, time_steps: int = 200, seed: int | None = 42) -> pd.DataFrame:
    data_dir, figures_dir, _ = project_paths(project_root)
    terrain = load_terrain(data_dir)
    salt_points = load_salt_points(data_dir)
    terrain_mod = modify_terrain(terrain, slope_modifier)
    salt_mod = modify_salt_points(salt_points, salt_modifier)

    model = IbexModel(terrain=terrain_mod, salt_points=salt_mod,
                      n_agents=n_agents, time_steps=time_steps, slope_modifier=slope_modifier, seed=seed)
    df = model.run()
    out_csv = data_dir / f"results_{name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{name}] saved to {out_csv}")
    # Basic figures
    heatmap_path = figures_dir / f"heatmap_{name}.png"
    ts_path = figures_dir / f"pop_{name}.png"
    plot_heatmap(df, heatmap_path)
    plot_population_timeseries(df, ts_path)
    print(f"[{name}] figures saved to {heatmap_path} and {ts_path}")
    return df
