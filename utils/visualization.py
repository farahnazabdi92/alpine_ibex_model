from __future__ import annotations
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, pandas as pd

def plot_heatmap(df: pd.DataFrame, out_path: Path, grid_size: int = 100):
    """Plot occupancy heatmap from agent positions over time."""
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    if xs.size == 0:
        return
    # Determine bounds
    x_max = int(np.ceil(xs.max())) + 1
    y_max = int(np.ceil(ys.max())) + 1
    H, xedges, yedges = np.histogram2d(ys, xs, bins=[min(y_max, grid_size), min(x_max, grid_size)])
    # Display
    plt.figure()
    plt.imshow(H, origin="lower", aspect="auto")
    plt.title("Agent Occupancy Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    #Output
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_population_timeseries(df: pd.DataFrame, out_path: Path):
    """Plot number of alive agents over time."""
    if df.empty:
        return
    alive_by_step = df.groupby("step")["alive"].sum()
    plt.figure()
    alive_by_step.plot()
    plt.title("Alive Ibex Over Time")
    plt.xlabel("Step")
    plt.ylabel("Alive")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()
