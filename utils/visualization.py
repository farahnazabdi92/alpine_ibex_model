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

def plot_heatmap(df, terrain, title, fig_dir, filename):
    if df is None or df.empty:
        print(f"No data for {title}, skipping.")
        return
    heat = np.zeros_like(terrain)
    for _, row in df.iterrows():
        if bool(row.get("alive", True)):
            x, y = int(row["x"]), int(row["y"])
            if 0 <= x < heat.shape[0] and 0 <= y < heat.shape[1]:
                heat[x, y] += 1
    plt.imshow(heat, cmap="hot", origin="lower")
    plt.title(title)
    plt.colorbar(label="Visit count")
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved: {path}")

def plot_salt_visits(df_dict, fig_dir):
    def salt_visit_counts(df):
        if df is None or df.empty or "consumed_salt_id" not in df.columns:
            return pd.Series(dtype=int)
        s = df["consumed_salt_id"].dropna()
        return s.astype(int).value_counts().sort_index() if not s.empty else pd.Series(dtype=int)

    vb = salt_visit_counts(df_dict.get("baseline"))
    vl = salt_visit_counts(df_dict.get("low_salt"))
    vs = salt_visit_counts(df_dict.get("steeper"))
    visits = pd.DataFrame({"baseline": vb, "low_salt": vl, "steeper": vs}).fillna(0).astype(int)
    if visits.empty:
        print("No salt visit data to plot.")
        return
    ax = visits.plot.bar(figsize=(8,4))
    ax.set_xlabel("Salt Point ID"); ax.set_ylabel("Visits")
    ax.set_title("Salt point visits across scenarios")
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "salt_visits_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved {path}")
