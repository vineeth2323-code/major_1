"""Static route plots with matplotlib (handy for headless reports)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .environment import CityGrid  # noqa: E402
from .routing import RouteResult  # noqa: E402


def plot_route(city: CityGrid, route: RouteResult, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(city.cols / 3, city.rows / 3 + 0.6), dpi=100)
    ax.imshow(np.where(city.passable, 0.85, 0.25), cmap="gray", vmin=0, vmax=1, origin="upper")
    if route.explored:
        er, ec = zip(*route.explored)
        ax.scatter(ec, er, s=12, c="tab:blue", alpha=0.35, label="explored")
    if route.path:
        pr, pc = zip(*route.path)
        ax.plot(pc, pr, color="gold", linewidth=2.5, label=f"A* path ({len(route.path) - 1} steps)")
    ir, ic = zip(*city.intersections)
    ax.scatter(ic, ir, s=10, c="black", marker="s", label="intersections")
    ax.scatter([city.start[1]], [city.start[0]], s=120, c="green", label="start", zorder=5)
    ax.scatter([city.goal[1]], [city.goal[0]], s=120, c="red", label="goal", zorder=5)
    ax.set_title("A* route through city grid")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
