"""Render training and evaluation charts.

    python plot_training.py

runs/training.png (3D, viridis = score per game-minute):
  left:  training waterfall - training steps x run x smoothed score/min
  right: evaluation bars - policy x world age (10-min blocks) x score/min, in fresh worlds
runs/training_2d.png: the same data as flat line and dot charts, plus deaths/kills/level.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = Path("runs")
OUT = RUNS / "training_2d.png"
OUT_3D = RUNS / "training.png"

# Reference categorical palette (light mode), assigned to runs in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BASELINE = "#8d8c87"

PANELS = [
    ("game/score_per_min", "Score per game-minute"),
    ("game/deaths_per_min", "Deaths per game-minute"),
    ("game/kills_per_min", "Kills per game-minute"),
    ("episode/max_level", "Highest level per life"),
]

# Milestones shown in the 3D evaluation chart (all results stay in runs/evals.json and the 2D chart).
MILESTONES = {
    "random actions": "random",
    "built-in bot": "built-in bot",
    "v3 (7M steps, from scratch)": "PPO from scratch",
    "imitation, transformer + 7 weapons": "imitation (NN bots)",
    "v5 PPO (1007616 steps)": "v5 PPO",
    "expert: bot via action space": "expert",
    "elite 4M, no guard": "elite, no guard",
    "elite 4M + guard v3 (hull-aware)": "elite + guard",
}

RUN_NOTES = {
    "ppo_v1": "v1: 16 agents + 32 bots / world",
    "ppo_v2": "v2: 4 agents + 32 bots, resumed from v1",
    "ppo_v3": "v3: GPU, new inputs, worlds restart every 30 min",
    "ppo_v4": "v4: PPO fine-tune from bot imitation (DAgger)",
    "ppo_v5": "v5: entity transformer, imitation + KL-anchored PPO",
    "ppo_elite": "elite: aggressive reward, salvos, vs frozen NN opponents",
}


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    series: dict[str, list[tuple[int, float]]] = {}
    for events in sorted(run_dir.glob("**/events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for tag, _ in PANELS:
            if tag in acc.Tags()["scalars"]:
                series.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    return {tag: np.array(sorted(points)) for tag, points in series.items()}


def ema(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def main() -> None:
    runs = sorted((p for p in RUNS.iterdir() if p.is_dir() and p.name.startswith("ppo_")), key=lambda p: (p.name == "ppo_elite", p.name))
    colors = {run.name: SERIES[i % len(SERIES)] for i, run in enumerate(runs)}
    data = {run.name: load_run(run) for run in runs}

    fig = plt.figure(figsize=(15, 9.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.42, wspace=0.28)

    for col, (tag, title) in enumerate(PANELS):
        ax = fig.add_subplot(grid[0, col])
        style(ax, title)
        for name, series in data.items():
            if tag not in series or len(series[tag]) == 0:
                continue
            steps = series[tag][:, 0] / 1e6
            values = series[tag][:, 1]
            ax.plot(steps, values, color=colors[name], linewidth=0.8, alpha=0.25)
            ax.plot(steps, ema(values), color=colors[name], linewidth=2, label=RUN_NOTES.get(name, name))
        ax.set_xlabel("training steps (millions)", fontsize=9, color=INK_2)
        ax.set_ylim(bottom=0)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=4,
               frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.06, 0.935, "Training worlds differ between runs, so compare top-row curves within a run; "
             "the bottom chart compares runs on equal terms.", fontsize=9, color=INK_2)

    ax = fig.add_subplot(grid[1, :])
    evals_path = RUNS / "evals.json"
    evals = json.loads(evals_path.read_text()) if evals_path.exists() else []
    style(ax, "Evaluation: score per game-minute in fresh worlds (2 agents + 32 bots each), mean ± 95% CI")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    if evals:
        evals = sorted(evals, key=lambda e: e["score"][0])
        for y, e in enumerate(evals):
            mean, ci = e["score"]
            run = next((r for r in colors if r in e["policy"]), None)
            color = colors[run] if run else BASELINE
            ax.errorbar(mean, y, xerr=ci, fmt="o", color=color, markersize=9, elinewidth=2, capsize=4,
                        markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(mean + ci + 0.6, y, f"{mean:.1f}  (K/D {e['kd']:.2f}, {e['minutes']:.0f} min)",
                    va="center", fontsize=9, color=INK_2)
        ax.set_yticks(range(len(evals)), [e["label"] for e in evals], fontsize=10, color=INK)
        ax.set_ylim(-0.6, len(evals) - 0.4)
        ax.set_xlim(left=0, right=max(e["score"][0] + e["score"][1] for e in evals) * 1.35)
    else:
        ax.text(0.5, 0.5, "no evaluations yet", transform=ax.transAxes, ha="center", color=INK_2)
    ax.set_xlabel("score per game-minute", fontsize=9, color=INK_2)

    fig.savefig(OUT, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    plot_3d(data, evals)


def short_label(label: str) -> str:
    """'imitation v3 (arcs + snap)' -> 'imitation v3'; 'expert: bot via ...' -> 'expert'."""
    return label.split(" (")[0].split(":")[0]


def plot_3d(data: dict, evals: list[dict]) -> None:
    from matplotlib import cm, colors
    from matplotlib.cm import ScalarMappable

    cmap = cm.viridis
    tag = "game/score_per_min"
    runs = [name for name in data if tag in data[name] and len(data[name][tag])]
    curves = {}
    for name in runs:
        steps, values = data[name][tag][:, 0] / 1e6, ema(data[name][tag][:, 1])
        keep = np.linspace(0, len(steps) - 1, min(len(steps), 300)).astype(int)
        curves[name] = (steps[keep], values[keep])
    blocks = [e for e in evals if e.get("score_by_10min")]
    shown = [e for e in blocks if e["label"] in MILESTONES]
    if shown:
        blocks = [dict(e, label=MILESTONES[e["label"]]) for e in shown]
    zmax = max([v.max() for _, v in curves.values()] + [max(e["score_by_10min"]) for e in blocks] + [1.0])
    norm = colors.Normalize(0, zmax)

    fig = plt.figure(figsize=(17, 8.5), facecolor=SURFACE)
    pane = (0.96, 0.96, 0.95, 1.0)

    def style3d(ax, title):
        ax.set_facecolor(SURFACE)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(pane)
            axis._axinfo["grid"]["color"] = GRID
            axis.label.set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=4)

    # Training waterfall: one curtain per run, colored by score.
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    style3d(ax, "Training: smoothed score per game-minute")
    # Newest run in front so earlier, finished runs don't hide it.
    runs = runs[::-1]
    for y, name in enumerate(runs):
        x, z = curves[name]
        X = np.vstack([x, x])
        Y = np.full_like(X, y)
        Z = np.vstack([np.zeros_like(z), z])
        face = cmap(norm(np.vstack([z, z])))
        ax.plot_surface(X, Y, Z, facecolors=face, shade=False, linewidth=0, antialiased=False, alpha=0.9)
        ax.plot(x, np.full_like(x, y), z, color=INK, linewidth=1.2)
    ax.set_yticks(range(len(runs)), [n.replace("ppo_", "") for n in runs])
    ax.set_xlabel("training steps (M)")
    ax.set_zlabel("score / min")
    ax.set_zlim(0, zmax)
    ax.view_init(elev=34, azim=-62)

    # Evaluation bars: policy x world-age block, colored by score.
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    style3d(ax, "Evaluation in fresh worlds: score per game-minute by world age")
    if blocks:
        blocks = sorted(blocks, key=lambda e: e["score"][0])
        n_blocks = max(len(e["score_by_10min"]) for e in blocks)
        for y, e in enumerate(blocks):
            for b, value in enumerate(e["score_by_10min"]):
                ax.bar3d(b, y, 0, 0.7, 0.6, value, color=cmap(norm(value)), edgecolor=(0, 0, 0, 0.25),
                         linewidth=0.4, shade=True)
        ax.set_yticks(np.arange(len(blocks)) + 0.3, [short_label(e["label"]) for e in blocks], fontsize=8,
                      verticalalignment="center", horizontalalignment="left")
        ax.set_xticks(np.arange(n_blocks) + 0.35, [f"{10 * b}-{10 * (b + 1)}" for b in range(n_blocks)])
        ax.set_xlabel("world age (game-minutes)")
        ax.set_zlabel("score / min")
        ax.set_zlim(0, zmax)
        ax.view_init(elev=26, azim=-130)

    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=0.55, pad=0.02, aspect=30)
    bar.set_label("score per game-minute", color=INK_2)
    bar.ax.tick_params(colors=INK_2, labelsize=8)
    bar.outline.set_visible(False)
    fig.savefig(OUT_3D, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_3D}")


if __name__ == "__main__":
    main()
