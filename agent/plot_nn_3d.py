"""3D views of the networks (viridis).

    python plot_nn_3d.py        # -> runs/nn_3d.png

1. Training losses of the elite run as a 3D waterfall (each metric scaled to 0-1, smoothed).
2. Loss landscape of the imitation network: its imitation loss on fresh game data over a 2D
   slice of weight space through the trained weights (filter-normalized random directions,
   Li et al. 2018).
3. Inside the elite network at one real game moment: every neuron, layer by layer, colored by
   its activation, with the decision it made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib.cm import ScalarMappable
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import entity_policy as ep
from evaluate import type_names
from mk48env import SERVER_BUILDS, Mk48VecEnv, ServerConfig
from plot_training import GRID, INK, INK_2, SURFACE, ema
from train_bc_entity import bc_loss

OUT = Path("runs/nn_3d.png")
CMAP = cm.viridis
LOSS_TAGS = [
    ("train/value_loss", "value loss"),
    ("train/pg_loss", "policy loss"),
    ("train/entropy", "entropy"),
    ("train/kl_ref", "KL to imitation"),
    ("train/approx_kl", "update size"),
    ("train/clipfrac", "clipped share"),
]


def style3d(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.96, 0.96, 0.95, 1.0))
        axis._axinfo["grid"]["color"] = GRID
        axis.label.set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=4)


def pad_actions(actions: np.ndarray, act_dim: int) -> np.ndarray:
    """Older 9-action policies on the 10-action server: no salvo."""
    if actions.shape[1] < act_dim:
        actions = np.hstack([actions, -np.ones((len(actions), act_dim - actions.shape[1]), np.float32)])
    return actions


# --- 1. training losses ---------------------------------------------------------------------------

def training_losses(run_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    series = {}
    for events in sorted(run_dir.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for tag, _ in LOSS_TAGS:
            if tag in acc.Tags()["scalars"]:
                series.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    out = []
    for tag, name in LOSS_TAGS:
        if tag in series:
            pts = np.array(sorted(series[tag]))
            out.append((name, pts[:, 0] / 1e6, ema(pts[:, 1], 0.05)))
    return out


# --- 2. loss landscape ----------------------------------------------------------------------------

def collect(policy: ep.Policy, steps: int) -> list[torch.Tensor]:
    """Fresh game data driven by `policy`, labelled with the built-in bot's choices."""
    env = Mk48VecEnv(n_procs=4, server=ServerConfig(agents=4, bots=24, expert_labels=True,
                                                    server_path=SERVER_BUILDS["v7"]))
    act_dim, e0 = env.action_space.shape[0], env.expert_offset
    obs = env.reset()
    data = ([], [], [])
    for _ in range(steps):
        expert = env.last_info[:, e0 : e0 + act_dim + 2]
        valid = expert[:, act_dim] > 0
        labels, masks = ep.from_expert(expert, obs, policy.layout)
        for store, value in zip(data, (obs[valid], labels[valid], masks[valid])):
            store.append(value)
        with torch.no_grad():
            discrete = ep.sample(policy(torch.as_tensor(obs)), deterministic=True).numpy()
        obs, *_ = env.step(pad_actions(ep.to_env(discrete, obs, policy.layout), act_dim))
    env.close()
    return [torch.as_tensor(np.concatenate(v)) for v in data]


def landscape(policy: ep.Policy, X, Y, M, n: int, span: float, device: str):
    policy = policy.to(device).eval()
    params = list(policy.parameters())
    base = [p.detach().clone() for p in params]
    gen = torch.Generator().manual_seed(0)

    def direction():
        out = []
        for p in base:
            r = torch.randn(p.shape, generator=gen).to(device)
            if p.dim() <= 1:
                r.zero_()  # biases and norms stay put (standard practice)
            else:
                pf, rf = p.reshape(p.shape[0], -1), r.reshape(r.shape[0], -1)
                rf *= pf.norm(dim=1, keepdim=True) / (rf.norm(dim=1, keepdim=True) + 1e-10)
            out.append(r)
        return out

    d1, d2 = direction(), direction()
    X, Y, M = X.float().to(device), Y.to(device), M.to(device)
    grid = np.linspace(-span, span, n)
    Z = np.zeros((n, n))
    with torch.no_grad():
        for i, a in enumerate(grid):
            for j, b in enumerate(grid):
                for p, p0, u, v in zip(params, base, d1, d2):
                    p.copy_(p0 + a * u + b * v)
                total = 0.0
                for k in range(0, len(X), 4096):
                    total += float(bc_loss(policy(X[k : k + 4096]), Y[k : k + 4096], M[k : k + 4096])) * len(X[k : k + 4096])
                Z[i, j] = total / len(X)
        for p, p0 in zip(params, base):
            p.copy_(p0)
    return grid, Z


# --- 3. inside the network ------------------------------------------------------------------------

def capture_moment(policy: ep.Policy, max_steps: int = 3000) -> np.ndarray:
    """An observation where the policy fires at a chosen target with plenty of contacts around."""
    layout = policy.layout
    env = Mk48VecEnv(n_procs=2, server=ServerConfig(agents=2, bots=24, server_path=SERVER_BUILDS["v7"]))
    act_dim = env.action_space.shape[0]
    obs = env.reset()
    moment = None
    for _ in range(max_steps):
        with torch.no_grad():
            discrete = ep.sample(policy(torch.as_tensor(obs)), deterministic=True).numpy()
        contacts = obs[:, layout.self_dim : layout.self_dim + ep.K * layout.contact_dim].reshape(-1, ep.K, layout.contact_dim)
        crowded = (contacts[:, :, 0] > 0.5).sum(1) >= 8
        firing = (discrete[:, ep.H["fire"]] == 1) & (discrete[:, ep.H["target"]] > 0)
        found = np.flatnonzero(crowded & firing)
        if len(found):
            moment = obs[found[0]].copy()
            break
        obs, *_ = env.step(pad_actions(ep.to_env(discrete, obs, layout), act_dim))
    env.close()
    return moment if moment is not None else obs[0].copy()


def activations(policy: ep.Policy, obs: np.ndarray):
    acts: dict[str, np.ndarray] = {}
    encoder = policy.encoder
    hooks = [encoder.transformer.register_forward_hook(
        lambda m, args, out: acts.__setitem__("tokens", args[0][0].detach().numpy()))]
    for k, layer in enumerate(encoder.transformer.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, args, out, k=k: acts.__setitem__(f"layer {k + 1}", out[0].detach().numpy())))
    hooks.append(encoder.out.register_forward_hook(
        lambda m, args, out: acts.__setitem__("latent", out[0].detach().numpy())))
    with torch.no_grad():
        logits = policy(torch.as_tensor(obs[None]))
    for h in hooks:
        h.remove()
    probs = {n: torch.softmax(logits[n][0], -1).numpy() for n in ep.HEAD_NAMES if n in logits}
    return acts, probs


def describe(obs: np.ndarray, probs: dict, layout: ep.Layout) -> str:
    names = type_names()
    choice = {n: int(p.argmax()) for n, p in probs.items()}
    contacts = obs[layout.self_dim : layout.self_dim + ep.K * layout.contact_dim].reshape(ep.K, layout.contact_dim)
    own = names.get(int(obs[layout.own_type]), "?")
    target = choice["target"]
    if target:
        c = contacts[target - 1]
        kind = names.get(int(c[layout.contact_type]) - 1, "unidentified contact") if c[layout.contact_type] else "unidentified contact"
        side = "friendly" if c[ep.FRIENDLY] > 0.5 else "enemy"
        target_text = f"{side} {kind}"
    else:
        target_text = "none"
    parts = [
        f"as {own}: steer {ep.STEER_BINS[choice['steer']] * 180:+.0f}°",
        f"throttle {ep.THROTTLE_BINS[choice['throttle']]:.0%}",
        f"target {target_text}",
    ]
    if choice["fire"]:
        parts.append(f"fire {ep.WEAPON_NAMES_V6[choice['weapon']]}" + (" + salvo" if choice.get("salvo") else ""))
    return "Decision " + ", ".join(parts)


def sheet(values: np.ndarray, rows: int, cols: int):
    """Grid coordinates (y across, z down) for a layer drawn as a sheet of neurons."""
    v = np.zeros(rows * cols)
    v[: values.size] = values.ravel()
    r, c = np.divmod(np.arange(rows * cols), cols)
    return c / max(cols - 1, 1), 1 - r / max(rows - 1, 1), v, values.size


# --- figure ---------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--elite", default="models/elite_15M.pt")
    p.add_argument("--imitation", default="models/bc_v6.pt")
    p.add_argument("--run", default="runs/ppo_elite")
    p.add_argument("--device", default="mps")
    p.add_argument("--grid", type=int, default=21)
    a = p.parse_args()

    fig = plt.figure(figsize=(22, 8), facecolor=SURFACE)

    # 1. Training losses.
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    style3d(ax, "Elite training losses (each scaled 0–1)")
    curves = training_losses(Path(a.run))
    for y, (name, x, v) in enumerate(curves):
        lo, hi = np.percentile(v, [2, 98])  # robust: early warm-up spikes don't flatten the rest
        z = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
        keep = np.linspace(0, len(x) - 1, min(len(x), 300)).astype(int)
        x, z = x[keep], z[keep]
        X = np.vstack([x, x])
        ax.plot_surface(X, np.full_like(X, y), np.vstack([np.zeros_like(z), z]),
                        facecolors=CMAP(np.vstack([z, z])), shade=False, linewidth=0, antialiased=False, alpha=0.9)
        ax.plot(x, np.full_like(x, y), z, color=INK, linewidth=1.0)
    if not curves:
        ax.text2D(0.5, 0.5, "no training logs yet:\nrun ppo_entity.py --run ppo_elite", transform=ax.transAxes,
                  ha="center", color=INK_2)
    ax.set_yticks(range(len(curves)), [c[0] for c in curves], fontsize=8)
    ax.set_xlabel("training steps (M)")
    ax.set_zlabel("scaled value")
    ax.view_init(elev=28, azim=-62)

    # 2. Loss landscape.
    cache = Path("runs/landscape.npz")
    if cache.exists() and np.load(cache)["grid"].size == a.grid:
        saved = np.load(cache)
        grid, Z, samples = saved["grid"], saved["Z"], int(saved["samples"])
    else:
        imitation = ep.load(a.imitation, "cpu")[0].eval()
        X, Y, M = collect(imitation, steps=250)
        grid, Z = landscape(imitation, X, Y, M, a.grid, 1.0, a.device)
        samples = len(X)
        np.savez(cache, grid=grid, Z=Z, samples=samples)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    style3d(ax, f"Imitation loss landscape\nred dot = trained weights, {samples:,} game moments")
    log = Z.max() / max(Z.min(), 1e-9) > 20
    show = np.log10(Z) if log else Z
    G1, G2 = np.meshgrid(grid, grid, indexing="ij")
    norm = colors.Normalize(show.min(), show.max())
    ax.plot_surface(G1, G2, show, facecolors=CMAP(norm(show)), shade=False, linewidth=0.2,
                    edgecolor=(0, 0, 0, 0.08), antialiased=True)
    centre = show[len(grid) // 2, len(grid) // 2]
    ax.scatter([0], [0], [centre], color="#e34948", s=40, depthshade=False, zorder=10)
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_zlabel("log₁₀ loss" if log else "loss")
    ax.view_init(elev=32, azim=-50)

    # 3. Inside the elite network.
    elite = ep.load(a.elite, "cpu")[0].eval()
    moment = capture_moment(elite)
    acts, probs = activations(elite, moment)
    def scaled(v):  # activation magnitude, scaled per layer by its 99th percentile
        v = np.abs(v)
        return np.clip(v / max(np.percentile(v, 99), 1e-9), 0, 1)

    layers = [("inputs\n1,201", sheet(np.clip(moment, -1, 1) * 0.5 + 0.5, 35, 35), 2.5)]
    for name in ("tokens", "layer 1", "layer 2"):
        v = acts[name]  # (26 tokens, 128)
        layers.append((f"{name}\n26×128", sheet(scaled(v), v.shape[0], v.shape[1]), 1.6))
    layers.append(("latent\n256", sheet(scaled(acts["latent"]), 16, 16), 10.0))
    width = max(len(v) for v in probs.values())
    decision = np.zeros((len(probs), width))
    for r, v in enumerate(probs.values()):
        decision[r, : len(v)] = v
    layers.append(("decisions\n" + str(len(probs)) + " heads", sheet(decision, len(probs), width), 14.0))

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    style3d(ax, "Inside the elite at one game moment")
    for x, (name, (yy, zz, v, count), size) in enumerate(layers):
        vis = np.arange(len(v)) < count
        ax.scatter(np.full(vis.sum(), x), yy[vis], zz[vis], c=CMAP(v[vis]), s=size, depthshade=False,
                   alpha=0.85, linewidths=0)
    ax.set_xticks(range(len(layers)), [l[0] for l in layers], fontsize=7)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((2.4, 1, 0.9))
    ax.view_init(elev=22, azim=-40)
    ax.text2D(0.02, -0.02, describe(moment, probs, elite.layout), transform=ax.transAxes, fontsize=9, color=INK)

    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(0, 1), cmap=CMAP), ax=fig.axes, shrink=0.5,
                       pad=0.02, aspect=30)
    bar.set_label("low → high (per panel: scaled loss, loss, activation / probability)", color=INK_2, fontsize=8)
    bar.set_ticks([])
    bar.outline.set_visible(False)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
