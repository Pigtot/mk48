"""DAgger imitation of the built-in bot for the entity-transformer policy (cross-entropy heads).

    python train_bc_entity.py --device mps

Same scheme as train_bc.py: iteration 0 follows the expert, later iterations follow the learner
while the server labels every visited observation with the bot's choice for that exact view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import entity_policy as ep
from mk48env import SCORE_DELTA, SERVER_BUILDS, Mk48VecEnv, ServerConfig
HEAD_WEIGHT = {"target": 2.0, "fire": 2.0}


def bc_loss(logits: dict[str, torch.Tensor], labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    total = 0.0
    for i, name in enumerate(ep.HEAD_NAMES):
        ce = F.cross_entropy(logits[name], labels[:, i], reduction="none")
        m = masks[:, i]
        total = total + HEAD_WEIGHT.get(name, 1.0) * (ce * m).sum() / m.sum().clamp(min=1.0)
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--steps-per-iter", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--bots", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="runs/bc_v7/policy.pt")
    p.add_argument("--layout", choices=["v6", "v7"], default="v7",
                   help="v7 = current server source (10 actions); v6 = the saved build bc_v6 used")
    a = p.parse_args()

    env = Mk48VecEnv(
        n_procs=a.procs,
        server=ServerConfig(agents=a.agents, bots=a.bots, expert_labels=True, server_path=SERVER_BUILDS[a.layout]),
        world_minutes=30,
    )
    layout = {"v6": ep.V6, "v7": ep.V7}[a.layout]
    assert env.observation_space.shape[0] == layout.obs_dim, env.observation_space.shape
    assert env.action_space.shape[0] == layout.act_dim, (env.action_space.shape, layout)
    policy = ep.Policy(layout).to(a.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    e0, act_dim = env.expert_offset, env.action_space.shape[0]
    rng = np.random.default_rng()

    data_obs, data_labels, data_masks = [], [], []
    obs = env.reset()
    for it in range(a.iterations):
        beta = [1.0, 0.5, 0.25, 0.1][it] if it < 4 else 0.0
        score, steps = 0.0, 0
        hits = np.zeros(len(layout.head_names))
        counts = np.zeros(len(layout.head_names))
        fire_tp = fire_label = fire_pred = 0
        policy.eval()
        for _ in range(a.steps_per_iter):
            expert = env.last_info[:, e0 : e0 + act_dim + 2]
            valid = expert[:, act_dim] > 0
            labels, masks = ep.from_expert(expert, obs, layout)
            data_obs.append(obs[valid].astype(np.float16))
            data_labels.append(labels[valid])
            data_masks.append(masks[valid])

            with torch.no_grad():
                learner = ep.sample(policy(torch.as_tensor(obs, device=a.device)), deterministic=True).cpu().numpy()
            hits += ((learner == labels) * masks)[valid].sum(0)
            counts += masks[valid].sum(0)
            f = ep.H["fire"]
            fire_label += int(labels[valid, f].sum())
            fire_pred += int(learner[valid, f].sum())
            fire_tp += int((labels[valid, f] & learner[valid, f]).sum())

            actions = ep.to_env(learner, obs, layout)
            use_expert = (rng.random(env.num_envs) < beta) & valid
            actions[use_expert] = expert[use_expert, :act_dim]
            obs, _, _, _ = env.step(actions)
            score += env.last_info[:, SCORE_DELTA].sum()
            steps += env.num_envs

        X = torch.as_tensor(np.concatenate(data_obs))
        Y = torch.as_tensor(np.concatenate(data_labels))
        M = torch.as_tensor(np.concatenate(data_masks))
        policy.train()
        losses = []
        for _ in range(a.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), 4096):
                idx = perm[start : start + 4096]
                logits = policy(X[idx].float().to(a.device))
                loss = bc_loss(logits, Y[idx].to(a.device), M[idx].to(a.device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

        acc = hits / np.maximum(counts, 1)
        minutes = steps * env.server_cfg.ticks_per_step * 0.1 / 60
        print(
            f"iter {it} beta={beta:.2f} samples={len(X):,} loss={np.mean(losses[-50:]):.3f} "
            f"score/min={score / minutes:.1f} | acc steer={acc[ep.H['steer']]:.2f} target={acc[ep.H['target']]:.2f} "
            f"weapon={acc[ep.H['weapon']]:.2f} | fire recall={fire_tp / max(fire_label, 1):.2f} "
            f"precision={fire_tp / max(fire_pred, 1):.2f}",
            flush=True,
        )
        ep.save(a.out, policy, iteration=it)

    env.close()
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
