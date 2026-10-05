"""Imitate the built-in bot with DAgger, producing a PPO-compatible starting policy.

    python train_bc.py --device mps

Each iteration plays the game with a mix of expert and learner actions (beta = share of expert),
labels every visited observation with the bot's choice for that exact view, aggregates all data
and retrains the policy mean. Iteration 0 is pure behavioral cloning; later iterations teach the
policy to recover from its own mistakes. The result loads directly into train_ppo.py --init.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from mk48env import SCORE_DELTA, SERVER_BC, Mk48VecEnv, ServerConfig

# Action layout: steer, throttle, aim_x, aim_y, fire, weapon_class, submerge, active, ship_group
AIM, FIRE, CLASS = [2, 3], 4, 5


def policy_mean(policy, obs: torch.Tensor) -> torch.Tensor:
    features = policy.extract_features(obs)
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


FIRE_LOGIT_SCALE = 3.0  # fire > 0 in action space <=> p(fire) > 0.5


def bc_loss(mean: torch.Tensor, label: torch.Tensor, aim_valid: torch.Tensor, fire_pos_weight: float) -> torch.Tensor:
    """MSE on continuous dims, binary cross-entropy on the fire decision."""
    err = (mean - label) ** 2
    weights = torch.ones_like(err)
    weights[:, AIM] *= aim_valid[:, None]  # aim only matters when the bot has a target
    fired = (label[:, FIRE] > 0).float()
    weights[:, CLASS] *= fired  # weapon class only matters when firing
    weights[:, FIRE] = 0.0  # handled by the classification term
    regression = (err * weights).sum() / weights.sum()
    fire = F.binary_cross_entropy_with_logits(
        FIRE_LOGIT_SCALE * mean[:, FIRE], fired, pos_weight=torch.tensor(fire_pos_weight, device=mean.device)
    )
    return regression + fire


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--steps-per-iter", type=int, default=2000, help="env steps per iteration (x envs samples)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--bots", type=int, default=32)
    p.add_argument("--net", default="512,512,256")
    p.add_argument("--device", default="mps")
    p.add_argument("--fire-pos-weight", type=float, default=1.5, help="favor recall on the fire decision")
    p.add_argument("--out", default="runs/bc")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = Mk48VecEnv(
        n_procs=a.procs,
        server=ServerConfig(agents=a.agents, bots=a.bots, expert_labels=True, server_path=SERVER_BC),
        world_minutes=30,
    )
    net = [int(x) for x in a.net.split(",")]
    model = PPO("MlpPolicy", env, device=a.device,
                policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.ReLU))
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    act_dim = env.action_space.shape[0]
    EXPERT = env.expert_offset
    rng = np.random.default_rng()

    data_obs, data_label, data_aim = [], [], []
    obs = env.reset()
    for it in range(a.iterations):
        beta = [1.0, 0.5, 0.25, 0.1][it] if it < 4 else 0.0
        score, steps = 0.0, 0
        fire_tp = fire_label = fire_pred = 0
        policy.eval()
        for _ in range(a.steps_per_iter):
            info = env.last_info
            label = info[:, EXPERT : EXPERT + act_dim]
            valid = info[:, EXPERT + act_dim] > 0
            aim_valid = info[:, EXPERT + act_dim + 1]
            data_obs.append(obs[valid].astype(np.float16))
            data_label.append(label[valid])
            data_aim.append(aim_valid[valid])

            with torch.no_grad():
                learner = policy_mean(policy, torch.as_tensor(obs, device=policy.device)).cpu().numpy()
            fire_label += int((label[valid, FIRE] > 0).sum())
            fire_pred += int((learner[valid, FIRE] > 0).sum())
            fire_tp += int(((label[valid, FIRE] > 0) & (learner[valid, FIRE] > 0)).sum())
            use_expert = (rng.random(env.num_envs) < beta) & valid
            actions = np.where(use_expert[:, None], label, learner)
            obs, _, _, _ = env.step(np.clip(actions, -1, 1).astype(np.float32))
            score += env.last_info[:, SCORE_DELTA].sum()
            steps += env.num_envs

        # Retrain on everything collected so far.
        X = torch.as_tensor(np.concatenate(data_obs))
        Y = torch.as_tensor(np.concatenate(data_label))
        M = torch.as_tensor(np.concatenate(data_aim))
        policy.train()
        losses = []
        for _ in range(a.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), 8192):
                idx = perm[start : start + 8192]
                mean = policy_mean(policy, X[idx].float().to(policy.device))
                loss = bc_loss(mean, Y[idx].to(policy.device), M[idx].to(policy.device), a.fire_pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        minutes = steps * env.server_cfg.ticks_per_step * 0.1 / 60
        print(f"iter {it}  beta={beta:.2f}  samples={len(X):,}  loss={np.mean(losses[-50:]):.4f}  "
              f"score/min while collecting={score / minutes:.1f}  "
              f"fire recall={fire_tp / max(fire_label, 1):.2f} precision={fire_tp / max(fire_pred, 1):.2f}", flush=True)

    # Moderate exploration for PPO fine-tuning.
    with torch.no_grad():
        policy.log_std.fill_(-1.0)
    model.save(out / "model")
    env.close()
    print(f"saved {out / 'model.zip'}")


if __name__ == "__main__":
    main()
