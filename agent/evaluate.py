"""Evaluate a policy in the headless mk48 world.

    python evaluate.py random
    python evaluate.py bot                 # agents run the built-in bot logic
    python evaluate.py runs/ppo/model.zip  # trained PPO policy

Rates are per game-minute alive-or-dead, with 95% confidence intervals computed over
(agent x game-minute) samples.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pathlib import Path

from mk48env import DEATH_CAUSE, DEATH_CAUSES, DIED, FIRED, KILLS, LEVEL, SCORE_DELTA, SERVER, Mk48VecEnv, ServerConfig


TYPES_FILE = Path(__file__).with_name("entity_types.tsv")  # from `server types`
V6_OWN_TYPE, V6_OBS_DIM, V6_CLASSES = 36, 1201, 7
WEAPON_NAMES = ["torpedo", "gun", "missile", "aircraft", "depth/mine", "SAM", "decoy"]


def type_names() -> dict[int, str]:
    if not TYPES_FILE.exists():
        return {}
    return {int(r[0]): r[1] for r in (line.split("\t") for line in TYPES_FILE.read_text().splitlines())}


def ci95(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def evaluate(policy: str, minutes: float, agents: int, bots: int, procs: int, device: str,
             stochastic: bool = False, server_path: Path = SERVER) -> dict:
    scripted = policy == "bot"
    expert = policy == "expert"  # the bot's choices replayed through the agent's action space
    env = Mk48VecEnv(
        n_procs=procs,
        server=ServerConfig(agents=agents, bots=bots, scripted_agents=scripted, expert_labels=expert,
                            server_path=server_path),
        max_episode_steps=10**9,
    )
    model = None
    entity = None
    if policy.endswith(".pt"):
        import torch

        import entity_policy as ep

        entity, _, _ = ep.load(policy, device)
        entity.eval()
    elif policy not in ("random", "bot", "expert"):
        from stable_baselines3 import PPO

        model = PPO.load(policy, device=device)

    steps_per_minute = int(60 / (env.server_cfg.ticks_per_step * 0.1))
    n_steps = int(minutes * steps_per_minute)
    obs = env.reset()
    per_min = {k: np.zeros((n_steps // steps_per_minute, env.num_envs)) for k in ("score", "kills", "deaths", "fired")}
    max_level = np.zeros(env.num_envs)
    causes = np.zeros(len(DEATH_CAUSES))
    # Per-vehicle stats (only when the observation carries the exact ship type).
    per_type = env.observation_space.shape[0] == V6_OBS_DIM
    by_type: dict[int, np.ndarray] = {}  # id -> [steps, score, kills, deaths, fires, fires per class...]
    t0 = time.perf_counter()
    for t in range(n_steps):
        act_dim = env.action_space.shape[0]
        if entity is not None:
            with torch.no_grad():
                discrete = ep.sample(entity(torch.as_tensor(obs, device=device)), deterministic=not stochastic)
            actions = ep.to_env(discrete.cpu().numpy(), obs, entity.layout)
            if actions.shape[1] < act_dim:  # older 9-action policy on a 10-action server: no salvo
                actions = np.hstack([actions, -np.ones((len(actions), act_dim - actions.shape[1]), np.float32)])
        elif model is not None:
            actions, _ = model.predict(obs, deterministic=not stochastic)
        elif expert:
            info = env.last_info
            actions = np.zeros((env.num_envs, act_dim), np.float32)
            if info is not None:
                e = env.expert_offset
                valid = info[:, e + act_dim] > 0
                actions[valid] = info[valid, e : e + act_dim]
        else:
            actions = np.random.uniform(-1, 1, (env.num_envs, env.action_space.shape[0])).astype(np.float32)
        prev_obs = obs
        obs, _, _, _ = env.step(actions)
        info = env.last_info
        if per_type:
            alive = prev_obs[:, 0] > 0.5
            classes = np.clip(((actions[:, 5] + 1) * 0.5 * V6_CLASSES).astype(int), 0, V6_CLASSES - 1)
            for i in np.flatnonzero(alive):
                row = by_type.setdefault(int(prev_obs[i, V6_OWN_TYPE]), np.zeros(5 + V6_CLASSES))
                row[0] += 1
                row[1] += info[i, SCORE_DELTA]
                row[2] += info[i, KILLS]
                row[3] += info[i, DIED]
                if info[i, FIRED] > 0:
                    row[4] += 1
                    if not scripted:  # the bot's internal weapon choice isn't visible here
                        row[5 + classes[i]] += 1
        m = t // steps_per_minute
        if m < len(per_min["score"]):
            per_min["score"][m] += info[:, SCORE_DELTA]
            per_min["kills"][m] += info[:, KILLS]
            per_min["deaths"][m] += info[:, DIED]
            per_min["fired"][m] += info[:, FIRED]
        max_level = np.maximum(max_level, info[:, LEVEL])
        if env.has_death_cause:
            for c in info[info[:, DIED] > 0, DEATH_CAUSE].astype(int):
                causes[c] += 1
    elapsed = time.perf_counter() - t0
    env.close()

    result = {k: ci95(v.ravel()) for k, v in per_min.items()}
    kills, deaths = per_min["kills"].sum(), per_min["deaths"].sum()
    result["kd"] = kills / max(deaths, 1)
    result["max_level_mean"] = float(max_level.mean())
    result["agent_steps_per_sec"] = n_steps * env.num_envs / elapsed
    if by_type:
        names = type_names()
        total = sum(r[0] for r in by_type.values())
        result["by_type"] = {}
        for tid, r in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
            minutes = r[0] / steps_per_minute
            mix = {WEAPON_NAMES[c]: round(float(r[5 + c] / max(r[4], 1)), 2) for c in range(V6_CLASSES) if r[5 + c]}
            result["by_type"][names.get(tid, str(tid))] = {
                "share": float(r[0] / total), "score_per_min": float(r[1] / minutes),
                "kd": float(r[2] / max(r[3], 1)), "fires_per_min": float(r[4] / minutes), "weapon_mix": mix,
            }
    if causes.sum():
        result["death_causes"] = {DEATH_CAUSES[i]: int(c) for i, c in enumerate(causes) if c}
    # Score/min per 10-minute block, to separate early game from steady state.
    blocks = per_min["score"].shape[0] // 10
    result["score_by_10min"] = [float(per_min["score"][10 * b : 10 * (b + 1)].mean()) for b in range(blocks)]
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="random | bot | expert | path/to/model.zip (SB3) | path/to/policy.pt (entity)")
    p.add_argument("--minutes", type=float, default=10, help="game minutes per agent")
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--bots", type=int, default=32)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--server", type=Path, default=SERVER, help="server binary (older models need older builds)")
    p.add_argument("--save", metavar="LABEL", help="append the result to runs/evals.json under this label")
    a = p.parse_args()
    r = evaluate(a.policy, a.minutes, a.agents, a.bots, a.procs, a.device, a.stochastic, a.server)
    print(f"policy={a.policy}  ({a.agents * a.procs} agents x {a.minutes} game-min)")
    for k in ("score", "kills", "deaths", "fired"):
        mean, ci = r[k]
        print(f"  {k + '/min':12s} {mean:8.2f} ± {ci:.2f}")
    print(f"  {'K/D':12s} {r['kd']:8.2f}")
    print(f"  {'max level':12s} {r['max_level_mean']:8.2f}")
    print(f"  throughput   {r['agent_steps_per_sec']:8.0f} agent-steps/s")
    if r.get("by_type"):
        print("  per vehicle (time share, score/min, K/D, fires/min, weapon mix):")
        for name, t in list(r["by_type"].items())[:12]:
            mix = ", ".join(f"{k} {v:.0%}" for k, v in sorted(t["weapon_mix"].items(), key=lambda kv: -kv[1]))
            print(f"    {name:13s} {t['share']:5.1%}  {t['score_per_min']:6.1f}  {t['kd']:5.2f}  "
                  f"{t['fires_per_min']:5.1f}  {mix}")
    if r.get("death_causes"):
        print("  death causes: " + ", ".join(f"{k} {v}" for k, v in r["death_causes"].items()))
    if r["score_by_10min"]:
        print("  score/min by 10-min block: " + " ".join(f"{x:.1f}" for x in r["score_by_10min"]))
    if a.save:
        path = Path("runs/evals.json")
        evals = json.loads(path.read_text()) if path.exists() else []
        evals = [e for e in evals if e["label"] != a.save]
        evals.append({"label": a.save, "policy": a.policy, "minutes": a.minutes,
                      "agents": a.agents * a.procs, "bots_per_world": a.bots, **r})
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(evals, indent=1))


if __name__ == "__main__":
    main()
