"""Evaluate a policy in the headless mk48 world.

    python evaluate.py random
    python evaluate.py bot                 # agents run the built-in bot logic
    python evaluate.py runs/ppo/model.zip  # trained PPO policy
    python evaluate.py runs/ppo_elite/policy.pt --agent-kind nn-bot  # as an NN bot in the playable server

Rates are per game-minute alive-or-dead, with 95% confidence intervals computed over
(agent x game-minute) samples.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pathlib import Path

from mk48env import (BOARD_RANK, DAMAGE_DEALT, DEATH_CAUSE, DEATH_CAUSES, DIED, FIRED, KILLS, LEVEL, SCORE, SCORE_DELTA,
                     SCORE_LOST, SERVER, Mk48VecEnv, ServerConfig)
from skills import SkillAudit, print_report


TYPES_FILE = Path(__file__).with_name("entity_types.tsv")  # from `server types`
V6_OWN_TYPE, V6_OBS_DIM, V6_CLASSES = 36, 1201, 7
WEAPON_NAMES = ["torpedo", "gun", "missile", "aircraft", "depth/mine", "SAM", "decoy"]


def type_names() -> dict[int, str]:
    return {i: t[0] for i, t in type_table().items()}


def type_table() -> dict[int, tuple[str, str, int]]:
    """id -> (name, sub kind, level)."""
    if not TYPES_FILE.exists():
        return {}
    rows = (line.split("\t") for line in TYPES_FILE.read_text().splitlines())
    return {int(r[0]): (r[1], r[3], int(r[4])) for r in rows}


def ship_family(sub_kind: str) -> str:
    """Ship families as in the server's `ship_group`."""
    if sub_kind == "Submarine":
        return "submarine"
    if sub_kind in ("Battleship", "Cruiser", "Destroyer", "Dreadnought", "Corvette", "MissileBoat", "Lcs", "Mtb"):
        return "surface"
    return "carrier" if sub_kind == "Carrier" else "other"


def ci95(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def policy_actions(net, obs: np.ndarray, act_dim: int, device: str, deterministic: bool = True) -> np.ndarray:
    """Actions of an entity policy, padded for older 9-action policies on a 10-action server (no salvo)."""
    import torch

    import entity_policy as ep

    with torch.no_grad():
        discrete = ep.sample(net(torch.as_tensor(obs, device=device)), deterministic=deterministic)
    actions = ep.to_env(discrete.cpu().numpy(), obs, net.layout)
    if actions.shape[1] < act_dim:
        actions = np.hstack([actions, -np.ones((len(actions), act_dim - actions.shape[1]), np.float32)])
    return actions


def evaluate(policy: str, minutes: float, agents: int, bots: int, procs: int, device: str,
             stochastic: bool = False, server_path: Path = SERVER, agent_kind: str = "player",
             ship_style: bool = False, lead_aim: bool = True, ship_ratings: Path | None = None,
             skills: bool = False, opponents: int = 0, opponent_policy: str | None = None,
             start_score: int = 0, bot_aggression: float = 1.0) -> dict:
    """With `opponents`, each world also has that many ships driven by `opponent_policy`, which play
    but aren't measured (e.g. the elite among imitation NN bots, as in the game)."""
    scripted = policy == "bot"
    expert = policy == "expert"  # the bot's choices replayed through the agent's action space
    is_network = policy.endswith((".pt", ".json"))  # a network, or a phased recipe of networks
    if opponents and (not is_network or not opponent_policy):
        raise ValueError("opponents need an entity policy (.pt or .json) and --opponent-policy")
    per_world = agents + opponents
    env = Mk48VecEnv(
        n_procs=procs,
        server=ServerConfig(agents=per_world, bots=bots, scripted_agents=scripted, expert_labels=expert,
                            agent_kind=agent_kind, ship_style=ship_style, lead_aim=lead_aim,
                            ship_ratings=ship_ratings, start_score=start_score, bot_aggression=bot_aggression,
                            server_path=server_path),
        max_episode_steps=10**9,
    )
    mine = np.array([i for i in range(env.num_envs) if i % per_world < agents])
    others = np.array([i for i in range(env.num_envs) if i % per_world >= agents], dtype=int)
    model = None
    entity = None
    if is_network:
        import torch

        import entity_policy as ep

        entity, _, _ = ep.load(policy, device)
        entity.eval()
        opp = ep.load(opponent_policy, device)[0].eval() if opponents else None
    elif policy not in ("random", "bot", "expert"):
        from stable_baselines3 import PPO

        model = PPO.load(policy, device=device)

    steps_per_minute = int(60 / (env.server_cfg.ticks_per_step * 0.1))
    n_steps = int(minutes * steps_per_minute)
    obs = env.reset()
    per_min = {k: np.zeros((n_steps // steps_per_minute, len(mine)))
               for k in ("score", "kills", "deaths", "fired", "damage", "lost")}
    # What the scoreboard shows: current score (which deaths can wipe, unlike score/min) and position.
    board = {k: np.zeros_like(per_min["score"]) for k in ("board_score", "board_rank")}
    max_level = np.zeros(len(mine))
    causes = np.zeros(len(DEATH_CAUSES))
    # Per-vehicle stats (only when the observation carries the exact ship type).
    per_type = env.observation_space.shape[0] == V6_OBS_DIM
    by_type: dict[int, np.ndarray] = {}  # id -> [steps, score, kills, deaths, fires, fires per class...]
    audit = SkillAudit(len(mine)) if skills else None
    t0 = time.perf_counter()
    for t in range(n_steps):
        act_dim = env.action_space.shape[0]
        if entity is not None:
            actions = policy_actions(entity, obs, act_dim, device, deterministic=not stochastic)
            if len(others):
                actions[others] = policy_actions(opp, obs[others], act_dim, device)
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
        if len(others):  # opponents play but only the evaluated agents are measured
            prev_obs, actions, info = prev_obs[mine], actions[mine], info[mine]
        if audit is not None:
            audit.update(prev_obs, actions, info)
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
            if env.has_damage_dealt:
                per_min["damage"][m] += info[:, DAMAGE_DEALT]
            if info.shape[1] > SCORE_LOST and env.has_board_rank:
                per_min["lost"][m] += info[:, SCORE_LOST]
            board["board_score"][m] += info[:, SCORE] / steps_per_minute
            if env.has_board_rank:
                board["board_rank"][m] += info[:, BOARD_RANK] / steps_per_minute
        max_level = np.maximum(max_level, info[:, LEVEL])
        if env.has_death_cause:
            for c in info[info[:, DIED] > 0, DEATH_CAUSE].astype(int):
                causes[c] += 1
    elapsed = time.perf_counter() - t0
    env.close()

    damage = per_min.pop("damage")
    lost = per_min.pop("lost")
    result_lost = ci95(lost.ravel())
    result = {k: ci95(v.ravel()) for k, v in per_min.items()}
    if env.has_damage_dealt:
        # Weapon damage dealt in boats' worth (1 = a whole boat), and per decision that fired.
        result["damage"] = ci95(damage.ravel())
        result["damage_per_shot"] = float(damage.sum() / max(per_min["fired"].sum(), 1))
    result["score_lost"] = result_lost  # score per minute that deaths wiped from the scoreboard
    result["board_score"] = ci95(board["board_score"].ravel())
    if env.has_board_rank:
        result["board_rank"] = ci95(board["board_rank"].ravel())
    kills, deaths = per_min["kills"].sum(), per_min["deaths"].sum()
    result["kd"] = kills / max(deaths, 1)
    result["max_level_mean"] = float(max_level.mean())
    result["agent_steps_per_sec"] = n_steps * len(mine) / elapsed
    if by_type:
        table = type_table()
        total = sum(r[0] for r in by_type.values())
        result["by_type"] = {}
        families: dict[str, float] = {}
        for tid, r in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
            minutes = r[0] / steps_per_minute
            mix = {WEAPON_NAMES[c]: round(float(r[5 + c] / max(r[4], 1)), 2) for c in range(V6_CLASSES) if r[5 + c]}
            name, sub_kind, level = table.get(tid, (str(tid), "", 0))
            family = ship_family(sub_kind)
            families[family] = families.get(family, 0.0) + float(r[0] / total)
            result["by_type"][name] = {
                "level": level, "family": family,
                "share": float(r[0] / total), "score_per_min": float(r[1] / minutes),
                "kd": float(r[2] / max(r[3], 1)), "fires_per_min": float(r[4] / minutes), "weapon_mix": mix,
            }
        result["family_share"] = families
    if audit is not None:
        result["skills"] = audit.report(steps_per_minute)
    if causes.sum():
        result["death_causes"] = {DEATH_CAUSES[i]: int(c) for i, c in enumerate(causes) if c}
    # Score/min per 10-minute block, to separate early game from steady state.
    blocks = per_min["score"].shape[0] // 10
    result["score_by_10min"] = [float(per_min["score"][10 * b : 10 * (b + 1)].mean()) for b in range(blocks)]
    if env.has_board_rank:
        result["board_rank_by_10min"] = [float(board["board_rank"][10 * b : 10 * (b + 1)].mean())
                                         for b in range(blocks)]
        # The opening, minute by minute.
        result["board_rank_by_min"] = [float(x) for x in board["board_rank"][:15].mean(1)]
        result["score_by_min"] = [float(x) for x in per_min["score"][:15].mean(1)]
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="random | bot | expert | path/to/model.zip (SB3) | path/to/policy.pt (entity) | "
                                  "path/to/recipe.json (phased: opening, then main policy)")
    p.add_argument("--minutes", type=float, default=10, help="game minutes per agent")
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--bots", type=int, default=32)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--server", type=Path, default=SERVER, help="server binary (older models need older builds)")
    p.add_argument("--agent-kind", choices=["player", "bot", "nn-bot"], default="player",
                   help="agents as players (training), or as the playable server's NN bots: nn-bot "
                        "(player rules) or bot (bot rules: score reset on death, random spawns)")
    p.add_argument("--ship-style", action="store_true",
                   help="random per-life vehicle preferences (as NN bots in the game) instead of the ship-family head")
    p.add_argument("--no-lead", action="store_true", help="aim at targets' current position instead of leading them")
    p.add_argument("--ship-ratings", type=Path, default=None,
                   help="with --ship-style: choose only among ships the network plays well (ship_ratings.py)")
    p.add_argument("--opponents", type=int, default=0,
                   help="ships per world driven by --opponent-policy (play but aren't measured)")
    p.add_argument("--opponent-policy", default=None)
    p.add_argument("--start-score", type=int, default=0, help="agents start with this score")
    p.add_argument("--bot-aggression", type=float, default=1.0, help="built-in bots' aggression (game default 1)")
    p.add_argument("--skills", action="store_true",
                   help="audit weapon and feature use: response rates, diving, sensors, target choice")
    p.add_argument("--save", metavar="LABEL", help="append the result to runs/evals.json under this label")
    a = p.parse_args()
    r = evaluate(a.policy, a.minutes, a.agents, a.bots, a.procs, a.device, a.stochastic, a.server, a.agent_kind,
                 a.ship_style, not a.no_lead, a.ship_ratings, a.skills, a.opponents, a.opponent_policy,
                 a.start_score, a.bot_aggression)
    print(f"policy={a.policy}  ({a.agents * a.procs} agents x {a.minutes} game-min, as {a.agent_kind})")
    for k in ("score", "kills", "deaths", "fired"):
        mean, ci = r[k]
        print(f"  {k + '/min':12s} {mean:8.2f} ± {ci:.2f}")
    if "damage" in r:
        mean, ci = r["damage"]
        print(f"  {'damage/min':12s} {mean:8.2f} ± {ci:.2f}   (boats' worth; per shot {r['damage_per_shot']:.3f})")
    mean, ci = r["score_lost"]
    print(f"  {'lost/min':12s} {mean:8.2f} ± {ci:.2f}   (score that deaths wiped)")
    mean, ci = r["board_score"]
    print(f"  {'board score':12s} {mean:8.1f} ± {ci:.1f}   (current score, as on the scoreboard)")
    if "board_rank" in r:
        mean, ci = r["board_rank"]
        print(f"  {'board rank':12s} {mean:8.0%} ± {ci:.0%}   (100% = top of the scoreboard, 50% = middle)")
    print(f"  {'K/D':12s} {r['kd']:8.2f}")
    print(f"  {'max level':12s} {r['max_level_mean']:8.2f}")
    print(f"  throughput   {r['agent_steps_per_sec']:8.0f} agent-steps/s")
    if r.get("by_type"):
        print("  per vehicle (time share, score/min, K/D, fires/min, weapon mix):")
        for name, t in list(r["by_type"].items())[:12]:
            mix = ", ".join(f"{k} {v:.0%}" for k, v in sorted(t["weapon_mix"].items(), key=lambda kv: -kv[1]))
            print(f"    {name:13s} {t['share']:5.1%}  {t['score_per_min']:6.1f}  {t['kd']:5.2f}  "
                  f"{t['fires_per_min']:5.1f}  {mix}")
    if r.get("family_share"):
        print("  time by family: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(r["family_share"].items(),
                                                                             key=lambda kv: -kv[1])))
        levels: dict[int, list[tuple[str, float]]] = {}
        for name, t in r["by_type"].items():
            levels.setdefault(t["level"], []).append((name, t["share"]))
        print("  ships by level (time share):")
        for level in sorted(levels):
            print(f"    {level:2d}: " + ", ".join(f"{n} {s:.1%}" for n, s in sorted(levels[level], key=lambda x: -x[1])))
    if r.get("skills"):
        print_report(r["skills"])
    if r.get("death_causes"):
        print("  death causes: " + ", ".join(f"{k} {v}" for k, v in r["death_causes"].items()))
    if r["score_by_10min"]:
        print("  score/min by 10-min block: " + " ".join(f"{x:.1f}" for x in r["score_by_10min"]))
    if r.get("board_rank_by_10min"):
        print("  board rank by 10-min block: " + " ".join(f"{x:.0%}" for x in r["board_rank_by_10min"]))
    if a.save:
        path = Path("runs/evals.json")
        evals = json.loads(path.read_text()) if path.exists() else []
        evals = [e for e in evals if e["label"] != a.save]
        evals.append({"label": a.save, "policy": a.policy, "minutes": a.minutes,
                      "agents": a.agents * a.procs, "bots_per_world": a.bots, "agent_kind": a.agent_kind,
                      "ship_style": a.ship_style, "lead_aim": not a.no_lead,
                      "ship_ratings": str(a.ship_ratings) if a.ship_ratings else None,
                      "opponents": a.opponents, "opponent_policy": a.opponent_policy,
                      "start_score": a.start_score, "bot_aggression": a.bot_aggression, **r})
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(evals, indent=1))


if __name__ == "__main__":
    main()
