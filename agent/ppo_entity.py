"""PPO for the entity-transformer policy, anchored to an imitation policy.

    python ppo_entity.py --init runs/bc_entity/policy.pt --device mps --run ppo_v5
    tensorboard --logdir runs

- Separate value network, trained alone for the first `--vf-warmup` steps.
- Loss adds beta * KL(imitation || policy), with beta decaying from --kl-start to --kl-end, so
  fine-tuning improves on the imitation policy instead of forgetting it.
- Optional frozen NN opponents in every world (--opponents, --opponent-policy) for a tougher league.
- --layout v7 adds the salvo action; a v6 starting policy is upgraded and still anchors the KL.
- Ctrl+C or SIGTERM saves and exits (kill switch).

Elite bot:
    python ppo_entity.py --init runs/ppo_v5/policy.pt --layout v7 --reward aggressive \
        --opponents 2 --opponent-policy runs/ppo_v5/policy.pt --kl-start 0.1 --kl-end 0.01 --run ppo_elite

Elite v2: scoreboard reward (a death costs the score it wipes), every ship family (personas), against
frozen copies of the elite:
    python ppo_entity.py --init runs/ppo_elite/policy.pt --reward board --ship-style --agents 4 --bots 40 \
        --opponents 4 --opponent-policy runs/ppo_elite/policy.pt --kl-start 0.05 --kl-end 0.01 --run ppo_elite2
"""

from __future__ import annotations

import argparse
import copy
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import entity_policy as ep
from mk48env import AGGRESSIVE, BOARD, DIED, KILLS, SCORE_DELTA, SERVER_BUILDS, Mk48VecEnv, RewardConfig, ServerConfig

SERVERS = SERVER_BUILDS
REWARDS = {"default": RewardConfig(), "aggressive": AGGRESSIVE, "board": BOARD}


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, n = x.mean(), x.var(), len(x)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total


def _stop(signum, frame):
    raise KeyboardInterrupt


def batched(fn, x: torch.Tensor, size: int = 4096) -> torch.Tensor:
    return torch.cat([fn(x[i : i + size]) for i in range(0, len(x), size)])


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000_000)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--bots", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--vf-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--kl-start", type=float, default=0.5)
    p.add_argument("--kl-end", type=float, default=0.05)
    p.add_argument("--vf-warmup", type=int, default=500_000)
    p.add_argument("--world-minutes", type=float, default=30)
    p.add_argument("--init", default=None, help="imitation policy to start from and anchor to")
    p.add_argument("--device", default="mps")
    p.add_argument("--run", default="ppo_v5")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on the GPU")
    p.add_argument("--layout", choices=["v6", "v7"], default=None, help="default: the --init policy's layout")
    p.add_argument("--reward", choices=list(REWARDS), default="default")
    p.add_argument("--opponents", type=int, default=0, help="frozen NN opponents per world")
    p.add_argument("--opponent-policy", default=None)
    p.add_argument("--ship-style", action="store_true",
                   help="random per-life vehicle preferences, so every ship family and its weapons get practice")
    p.add_argument("--agent-kind", choices=["player", "bot", "nn-bot"], default="player")
    p.add_argument("--free-heads", default="",
                   help="comma-separated heads left out of the KL anchor, for skills the starting policy never "
                        "learned (e.g. submerge,active)")
    p.add_argument("--defense-coef", type=float, default=0.0,
                   help="weight of an auxiliary imitation loss teaching SAMs against incoming missiles and "
                        "aircraft and decoys against closing torpedoes and missiles (ep.defense_labels)")
    p.add_argument("--doctrine-coef", type=float, default=0.0,
                   help="weight of an auxiliary imitation loss teaching carriers to launch aircraft at ships "
                        "in reach and submerged submarines to keep active sonar off (ep.doctrine_labels)")
    p.add_argument("--dive-bias", type=float, default=0.0,
                   help="added to the starting policy's 'dive' logit so submarines explore diving "
                        "(only submarines can dive; the server ignores it for other ships)")
    p.add_argument("--server", type=Path, default=None,
                   help="server binary (default: the build for the layout); a copy keeps rebuilds out of a long run")
    a = p.parse_args()

    dev = a.device
    run_dir = Path("runs") / a.run
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    reward_cfg = REWARDS[a.reward]

    init_policy = init_value = None
    if a.init:
        init_policy, init_value, _ = ep.load(a.init, dev)
    layout = {"v6": ep.V6, "v7": ep.V7}[a.layout] if a.layout else (init_policy.layout if init_policy else ep.V6)
    version = "v7" if layout.salvo else "v6"

    per_world = a.agents + a.opponents
    env = Mk48VecEnv(
        n_procs=a.procs,
        server=ServerConfig(agents=per_world, bots=a.bots, ship_style=a.ship_style, agent_kind=a.agent_kind,
                            server_path=a.server or SERVERS[version]),
        reward=reward_cfg,
        world_minutes=a.world_minutes,
    )
    # Learners and frozen opponents share each world; only learner rows are trained on.
    learner = np.array([i for i in range(env.num_envs) if i % per_world < a.agents])
    opponent = np.array([i for i in range(env.num_envs) if i % per_world >= a.agents], dtype=int)
    n_envs = len(learner)
    step_seconds = env.server_cfg.ticks_per_step * 0.1
    assert env.observation_space.shape[0] == layout.obs_dim, (env.observation_space.shape, layout)
    assert env.action_space.shape[0] == layout.act_dim, (env.action_space.shape, layout)

    opp = None
    if a.opponents:
        opp, _, _ = ep.load(a.opponent_policy or a.init, dev)
        opp.eval()
        print(f"{a.opponents} frozen opponents per world from {a.opponent_policy or a.init}")

    policy, value, ref = ep.Policy(layout).to(dev), ep.Value(layout).to(dev), None
    if init_policy is not None:
        if init_policy.layout == layout:
            policy.load_state_dict(init_policy.state_dict())
        else:
            policy = ep.upgrade(init_policy, layout)
        if init_value is not None:
            value.load_state_dict(init_value.state_dict())
        ref = init_policy.eval()  # KL anchor covers the heads both policies have
        for param in ref.parameters():
            param.requires_grad_(False)
        print(f"initialized from {a.init}; anchoring with KL {a.kl_start} -> {a.kl_end}")
    if a.dive_bias:
        with torch.no_grad():
            policy.heads["submerge"].bias[1] += a.dive_bias
        print(f"dive logit +{a.dive_bias}")
    free_heads = {h for h in a.free_heads.split(",") if h}
    if free_heads:
        print(f"not anchored: {sorted(free_heads)}")
    opt_pi = torch.optim.Adam(policy.parameters(), lr=a.lr, eps=1e-5)
    opt_v = torch.optim.Adam(value.parameters(), lr=a.vf_lr, eps=1e-5)
    # fp16 autocast on the GPU (~1.6x faster updates); logits/values are cast back to fp32.
    amp = dev != "cpu" and not a.no_amp
    scaler_pi = torch.amp.GradScaler(dev, enabled=amp)
    scaler_v = torch.amp.GradScaler(dev, enabled=amp)

    def pi(o):
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            out = policy(o)
        return {k: v.float() for k, v in out.items()}

    def vf(o):
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            return value(o).float()

    def ref_pi(o):
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            out = ref(o)
        return {k: v.float() for k, v in out.items()}

    def apply(loss, opt, scaler, params):
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        scaler.step(opt)
        scaler.update()
    def opponent_actions(o):
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=amp):
            out = opp(torch.as_tensor(o, device=dev))
        discrete = ep.sample({k: v.float() for k, v in out.items()}).cpu().numpy()
        acts = ep.to_env(discrete, o, opp.layout)
        if acts.shape[1] < layout.act_dim:  # older opponent: no salvo
            acts = np.hstack([acts, -np.ones((len(acts), layout.act_dim - acts.shape[1]), np.float32)])
        return acts

    print(f"device={dev} learners={n_envs} layout={version} reward={reward_cfg}", flush=True)

    ret_rms, disc_ret = RunningMeanStd(), np.zeros(n_envs)
    obs_all = env.reset()
    obs = obs_all[learner]
    global_step, next_ckpt = 0, 1_000_000
    n_updates = a.steps // (a.n_steps * n_envs)
    try:
        for update in range(n_updates):
            frac = update / max(n_updates - 1, 1)
            beta = a.kl_start + (a.kl_end - a.kl_start) * frac if ref is not None else 0.0
            warm = global_step < a.vf_warmup
            t0 = time.perf_counter()

            b_obs = np.zeros((a.n_steps, n_envs, layout.obs_dim), np.float32)
            b_act = np.zeros((a.n_steps, n_envs, len(layout.head_names)), np.int64)
            b_lp = np.zeros((a.n_steps, n_envs), np.float32)
            b_rew = np.zeros((a.n_steps, n_envs), np.float32)
            b_done = np.zeros((a.n_steps, n_envs), np.float32)
            truncated = []  # (t, env index, terminal obs)
            sums = {"score": 0.0, "kills": 0.0, "deaths": 0.0}
            episodes = []

            policy.eval()
            for t in range(a.n_steps):
                with torch.no_grad():
                    logits = pi(torch.as_tensor(obs, device=dev))
                    actions = ep.sample(logits)
                    b_lp[t] = ep.log_prob(logits, actions).cpu().numpy()
                actions = actions.cpu().numpy()
                b_obs[t], b_act[t] = obs, actions
                env_actions = np.zeros((env.num_envs, layout.act_dim), np.float32)
                env_actions[learner] = ep.to_env(actions, b_obs[t], layout)
                if len(opponent):
                    env_actions[opponent] = opponent_actions(obs_all[opponent])
                obs_all, reward, done, infos = env.step(env_actions)
                obs, reward, done = obs_all[learner], reward[learner], done[learner]
                infos = [infos[i] for i in learner]

                disc_ret = disc_ret * a.gamma + reward
                ret_rms.update(disc_ret)
                disc_ret[done] = 0.0
                b_rew[t] = reward / np.sqrt(ret_rms.var + 1e-8)
                b_done[t] = done
                for i, info in enumerate(infos):
                    if info.get("TimeLimit.truncated"):
                        truncated.append((t, i, info["terminal_observation"]))
                    if "episode" in info:
                        episodes.append(info["episode"])
                last = env.last_info[learner]
                sums["score"] += last[:, SCORE_DELTA].sum()
                sums["kills"] += last[:, KILLS].sum()
                sums["deaths"] += last[:, DIED].sum()
            global_step += a.n_steps * n_envs
            rollout_time = time.perf_counter() - t0

            # Values in one batch, plus bootstrap for truncated episodes.
            flat_obs = torch.as_tensor(b_obs.reshape(-1, layout.obs_dim), device=dev)
            value.eval()
            with torch.no_grad():
                b_val = batched(vf, flat_obs).cpu().numpy().reshape(a.n_steps, n_envs)
                next_val = vf(torch.as_tensor(obs, device=dev)).cpu().numpy()
                if truncated:
                    term = torch.as_tensor(np.stack([x[2] for x in truncated]), device=dev)
                    term_val = batched(vf, term).cpu().numpy()
                    for (t, i, _), v in zip(truncated, term_val):
                        b_rew[t, i] += a.gamma * v

            adv = np.zeros_like(b_rew)
            running = np.zeros(n_envs, np.float32)
            for t in reversed(range(a.n_steps)):
                nv = next_val if t == a.n_steps - 1 else b_val[t + 1]
                nonterminal = 1.0 - b_done[t]
                delta = b_rew[t] + a.gamma * nv * nonterminal - b_val[t]
                running = delta + a.gamma * a.lam * nonterminal * running
                adv[t] = running
            ret = adv + b_val

            f_act = torch.as_tensor(b_act.reshape(-1, len(layout.head_names)), device=dev)
            f_lp = torch.as_tensor(b_lp.reshape(-1), device=dev)
            f_adv = torch.as_tensor(adv.reshape(-1), device=dev)
            f_ret = torch.as_tensor(ret.reshape(-1), device=dev)

            policy.train()
            value.train()
            logs = {"value_loss": [], "pg_loss": [], "entropy": [], "kl_ref": [], "approx_kl": [], "clipfrac": [],
                    "defense_lesson": [], "doctrine_lesson": []}
            total = len(f_ret)
            for _ in range(a.epochs):
                perm = torch.randperm(total, device=dev)
                for start in range(0, total, a.minibatch):
                    idx = perm[start : start + a.minibatch]
                    o = flat_obs[idx]
                    v_loss = 0.5 * ((vf(o) - f_ret[idx]) ** 2).mean()
                    apply(v_loss, opt_v, scaler_v, value.parameters())
                    logs["value_loss"].append(v_loss.item())
                    if warm:
                        continue

                    logits = pi(o)
                    new_lp = ep.log_prob(logits, f_act[idx])
                    log_ratio = new_lp - f_lp[idx]
                    ratio = log_ratio.exp()
                    mb_adv = f_adv[idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = -torch.min(ratio * mb_adv, ratio.clamp(1 - a.clip, 1 + a.clip) * mb_adv).mean()
                    ent = ep.entropy(logits).mean()
                    loss = pg_loss - a.ent_coef * ent
                    if ref is not None:
                        with torch.no_grad():
                            ref_logits = ref_pi(o)
                        ref_logits = {n: v for n, v in ref_logits.items() if n not in free_heads}
                        kl_ref = ep.kl(ref_logits, logits).mean()
                        loss = loss + beta * kl_ref
                        logs["kl_ref"].append(kl_ref.item())
                    if a.defense_coef:
                        rows, target, weapon = ep.defense_labels(o, layout)
                        if rows.any():
                            def lp(head, label):
                                return torch.log_softmax(logits[head][rows].float(), -1).gather(1, label[rows, None])[:, 0]
                            lesson = -(lp("fire", torch.ones_like(target)) + lp("weapon", weapon) + lp("target", target)).mean()
                            loss = loss + a.defense_coef * lesson
                            logs["defense_lesson"].append(lesson.item())
                    if a.doctrine_coef:
                        air, air_target, stealth = ep.doctrine_labels(o, layout)
                        terms = []
                        if air.any():
                            def air_lp(head, label):
                                return torch.log_softmax(logits[head][air].float(), -1).gather(1, label[air, None])[:, 0]
                            weapon = torch.full_like(air_target, 3)
                            terms.append(-(air_lp("fire", torch.ones_like(air_target)) + air_lp("weapon", weapon)
                                           + air_lp("target", air_target)).mean())
                        if stealth.any():
                            terms.append(-torch.log_softmax(logits["active"][stealth].float(), -1)[:, 0].mean())
                        if terms:
                            lesson = sum(terms)
                            loss = loss + a.doctrine_coef * lesson
                            logs["doctrine_lesson"].append(lesson.item())
                    apply(loss, opt_pi, scaler_pi, policy.parameters())
                    with torch.no_grad():
                        logs["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                        logs["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    logs["pg_loss"].append(pg_loss.item())
                    logs["entropy"].append(ent.item())

            minutes = a.n_steps * n_envs * step_seconds / 60
            elapsed = time.perf_counter() - t0
            for k, v in sums.items():
                writer.add_scalar(f"game/{k}_per_min", v / minutes, global_step)
            writer.add_scalar("game/kd", sums["kills"] / max(sums["deaths"], 1), global_step)
            if episodes:
                writer.add_scalar("episode/max_level", np.mean([e["max_level"] for e in episodes]), global_step)
                for k in episodes[0]:
                    if k.startswith("rew_"):
                        writer.add_scalar(f"episode/{k}", np.mean([e[k] for e in episodes]), global_step)
            for k, v in logs.items():
                if v:
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
            writer.add_scalar("train/kl_beta", beta, global_step)
            writer.add_scalar("perf/fps", a.n_steps * n_envs / elapsed, global_step)
            if update % 5 == 0:
                print(
                    f"step {global_step:>10,} fps {a.n_steps * n_envs / elapsed:5.0f} (rollout {rollout_time:.1f}s) "
                    f"score/min {sums['score'] / minutes:5.1f} kills/min {sums['kills'] / minutes:.2f} "
                    f"deaths/min {sums['deaths'] / minutes:.2f} kl_ref {np.mean(logs['kl_ref'] or [0]):.3f} "
                    f"{'(value warmup)' if warm else ''}",
                    flush=True,
                )
            if global_step >= next_ckpt:
                ep.save(run_dir / "checkpoints" / f"policy_{global_step}.pt", policy, value, step=global_step)
                next_ckpt += 1_000_000
    except KeyboardInterrupt:
        print("interrupted: saving", flush=True)
    finally:
        ep.save(run_dir / "policy.pt", policy, value, step=global_step)
        env.close()
        writer.close()
        print(f"saved {run_dir / 'policy.pt'}", flush=True)


if __name__ == "__main__":
    main()
