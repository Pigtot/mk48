"""Vectorized environment backed by the mk48 server's headless `train` mode.

Each server process runs one world with `agents` RL-controlled ships plus built-in bots.
Every agent is one env slot in an SB3 VecEnv (shared policy, independent episodes).
An episode is one life: it ends on death (terminated) or after `max_episode_steps` (truncated).
"""

from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

MAGIC = 0x4D6B3438
ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server" / "target" / "release" / "server"
# Training server builds by interface. The normal release build (`server/target`) also has the
# `train` subcommand and is the current v7 interface (10 actions, salvo, damage dealt, guard).
# v6 (9 actions) needs a build of the older source that bc_v6 was trained with.
SERVER_BUILDS = {
    "v6": ROOT / "server" / "target-v6" / "release" / "server",
    "v7": ROOT / "server" / "target" / "release" / "server",
}

# Indices into the per-agent info vector written by the server.
SCORE, ALIVE, DIED, KILLS, SCORE_DELTA, HEALTH_LOST, FIRED, LEVEL = range(8)
# Newer builds append a death cause (0 none, 1 terrain, 2 border, 3 weapon, 4 ram, 5 obstacle, 6 other)
# and damage dealt this step (boats' worth of weapon damage).
DEATH_CAUSE = 8
DAMAGE_DEALT = 9
DEATH_CAUSES = ["none", "terrain", "border", "weapon", "ram/collision", "obstacle", "other"]
# Then the scoreboard position among everyone in the world, bots included (1 = top, 0 = bottom),
# and the score a death cost (the scoreboard shows it; SCORE_DELTA only counts gains while alive).
BOARD_RANK = 10
SCORE_LOST = 11
# Most a single death can cost in reward through `death_loss`, so one outlier can't swamp a batch.
DEATH_LOSS_CAP = 50.0
# With expert_labels, the last act_dim + 2 info values are the built-in bot's action for the same
# observation, then valid and aim_valid flags. Use Mk48VecEnv.expert_offset to locate them.
SERVER_BC = ROOT / "server" / "target-bc" / "release" / "server"


@dataclass
class RewardConfig:
    """Reward = sum of weighted components. All components are logged per episode."""

    score: float = 0.05  # per score point gained (pickups + kills)
    kill: float = 1.0  # per kill, on top of its score
    death: float = -2.0
    damage_taken: float = -0.5  # per full health bar lost
    alive: float = 0.0  # per step survived
    damage_dealt: float = 0.0  # per boat's worth of weapon damage dealt (needs a build with it)
    death_loss: float = 0.0  # per point of score a death cost (capped per death; needs a build with it)

    def components(self, info: np.ndarray) -> dict[str, np.ndarray]:
        out = {
            "score": self.score * info[:, SCORE_DELTA],
            "kill": self.kill * info[:, KILLS],
            "death": self.death * info[:, DIED],
            "damage_taken": self.damage_taken * info[:, HEALTH_LOST],
            "alive": self.alive * info[:, ALIVE],
        }
        if self.damage_dealt and info.shape[1] > DAMAGE_DEALT:
            out["damage_dealt"] = self.damage_dealt * info[:, DAMAGE_DEALT]
        if self.death_loss and info.shape[1] > SCORE_LOST:
            out["death_loss"] = -np.minimum(self.death_loss * info[:, SCORE_LOST], DEATH_LOSS_CAP)
        return out


# Elite bot: hunts (damage and kills pay well) but dodges and hides (being hit and dying cost more).
AGGRESSIVE = RewardConfig(score=0.05, kill=3.0, death=-4.0, damage_taken=-1.5, damage_dealt=2.0)
# Elite tuned for the scoreboard: as AGGRESSIVE, but a death also costs the score it wipes, at the
# same rate as score gained. Bold when poor, careful when rich ("protect the lead").
BOARD = RewardConfig(score=0.05, kill=3.0, death=-4.0, damage_taken=-1.5, damage_dealt=2.0, death_loss=0.05)
# Opening specialist: as BOARD, with points gained worth twice as much (climbing fast is the opening's job).
OPENING = RewardConfig(score=0.1, kill=3.0, death=-4.0, damage_taken=-1.5, damage_dealt=2.0, death_loss=0.05)


@dataclass
class ServerConfig:
    agents: int = 8
    bots: int = 24
    ticks_per_step: int = 2  # 10 ticks/s, so 2 = 5 decisions per second
    bot_aggression: float = 1.0
    scripted_agents: bool = False  # agents run the built-in bot (baseline)
    expert_labels: bool = False  # append the built-in bot's action for each observation
    spawn_type: str | None = None
    # "player" (training), or as the playable server's NN bots: "nn-bot" (player rules, current)
    # or "bot" (bot rules: score reset to level 1-2 on death, random spawns; before the fix).
    agent_kind: str = "player"
    ship_style: bool = False  # random per-life vehicle preferences, as NN bots in the game
    lead_aim: bool = True  # aim ahead of moving targets (False: at their current position)
    ship_ratings: Path | None = None  # with ship_style: only ships the network plays well (ship_ratings.py)
    start_score: int = 0  # agents' score when they join (e.g. to test protecting a lead)
    server_path: Path = SERVER

    def args(self) -> list[str]:
        args = [
            str(self.server_path), "train",
            "--agents", str(self.agents),
            "--bots", str(self.bots),
            "--ticks-per-step", str(self.ticks_per_step),
            "--bot-aggression", str(self.bot_aggression),
        ]
        if self.scripted_agents:
            args.append("--scripted-agents")
        if self.expert_labels:
            args.append("--expert-labels")
        if self.spawn_type:
            args += ["--spawn-type", self.spawn_type]
        if self.agent_kind != "player":
            args += ["--agent-kind", self.agent_kind]
        if self.ship_style:
            args.append("--ship-style")
        if not self.lead_aim:
            args.append("--no-lead")
        if self.ship_ratings:
            args += ["--ship-ratings", str(self.ship_ratings)]
        if self.start_score:
            args += ["--start-score", str(self.start_score)]
        return args


class _Server:
    def __init__(self, cfg: ServerConfig):
        if not cfg.server_path.exists():
            raise FileNotFoundError(f"server binary not found: {cfg.server_path}")
        # Dedicated pipes: the game's own debug prints go to stdout, which we discard.
        act_r, act_w = os.pipe()
        obs_r, obs_w = os.pipe()
        self.proc = subprocess.Popen(
            cfg.args() + ["--in-fd", str(act_r), "--out-fd", str(obs_w)],
            pass_fds=(act_r, obs_w),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        os.close(act_r)
        os.close(obs_w)
        self._act = os.fdopen(act_w, "wb", buffering=0)
        self._obs = os.fdopen(obs_r, "rb", buffering=0)
        magic, self.n, self.obs_dim, self.act_dim, self.info_dim = struct.unpack(
            "<5I", self._read(20)
        )
        if magic != MAGIC:
            raise RuntimeError(f"bad handshake from server: {magic:#x}")
        self.row = self.obs_dim + self.info_dim

    def _read(self, size: int) -> bytes:
        buf = bytearray()
        while len(buf) < size:
            chunk = self._obs.read(size - len(buf))
            if not chunk:
                raise RuntimeError(f"server exited (code {self.proc.poll()})")
            buf += chunk
        return bytes(buf)

    def send(self, actions: np.ndarray) -> None:
        self._act.write(np.ascontiguousarray(actions, dtype="<f4").tobytes())

    def recv(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.frombuffer(self._read(self.n * self.row * 4), dtype="<f4").reshape(self.n, self.row)
        return data[:, : self.obs_dim].copy(), data[:, self.obs_dim :].copy()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._act.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self._obs.close()


class Mk48VecEnv(VecEnv):
    def __init__(
        self,
        n_procs: int = 1,
        server: ServerConfig | None = None,
        reward: RewardConfig | None = None,
        max_episode_steps: int = 3000,
        world_minutes: float | None = None,
    ):
        """`world_minutes`: restart each world (staggered) after this many game-minutes, so
        training keeps covering the early game, like a fresh evaluation world."""
        self.server_cfg = server or ServerConfig()
        self.reward_cfg = reward or RewardConfig()
        self.max_episode_steps = max_episode_steps
        self.servers = [_Server(self.server_cfg) for _ in range(n_procs)]
        step_seconds = self.server_cfg.ticks_per_step * 0.1
        self.restart_steps = int(world_minutes * 60 / step_seconds) if world_minutes else None
        # Stagger restarts so the worlds are always at different stages.
        self._world_age = [k * (self.restart_steps or 0) // n_procs for k in range(n_procs)]
        s = self.servers[0]
        observation_space = spaces.Box(-np.inf, np.inf, (s.obs_dim,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (s.act_dim,), np.float32)
        self.render_mode = None
        expert_width = s.act_dim + 2 if self.server_cfg.expert_labels else 0
        self.expert_offset = s.info_dim - expert_width if expert_width else None
        self.has_death_cause = s.info_dim - expert_width > DEATH_CAUSE
        self.has_board_rank = s.info_dim - expert_width > BOARD_RANK
        self.has_damage_dealt = s.info_dim - expert_width > DAMAGE_DEALT
        super().__init__(s.n * n_procs, observation_space, action_space)
        self.last_info = None
        self._ep = None
        self._reset_episode_stats(np.arange(self.num_envs))

    # --- episode bookkeeping -------------------------------------------------
    def _reset_episode_stats(self, idx: np.ndarray) -> None:
        if self._ep is None:
            keys = ["r", "l", "score_gain", "kills", "fired", "max_level"]
            keys += [f"rew_{k}" for k in asdict(self.reward_cfg)]
            self._ep = {k: np.zeros(self.num_envs, dtype=np.float64) for k in keys}
        for v in self._ep.values():
            v[idx] = 0.0

    # --- VecEnv API ----------------------------------------------------------
    def reset(self):
        # The servers stream continuously; a no-op step yields the current observations.
        self.step_async(np.zeros((self.num_envs, self.action_space.shape[0]), np.float32))
        obs, _, info, _ = self._step()
        self.last_info = info
        self._reset_episode_stats(np.arange(self.num_envs))
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        n = self.servers[0].n
        for i, server in enumerate(self.servers):
            server.send(actions[i * n : (i + 1) * n])

    def _step(self):
        obs, info = zip(*(server.recv() for server in self.servers))
        obs = np.concatenate(obs)
        info = np.concatenate(info)
        components = self.reward_cfg.components(info)
        reward = sum(components.values()).astype(np.float32)
        return obs, reward, info, components

    def step_wait(self):
        obs, reward, info, components = self._step()
        self.last_info = info
        ep = self._ep
        ep["r"] += reward
        ep["l"] += 1
        ep["score_gain"] += info[:, SCORE_DELTA]
        ep["kills"] += info[:, KILLS]
        ep["fired"] += info[:, FIRED]
        ep["max_level"] = np.maximum(ep["max_level"], info[:, LEVEL])
        for k, v in components.items():
            ep[f"rew_{k}"] += v

        terminated = info[:, DIED] > 0
        truncated = (ep["l"] >= self.max_episode_steps) & ~terminated
        dones = terminated | truncated

        # Episode info must use the old world's last observation, so copy before restarting.
        obs_for_info = obs
        if self.restart_steps:
            n = self.servers[0].n
            for k in range(len(self.servers)):
                self._world_age[k] += 1
                if self._world_age[k] < self.restart_steps:
                    continue
                self._world_age[k] = 0
                if obs_for_info is obs:
                    obs_for_info = obs.copy()
                idx = np.arange(k * n, (k + 1) * n)
                truncated[idx] |= ~terminated[idx]
                dones[idx] = True
                obs[idx] = self._restart(k)

        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["episode"] = {k: float(v[i]) for k, v in ep.items()}
            infos[i]["episode"]["died"] = bool(terminated[i])
            infos[i]["terminal_observation"] = obs_for_info[i]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i])
        self._reset_episode_stats(np.flatnonzero(dones))
        return obs, reward, dones, infos

    def _restart(self, k: int) -> np.ndarray:
        """Replaces world k with a fresh one and returns its first observations."""
        self.servers[k].close()
        self.servers[k] = _Server(self.server_cfg)
        self.servers[k].send(np.zeros((self.servers[k].n, self.action_space.shape[0]), np.float32))
        obs, _ = self.servers[k].recv()
        return obs

    def close(self) -> None:
        for server in self.servers:
            server.close()

    # Minimal implementations of the remaining abstract methods.
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def seed(self, seed=None):
        return [None] * self.num_envs

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices
