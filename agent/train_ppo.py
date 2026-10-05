"""Train a state-based PPO agent (the "teacher") in the headless mk48 world.

    python train_ppo.py --steps 20_000_000 --device mps
    tensorboard --logdir runs

Ctrl+C stops training and saves the current model (kill switch).
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from mk48env import DIED, KILLS, SCORE_DELTA, SERVER, Mk48VecEnv, RewardConfig, ServerConfig


class ValueWarmup(BaseCallback):
    """Freezes the actor for the first `steps` timesteps so the value net can catch up with a
    pretrained (e.g. imitation-learned) policy before policy gradients start moving it."""

    def __init__(self, steps: int):
        super().__init__()
        self.steps = steps
        self.frozen = False

    def _actor_params(self):
        policy = self.model.policy
        return [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(), policy.log_std]

    def _on_training_start(self) -> None:
        if self.steps > 0:
            for param in self._actor_params():
                param.requires_grad_(False)
            self.frozen = True

    def _on_step(self) -> bool:
        if self.frozen and self.num_timesteps >= self.steps:
            for param in self._actor_params():
                param.requires_grad_(True)
            self.frozen = False
            print(f"value warmup done at {self.num_timesteps:,} steps: actor unfrozen", flush=True)
        return True


class GameStats(BaseCallback):
    """Logs per-game-minute rates and every reward component to TensorBoard."""

    def __init__(self, step_seconds: float):
        super().__init__()
        self.step_seconds = step_seconds
        self._reset()

    def _reset(self):
        self.sums = {"score": 0.0, "kills": 0.0, "deaths": 0.0}
        self.agent_steps = 0
        self.episodes = []
        self.t0 = time.perf_counter()

    def _on_step(self) -> bool:
        info = self.training_env.unwrapped.last_info if hasattr(self.training_env, "unwrapped") else None
        if info is None:
            info = self.training_env.venv.last_info
        self.sums["score"] += info[:, SCORE_DELTA].sum()
        self.sums["kills"] += info[:, KILLS].sum()
        self.sums["deaths"] += info[:, DIED].sum()
        self.agent_steps += len(info)
        for i in self.locals["infos"]:
            if "episode" in i:
                self.episodes.append(i["episode"])
        return True

    def _on_rollout_end(self) -> None:
        minutes = self.agent_steps * self.step_seconds / 60
        for k, v in self.sums.items():
            self.logger.record(f"game/{k}_per_min", v / max(minutes, 1e-9))
        self.logger.record("game/kd", self.sums["kills"] / max(self.sums["deaths"], 1))
        self.logger.record("perf/agent_steps_per_sec", self.agent_steps / (time.perf_counter() - self.t0))
        if self.episodes:
            for k in self.episodes[0]:
                if k.startswith("rew_") or k in ("max_level", "fired"):
                    self.logger.record(f"episode/{k}", float(np.mean([e[k] for e in self.episodes])))
        self._reset()


def _stop(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    # Kill switch: Ctrl+C or SIGTERM saves and exits. Set explicitly, because processes started in
    # the background by a non-interactive shell inherit "ignore SIGINT".
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--agents", type=int, default=16, help="agents per server process")
    p.add_argument("--bots", type=int, default=32, help="built-in bots per server process")
    p.add_argument("--device", default="mps")
    p.add_argument("--run", default="ppo")
    p.add_argument("--resume", default=None, help="path to model.zip to continue from")
    p.add_argument("--init", default=None, help="start a new run from this model's weights (e.g. runs/bc/model.zip)")
    p.add_argument("--vf-warmup", type=int, default=0, help="timesteps to train only the value net first")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--server", type=Path, default=SERVER, help="server binary")
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--net", default="256,256", help="hidden layer sizes for policy and value nets")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--world-minutes", type=float, default=30, help="restart each world after this many game-minutes (0 = never)")
    a = p.parse_args()

    net = [int(x) for x in a.net.split(",")]
    run_dir = Path("runs") / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    reward = RewardConfig()
    server = ServerConfig(agents=a.agents, bots=a.bots, server_path=a.server)
    env = Mk48VecEnv(n_procs=a.procs, server=server, reward=reward, world_minutes=a.world_minutes or None)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.995)

    if a.resume:
        model = PPO.load(a.resume, env=env, device=a.device, tensorboard_log=str(run_dir))
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=a.n_steps,
            batch_size=a.batch,
            n_epochs=4,
            learning_rate=a.lr,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=a.ent_coef,
            policy_kwargs=dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.ReLU),
            tensorboard_log=str(run_dir),
            device=a.device,
            verbose=1,
        )
    if a.init:
        model.policy.load_state_dict(PPO.load(a.init, device=a.device).policy.state_dict())
        print(f"initialized weights from {a.init}")
    print(f"device={model.device}  envs={env.num_envs}  reward={reward}")

    step_seconds = server.ticks_per_step * 0.1
    callbacks = [
        ValueWarmup(a.vf_warmup),
        GameStats(step_seconds),
        CheckpointCallback(save_freq=max(1_000_000 // env.num_envs, 1), save_path=str(run_dir / "checkpoints")),
    ]
    try:
        model.learn(total_timesteps=a.steps, callback=callbacks, reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:
        print("interrupted: saving")
    finally:
        model.save(run_dir / "model")
        env.save(str(run_dir / "vecnormalize.pkl"))
        env.close()
        print(f"saved {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
