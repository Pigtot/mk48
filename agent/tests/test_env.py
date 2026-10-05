import numpy as np
import pytest

from mk48env import ALIVE, Mk48VecEnv, ServerConfig


@pytest.fixture
def env():
    env = Mk48VecEnv(server=ServerConfig(agents=4, bots=8))
    yield env
    env.close()


def test_shapes_and_values(env):
    obs = env.reset()
    assert obs.shape == (4, env.observation_space.shape[0])
    for _ in range(50):
        actions = np.random.uniform(-1, 1, (4, env.action_space.shape[0])).astype(np.float32)
        obs, reward, dones, infos = env.step(actions)
        assert obs.shape[0] == 4 and reward.shape == (4,) and dones.shape == (4,)
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
    # Agents spawn and are alive almost all the time.
    assert obs[:, 0].mean() > 0.5


def test_scripted_agents_score():
    """Built-in bot logic driving agent slots should pick up score."""
    env = Mk48VecEnv(server=ServerConfig(agents=4, bots=8, scripted_agents=True))
    try:
        env.reset()
        total = 0.0
        for _ in range(300):
            _, _, _, _ = env.step(np.zeros((4, env.action_space.shape[0]), np.float32))
            total += env._ep["score_gain"].sum()
        assert total > 0
    finally:
        env.close()


def test_server_exits_on_close(env):
    env.reset()
    env.close()
    assert all(s.proc.poll() is not None for s in env.servers)


def test_world_restart_truncates_and_continues():
    # 0.1 game-minutes = 30 steps at 5 decisions/s.
    env = Mk48VecEnv(n_procs=2, server=ServerConfig(agents=2, bots=4), world_minutes=0.1)
    try:
        env.reset()
        old_pids = [s.proc.pid for s in env.servers]
        restarted = 0
        for _ in range(35):
            obs, _, dones, infos = env.step(np.zeros((4, env.action_space.shape[0]), np.float32))
            restarted += sum(i.get("TimeLimit.truncated", False) for i in infos)
            assert np.isfinite(obs).all()
        # World 1 is staggered to restart at step 15, world 0 at step 30.
        assert [s.proc.pid for s in env.servers] != old_pids
        assert restarted >= 4
        assert obs[:, 0].mean() > 0.5  # agents alive in the fresh worlds
    finally:
        env.close()
