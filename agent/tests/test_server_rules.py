import subprocess

import numpy as np

from mk48env import BOARD_RANK, DIED, SCORE, SCORE_LOST, SERVER, Mk48VecEnv, ServerConfig


def test_server_self_test():
    """The server's own checks: lead aim, ship personas and ratings, player rules for NN bots."""
    out = subprocess.run([str(SERVER), "self-test"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "0 failed" in out.stdout


def test_start_score_and_score_lost():
    """Agents can start rich, and a death reports the score it wiped."""
    env = Mk48VecEnv(server=ServerConfig(agents=2, bots=6, start_score=1500))
    try:
        env.reset()
        info = env.last_info
        assert (info[:, SCORE] >= 1500).all()
        for _ in range(100):
            env.step(np.zeros((2, env.action_space.shape[0]), np.float32))
            info = env.last_info
            assert (info[:, SCORE_LOST] >= 0).all()
            assert (info[info[:, DIED] == 0, SCORE_LOST] == 0).all()
            assert ((info[:, BOARD_RANK] >= 0) & (info[:, BOARD_RANK] <= 1)).all()
    finally:
        env.close()
