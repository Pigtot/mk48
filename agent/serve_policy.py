"""Drives the NN ships of a normal mk48 server (see server/src/nn_bots.rs).

    serve_policy.py MAIN_POLICY [ELITE_POLICY]

The server writes a header and then, per decision, one row per NN ship: the observation plus one
value choosing the policy (0 = main, 1 = elite). This process answers with one action row per ship
on stdout; nothing else may be written there. Actions from older policies (no salvo) are padded.
"""

from __future__ import annotations

import struct
import sys

import numpy as np
import torch

import entity_policy as ep

MAGIC = 0x4D6B3438


def read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def main() -> None:
    # The game waits for every decision. A batch of ~16 ships is answered faster and more steadily
    # on one thread (median 3.9 ms vs 7 ms on 8, worst case 5 vs 12 ms with other programs busy).
    torch.set_num_threads(1)
    paths = sys.argv[1:3]
    policies = [ep.load(p, "cpu")[0].eval() for p in paths]
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    magic, n, obs_dim, act_dim, info_dim = struct.unpack("<5I", read_exact(inp, 20))
    for path, policy in zip(paths, policies):
        if magic != MAGIC or obs_dim != policy.layout.obs_dim:
            sys.exit(f"serve_policy: server obs_dim {obs_dim} does not match {path} ({policy.layout.obs_dim})")
    names = ", ".join(f"{label} {path}" for label, path in zip(["main", "elite"], paths))
    print(f"serve_policy: {n} NN slots; {names}", file=sys.stderr, flush=True)
    row = obs_dim + info_dim
    try:
        while True:
            data = np.frombuffer(read_exact(inp, n * row * 4), dtype="<f4").reshape(n, row)
            obs = np.array(data[:, :obs_dim])
            choice = data[:, obs_dim].astype(int) if info_dim else np.zeros(n, int)
            actions = np.full((n, act_dim), -1.0, np.float32)
            actions[:, :4] = 0.0
            for k, policy in enumerate(policies):
                rows = np.flatnonzero(np.minimum(choice, len(policies) - 1) == k)
                if len(rows) == 0:
                    continue
                with torch.no_grad():
                    discrete = ep.sample(policy(torch.as_tensor(obs[rows])), deterministic=True).numpy()
                acts = ep.to_env(discrete, obs[rows], policy.layout)
                actions[rows, : acts.shape[1]] = acts[:, :act_dim]
            out.write(np.ascontiguousarray(actions, dtype="<f4").tobytes())
            out.flush()
    except (EOFError, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
