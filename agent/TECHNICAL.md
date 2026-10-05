# mk48 neural-network bots: technical details

*The plain-language overview is in [README.md](README.md). This page is for programmers.*

Neural-network players for [mk48](https://github.com/SoftbearStudios/mk48), the open-source naval
combat game (AGPL-3.0), trained on a server you run yourself. You can play against them, or watch
the game through their eyes, in the real web client.

Everything runs on your machine, against the game in this repository. Nothing here connects to
mk48.io, CrazyGames or any other public server, and there is no anti-detection code of any kind.

![Training and evaluation](results/training.png)

---

## Setup (once)

From the repository root, on macOS (Apple Silicon) or Linux:

```bash
# Rust: the game pins a nightly toolchain; the web client is built with trunk.
rustup toolchain install nightly-2024-04-20 && rustup override set nightly-2024-04-20
rustup target add wasm32-unknown-unknown
cargo install --locked trunk --version 0.21.7
(cd client && trunk build --release --minify --no-sri --skip-version-check --filehash false)
(cd server && cargo build --release)   # one binary: the game server and the `train` mode

# Python 3.12 environment for the networks
cd agent
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch gymnasium stable-baselines3 numpy tensorboard matplotlib pytest
.venv/bin/python -m pytest -q
```

On macOS the C compiler needs the Xcode license accepted once (`sudo xcodebuild -license`).
Build the client before the server: the server embeds the built client.

## Quick start

```bash
./run_game.sh models/bc_v6.pt 12 40 models/elite_15M.pt
```

Arguments: main policy, NN bots, built-in bots, optional elite policy. Run it from this folder;
it keeps running until Ctrl+C, and refuses to start if a server is already on port 8443.

1. Open **https://localhost:8443** and click past the certificate warning (your own server uses a
   self-signed certificate).
2. Pick a name and press Play:
   - **`AI Elite`**: the elite network drives your ship and you see exactly what it sees.
   - **`AI`**: a regular NN bot drives your ship.
   - anything else: you play yourself (hold the mouse to steer, click to fire, 1–9 pick a
     weapon, R dives/surfaces a submarine, scroll to zoom).
3. NN bots are named **NN 1, NN 2, …** and the elite is **NN Elite**. The scoreboard lists
   everyone, bots included (it scrolls).

mk48 is a top-down game, so there is no first-person camera; "through its eyes" means your
browser shows the NN ship's own view.

---

## Results

Every policy is measured the same way: **fresh worlds** (everyone starts at level 1), 2 agents
plus 32 built-in bots per world, 8 worlds, 60 game-minutes each. "Score" is the game's own score
(pickups and kills) per game-minute.

| Policy | Score/min | Kills/min | Deaths/min | K/D |
|---|---|---|---|---|
| **Elite, final (15M steps, `models/elite_15M.pt`)** | **55.0 ± 3.9** | **0.81** | 0.04 | 19.0 |
| Elite at 6M steps (`models/elite_6M.pt`) | 48.0–49.5 | 0.52–0.62 | 0.02–0.04 | 13–24 |
| Elite at 10M steps | 51.1 ± 3.6 | 0.63 | 0.06 | 11.0 |
| Elite at 4M steps | 45.7–49.2 | 0.57–0.59 | 0.04 | 14–17 |
| Expert: the built-in bot's logic, played through the NN action interface | 42–59 | 0.38–0.60 | 0.09–0.12 | 4–5 |
| Imitation policy (`bc_v6`, the regular NN bots) | 33.5 ± 3.2 | 0.30 | 0.16 | 1.9 |
| Built-in bot | 23–30 | 0.11 | 0.08 | 1.3 |
| Best PPO trained from scratch (v3) | 23.9 ± 2.8 | 0.18 | 0.14 | 1.3 |
| Random actions | 3.9 ± 1.1 | 0.03 | 0.15 | 0.2 |

The final elite scores about 2× as much as the built-in bot, kills about 7× as often and dies
about half as often. In the same test batch, 15M beat 6M on kills (0.81 vs 0.52/min) with the
same death rate.
The ± is a 95% interval within one evaluation batch; separate batches of the same policy have
varied by more than that (up to ~±5), so only same-batch comparisons are used for decisions.

All raw results: `results/evals.json` (a snapshot; new evaluations are appended to
`runs/evals.json`). Per-vehicle breakdowns and death causes:
`python evaluate.py <policy> --server ../server/target/release/server ...`.

---

## How the network works

### What it sees

Only what a real player's client is sent (`World::get_player_complete` on the server), encoded
as 1,201 numbers per decision:

- **Own ship (40):** health, speed, heading, position, distance to the world edge, level and
  progress to the next, submerged, which weapon types are ready, sensor ranges, own active-sensor
  setting, time since spawn, upgrade available, exact ship type.
- **24 nearest contacts × 39:** position, velocity and heading relative to the ship; kind (ship,
  weapon, aircraft, pickup, obstacle, …); exact type, or "unidentified blip" if sensors can't tell;
  friendly or not; level; altitude; weapon type; pickup value (coin 10, crate 2, barrel 1); and for
  each of 7 weapon types, *can a ready weapon hit this contact right now* (the player sees turret
  arcs and reload bars).
- **Map:** a 15×15 grid of land and border around the ship, rotated to the ship's heading.

### What it can do

Decisions are made 5 times a second, as a set of choices:

| Head | Options |
|---|---|
| Steering | 16 directions relative to the current heading |
| Throttle | 5 levels (including reverse) |
| Target | point at one of the visible contacts, or "none" |
| Fire | yes / no |
| Weapon type | torpedo, gun, missile/rocket, aircraft, depth charge/mine, SAM, decoy |
| Salvo (elite) | also fire every other ready weapon that can hit the target |
| Submerge | yes / no (submarines) |
| Active sensors | on / off (they reveal you) |
| Ship family | preferred family for upgrades and respawns |

These are turned into the game's normal control message (heading, speed, aim point, fire weapon
N, submerge, sensors), so the network uses exactly the controls a player has. Upgrades happen as
soon as they are affordable.

### Architecture

An **entity transformer** (`entity_policy.py`):

- Every contact becomes a token (small MLP), plus one token for the own ship and one for the map
  (small CNN over the 15×15 grid). Ship types get a learned embedding, so the same network can
  play a submarine differently from a destroyer.
- 2 transformer layers (128 wide, 4 attention heads) let contacts be compared with each other,
  e.g. which ship is the threat and which is the best target.
- One categorical head per decision; the **target head is a pointer**: it scores each contact
  token directly, so aiming is "pick a ship" rather than regressing coordinates.
- A separate value network with the same shape is used for PPO.

![Inside the networks](results/nn_3d.png)

Left: the elite's training losses. Middle: the imitation network's loss over a 2D slice of weight
space; the trained weights sit at the bottom of the bowl. Right: every neuron of the elite at one
real game moment, from the 1,201 inputs through the token embeddings, both transformer layers and
the latent vector to the 9 decision heads (here: torpedo an enemy with a salvo).

### Rules outside the network

The network makes the decisions; a few fixed rules sit around it:

- **Collision guard** (`agent_commands` in `../server/src/train.rs`): if the ship touches an
  oil platform or land, or its commanded course would hit an obstacle, land or the world border
  within ~2 s, it takes the clear heading closest to what the network wanted. Oil platforms kill
  after ~6 s of contact; the network saw them in every case and still pushed in (a habit copied
  from the built-in bot). Same checkpoint, before the guard and with its final version:
  navigation deaths 81 → 3, deaths/min 0.15 → 0.04, K/D 3.1 → 14.1.
- **Aim snapping:** the aim point snaps to a visible enemy within 15% of sensor range, like
  clicking on a ship rather than beside it.
- **Weak ships avoided:** NN bots don't pick the Olympias (ram), Dredger, Lublin (minelayer, whose
  mines drop behind it) or Tanker. They came last at their level in every evaluation and took
  ~23% of the elite's time.

---

## How it was trained

### 1. A headless training world (fast simulation)

`server train` (`../server/src/train.rs`) runs the real game world with N network-controlled
ships and built-in bots, with no networking and no wall clock: about **18,000 agent-steps per
second per process**, versus ~10 for a bot playing through a browser. Python talks to it over
pipes (`mk48env.py`, a vectorized Gymnasium/Stable-Baselines3-style environment).

- One episode is one life; worlds restart every 30 game-minutes (staggered) so training keeps
  covering the early game.
- Reward is a weighted sum of score gained, kills, death, damage taken and damage dealt, and every
  part is logged separately.

### 2. Imitation of the built-in bot (DAgger)

Reinforcement learning from scratch plateaued below the built-in bot, so the network first learns
to copy it (`train_bc_entity.py`):

- Each NN ship has a "shadow" copy of the built-in bot. Every step the server asks the shadow what
  it would do *from that ship's exact view* and sends that as the label.
- Round 1 follows the bot; later rounds follow the network while still collecting the bot's
  labels, so it also learns to recover from its own mistakes (DAgger). 10 rounds, ~640k samples.
- The bot fires randomly even when it has a shot; the labels use its *firing solution* instead
  ("a ready weapon can hit this target"), which is learnable.

Result: `bc_v6`, 33.5 score/min. That's above the built-in bot, but below the bot's own logic played
through the NN interface (the "expert", 42–59).

### 3. Reinforcement learning from the imitation policy (PPO)

`ppo_entity.py`, a custom PPO on the GPU (fp16):

- Starts from the imitation weights. The value network is trained alone for the first 0.5M steps.
- A KL penalty keeps the policy close to the imitation policy, fading over the run. Without it,
  fine-tuning wiped out the imitated combat skills within a few million steps (v4: 28 → 18).
- Exploration samples from the discrete choices instead of adding noise to aim and trigger.

### 4. The elite

The elite is trained like step 3, with:

- an **aggressive reward**: damage dealt +2 per ship's worth, kill +3, death −4, damage taken −1.5;
- the **salvo** action (several weapons at once);
- **frozen NN opponents**: each world has 4 learning agents, 2 frozen copies of the imitation
  policy and 32 built-in bots;
- a looser KL anchor (0.1 → 0.01) so it can develop its own style;
- the v5 PPO policy as its starting point.

Checkpoints are evaluated and swapped into the game only when they win a same-batch comparison
(4M → 6M: deaths halved, K/D 16.7 → 24.3; 10M tied 6M; the final 15M beat 6M with 56% more
kills at the same death rate). Late in the run the policy drifted far from the imitation anchor
(KL ≈ 1.1 at 15M), so it developed its own, more aggressive style.

---

## What didn't work (and what it taught)

| Attempt | What happened | Lesson / fix |
|---|---|---|
| v1: PPO, MLP, 16 learning agents per world | 50+ score/min in training, 19.6 in fresh worlds | Agents farmed each other. Evaluate in bot-dominated **fresh** worlds only. |
| v2: same, 4 agents per world, worlds never restart | Got *worse* in fresh worlds (13.9) | It overfit to long-running worlds and lost the early game. Restart worlds every 30 min. |
| Running the small MLP on the GPU | ~2× slower than CPU | GPU pays off only with the transformer. |
| Imitation with an MLP and averaged (MSE) outputs | 21–25 score/min, rarely hit anything | Averaging "aim at ship A or B" or "turn left or right" gives a bad answer. Use discrete heads and a target pointer. |
| Copying the bot's trigger pulls | Never learned to fire | The bot fires at random; label its firing solution instead. |
| v4: PPO from imitation, no anchor | 28 → 18 score/min, K/D 0.95 → 0.26 | Add a KL anchor and value warm-up (v5, elite). |
| Elite before the guard | ~40% of deaths from oil platforms | The platform was always visible; added the collision guard. |
| First guard versions | Fixed platforms, but ships ran onto land / off the map | Check the whole hull and the path, not just the centerline. |

---

## Files

| File | What it does |
|---|---|
| `../server/src/train.rs` | Headless training mode, observation/action encoding, expert labels, collision guard |
| `../server/src/nn_bots.rs` | NN control of engine bots and `AI…` autopilot players in the normal server |
| `../server/src/bot.rs` | Built-in bot (now also exposes its firing solution for labels) |
| `../server/src/world_mutation.rs` | Damage-dealt statistic used by the elite reward |
| `../server/src/server.rs` | Scoreboard shows everyone (`LEADERBOARD_SIZE`, `LIVEBOARD_BOTS`); NN hooks |
| `../vendor/kodiak` | Local copy of the game engine; the client's 10-row scoreboard limit raised |
| `mk48env.py` | Vectorized environment, rewards, world restarts |
| `entity_policy.py` | Entity-transformer policy, action/label conversion, save/load |
| `train_bc_entity.py` | Imitation (DAgger) |
| `ppo_entity.py` | PPO with KL anchor, fp16, frozen opponents, reward presets |
| `evaluate.py` | Fresh-world evaluation, per-vehicle stats, death causes |
| `serve_policy.py` | Runs main + elite policies for the game server |
| `plot_training.py` | 3D viridis chart (`runs/training.png`) and 2D chart |
| `plot_nn_3d.py` | 3D views of the networks: training losses, loss landscape, the elite's neurons at one game moment |
| `run_game.sh` | Starts the playable server with NN bots |
| `models/` | Trained networks: `bc_v6.pt` (imitation), `ppo_v5.pt`, `elite_6M.pt`, `elite_15M.pt` (final elite, the best) |
| `results/` | Evaluation results (`evals.json`) and charts, including the final elite |
| `train_ppo.py`, `train_bc.py` | Earlier Stable-Baselines3 MLP versions (v1–v4), kept for reference |

One server build (`server/target`) serves both the game and training (`server train`). The
networks in `models/` were trained during development against intermediate builds; `bc_v6.pt` and
`ppo_v5.pt` use the earlier 9-action interface, and the game server and `evaluate.py` pad their
actions (no salvo), so they run as-is.

---

## Reproduce

After Setup, from this folder:

```bash
.venv/bin/python train_bc_entity.py --device mps                          # -> runs/bc_v7
.venv/bin/python ppo_entity.py --init runs/bc_v7/policy.pt --steps 10000000 --run ppo_v5
.venv/bin/python ppo_entity.py --init runs/ppo_v5/policy.pt --reward aggressive \
    --opponents 2 --opponent-policy runs/bc_v7/policy.pt --kl-start 0.1 --kl-end 0.01 \
    --steps 15000000 --run ppo_elite
.venv/bin/python evaluate.py runs/ppo_elite/policy.pt --server ../server/target/release/server \
    --device mps --agents 2 --bots 32 --procs 8 --minutes 60 --save "elite"
.venv/bin/python plot_training.py
.venv/bin/python -m pytest -q
```

Ctrl+C or SIGTERM during training saves the model and exits. Checkpoints are written every 1M steps.

Training from this source uses the 10-action interface (with salvo). The shipped `bc_v6` and
`ppo_v5` models predate it (9 actions); the elite was upgraded from 9 to 10 actions when its
training started.

---

## Limitations and next steps

- **Navigation is partly scripted.** The collision guard is a fixed rule. A future version could
  learn avoidance itself, for example with a contact penalty and an "about to collide" input.
- **No memory.** The network reacts to the current view. A recurrent layer (GRU/LSTM) would help
  it track ships that dive or move out of view.
- **Opponents are bots and frozen copies.** A league of past elite versions (self-play) would make
  it more robust against human tactics.
- **Ship choice is coarse:** a family preference plus excluded weak ships. Learning the exact
  upgrade would let it pick, say, the Kolkata or Arleigh Burke on purpose.
- **Not done: playing from screen pixels.** The original plan's last step is a vision network that
  plays through the browser from screenshots. Here, a pixel network would learn by copying this
  state-based policy (teacher → student).
