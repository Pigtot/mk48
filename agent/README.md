# agent/

The Python side of the project: the network, training, evaluation, and the process that drives NN ships in the
playable game. The [top-level README](../README.md) explains what it does and how well it plays;
[TECHNICAL.md](TECHNICAL.md) has the full details. This page covers running it and what each file does.

## Running the game with NN bots

After the setup in the top-level README, from this folder:

```bash
./run_game.sh models/bc_v6.pt 12 40 models/elite_v2b_3M.pt
```

The arguments are the policy for the regular NN bots, the number of NN bots, the number of built-in bots, and an
optional policy for the last NN bot ("NN Elite"). Ctrl+C stops it. It refuses to start if something is already
listening on port 8443.

Open https://localhost:8443 and click past the certificate warning (the server uses a self-signed certificate).
The name you enter decides who controls your ship:

| Name | Who controls your ship |
|---|---|
| `AI Elite` | the elite network; your browser shows its view, and it respawns automatically |
| `AI` | the regular NN-bot network |
| anything else | you: hold the mouse to steer, click to fire, 1–9 to pick a weapon, R to dive, scroll to zoom |

Mk48 is a top-down game, so "its view" is the normal game view centred on the ship the network is controlling.

Two other setups:

- `./run_game.sh models/elite_v4.pt 12 40 models/elite_v2b_3M.pt`: the regular NN bots use v4, which keeps its
  sonar off underwater and fires SAMs and decoys more often than NN Elite (94% and 97% of the time), but scores less.
- `./run_game.sh models/bc_v6.pt 12 40 models/elite_opening.json`: NN Elite uses a separate opening network until
  level 5, then switches to the elite. It does better in the first 10 minutes and the same over an hour.

## Models

| File | What it is | Used for |
|---|---|---|
| `models/bc_v6.pt` | step 1: imitation of the built-in bot | the NN 1, NN 2, … bots |
| `models/ppo_v5.pt` | first PPO run from the imitation weights (2M decisions) | starting point of the elite |
| `models/elite_6M.pt`, `models/elite_15M.pt` | step 2 checkpoints; `elite_15M` is "Elite 15M" in the results chart | comparisons |
| `models/elite_v2b_3M.pt` | step 3: Elite 15M after 3M more decisions | NN Elite in the game |
| `models/elite_v4.pt` | a step 3 variant that uses SAMs and decoys more and scores less | optional NN-bot style |
| `models/opening_c.pt`, `models/elite_opening.json` | an opening network, and the config that hands over to the elite at level 5 | optional |

`bc_v6.pt` and `ppo_v5.pt` predate the salvo action (9 actions instead of 10); the server and `evaluate.py` pad
their output, so they run as they are.

## Files

| File | What it does |
|---|---|
| `entity_policy.py` | the network: entity transformer, action heads, the step-3 auxiliary labels, opening configs |
| `mk48env.py` | vectorised environment around the server's `train` mode; rewards; world restarts |
| `train_bc_entity.py` | step 1: imitation (DAgger) |
| `ppo_entity.py` | steps 2 and 3: PPO with a KL anchor, frozen opponents and reward presets |
| `evaluate.py` | the standard test, per-ship stats, death causes, scoreboard rank |
| `pressure_test.py`, `skills.py` | scoreboard scenarios with pass/fail targets; how often each weapon answers its threat |
| `ship_ratings.py`, `ship_ratings.tsv` | score per ship for a policy; the game uses it to skip ships the network plays badly |
| `serve_policy.py`, `run_game.sh` | run the networks for the playable server |
| `plot_readme.py` | the figures in the top-level README (written to `figures/`) |
| `plot_training.py`, `plot_nn_3d.py` | the diagnostic plots in TECHNICAL.md |
| `results/` | saved evaluations (`evals.json`), pressure tests (`pressure/`), the screenshot and plots |
| `train_ppo.py`, `train_bc.py` | the earlier Stable-Baselines3 MLP versions (v1–v4), kept for reference |
| `tests/` | `.venv/bin/python -m pytest -q` |

The game-side changes are in `../server/src/`: `train.rs` (training mode, observation and action encoding,
collision guard, lead aim, per-life upgrade preferences), `nn_bots.rs` (NN control in the playable server) and `player.rs`
(player rules for NN bots).

## What didn't work

| What I tried | What happened | What I changed |
|---|---|---|
| 16 learning ships per training world | over 50 points per minute in training, 19.6 in fresh games: they had learned to farm each other | test only in fresh games dominated by built-in bots |
| Never restarting training worlds | worse in fresh games (13.9): it fitted the late game and lost the start | restart worlds every 30 game-minutes |
| An MLP imitating the bot with averaged (MSE) outputs | 21–26 points per minute and few hits: averaging "aim at A or B" aims between them | discrete action heads and a pointer to the target |
| Copying the bot's trigger pulls | never learned when to fire, because the bot's trigger is partly random | label the bot's firing solution instead |
| PPO from the imitation weights with no anchor | it drifted away from what it had copied and got worse | a KL penalty toward the imitation policy, fading over the run |
| Trusting the network to avoid oil platforms | 68 of 140 deaths in one test were platform crashes | the collision guard |
| An exploration bonus for diving (step 3, run v2) | submarines dived 99% of the time and stopped attacking: the bonus taught the move, not when to use it | no fix in training; in the game, ship ratings (score per ship) steer NN bots toward ships the network scores well in |
