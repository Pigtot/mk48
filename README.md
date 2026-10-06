# Teaching a Neural Network to Play Mk48

I modified a local copy of the open-source naval game [Mk48.io](https://github.com/SoftbearStudios/mk48) so that
ships controlled by a neural network could train in it and then play in the real game. The network gets the game
state from the server as 1,201 numbers, not screenshots.

![The game running locally, with labels for the network-controlled ship, a built-in bot and the scoreboard](agent/figures/gameplay_annotated.png)

The game running on my machine. I joined as `AI Elite`, so the server gave control of my ship (centre) to my final
model, NN Elite, and the browser just shows what it does. On the scoreboard, NN 1, NN 2, … and NN Elite are bots
driven by networks. Every other name is one of the game's built-in bots.

## What I built

- A headless training mode for the game server: the same world simulation with no graphics, networking or
  real-time clock (`server/src/train.rs`). The main PPO run (15M decisions) took about 4 hours; one ship
  playing in real time, at 5 decisions per second, would need about 35 days.
- A transformer policy (~880k weights, PyTorch) trained in three steps: copying the built-in bot,
  reinforcement learning with PPO, then fine-tuning against copies of itself (`agent/`).
- NN bots in the playable game. The server can hand any bot, or any player whose name starts with `AI`, to the
  network (`server/src/nn_bots.rs`). They score and respawn under player rules and act through the commands a
  player sends.
- An evaluation harness that drops a model into fresh games against the built-in bots (`agent/evaluate.py`,
  `agent/pressure_test.py`). In it, the model after step 2 scores 2.1× the built-in bot's points per minute
  and is sunk half as often.

Everything runs locally. Nothing in this repository connects to, or plays on, the public mk48.io servers.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="agent/figures/overview-dark.svg">
  <img src="agent/figures/overview.svg" width="760" alt="The game server sends 1,201 numbers (own ship, 24 nearest objects, local map) to a transformer, which returns steer, throttle, target, fire, weapon, salvo, dive, sonar/radar and upgrade choices as player commands. No screenshots are used.">
</picture>

Five times per game-second, the server takes what it would send that ship's player and encodes it as 1,201
numbers: 40 about the ship itself, 39 for each of the 24 nearest objects (ships, torpedoes, missiles, aircraft,
barrels, oil platforms) and a 15 × 15 grid of land around it. Each object becomes one token for a small
transformer, which compares the objects with each other, for example to decide which enemy to aim at. The
network outputs one choice per control: heading, throttle, which object to target (or none), whether to fire
and with which weapon, whether to dive, and whether sonar/radar is on. The server applies that as an ordinary
player command. The only convolution in the model runs over the 15 × 15 land grid, which is built from map data.

## How it learned

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="agent/figures/training_pipeline-dark.svg">
  <img src="agent/figures/training_pipeline.svg" width="940" alt="Four stages: the built-in bot as teacher (26.0 points per minute), imitation learning with DAgger (33.5), reinforcement learning with PPO (55.0, Elite 15M), and tougher opponents with new skills (85% average scoreboard place, NN Elite).">
</picture>

The built-in Mk48 bot is a set of hand-written rules (`server/src/bot.rs`). It isn't strong, but it plays a
complete game, so I used it as the teacher.

1. **Imitation (DAgger).** Each training ship has a hidden copy of the bot that reports, at every step, what it
   would do from that ship's view. The network drives and is trained to match those labels, so it also gets
   labels for the situations its own mistakes lead to. After 10 rounds (640k labelled decisions) it scored 33.5
   points per minute against the bot's 26.0. The labels use the bot's firing solution rather than its trigger,
   which fires at partly random moments.
2. **Reinforcement learning (PPO).** Starting from the imitation weights, the network plays on its own and is
   trained on a reward: plus for score, damage dealt and kills, minus for damage taken and sinking. A KL penalty
   keeps it close to the imitation policy at first; an earlier run without it drifted away from what it had
   copied and got worse. The final run, Elite 15M, also had a salvo action and frozen imitation bots as
   opponents.
3. **Tougher opponents and new skills.** In the playable game what shows is the live scoreboard, where a death
   wipes about 40% of your score, and there Elite 15M averaged 78%. I fine-tuned it for 3M more decisions against
   copies of itself, in every ship type, with a reward that charges for the points a death actually costs. An
   extra imitation loss on scripted examples covered moves PPO almost never tried: diving, firing a SAM at an
   incoming missile, launching a decoy at a torpedo. The result is NN Elite, the model in the game.

The reward is only used for training. The results below use the game's own score or the scoreboard position.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="agent/figures/results-dark.svg">
  <img src="agent/figures/results.svg" width="760" alt="Game score per minute: random 3.9, built-in bot 26.0, imitation 33.5, built-in bot through the network's controls 49.4, Elite 15M 55.0. Ships sunk per minute: 0.11, 0.30, 0.49, 0.81. Times sunk per minute: 0.08, 0.16, 0.11, 0.04.">
</picture>

Two models appear below: Elite 15M (the end of step 2) and NN Elite (Elite 15M after step 3). Points per
minute come from the standard test: fresh games with 2 test ships and 32 built-in bots each, 60 game-minutes,
8 separate games (`agent/evaluate.py`; raw numbers in [`agent/results/evals.json`](agent/results/evals.json)).

Compared with the built-in bot, Elite 15M scored 2.1× as many points per minute (55.0 vs 26.0). Over the same
960 ship-minutes it sank 780 ships against the bot's 102 and was sunk 41 times against 81. That is a
kills-per-death ratio of 19 against 1.3, but with only 41 deaths the ratio is noisy: three tests of the 6M
checkpoint gave 12.9, 14.8 and 24.3.

Part of the gap comes from the controls, not from learning. When the bot's own decisions go through the
network's control interface (aim snaps onto the target, and it fires whenever a weapon can hit), it already
scores 49.4. Against that version the network's advantage is mostly in combat: 1.7× the kills and 0.4× the
deaths per minute.

Step 3 targeted scoreboard position, not points per minute, so I measured NN Elite on that, in the game's setup
(NN Elite, 11 other NN bots and 40 built-in bots, 60 minutes, 8 games). Its average position (100% = first) went
from 78% to 85%, the same in two separate runs. From minute 20 on it averaged 93%. My target was 90% over the
whole hour. It misses that because everyone starts at zero and the smallest boats score very little.

All comparisons are against the game's bots, not human players. Repeating the same test has moved a model's score
by up to 6 points per minute, so I only compare models tested in the same batch when choosing between them.

## What the network learned, and what I added by hand

`agent/skills.py` counts how often each weapon is used in the situations it exists for (32 ships of every type,
30 game-minutes):

| Behaviour | NN Elite | Elite 15M |
|---|---|---|
| Submarines submerged (when a threat is near) | 91% (95%) | 0% (0%) |
| SAM fired at an incoming missile or aircraft in reach | 63% | 5% |
| Decoy launched at a closing torpedo or missile | 26% | 0% |

Other things the network does on its own:

- Weapon mix by ship type. The ship type is one of the inputs. In the standard test the Skjold and Buyan missile
  boats fired 91% missiles, the Montana battleship 82% guns, and the Fletcher destroyer split 64/18/18 between
  torpedoes, guns and depth charges.
- Fewer deaths. With the step-3 reward, which charges for the points a death wipes, deaths in the scoreboard
  test fell from 0.087 to 0.040 per minute, and from 0.108 to 0.075 per minute against bots set to double aggression.

Some behaviour is fixed code in `server/src/train.rs`, not the network:

- Lead aim. The network picks the target and when to fire; the aim point is then moved to where a moving ship
  will be when the shot arrives. For Elite 15M in the standard test, this gave 18% more damage and 16% more
  kills.
- Collision guard. If the ship is about to hit land, an oil platform or the map edge, it takes the nearest clear
  heading. In one test, 81 of 140 deaths were navigation deaths without the guard and 3 with it.
- Upgrade paths. Each NN ship draws random upgrade preferences every life, so NN bots end up in submarines,
  carriers and battleships instead of all following one path. Left to itself, the network always upgraded to
  surface ships.

## Try it

macOS or Linux. Build once (the server embeds the web client, so build the client first):

```bash
rustup toolchain install nightly-2024-04-20 && rustup override set nightly-2024-04-20
rustup target add wasm32-unknown-unknown
cargo install --locked trunk --version 0.21.7
(cd client && trunk build --release --minify --no-sri --skip-version-check --filehash false)
(cd server && cargo build --release)

cd agent
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch gymnasium stable-baselines3 numpy tensorboard matplotlib pytest
```

Then, from `agent/`, start a game with 12 NN bots (the last one is NN Elite) and 40 built-in bots:

```bash
./run_game.sh models/bc_v6.pt 12 40 models/elite_v2b_3M.pt
```

Open https://localhost:8443 and accept the certificate warning (it's your own server). Enter `AI Elite` as your
name to watch the elite play, or any other name to play against the bots. More options are in
[agent/README.md](agent/README.md).

## Technical details

[agent/TECHNICAL.md](agent/TECHNICAL.md) covers the observation and action encoding, the architecture, every
evaluation table, the training commands, diagnostic plots and what didn't work. [agent/README.md](agent/README.md)
lists the models and what each file does. The figures on this page are generated by
[`agent/plot_readme.py`](agent/plot_readme.py) from the result files.

## Original Mk48 and attribution

Mk48.io is an online naval combat game made by [Softbear](https://github.com/SoftbearStudios). This repository is
a fork of [SoftbearStudios/mk48](https://github.com/SoftbearStudios/mk48), imported at commit `277fa87`, with
Softbear's Kodiak engine vendored in `vendor/kodiak`. The game, its art and its engine are their work. My
additions are `agent/`, the training and NN-bot code in `server/src/` (`train.rs`, `nn_bots.rs` and smaller
changes to existing server files) and a scoreboard change in the vendored engine.

The game is licensed under [AGPL-3.0](LICENSE) and the engine under [LGPL-3.0](vendor/kodiak/LICENSE); this fork
keeps those licenses. For the original project and its build instructions, see the
[upstream README](https://github.com/SoftbearStudios/mk48#readme).

Mk48.io is a trademark of Softbear, Inc. This project is not affiliated with or endorsed by Softbear.
