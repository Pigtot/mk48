# mk48

mk48 neural network on mac

This repository is [mk48](https://github.com/SoftbearStudios/mk48), the open-source naval combat
game by Softbear (AGPL-3.0), plus neural-network bots trained on a server you run yourself.

**[agent/README.md](agent/README.md)** explains what the networks see and do, how they were trained,
the results, and how to play against them or watch the game through their eyes.

![Training and evaluation](agent/results/training.png)

Changes to the game itself:

- `server/src/train.rs`: headless training mode (`server train`), the observation and action
  encoding, expert labels for imitation, and a collision guard for NN ships.
- `server/src/nn_bots.rs`: NN-driven bots ("NN 1", ..., "NN Elite") and an autopilot for players
  named "AI..." in the normal game.
- `server/src/world_mutation.rs`, `player.rs`, `bot.rs`, `server.rs`: damage-dealt statistic, the
  bot's firing solution for labels, NN hooks, and a scoreboard that lists everyone.
- `vendor/kodiak`: the game engine (kodiak 0.2.0), vendored so its scoreboard can show everyone.

Nothing here connects to mk48.io or any public server.

---

*The original mk48 README follows.*

# Mk48.io Game

[![Build](https://github.com/SoftbearStudios/mk48/actions/workflows/build.yml/badge.svg)](https://github.com/SoftbearStudios/mk48/actions/workflows/build.yml)
<a href='https://discord.gg/YMheuFQWTX'>
  <img src='https://img.shields.io/badge/Mk48.io-%23announcements-blue.svg' alt='Mk48.io Discord' />
</a>

![Logo](/client/logo-712.png)

[Mk48.io](https://mk48.io) is an online multiplayer naval combat game, in which you take command of a ship and sail your way to victory. Watch out for torpedoes!

- [Ship Suggestions](https://github.com/SoftbearStudios/mk48/discussions/132)

## Build Instructions

1. Install `rustup` ([see instructions here](https://rustup.rs/))
2. Install `gmake` and `gcc` if they are not already installed.
3. Install trunk, Rust Nightly, and the WebAssembly target

```console
make rustup
make trunk
```

4. Build client

```console
cd client
make release
```

5. Build and run server

```console
cd server
make run_release
```

6. Navigate to `https://localhost:8443/` and play!

## Developing

If you follow the *Building* steps, you have a fully functioning game (could be used to host a private server). If your goal
is to modify the game, you may want to read more :)

### Entity data

Entities (ships, weapons, aircraft, collectibles, obstacles, decoys, etc.) are defined at the bottom of
`common/src/entity/_type.rs`.

### Entity textures

Each entity type must be accompanied by a texture of the same name in the spritesheet, which comes with the
repository. If entity textures need to be changed, see instructions in the `sprite_sheet_packer` directory.

## Contributing
See [Contributing](https://github.com/SoftbearStudios/mk48/wiki/Contributing) Wiki page.

## Trademark

Mk48.io is a trademark of Softbear, Inc.
