// SPDX-FileCopyrightText: 2024 Softbear, Inc.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Headless training mode: `server train [options]`.
//!
//! Runs a [`World`] with built-in bots and externally controlled agents, stepping as fast as the
//! CPU allows (no networking, no wall clock). A trainer drives it over two pipes
//! (`--in-fd`/`--out-fd`, preferred because game code may print to stdout; else stdin/stdout),
//! little-endian:
//!
//! - startup, server -> trainer: `[MAGIC, n_agents, OBS_DIM, ACT_DIM, INFO_DIM]` as u32
//! - each step, trainer -> server: `n_agents * ACT_DIM` f32 actions in `[-1, 1]`
//! - each step, server -> trainer: `n_agents * (OBS_DIM + INFO_DIM)` f32
//!
//! EOF on the input ends the process. Agents observe only [`World::get_player_complete`], which is
//! exactly what a real client is sent, and act only through the normal [`Command`] path.

use crate::bot::Bot;
use crate::player::{PlayerTuple, PlayerTupleRepo, Status, TempPlayer};
use common::death_reason::DeathReason;
use crate::protocol::{AsCommandTrait, CommandTrait};
use crate::server::Server;
use crate::team::TeamRepo;
use crate::world::World;
use common::altitude::Altitude;
use common::angle::Angle;
use common::complete::CompleteTrait;
use common::contact::ContactTrait;
use common::entity::{EntityData, EntityKind, EntitySubKind, EntityType};
use common::guidance::Guidance;
use common::protocol::{Command, Control, Fire, Spawn, Upgrade};
use common::terrain;
use common::ticks::Ticks;
use common::util::level_to_score;
use common::velocity::Velocity;
use kodiak_server::glam::Vec2;
use kodiak_server::rand::seq::IteratorRandom;
use kodiak_server::rand::{thread_rng, Rng};
use kodiak_server::{BotAction, PlayerId};
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::os::fd::FromRawFd;
use std::process::ExitCode;
use std::sync::Arc;

pub(crate) const MAGIC: u32 = 0x4d6b_3438; // "Mk48"

const SELF_FEATURES: usize = 40;
const MAX_CONTACTS: usize = 24;
const CONTACT_FEATURES: usize = 39;
/// Aim snaps to the nearest visible enemy within this fraction of sensor range of the aim point,
/// like a player clicking on a ship rather than beside it.
const AIM_SNAP: f32 = 0.15;
/// Collision guard: closer than this (meters, hull to hull) to an obstacle counts as touching.
/// Touching an obstacle kills a ship after ~6 s (`Mutation::CollidedWithObstacle`).
const OBSTACLE_TOUCH: f32 = 5.0;
const TERRAIN_GRID: usize = 15;
pub const OBS_DIM: usize =
    SELF_FEATURES + MAX_CONTACTS * CONTACT_FEATURES + TERRAIN_GRID * TERRAIN_GRID;

/// steer, throttle, aim_x, aim_y, fire, weapon_class, submerge, active_sensors, ship_group, salvo
/// (salvo > 0 with fire > 0: also fire every other ready weapon that can engage the target)
pub const ACT_DIM: usize = 10;

/// score, alive, died, kills, score_delta, health_lost, fired, level, death_cause, damage_dealt
/// (death_cause: 0 none, 1 terrain, 2 border, 3 weapon, 4 ram/collision, 5 obstacle, 6 other;
/// damage_dealt: boats' worth of weapon damage dealt this step)
pub const INFO_DIM: usize = 10;
/// With `--expert-labels`, each info row is followed by the built-in bot's choice for the same
/// observation, expressed in the agent's action space: `ACT_DIM` values, then valid, aim_valid.
pub const EXPERT_DIM: usize = ACT_DIM + 2;

/// torpedo, gun, missile/rocket, aircraft, depth charge/mine, SAM, decoy
const WEAPON_CLASSES: usize = 7;
const SHIP_GROUPS: usize = 4;
/// Spawn protection lasts 20s (`entity_extension.rs`); the player can count this themselves.
const SPAWN_PROTECTION_TICKS: u32 = 200;

struct Config {
    agents: usize,
    bots: usize,
    ticks_per_step: u32,
    bot_aggression: f32,
    /// Agents ignore actions and run the built-in bot logic (baseline measurement).
    scripted_agents: bool,
    spawn_type: Option<EntityType>,
    in_fd: Option<i32>,
    out_fd: Option<i32>,
    expert_labels: bool,
}

impl Config {
    fn info_dim(&self) -> usize {
        INFO_DIM + if self.expert_labels { EXPERT_DIM } else { 0 }
    }

    fn parse(args: &[String]) -> Result<Self, String> {
        let mut cfg = Self {
            agents: 8,
            bots: 24,
            ticks_per_step: 2,
            bot_aggression: 1.0,
            scripted_agents: false,
            spawn_type: None,
            in_fd: None,
            out_fd: None,
            expert_labels: false,
        };
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if flag == "--scripted-agents" {
                cfg.scripted_agents = true;
                continue;
            }
            if flag == "--expert-labels" {
                cfg.expert_labels = true;
                continue;
            }
            let value = it.next().ok_or(format!("missing value for {flag}"))?;
            let bad = |e: &dyn std::fmt::Display| format!("bad value for {flag} ({value}): {e}");
            match flag.as_str() {
                "--agents" => cfg.agents = value.parse().map_err(|e| bad(&e))?,
                "--bots" => cfg.bots = value.parse().map_err(|e| bad(&e))?,
                "--ticks-per-step" => cfg.ticks_per_step = value.parse().map_err(|e| bad(&e))?,
                "--in-fd" => cfg.in_fd = Some(value.parse().map_err(|e| bad(&e))?),
                "--out-fd" => cfg.out_fd = Some(value.parse().map_err(|e| bad(&e))?),
                "--bot-aggression" => cfg.bot_aggression = value.parse().map_err(|e| bad(&e))?,
                "--spawn-type" => {
                    cfg.spawn_type = Some(
                        EntityType::iter()
                            .find(|t| t.as_str() == value)
                            .ok_or(format!("unknown entity type {value}"))?,
                    )
                }
                _ => return Err(format!("unknown flag {flag}")),
            }
        }
        if cfg.agents == 0 || cfg.ticks_per_step == 0 {
            return Err("--agents and --ticks-per-step must be positive".into());
        }
        Ok(cfg)
    }
}

pub(crate) struct Agent {
    pub(crate) tuple: Arc<PlayerTuple>,
    pub(crate) player_id: PlayerId,
    scripted: Option<Bot>,
    was_alive: bool,
    prev_score: u32,
    prev_health: f32,
    prev_damage_dealt: f32,
    kills: u32,
    fired: bool,
    /// Ticks since the current boat spawned.
    pub(crate) ticks_alive: u32,
    /// Last commanded active-sensor state (the client knows its own toggle).
    pub(crate) active: bool,
    /// Preferred ship group for upgrades and respawns (learned via the last action).
    pub(crate) ship_group: usize,
    /// Built-in bot asked for its choice on the agent's own view (`--expert-labels`).
    shadow: Option<Bot>,
}

impl Agent {
    pub(crate) fn new(tuple: Arc<PlayerTuple>, player_id: PlayerId) -> Self {
        Self {
            tuple,
            player_id,
            scripted: None,
            was_alive: false,
            prev_score: 0,
            prev_health: 1.0,
            prev_damage_dealt: 0.0,
            kills: 0,
            fired: false,
            ticks_alive: 0,
            active: false,
            ship_group: thread_rng().gen_range(0..SHIP_GROUPS),
            shadow: None,
        }
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.tuple.borrow_player().is_alive()
    }
}

struct BotSlot {
    tuple: Arc<PlayerTuple>,
    player_id: PlayerId,
    bot: Bot,
}

pub fn run(args: &[String]) -> ExitCode {
    let cfg = match Config::parse(args) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("train: {e}");
            return ExitCode::FAILURE;
        }
    };

    let total = cfg.agents + cfg.bots;
    let mut world = World::new(World::target_radius(
        total as f32 * EntityType::FairmileD.data().visual_area(),
    ));
    // Pre-populate collectibles and obstacles.
    for _ in 0..100 {
        world.spawn_statics(Ticks::from_whole_secs(10));
    }

    let mut players = PlayerTupleRepo::default();
    let mut teams = TeamRepo::<Server>::default();

    let mut new_player = |player_id: PlayerId| {
        let tuple = Arc::new(PlayerTuple::new(TempPlayer::new(player_id, None)));
        players.insert(player_id, Arc::clone(&tuple));
        tuple
    };

    let mut agents: Vec<Agent> = (0..cfg.agents)
        .map(|i| {
            let player_id = PlayerId::nth_client(i).unwrap();
            let mut agent = Agent::new(new_player(player_id), player_id);
            agent.scripted = cfg.scripted_agents.then(Bot::default);
            agent.shadow = cfg.expert_labels.then(Bot::default);
            agent
        })
        .collect();

    let mut bots: Vec<BotSlot> = (0..cfg.bots)
        .map(|i| {
            let player_id = PlayerId::nth_bot(i).unwrap();
            BotSlot {
                tuple: new_player(player_id),
                player_id,
                bot: Bot::default(),
            }
        })
        .collect();

    for agent in &mut agents {
        agent.ship_group = thread_rng().gen_range(0..SHIP_GROUPS);
        spawn_agent(&mut world, agent, &players, &mut teams, cfg.spawn_type);
        agent.was_alive = agent.tuple.borrow_player().is_alive();
    }

    // Safety: the trainer passes us ownership of these descriptors.
    let mut input: Box<dyn Read> = match cfg.in_fd {
        Some(fd) => Box::new(BufReader::new(unsafe { File::from_raw_fd(fd) })),
        None => Box::new(std::io::stdin()),
    };
    let mut output: BufWriter<Box<dyn Write>> = BufWriter::new(match cfg.out_fd {
        Some(fd) => Box::new(unsafe { File::from_raw_fd(fd) }),
        None => Box::new(std::io::stdout()),
    });

    let info_dim = cfg.info_dim();
    let header = [
        MAGIC,
        cfg.agents as u32,
        OBS_DIM as u32,
        ACT_DIM as u32,
        info_dim as u32,
    ];
    let header_bytes: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
    if output.write_all(&header_bytes).and_then(|_| output.flush()).is_err() {
        return ExitCode::FAILURE;
    }
    eprintln!(
        "train: agents={} bots={} ticks_per_step={} obs_dim={OBS_DIM} act_dim={ACT_DIM}",
        cfg.agents, cfg.bots, cfg.ticks_per_step
    );

    let mut action_bytes = vec![0u8; cfg.agents * ACT_DIM * 4];
    let mut actions = vec![0f32; cfg.agents * ACT_DIM];
    let mut out = vec![0f32; cfg.agents * (OBS_DIM + info_dim)];
    let mut out_bytes = Vec::with_capacity(out.len() * 4);
    let mut kill_log: Vec<PlayerId> = Vec::new();

    loop {
        match input.read_exact(&mut action_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("train: read error {e}");
                return ExitCode::FAILURE;
            }
        }
        for (a, chunk) in actions.iter_mut().zip(action_bytes.chunks_exact(4)) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            *a = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        }

        // Agents act once per step (a decision), then the world advances `ticks_per_step` ticks.
        for (i, agent) in agents.iter_mut().enumerate() {
            agent.kills = 0;
            agent.fired = false;
            let action = &actions[i * ACT_DIM..(i + 1) * ACT_DIM];
            act_agent(&mut world, agent, action, &players, &mut teams, cfg.bot_aggression);
        }

        for _ in 0..cfg.ticks_per_step {
            for slot in &mut bots {
                let action = {
                    let update = world.get_player_complete(&slot.tuple);
                    slot.bot.act(update, slot.player_id, cfg.bot_aggression)
                };
                if let BotAction::Some(command) = action {
                    let _ = command.as_command().apply(
                        &mut world,
                        &slot.tuple,
                        &players,
                        &mut teams,
                        None,
                        None,
                    );
                }
            }
            world.update(Ticks::ONE, &mut |killer, _dead| kill_log.push(killer));
            world.terrain.pre_update();
            world.terrain.post_update();
        }

        for agent in &mut agents {
            agent.kills = kill_log.iter().filter(|k| **k == agent.player_id).count() as u32;
            agent.ticks_alive += cfg.ticks_per_step;
        }
        kill_log.clear();

        for (i, agent) in agents.iter_mut().enumerate() {
            let row = &mut out[i * (OBS_DIM + info_dim)..(i + 1) * (OBS_DIM + info_dim)];
            let (obs, info) = row.split_at_mut(OBS_DIM);

            let alive = agent.tuple.borrow_player().is_alive();
            let died = agent.was_alive && !alive;
            let death_cause = if died {
                match &agent.tuple.borrow_player().status {
                    Status::Dead { reason, .. } => match reason {
                        DeathReason::Terrain => 1.0,
                        DeathReason::Border => 2.0,
                        DeathReason::Weapon { .. } => 3.0,
                        DeathReason::Ram { .. } | DeathReason::Boat { .. } => 4.0,
                        DeathReason::Obstacle(_) => 5.0,
                        _ => 6.0,
                    },
                    _ => 6.0,
                }
            } else {
                0.0
            };
            if !alive {
                // Respawn immediately so the next observation starts a new life.
                spawn_agent(&mut world, agent, &players, &mut teams, cfg.spawn_type);
                agent.ticks_alive = 0;
                agent.active = false;
            }

            let (score, health, level) = observe(&world, agent, obs);
            let now_alive = agent.tuple.borrow_player().is_alive();

            let continuing = agent.was_alive && alive;
            info[0] = score as f32;
            info[1] = now_alive as u8 as f32;
            info[2] = died as u8 as f32;
            info[3] = agent.kills as f32;
            info[4] = if continuing {
                score as f32 - agent.prev_score as f32
            } else {
                0.0
            };
            info[5] = if continuing {
                (agent.prev_health - health).max(0.0)
            } else {
                0.0
            };
            info[6] = agent.fired as u8 as f32;
            info[7] = level as f32;
            info[8] = death_cause;
            let damage_dealt = agent.tuple.borrow_player().damage_dealt;
            info[9] = damage_dealt - agent.prev_damage_dealt;
            agent.prev_damage_dealt = damage_dealt;

            agent.was_alive = now_alive;
            agent.prev_score = score;
            agent.prev_health = health;

            if let Some(mut shadow) = agent.shadow.take() {
                let command = {
                    let update = world.get_player_complete(&agent.tuple);
                    shadow.act(update, agent.player_id, cfg.bot_aggression)
                };
                agent.shadow = Some(shadow);
                let expert = &mut info[INFO_DIM..INFO_DIM + EXPERT_DIM];
                expert.fill(0.0);
                if let BotAction::Some(Command::Control(control)) = &command {
                    let solution = agent.shadow.as_ref().and_then(|b| b.last_solution);
                    command_to_action(&world, agent, control, solution, expert);
                }
            }
        }

        out_bytes.clear();
        for v in &out {
            let v = if v.is_finite() { *v } else { 0.0 };
            out_bytes.extend_from_slice(&v.to_le_bytes());
        }
        if output.write_all(&out_bytes).and_then(|_| output.flush()).is_err() {
            // Trainer went away.
            return ExitCode::SUCCESS;
        }
    }
}

/// Highest-level affordable boat, preferring the given ship family.
pub(crate) fn pick_spawn_type(score: u32, ship_group: usize) -> Option<EntityType> {
    let max_level = EntityType::spawn_options(score, false)
        .map(|t| t.data().level)
        .max()?;
    choose_in_group(
        EntityType::spawn_options(score, false).filter(|t| t.data().level == max_level),
        ship_group,
        &mut thread_rng(),
    )
}

fn spawn_agent(
    world: &mut World,
    agent: &mut Agent,
    players: &PlayerTupleRepo,
    teams: &mut TeamRepo<Server>,
    spawn_type: Option<EntityType>,
) {
    let score = agent.tuple.borrow_player().score;
    let entity_type = spawn_type
        .filter(|t| t.can_spawn_as(score, false))
        .or_else(|| pick_spawn_type(score, agent.ship_group));
    let Some(entity_type) = entity_type else {
        return;
    };
    let spawn = Command::Spawn(Spawn {
        alias: None,
        entity_type,
    });
    for _ in 0..5 {
        if spawn
            .as_command()
            .apply(world, &agent.tuple, players, teams, None, None)
            .is_ok()
        {
            break;
        }
    }
}

/// Coarse ship families, so the policy can express a preference when upgrading / respawning.
fn ship_group(sub_kind: EntitySubKind) -> Option<usize> {
    use EntitySubKind::*;
    match sub_kind {
        Submarine => Some(0),
        Battleship | Cruiser | Destroyer | Dreadnought | Corvette | MissileBoat | Lcs | Mtb => {
            Some(1)
        }
        Carrier => Some(2),
        Ram | Dredger | Icebreaker | Hovercraft | Minelayer | Tanker | Pirate => Some(3),
        _ => None,
    }
}

/// Ships that measured worst at their level in every evaluation (no usable weapons for the agent:
/// a ram, a dredger, a minelayer whose mines drop astern, a tanker). NN agents avoid them.
fn is_weak_ship(entity_type: EntityType) -> bool {
    use EntitySubKind::*;
    matches!(entity_type.data().sub_kind, Ram | Dredger | Minelayer | Tanker)
}

/// Picks a random option in the preferred group, or any option if the group has none. Weak ships
/// are only picked when nothing else is available.
fn choose_in_group(
    options: impl Iterator<Item = EntityType>,
    group: usize,
    rng: &mut impl Rng,
) -> Option<EntityType> {
    let all: Vec<EntityType> = options.collect();
    let strong: Vec<EntityType> = all.iter().copied().filter(|t| !is_weak_ship(*t)).collect();
    let options = if strong.is_empty() { all } else { strong };
    options
        .iter()
        .copied()
        .filter(|t| ship_group(t.data().sub_kind) == Some(group))
        .choose(rng)
        .or_else(|| options.into_iter().choose(rng))
}

fn collectible_value(entity_type: EntityType) -> f32 {
    match entity_type {
        EntityType::Barrel => 1.0,
        EntityType::Coin => EntityData::COIN_VALUE as f32,
        EntityType::Crate | EntityType::Scrap => 2.0,
        _ => 0.0,
    }
}

fn weapon_class(sub_kind: EntitySubKind) -> Option<usize> {
    use EntitySubKind::*;
    match sub_kind {
        Torpedo | RocketTorpedo => Some(0),
        Shell => Some(1),
        Missile | Rocket => Some(2),
        Plane | Heli => Some(3),
        DepthCharge | Mine => Some(4),
        Sam => Some(5),
        Sonar => Some(6),
        _ => None,
    }
}

/// Prints every entity type as `id<TAB>name<TAB>kind<TAB>sub_kind<TAB>level` (for analysis tools).
pub fn print_types() -> ExitCode {
    for t in EntityType::iter() {
        let data = t.data();
        println!(
            "{}\t{}\t{:?}\t{:?}\t{}",
            t as usize,
            t.as_str(),
            data.kind,
            data.sub_kind,
            data.level
        );
    }
    ExitCode::SUCCESS
}

/// Converts a normalized action into a normal client [`Command::Control`] and applies it.
pub(crate) fn act_agent(
    world: &mut World,
    agent: &mut Agent,
    action: &[f32],
    players: &PlayerTupleRepo,
    teams: &mut TeamRepo<Server>,
    bot_aggression: f32,
) {
    if let Some(bot) = agent.scripted.as_mut() {
        let command = {
            let update = world.get_player_complete(&agent.tuple);
            bot.act(update, agent.player_id, bot_aggression)
        };
        if let BotAction::Some(command) = command {
            let fired = matches!(&command, Command::Control(c) if c.fire.is_some());
            let ok = command
                .as_command()
                .apply(world, &agent.tuple, players, teams, None, None)
                .is_ok();
            agent.fired = fired && ok;
        }
        return;
    }

    let Some((control, upgrade, salvo)) = agent_commands(world, agent, action) else {
        return;
    };
    agent.fired = apply_commands(world, agent, control, upgrade, salvo, players, teams);
}

/// Converts a normalized action into the client commands it stands for (a control, plus an
/// upgrade when one is affordable), without applying them. `None` if the agent has no boat.
pub(crate) fn agent_commands(
    world: &World,
    agent: &mut Agent,
    action: &[f32],
) -> Option<(Control, Option<EntityType>, Vec<u8>)> {
    let (control, upgrade, salvo) = {
        let mut update = world.get_player_complete(&agent.tuple);
        let score = update.score();
        let mut contacts = update.contacts();
        let boat = contacts
            .next()
            .filter(|c| c.is_boat() && c.player_id() == Some(agent.player_id))?;
        let others: Vec<_> = contacts.collect();
        let boat_type = boat.entity_type().unwrap();
        let data: &EntityData = boat_type.data();
        let transform = *boat.transform();
        let forward = transform.direction.to_vec();
        let left = forward.perp();

        let mut direction_target = transform.direction + Angle::from_radians(action[0] * PI);
        let mut throttle = action[1];
        if data.level <= 1 {
            // Matches the client: level 1 ships can't reverse.
            throttle = throttle.max(0.0);
        }
        let mut velocity_target = Velocity::from_mps(throttle * data.speed.to_mps());

        // Collision guard (a reflex, like a car's collision avoidance). If the ship is touching an
        // obstacle, or its commanded course runs into an obstacle, land or the world border within
        // ~2 s, it takes the clear heading closest to the one it wanted (straight away from a
        // touched obstacle). Uses only what the player can see; the network decides everything else.
        let reverse = velocity_target.to_mps() < 0.0;
        let look_ahead = 15.0 + data.radius + 2.0 * transform.velocity.to_mps().abs();
        let obstacles: Vec<(f32, Vec2)> = others
            .iter()
            .filter_map(|c| {
                let obstacle = c.entity_type()?.data();
                (obstacle.kind == EntityKind::Obstacle).then(|| {
                    let away = transform.position - c.transform().position;
                    (away.length() - obstacle.radius - data.radius, away)
                })
            })
            .filter(|(clearance, _)| *clearance < look_ahead)
            .collect();
        let world_radius = update.world_radius();
        let terrain = update.terrain();
        let land = |p: Vec2| is_land_or_border(p, terrain, world_radius);
        // A heading is clear if the hull-wide lane ahead is free of land/border and doesn't point
        // at a nearby obstacle.
        let half_width = 0.5 * data.width + 5.0;
        let clear = |heading: Angle| {
            let dir = heading.to_vec() * if reverse { -1.0 } else { 1.0 };
            let side = dir.perp() * half_width;
            obstacles.iter().all(|(_, away)| dir.dot(*away) >= 0.0)
                && (1..=4).all(|k| {
                    let p = transform.position + dir * (0.5 * data.length + look_ahead * k as f32 / 4.0);
                    !land(p) && !land(p + side) && !land(p - side)
                })
        };
        // Land touching (or about to touch) any part of the hull: probe a ring around the ship.
        let ring = 0.5 * data.length + 10.0;
        let land_push: Vec2 = (0..16)
            .map(|i| Angle::from_radians(i as f32 * (2.0 * PI / 16.0)).to_vec())
            .filter(|d| land(transform.position + *d * ring))
            .fold(Vec2::ZERO, |sum, d| sum - d);
        let touching = obstacles
            .iter()
            .filter(|(clearance, _)| *clearance < OBSTACLE_TOUCH)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, away)| *away)
            .or((land_push != Vec2::ZERO).then_some(land_push));
        if touching.is_some() || !clear(direction_target) {
            // Preferred heading: away from what we touch, else the commanded one.
            let base = touching.map_or(direction_target, |away| {
                Angle::from(away) + if reverse { Angle::PI } else { Angle::ZERO }
            });
            let escape = (0..=6)
                .flat_map(|k| [k, -k])
                .map(|k| base + Angle::from_degrees(30.0 * k as f32))
                .find(|h| clear(*h));
            if let Some(heading) = escape {
                direction_target = heading;
            }
            if touching.is_some() {
                velocity_target = Velocity::from_mps(0.6 * data.speed.to_mps());
            }
        }

        let aim_local = Vec2::new(action[2], action[3]).clamp_length_max(1.0);
        let aim_range = data.sensors.max_range();
        let mut aim = transform.position + (forward * aim_local.x + left * aim_local.y) * aim_range;
        let snap_radius_squared = (AIM_SNAP * aim_range).powi(2);
        if let Some((_, target)) = others
            .iter()
            .filter(|c| c.player_id() != Some(agent.player_id) && is_target(*c))
            .map(|c| (c.transform().position.distance_squared(aim), c.transform().position))
            .filter(|(d, _)| *d < snap_radius_squared)
            .min_by(|a, b| a.0.total_cmp(&b.0))
        {
            aim = target;
        }

        let fire = if action[4] > 0.0 {
            let class = ((action[5] + 1.0) * 0.5 * WEAPON_CLASSES as f32)
                .clamp(0.0, WEAPON_CLASSES as f32 - 1.0) as usize;
            best_armament(&boat, data, aim, class).map(|armament_index| Fire { armament_index })
        } else {
            None
        };
        let salvo = if action[4] > 0.0 && action[9] > 0.0 {
            salvo_armaments(&boat, data, aim, fire.as_ref().map(|f| f.armament_index))
        } else {
            Vec::new()
        };

        agent.active = action[7] > 0.0;
        agent.ship_group = ((action[8] + 1.0) * 0.5 * SHIP_GROUPS as f32)
            .clamp(0.0, SHIP_GROUPS as f32 - 1.0) as usize;
        let upgrade = choose_in_group(
            boat_type
                .upgrade_options(score, false)
                .filter(|t| t.data().level == data.level + 1),
            agent.ship_group,
            &mut thread_rng(),
        );

        (
            Control {
                guidance: Some(Guidance {
                    direction_target,
                    velocity_target,
                }),
                submerge: action[6] > 0.0,
                aim_target: Some(aim),
                active: action[7] > 0.0,
                fire,
                pay: None,
                hint: None,
            },
            upgrade,
            salvo,
        )
    };
    Some((control, upgrade, salvo))
}

/// Every other ready offensive weapon (torpedo, gun, missile/rocket, aircraft, depth charge/mine)
/// that can engage `aim` right now: what "fire everything" means.
fn salvo_armaments<C: ContactTrait>(boat: &C, data: &EntityData, aim: Vec2, except: Option<u8>) -> Vec<u8> {
    (0..WEAPON_CLASSES.min(5))
        .flat_map(|class| ready_armaments(boat, data, aim, class))
        .filter(|i| Some(*i) != except)
        .collect()
}

/// Ready armaments of a class that can engage `aim` (same rules as [`best_armament`]).
fn ready_armaments<C: ContactTrait>(boat: &C, data: &EntityData, aim: Vec2, class: usize) -> Vec<u8> {
    let reloads = boat.reloads();
    let mut out = Vec::new();
    for (i, armament) in data.armaments.iter().enumerate() {
        if !reloads.get(i).map(|r| *r).unwrap_or(false) {
            continue;
        }
        let armament_data = armament.entity_type.data();
        if !matches!(armament_data.kind, EntityKind::Weapon | EntityKind::Aircraft)
            || weapon_class(armament_data.sub_kind) != Some(class)
        {
            continue;
        }
        if let Some(turret_index) = armament.turret {
            if !data.turrets[turret_index].within_azimuth(boat.turrets()[turret_index]) {
                continue;
            }
        }
        let transform = *boat.transform() + data.armament_transform(boat.turrets(), i);
        let angle_diff = (Angle::from(aim - transform.position) - transform.direction).abs();
        if armament.vertical || armament_data.kind == EntityKind::Aircraft || angle_diff <= Angle::from_degrees(60.0) {
            out.push(i as u8);
        }
    }
    out
}

/// Applies an agent's commands (control, salvo, upgrade). Returns whether anything fired.
pub(crate) fn apply_commands(
    world: &mut World,
    agent: &Agent,
    control: Control,
    upgrade: Option<EntityType>,
    salvo: Vec<u8>,
    players: &PlayerTupleRepo,
    teams: &mut TeamRepo<Server>,
) -> bool {
    let primary = control.fire.is_some();
    let ok = Command::Control(control)
        .as_command()
        .apply(world, &agent.tuple, players, teams, None, None)
        .is_ok();
    let mut fired = primary && ok;
    for armament_index in salvo {
        fired |= Fire { armament_index }
            .apply(world, &agent.tuple, players, teams, None, None)
            .is_ok();
    }
    if let Some(entity_type) = upgrade {
        let _ = Command::Upgrade(Upgrade { entity_type })
            .as_command()
            .apply(world, &agent.tuple, players, teams, None, None);
    }
    fired
}

/// Expresses a built-in bot [`Control`] in the agent's normalized action space (the inverse of
/// [`act_agent`]). Writes `ACT_DIM` values, then valid and aim_valid flags.
///
/// Firing uses the bot's firing *solution* rather than its randomized trigger pull: "fire" means
/// a ready weapon has a valid target, which is a deterministic function of what the agent sees.
fn command_to_action(
    world: &World,
    agent: &Agent,
    control: &Control,
    solution: Option<(u8, Vec2)>,
    out: &mut [f32],
) {
    let mut update = world.get_player_complete(&agent.tuple);
    let mut contacts = update.contacts();
    let Some(boat) = contacts
        .next()
        .filter(|c| c.is_boat() && c.player_id() == Some(agent.player_id))
    else {
        return;
    };
    let data: &EntityData = boat.entity_type().unwrap().data();
    let transform = *boat.transform();
    let forward = transform.direction.to_vec();
    let left = forward.perp();
    let class_value = |class: usize| -1.0 + (2 * class + 1) as f32 / WEAPON_CLASSES as f32;

    let guidance = control.guidance.unwrap_or(*boat.guidance());
    out[0] = (guidance.direction_target - transform.direction).to_radians() / PI;
    out[1] = (guidance.velocity_target.to_mps() / data.speed.to_mps().max(1.0)).clamp(-1.0, 1.0);
    if let Some(aim) = solution.map(|(_, target)| target).or(control.aim_target) {
        let local = aim - transform.position;
        let local = Vec2::new(local.dot(forward), local.dot(left)) / data.sensors.max_range().max(1.0);
        let local = local.clamp_length_max(1.0);
        out[2] = local.x;
        out[3] = local.y;
        out[ACT_DIM + 1] = 1.0;
    }
    out[4] = -1.0;
    out[5] = class_value(1);
    if let Some(armament_index) = solution
        .map(|(i, _)| i)
        .or(control.fire.as_ref().map(|f| f.armament_index))
    {
        if let Some(class) = data
            .armaments
            .get(armament_index as usize)
            .and_then(|a| weapon_class(a.entity_type.data().sub_kind))
        {
            out[4] = 1.0;
            out[5] = class_value(class);
        }
    }
    out[6] = if control.submerge { 1.0 } else { -1.0 };
    out[7] = if control.active { 1.0 } else { -1.0 };
    let group = ship_group(data.sub_kind).unwrap_or(1);
    out[8] = -1.0 + (2 * group + 1) as f32 / SHIP_GROUPS as f32;
    out[9] = -1.0;
    out[ACT_DIM] = 1.0;
}

/// Picks the ready armament of the given class that best points at `aim` (same rules as the
/// built-in bot: turret must be within azimuth, and non-vertical weapons within 60 degrees).
fn best_armament<C: ContactTrait>(
    boat: &C,
    data: &EntityData,
    aim: Vec2,
    class: usize,
) -> Option<u8> {
    let reloads = boat.reloads();
    let mut best: Option<(u8, Angle)> = None;
    for (i, armament) in data.armaments.iter().enumerate() {
        if !reloads.get(i).map(|r| *r).unwrap_or(false) {
            continue;
        }
        let armament_data = armament.entity_type.data();
        if !matches!(
            armament_data.kind,
            EntityKind::Weapon | EntityKind::Aircraft | EntityKind::Decoy
        ) || weapon_class(armament_data.sub_kind) != Some(class)
        {
            continue;
        }
        if let Some(turret_index) = armament.turret {
            if !data.turrets[turret_index].within_azimuth(boat.turrets()[turret_index]) {
                continue;
            }
        }
        let transform = *boat.transform() + data.armament_transform(boat.turrets(), i);
        let mut angle_diff = (Angle::from(aim - transform.position) - transform.direction).abs();
        if armament.vertical
            || matches!(
                armament_data.kind,
                EntityKind::Aircraft | EntityKind::Decoy
            )
        {
            angle_diff = Angle::ZERO;
        }
        if angle_diff > Angle::from_degrees(60.0) {
            continue;
        }
        if best.map_or(true, |(_, d)| angle_diff < d) {
            best = Some((i as u8, angle_diff));
        }
    }
    best.map(|(i, _)| i)
}

/// Contacts worth shooting at (same categories the built-in bot engages).
fn is_target<C: ContactTrait>(contact: &C) -> bool {
    contact.entity_type().map_or(false, |t| {
        let data = t.data();
        matches!(data.kind, EntityKind::Boat | EntityKind::Aircraft)
            || matches!(data.sub_kind, EntitySubKind::Torpedo | EntitySubKind::Missile)
    })
}

fn is_land_or_border(pos: Vec2, terrain: &terrain::Terrain, world_radius: f32) -> bool {
    pos.length_squared() > world_radius.powi(2)
        || terrain.sample(pos).unwrap_or(Altitude::MIN) >= terrain::SAND_LEVEL
}

fn kind_index(kind: EntityKind) -> usize {
    match kind {
        EntityKind::Aircraft => 0,
        EntityKind::Boat => 1,
        EntityKind::Collectible => 2,
        EntityKind::Decoy => 3,
        EntityKind::Obstacle => 4,
        EntityKind::Turret => 5,
        EntityKind::Weapon => 6,
    }
}

/// Writes the agent's observation (only what its client would see) into `obs`.
/// Returns (score, health fraction, level).
pub(crate) fn observe(world: &World, agent: &Agent, obs: &mut [f32]) -> (u32, f32, u8) {
    obs.fill(0.0);
    let mut update = world.get_player_complete(&agent.tuple);
    let score = update.score();
    let world_radius = update.world_radius();
    let mut contacts = update.contacts();
    let terrain = update.terrain();

    let Some(boat) = contacts
        .next()
        .filter(|c| c.is_boat() && c.player_id() == Some(agent.player_id))
    else {
        return (score, 1.0, 0);
    };

    let data: &EntityData = boat.entity_type().unwrap().data();
    let transform = *boat.transform();
    let pos = transform.position;
    let forward = transform.direction.to_vec();
    let left = forward.perp();
    let to_local = |v: Vec2| Vec2::new(v.dot(forward), v.dot(left));
    let max_speed = data.speed.to_mps().max(1.0);
    let sensor_range = data.sensors.max_range().max(1.0);
    let health = 1.0 - boat.damage().to_secs() / data.max_health().to_secs();
    let max_level = EntityData::MAX_BOAT_LEVEL as f32;

    // Own ship.
    let s = &mut obs[..SELF_FEATURES];
    s[0] = 1.0;
    s[1] = health;
    s[2] = transform.velocity.to_mps() / max_speed;
    s[3] = forward.x;
    s[4] = forward.y;
    s[5] = pos.x / world_radius;
    s[6] = pos.y / world_radius;
    s[7] = ((world_radius - pos.length()) / sensor_range).clamp(0.0, 1.0);
    s[8] = data.level as f32 / max_level;
    let level_score = level_to_score(data.level) as f32;
    let next_score = level_to_score(data.level + 1) as f32;
    s[9] = ((score as f32 - level_score) / (next_score - level_score).max(1.0)).clamp(0.0, 1.0);
    s[10] = boat.altitude().is_submerged() as u8 as f32;
    s[11] = (data.sub_kind == EntitySubKind::Submarine) as u8 as f32;
    let reloads = boat.reloads();
    let mut ready = 0usize;
    for (i, armament) in data.armaments.iter().enumerate() {
        if reloads.get(i).map(|r| *r).unwrap_or(false) {
            ready += 1;
            if let Some(class) = weapon_class(armament.entity_type.data().sub_kind) {
                s[13 + class] = 1.0;
            }
        }
    }
    s[12] = ready as f32 / data.armaments.len().max(1) as f32;
    s[20] = boat.guidance().velocity_target.to_mps() / max_speed;
    s[21] = boat.altitude().to_norm();
    s[22] = data.sensors.visual.range / 2000.0;
    s[23] = data.sensors.radar.range / 2000.0;
    s[24] = data.sensors.sonar.range / 2000.0;
    s[25] = (score as f32 + 1.0).ln() / 10.0;
    s[26] = data.length / 200.0;
    s[27] = (boat.guidance().direction_target - transform.direction)
        .to_radians()
        .sin();
    s[28] = agent.active as u8 as f32;
    s[29] = (agent.ticks_alive as f32 / SPAWN_PROTECTION_TICKS as f32).min(1.0);
    s[30] = (score >= level_to_score(data.level + 1)) as u8 as f32;
    s[31] = agent.ship_group as f32 / (SHIP_GROUPS - 1) as f32;
    if let Some(group) = ship_group(data.sub_kind) {
        s[32 + group] = 1.0;
    }
    // Exact ship type (the policy embeds it), so strategy can differ per vehicle.
    s[36] = boat.entity_type().unwrap() as usize as f32;

    // Nearest visible contacts, in the ship's frame.
    let own_velocity = forward * transform.velocity.to_mps();
    let mut nearby: Vec<_> = contacts
        .map(|c| (c.transform().position.distance_squared(pos), c))
        .collect();
    nearby.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (k, (distance_squared, contact)) in nearby.iter().take(MAX_CONTACTS).enumerate() {
        let base = SELF_FEATURES + k * CONTACT_FEATURES;
        let c = &mut obs[base..base + CONTACT_FEATURES];
        let ct = contact.transform();
        let rel = to_local(ct.position - pos) / sensor_range;
        let velocity = ct.direction.to_vec() * ct.velocity.to_mps();
        let rel_velocity = to_local(velocity - own_velocity) / 20.0;
        let rel_heading = (ct.direction - transform.direction).to_vec();
        c[0] = 1.0;
        c[1] = rel.x;
        c[2] = rel.y;
        c[3] = rel_velocity.x;
        c[4] = rel_velocity.y;
        c[5] = rel_heading.x;
        c[6] = rel_heading.y;
        c[7] = distance_squared.sqrt() / sensor_range;
        match contact.entity_type().map(EntityType::data) {
            Some(contact_data) => {
                c[8 + kind_index(contact_data.kind)] = 1.0;
                if contact_data.kind == EntityKind::Boat {
                    c[17] = contact_data.level as f32 / max_level;
                }
                if let Some(class) = weapon_class(contact_data.sub_kind) {
                    c[19 + class] = 1.0;
                }
                if contact_data.kind == EntityKind::Collectible {
                    c[26] = collectible_value(contact.entity_type().unwrap()) / 10.0;
                }
                if contact_data.kind == EntityKind::Boat {
                    if let Some(group) = ship_group(contact_data.sub_kind) {
                        c[27 + group] = 1.0;
                    }
                }
                // Exact type id + 1 (0 = unidentified blip).
                c[38] = (contact.entity_type().unwrap() as usize + 1) as f32;
            }
            None => c[15] = 1.0,
        }
        c[16] = (contact.player_id() == Some(agent.player_id)) as u8 as f32;
        c[18] = contact.altitude().to_norm();
        // Firing arcs: can a ready weapon of each class engage this contact right now?
        // (A player sees their turrets and reload bars.)
        if c[16] == 0.0 {
            for class in 0..WEAPON_CLASSES {
                c[31 + class] =
                    best_armament(&boat, data, ct.position, class).is_some() as u8 as f32;
            }
        }
    }

    // Land / border around the ship, in the ship's frame.
    let span = (0.5 * data.sensors.visual.range).clamp(200.0, 800.0);
    let half = (TERRAIN_GRID / 2) as f32;
    let base = SELF_FEATURES + MAX_CONTACTS * CONTACT_FEATURES;
    for gy in 0..TERRAIN_GRID {
        for gx in 0..TERRAIN_GRID {
            let offset = Vec2::new(gx as f32 - half, gy as f32 - half) / half * span;
            let world_pos = pos + forward * offset.x + left * offset.y;
            obs[base + gy * TERRAIN_GRID + gx] =
                is_land_or_border(world_pos, terrain, world_radius) as u8 as f32;
        }
    }

    (score, health, data.level)
}
