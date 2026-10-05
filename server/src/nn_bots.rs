// SPDX-FileCopyrightText: 2024 Softbear, Inc.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Neural-network control for the normal (networked) server.
//!
//! Enabled with `MK48_NN_BOTS=<count>` and `MK48_NN_POLICY=<policy file>`; optional
//! `MK48_NN_ELITE_POLICY`, `MK48_NN_PYTHON`, `MK48_NN_SCRIPT`.
//!
//! - Engine bots number `0..count` are driven by the network instead of the hand-written logic.
//!   They are ordinary bots otherwise (leaderboard, names). A count >= `--bots` makes every bot an
//!   NN bot. With an elite policy, the last of them ("NN Elite") is driven by it.
//! - Autopilot: a real player whose name starts with "AI" or "NN" has their ship driven by the
//!   network (their own steering, firing and upgrades are ignored, and they respawn automatically),
//!   so their browser shows exactly what the NN bot sees. Names containing "ELITE" use the elite
//!   policy.
//!
//! Every `TICKS_PER_DECISION` ticks, each NN ship's observation (exactly what a client would see,
//! encoded as in training) goes to the policy process and the returned actions are applied
//! through the normal command path.
//!
//! Protocol (little-endian), server -> policy: `[MAGIC, rows, OBS_DIM, ACT_DIM, 1]` as u32 once,
//! then per decision `rows * (OBS_DIM + 1)` f32, the extra value being the policy to use
//! (0 = main, 1 = elite); policy -> server: `rows * ACT_DIM` f32.

use crate::player::{PlayerTuple, PlayerTupleRepo};
use crate::protocol::AsCommandTrait;
use crate::server::Server;
use crate::team::TeamRepo;
use crate::train::{
    agent_commands, apply_commands, observe, pick_spawn_type, Agent, ACT_DIM, MAGIC, OBS_DIM,
};
use crate::world::World;
use common::protocol::{Command, Spawn};
use kodiak_server::log::{error, info};
use kodiak_server::{PlayerAlias, PlayerId};
use std::io::{BufReader, BufWriter, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command as Process, Stdio};
use std::sync::Arc;

/// Same decision rate as training (10 ticks/s / 2 = 5 decisions per second).
const TICKS_PER_DECISION: u32 = 2;
/// Real players that can be on autopilot at once.
const MAX_AUTOPILOT: usize = 4;

/// Players named "AI..." or "NN..." are driven by the network.
pub fn is_autopilot_alias(alias: &PlayerAlias) -> bool {
    let name = alias.as_str().to_ascii_uppercase();
    name.starts_with("AI") || name.starts_with("NN")
}

fn is_elite_alias(alias: &PlayerAlias) -> bool {
    alias.as_str().to_ascii_uppercase().contains("ELITE")
}

pub struct NnBots {
    count: usize,
    elite: bool,
    bots: Vec<Agent>,
    autopilot: Vec<Agent>,
    child: Child,
    to_policy: BufWriter<ChildStdin>,
    from_policy: BufReader<ChildStdout>,
    tick: u32,
}

impl NnBots {
    pub fn from_env() -> Option<Self> {
        let count: usize = std::env::var("MK48_NN_BOTS").ok()?.parse().ok()?;
        let policy = std::env::var("MK48_NN_POLICY").ok()?;
        let elite_policy = std::env::var("MK48_NN_ELITE_POLICY").ok();
        let python = std::env::var("MK48_NN_PYTHON").unwrap_or_else(|_| "python3".into());
        let script = std::env::var("MK48_NN_SCRIPT").unwrap_or_else(|_| "serve_policy.py".into());
        let mut process = Process::new(&python);
        process.arg(&script).arg(&policy);
        if let Some(elite) = &elite_policy {
            process.arg(elite);
        }
        let mut child = match process
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                error!("NN bots disabled: could not start {python} {script}: {e}");
                return None;
            }
        };
        let to_policy = BufWriter::new(child.stdin.take().unwrap());
        let from_policy = BufReader::new(child.stdout.take().unwrap());
        info!("NN bots enabled: engine bots 0..{count} driven by {policy}");
        Some(Self {
            count,
            elite: elite_policy.is_some(),
            bots: Vec::new(),
            autopilot: Vec::new(),
            child,
            to_policy,
            from_policy,
            tick: 0,
        })
    }

    /// Whether this engine bot is driven by the network (its hand-written logic is then skipped).
    pub fn controls(&self, player_id: PlayerId) -> bool {
        player_id.bot_number().map_or(false, |n| n < self.count)
    }

    /// Whether this real player's ship is on autopilot (their own controls are then ignored).
    pub fn is_autopilot(&self, player_id: PlayerId) -> bool {
        self.autopilot.iter().any(|a| a.player_id == player_id)
    }

    fn is_elite_bot(&self, bot_number: usize) -> bool {
        self.elite && bot_number + 1 == self.count
    }

    /// Called once per server tick, after the world updates. Returns false if the bridge failed
    /// (the caller should then drop it).
    pub fn tick(
        &mut self,
        world: &mut World,
        players: &PlayerTupleRepo,
        teams: &mut TeamRepo<Server>,
    ) -> bool {
        self.tick += 1;
        if self.tick % TICKS_PER_DECISION != 0 {
            return true;
        }
        let rows = self.count + MAX_AUTOPILOT;
        if self.tick == TICKS_PER_DECISION {
            let header = [MAGIC, rows as u32, OBS_DIM as u32, ACT_DIM as u32, 1];
            let header: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = self.to_policy.write_all(&header) {
                error!("NN bots stopped: {e}");
                return false;
            }
        }

        // Track NN engine bots (recycled after quitting, so match the tuple too) and autopilots.
        let bots: Vec<(PlayerId, Arc<PlayerTuple>)> = (0..self.count)
            .filter_map(PlayerId::nth_bot)
            .filter_map(|id| players.get(id).map(|t| (id, Arc::clone(t))))
            .collect();
        sync(&mut self.bots, bots);
        let mut pilots: Vec<(PlayerId, Arc<PlayerTuple>)> = players
            .iter()
            .filter(|t| {
                let p = t.borrow_player();
                !p.is_bot() && !p.flags.left_game && is_autopilot_alias(&p.alias)
            })
            .map(|t| (t.borrow_player().player_id, Arc::clone(t)))
            .collect();
        pilots.sort_by_key(|(id, _)| id.0);
        pilots.truncate(MAX_AUTOPILOT);
        sync(&mut self.autopilot, pilots);

        // Respawn dead NN ships (engine bots get NN names; autopilots keep theirs).
        for (agent, name) in self
            .bots
            .iter_mut()
            .map(|a| {
                let n = a.player_id.bot_number().unwrap();
                let name = if self.elite && n + 1 == self.count {
                    "NN Elite".to_owned()
                } else {
                    format!("NN {}", n + 1)
                };
                (a, Some(name))
            })
            .chain(self.autopilot.iter_mut().map(|a| (a, None)))
        {
            if agent.is_alive() {
                agent.ticks_alive += TICKS_PER_DECISION;
                continue;
            }
            agent.ticks_alive = 0;
            agent.active = false;
            let score = agent.tuple.borrow_player().score;
            if let Some(entity_type) = pick_spawn_type(score, agent.ship_group) {
                let alias = name.map(|n| PlayerAlias::new_sanitized(&n));
                let _ = Command::Spawn(Spawn { alias, entity_type })
                    .as_command()
                    .apply(world, &agent.tuple, players, teams, None, None);
            }
        }

        // Fixed-size batch: engine bots by bot number, then autopilot slots; empty rows are zeros.
        let row_len = OBS_DIM + 1;
        let mut obs = vec![0f32; rows * row_len];
        for agent in &self.bots {
            let n = agent.player_id.bot_number().unwrap();
            let row = &mut obs[n * row_len..(n + 1) * row_len];
            observe(world, agent, &mut row[..OBS_DIM]);
            row[OBS_DIM] = self.is_elite_bot(n) as u8 as f32;
        }
        for (i, agent) in self.autopilot.iter().enumerate() {
            let n = self.count + i;
            let row = &mut obs[n * row_len..(n + 1) * row_len];
            observe(world, agent, &mut row[..OBS_DIM]);
            let elite = self.elite && is_elite_alias(&agent.tuple.borrow_player().alias);
            row[OBS_DIM] = elite as u8 as f32;
        }
        let bytes: Vec<u8> = obs
            .iter()
            .flat_map(|v| (if v.is_finite() { *v } else { 0.0 }).to_le_bytes())
            .collect();
        if let Err(e) = self.to_policy.write_all(&bytes).and_then(|_| self.to_policy.flush()) {
            error!("NN bots stopped: write to policy failed: {e}");
            return false;
        }
        let mut action_bytes = vec![0u8; rows * ACT_DIM * 4];
        if let Err(e) = self.from_policy.read_exact(&mut action_bytes) {
            error!("NN bots stopped: read from policy failed: {e}");
            return false;
        }

        let count = self.count;
        let slots = self
            .bots
            .iter_mut()
            .map(|a| (a.player_id.bot_number().unwrap(), a))
            .chain(self.autopilot.iter_mut().enumerate().map(|(i, a)| (count + i, a)));
        for (row, agent) in slots {
            let mut action = [0f32; ACT_DIM];
            for (a, b) in action
                .iter_mut()
                .zip(action_bytes[row * ACT_DIM * 4..(row + 1) * ACT_DIM * 4].chunks_exact(4))
            {
                let v = f32::from_le_bytes(b.try_into().unwrap());
                *a = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
            }
            if let Some((control, upgrade, salvo)) = agent_commands(world, agent, &action) {
                apply_commands(world, agent, control, upgrade, salvo, players, teams);
            }
        }
        true
    }
}

/// Keeps `agents` in step with the players that should be driven.
fn sync(agents: &mut Vec<Agent>, wanted: Vec<(PlayerId, Arc<PlayerTuple>)>) {
    agents.retain(|a| wanted.iter().any(|(id, t)| *id == a.player_id && Arc::ptr_eq(t, &a.tuple)));
    for (id, tuple) in wanted {
        if !agents.iter().any(|a| a.player_id == id) {
            agents.push(Agent::new(tuple, id));
        }
    }
}

impl Drop for NnBots {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
