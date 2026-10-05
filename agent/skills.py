"""Skill audit: does the network use every weapon and ship feature when it should?

Used by `evaluate.py --skills`. Everything is measured from what the network sees (its
observation) and what it chose (its action), per decision:

- For each weapon class, *situations* where a ready weapon of that class can hit a fitting target
  (SAM: an incoming missile or aircraft; depth charges: an enemy submarine; decoys: a torpedo or
  missile closing in; torpedoes, guns, missiles and aircraft: an enemy ship), and how often it
  answered that situation by firing that class before it passed ("response rate").
- Per ship family: time spent submerged (submarines), submerged when threatened, active sensors
  on, health lost per minute.
- Target choice: when it fires at a ship, how often that ship is a higher level (worth more).
"""

from __future__ import annotations

import numpy as np

from mk48env import FIRED, HEALTH_LOST

SELF, K, C = 40, 24, 39  # own-ship features, contacts, features per contact (train.rs `observe`)
CLASSES = ["torpedo", "gun", "missile/rocket", "aircraft", "depth charge/mine", "SAM", "decoy"]
FAMILIES = ["submarine", "surface", "carrier", "other"]
NEAR = 0.35  # share of sensor range that counts as close


class SkillAudit:
    def __init__(self, n: int):
        self.opp_active = np.zeros((n, len(CLASSES)), bool)
        self.opp_answered = np.zeros((n, len(CLASSES)), bool)
        self.situations = np.zeros(len(CLASSES))
        self.answered = np.zeros(len(CLASSES))
        self.fires = np.zeros((len(FAMILIES), len(CLASSES)))
        self.steps = np.zeros(len(FAMILIES))
        self.active = np.zeros(len(FAMILIES))
        self.health_lost = np.zeros(len(FAMILIES))
        self.sub_submerged = self.sub_threat = self.sub_threat_submerged = 0.0
        self.sub_submerged_active = 0.0
        self.boat_shots = self.shots_at_bigger = 0.0

    def update(self, obs: np.ndarray, actions: np.ndarray, info: np.ndarray) -> None:
        """obs: what the agents saw when choosing `actions`; info: the outcome of that step."""
        s = obs[:, :SELF]
        c = obs[:, SELF : SELF + K * C].reshape(len(obs), K, C)
        alive = s[:, 0] > 0.5
        family = np.where(s[:, 32:36].max(1) > 0.5, s[:, 32:36].argmax(1), 1)

        enemy = (c[..., 0] > 0.5) & (c[..., 16] < 0.5)
        boat, aircraft, weapon = c[..., 9] > 0.5, c[..., 8] > 0.5, c[..., 14] > 0.5
        missile, torpedo = weapon & (c[..., 21] > 0.5), weapon & (c[..., 19] > 0.5)
        closing = (c[..., 1] * c[..., 3] + c[..., 2] * c[..., 4]) < 0
        near = c[..., 7] < NEAR
        can = c[..., 31:38] > 0.5

        situation = np.zeros((len(obs), len(CLASSES)), bool)
        for k in range(4):  # torpedo, gun, missile, aircraft against ships
            situation[:, k] = (enemy & boat & can[..., k]).any(1)
        situation[:, 4] = (enemy & boat & (c[..., 27] > 0.5) & can[..., 4]).any(1)
        situation[:, 5] = (enemy & (aircraft | missile) & can[..., 5]).any(1)
        situation[:, 6] = (enemy & (torpedo | missile) & closing & near).any(1) & (s[:, 19] > 0.5)
        situation &= alive[:, None]

        fired = info[:, FIRED] > 0
        fire_class = np.clip(((actions[:, 5] + 1) * 0.5 * len(CLASSES)).astype(int), 0, len(CLASSES) - 1)
        shot = np.zeros_like(situation)
        shot[np.arange(len(obs)), fire_class] = fired

        # Situations as episodes: answered if that class fired before the situation passed.
        start = situation & ~self.opp_active
        self.situations += start.sum(0)
        self.opp_answered[start] = False
        self.opp_active |= start
        newly = shot & self.opp_active & ~self.opp_answered
        self.answered += newly.sum(0)
        self.opp_answered |= newly
        over = self.opp_active & ~situation
        self.opp_active[over] = False

        for f in range(len(FAMILIES)):
            rows = alive & (family == f)
            self.steps[f] += rows.sum()
            self.active[f] += (rows & (actions[:, 7] > 0)).sum()
            self.health_lost[f] += info[rows, HEALTH_LOST].sum()
            self.fires[f] += shot[rows].sum(0)

        sub = alive & (s[:, 11] > 0.5)
        submerged = s[:, 10] > 0.5
        threat = (enemy & ((boat & near) | ((torpedo | missile) & closing & near))).any(1)
        self.sub_submerged += (sub & submerged).sum()
        self.sub_threat += (sub & threat).sum()
        self.sub_threat_submerged += (sub & threat & submerged).sum()
        self.sub_submerged_active += (sub & submerged & (actions[:, 7] > 0)).sum()

        # The target is the contact the aim point was set to (to_env aims at its position).
        dist = np.hypot(c[..., 1] - actions[:, None, 2], c[..., 2] - actions[:, None, 3])
        target = np.where(enemy & boat, dist, np.inf).argmin(1)
        hit_boat = fired & (np.take_along_axis(dist, target[:, None], 1)[:, 0] < 0.02)
        bigger = c[np.arange(len(obs)), target, 17] > s[:, 8] + 1e-6
        self.boat_shots += hit_boat.sum()
        self.shots_at_bigger += (hit_boat & bigger).sum()

    def report(self, steps_per_minute: float) -> dict:
        sub_steps = self.steps[0]
        return {
            "response_rate": {CLASSES[k]: float(self.answered[k] / self.situations[k])
                              for k in range(len(CLASSES)) if self.situations[k]},
            "situations": {CLASSES[k]: int(self.situations[k]) for k in range(len(CLASSES))},
            "fire_mix_by_family": {
                FAMILIES[f]: {CLASSES[k]: float(self.fires[f, k] / max(self.fires[f].sum(), 1))
                              for k in range(len(CLASSES)) if self.fires[f, k]}
                for f in range(len(FAMILIES)) if self.fires[f].sum()},
            "active_sensors_share": {FAMILIES[f]: float(self.active[f] / self.steps[f])
                                     for f in range(len(FAMILIES)) if self.steps[f]},
            "health_lost_per_min": {FAMILIES[f]: float(self.health_lost[f] / self.steps[f] * steps_per_minute)
                                    for f in range(len(FAMILIES)) if self.steps[f]},
            "sub_submerged_share": float(self.sub_submerged / sub_steps) if sub_steps else None,
            "sub_submerged_under_threat": float(self.sub_threat_submerged / self.sub_threat) if self.sub_threat else None,
            "sub_active_while_submerged": float(self.sub_submerged_active / self.sub_submerged) if self.sub_submerged else None,
            "shots_at_bigger_ships": float(self.shots_at_bigger / self.boat_shots) if self.boat_shots else None,
        }


def print_report(r: dict) -> None:
    print("  skills (response = fired that weapon before the situation passed):")
    for k, rate in r["response_rate"].items():
        print(f"    {k:18s} {rate:5.0%} of {r['situations'][k]} situations")
    for family, mix in r["fire_mix_by_family"].items():
        print(f"    fires as {family:9s}: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))
    print("    active sensors on: " + ", ".join(f"{k} {v:.0%}" for k, v in r["active_sensors_share"].items()))
    print("    health lost/min:   " + ", ".join(f"{k} {v:.2f}" for k, v in r["health_lost_per_min"].items()))
    if r["sub_submerged_share"] is not None:
        print(f"    submarines: submerged {r['sub_submerged_share']:.0%} of the time, "
              f"{r['sub_submerged_under_threat'] or 0:.0%} when threatened, "
              f"active sensors while submerged {r['sub_active_while_submerged'] or 0:.0%}")
    if r["shots_at_bigger_ships"] is not None:
        print(f"    shots at ships of a higher level: {r['shots_at_bigger_ships']:.0%}")
