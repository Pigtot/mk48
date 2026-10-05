"""Entity-transformer policy with discrete action heads and a target pointer.

Observation layout follows `server/src/train.rs` (see `Layout`): own ship, 24 contacts, 15x15
terrain. Each contact is a token; the policy picks a target by pointing at a contact (or "none")
instead of regressing aim coordinates, and every other decision is a small categorical head.
Discrete actions are converted to the server's 9 continuous action values by `to_env`.

Layout V6 adds 7 weapon types and exact ship-type ids (embedded), so weapon strategy can differ
per vehicle. V7 adds a "salvo" head: fire every other ready weapon that can hit the target too.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

K, G = 24, 15
FRIENDLY = 16  # contact feature: belongs to us
SNAP = 0.15  # server snaps aim to an enemy within this fraction of sensor range


@dataclass(frozen=True)
class Layout:
    self_dim: int
    contact_dim: int
    weapon_classes: int
    own_type: int | None = None  # self feature holding our exact ship type id
    contact_type: int | None = None  # contact feature holding type id + 1 (0 = unidentified)
    n_types: int = 0
    salvo: bool = False  # extra action: fire all weapons that can engage the target

    @property
    def obs_dim(self) -> int:
        return self.self_dim + K * self.contact_dim + G * G

    @property
    def class_values(self) -> np.ndarray:
        return -1 + (2 * np.arange(self.weapon_classes) + 1) / self.weapon_classes

    @property
    def act_dim(self) -> int:
        return 10 if self.salvo else 9

    def heads(self) -> dict[str, int]:
        heads = {"steer": 16, "throttle": 5, "target": K + 1, "fire": 2, "weapon": self.weapon_classes,
                 "submerge": 2, "active": 2, "family": 4}
        if self.salvo:
            heads["salvo"] = 2
        return heads

    @property
    def head_names(self) -> list[str]:
        return list(self.heads())


V5 = Layout(32, 30, 3)  # underwater / surface+air / defensive
V6 = Layout(40, 39, 7, own_type=36, contact_type=38, n_types=138)
V7 = Layout(40, 39, 7, own_type=36, contact_type=38, n_types=138, salvo=True)
WEAPON_NAMES_V6 = ["torpedo", "gun", "missile/rocket", "aircraft", "depth charge/mine", "SAM", "decoy"]

STEER_BINS = np.arange(-8, 8) / 8.0  # relative heading / pi; index 8 = straight ahead
THROTTLE_BINS = np.array([-0.5, 0.0, 0.33, 0.66, 1.0])
FAMILY_VALUES = np.array([-0.75, -0.25, 0.25, 0.75])  # submarine, gun/missile, carrier, special
NO_TARGET_AIM = (0.5, 0.0)  # straight ahead, half sensor range

# Head order is the same in every layout; newer heads are appended, so older layouts are a prefix.
HEAD_NAMES = list(V7.heads())
H = {name: i for i, name in enumerate(HEAD_NAMES)}


def _names(logits: dict[str, torch.Tensor]) -> list[str]:
    return [n for n in HEAD_NAMES if n in logits]


def split_obs(obs: torch.Tensor, layout: Layout):
    sd, cd = layout.self_dim, layout.contact_dim
    s = obs[:, :sd]
    c = obs[:, sd : sd + K * cd].reshape(-1, K, cd)
    t = obs[:, sd + K * cd :].reshape(-1, 1, G, G)
    return s, c, t


def _contacts(obs: np.ndarray, layout: Layout) -> np.ndarray:
    return obs[:, layout.self_dim : layout.self_dim + K * layout.contact_dim].reshape(-1, K, layout.contact_dim)


class Encoder(nn.Module):
    def __init__(self, layout: Layout, d: int = 128, layers: int = 2, heads: int = 4, out: int = 256):
        super().__init__()
        self.layout = layout
        self.self_embed = nn.Sequential(nn.Linear(layout.self_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.contact_embed = nn.Sequential(nn.Linear(layout.contact_dim, d), nn.ReLU(), nn.Linear(d, d))
        # One table for own and contact ship types (index = type id + 1, 0 = unknown).
        self.type_embed = nn.Embedding(layout.n_types + 1, d) if layout.n_types else None
        self.terrain = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 8 * 8, d),
        )
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(3 * d, out), nn.ReLU())

    def forward(self, obs: torch.Tensor):
        layout = self.layout
        s, c, t = split_obs(obs, layout)
        present = c[:, :, 0] > 0.5
        if self.type_embed is not None:
            own = s[:, layout.own_type].long().clamp(0, layout.n_types - 1) + 1
            other = c[:, :, layout.contact_type].long().clamp(0, layout.n_types)
            s, c = s.clone(), c.clone()
            s[:, layout.own_type] = 0.0
            c[:, :, layout.contact_type] = 0.0
            self_token = self.self_embed(s) + self.type_embed(own)
            contact_tokens = self.contact_embed(c) + self.type_embed(other)
        else:
            self_token, contact_tokens = self.self_embed(s), self.contact_embed(c)
        tokens = torch.cat([self_token[:, None], self.terrain(t)[:, None], contact_tokens], 1)
        padding = torch.cat([torch.zeros_like(present[:, :2]), ~present], 1)
        x = self.transformer(tokens, src_key_padding_mask=padding)
        contacts = x[:, 2:]
        weight = present.float()[..., None]
        pooled = (contacts * weight).sum(1) / weight.sum(1).clamp(min=1.0)
        return self.out(torch.cat([x[:, 0], x[:, 1], pooled], -1)), contacts, present


class Policy(nn.Module):
    def __init__(self, layout: Layout = V6, d: int = 128, hidden: int = 256):
        super().__init__()
        self.layout = layout
        self.encoder = Encoder(layout, d, out=hidden)
        self.heads = nn.ModuleDict({n: nn.Linear(hidden, k) for n, k in layout.heads().items() if n != "target"})
        self.query = nn.Linear(hidden, d)
        self.key = nn.Linear(d, d)
        self.no_target = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h, contacts, present = self.encoder(obs)
        logits = {n: head(h) for n, head in self.heads.items()}
        pointer = (self.query(h)[:, None] * self.key(contacts)).sum(-1) / math.sqrt(contacts.shape[-1])
        # -1e4 rather than -inf/-1e9: stays finite in fp16, so KL terms never see inf - inf.
        logits["target"] = torch.cat([self.no_target(h), pointer.masked_fill(~present, -1e4)], 1)
        return logits


class Value(nn.Module):
    def __init__(self, layout: Layout = V6, d: int = 128, hidden: int = 256):
        super().__init__()
        self.encoder = Encoder(layout, d, out=hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs)[0]).squeeze(-1)


# --- distributions -------------------------------------------------------------------------------

def _dists(logits: dict[str, torch.Tensor]) -> list[Categorical]:
    return [Categorical(logits=logits[n]) for n in _names(logits)]


def _weapon_mask(logits: dict[str, torch.Tensor], actions: torch.Tensor) -> list[torch.Tensor | float]:
    """Weapon class and salvo only matter when firing."""
    firing = (actions[:, H["fire"]] == 1).float()
    return [firing if n in ("weapon", "salvo") else 1.0 for n in _names(logits)]


def sample(logits: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
    if deterministic:
        return torch.stack([logits[n].argmax(-1) for n in _names(logits)], 1)
    return torch.stack([d.sample() for d in _dists(logits)], 1)


def log_prob(logits: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
    masks = _weapon_mask(logits, actions)
    return sum(d.log_prob(actions[:, i]) * m for i, (d, m) in enumerate(zip(_dists(logits), masks)))


def entropy(logits: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(d.entropy() for d in _dists(logits))


def kl(ref_logits: dict[str, torch.Tensor], logits: dict[str, torch.Tensor]) -> torch.Tensor:
    """KL(reference || current), summed over heads."""
    total = 0.0
    for n in _names(logits):
        if n not in ref_logits:
            continue
        p = torch.log_softmax(ref_logits[n], -1)
        q = torch.log_softmax(logits[n], -1)
        total = total + (p.exp() * (p - q)).sum(-1)
    return total


def defense_labels(obs: torch.Tensor, layout: Layout = V7, near: float = 0.35):
    """Defensive moves the elite never learned (it fired SAMs in 5% of the situations they exist
    for and decoys in none), as labels for an auxiliary imitation loss: launch a decoy against the
    nearest torpedo or missile closing in within `near` of sensor range, else fire a SAM at the
    nearest incoming missile or aircraft that a ready SAM can hit.

    Returns (rows with a lesson, target index for the pointer head (contact + 1), weapon class).
    Feature indices follow `observe` in server/src/train.rs."""
    s = obs[:, : layout.self_dim]
    c = obs[:, layout.self_dim : layout.self_dim + K * layout.contact_dim].reshape(len(obs), K, layout.contact_dim)
    enemy = (c[..., 0] > 0.5) & (c[..., FRIENDLY] < 0.5)
    aircraft, weapon = c[..., 8] > 0.5, c[..., 14] > 0.5
    torpedo, missile = weapon & (c[..., 19] > 0.5), weapon & (c[..., 21] > 0.5)
    closing = (c[..., 1] * c[..., 3] + c[..., 2] * c[..., 4]) < 0
    decoy = enemy & (torpedo | missile) & closing & (c[..., 7] < near) & (s[:, 19:20] > 0.5)
    sam = enemy & (aircraft | missile) & (c[..., 36] > 0.5)
    inf = torch.full_like(c[..., 7], float("inf"))
    use_decoy = decoy.any(1)
    nearest = torch.where(use_decoy[:, None], torch.where(decoy, c[..., 7], inf), torch.where(sam, c[..., 7], inf))
    rows = use_decoy | sam.any(1)
    weapon_class = torch.where(use_decoy, 6, 5)
    return rows, nearest.argmin(1) + 1, weapon_class


def doctrine_labels(obs: torch.Tensor, layout: Layout = V7):
    """Two more skills the elite lost or never had, for the same auxiliary loss: a carrier launches
    aircraft at the nearest enemy ship that a ready aircraft can reach (v2b answered 17% of those
    situations, the 15M elite 64%), and a submerged submarine keeps its active sonar off (sonar
    pings reveal it; the elite left it on ~90% of the time).

    Returns (aircraft rows, target index, stealth rows)."""
    s = obs[:, : layout.self_dim]
    c = obs[:, layout.self_dim : layout.self_dim + K * layout.contact_dim].reshape(len(obs), K, layout.contact_dim)
    enemy_ship = (c[..., 0] > 0.5) & (c[..., FRIENDLY] < 0.5) & (c[..., 9] > 0.5)
    reach = enemy_ship & (c[..., 34] > 0.5)
    carrier = s[:, 34] > 0.5  # own ship family: carrier
    air_rows = carrier & reach.any(1)
    nearest = torch.where(reach, c[..., 7], torch.full_like(c[..., 7], float("inf"))).argmin(1) + 1
    stealth_rows = (s[:, 11] > 0.5) & (s[:, 10] > 0.5)  # a submarine, submerged
    return air_rows, nearest, stealth_rows


# --- conversion to / from the server's continuous action vector ----------------------------------

def to_env(actions: np.ndarray, obs: np.ndarray, layout: Layout = V6) -> np.ndarray:
    """Discrete actions (B, heads) -> server actions (B, 9)."""
    a = actions
    b = np.arange(len(a))
    contacts = _contacts(obs, layout)
    target = a[:, H["target"]]
    aim = np.tile(np.array(NO_TARGET_AIM, np.float32), (len(a), 1))
    has = target > 0
    aim[has] = contacts[b[has], target[has] - 1, 1:3]
    out = np.empty((len(a), layout.act_dim), np.float32)
    out[:, 0] = STEER_BINS[a[:, H["steer"]]]
    out[:, 1] = THROTTLE_BINS[a[:, H["throttle"]]]
    out[:, 2:4] = aim
    out[:, 4] = np.where(a[:, H["fire"]] == 1, 1.0, -1.0)
    out[:, 5] = layout.class_values[a[:, H["weapon"]]]
    out[:, 6] = np.where(a[:, H["submerge"]] == 1, 1.0, -1.0)
    out[:, 7] = np.where(a[:, H["active"]] == 1, 1.0, -1.0)
    out[:, 8] = FAMILY_VALUES[a[:, H["family"]]]
    if layout.salvo:
        out[:, 9] = np.where(a[:, H["salvo"]] == 1, 1.0, -1.0)
    return out


def from_expert(expert: np.ndarray, obs: np.ndarray, layout: Layout = V6) -> tuple[np.ndarray, np.ndarray]:
    """Expert labels (B, act_dim + valid + aim_valid) -> discrete actions (B, heads) and loss masks.

    The expert never fires salvos, so the salvo head is left to reinforcement learning (mask 0)."""
    e = expert
    n = len(e)
    aim_valid = e[:, -1]
    labels = np.zeros((n, len(layout.head_names)), np.int64)
    labels[:, H["steer"]] = (np.round(e[:, 0] * 8).astype(int) + 8) % 16
    labels[:, H["throttle"]] = np.abs(e[:, 1:2] - THROTTLE_BINS).argmin(1)
    contacts = _contacts(obs, layout)
    enemy = (contacts[:, :, 0] > 0.5) & (contacts[:, :, FRIENDLY] < 0.5)
    dist = np.linalg.norm(contacts[:, :, 1:3] - e[:, None, 2:4], axis=-1)
    dist[~enemy] = np.inf
    nearest = dist.argmin(1)
    hit = (aim_valid > 0) & (dist[np.arange(n), nearest] < SNAP)
    labels[:, H["target"]] = np.where(hit, nearest + 1, 0)
    labels[:, H["fire"]] = e[:, 4] > 0
    labels[:, H["weapon"]] = np.abs(e[:, 5:6] - layout.class_values).argmin(1)
    labels[:, H["submerge"]] = e[:, 6] > 0
    labels[:, H["active"]] = e[:, 7] > 0
    labels[:, H["family"]] = np.abs(e[:, 8:9] - FAMILY_VALUES).argmin(1)
    masks = np.ones((n, len(layout.head_names)), np.float32)
    masks[:, H["weapon"]] = labels[:, H["fire"]]
    if layout.salvo:
        masks[:, H["salvo"]] = 0.0
    return labels, masks


# --- persistence ---------------------------------------------------------------------------------

def save(path: Path | str, policy: Policy, value: Value | None = None, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {"policy": policy.state_dict(), "layout": asdict(policy.layout), **extra}
    if value is not None:
        state["value"] = value.state_dict()
    torch.save(state, path)


def upgrade(policy: Policy, layout: Layout, salvo_bias: float = 1.0) -> Policy:
    """Copies a policy into a newer layout (same observation); new heads start fresh, with the
    salvo head initially preferring single shots (`salvo_bias`)."""
    new = Policy(layout).to(next(policy.parameters()).device)
    missing, unexpected = new.load_state_dict(policy.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("heads.salvo") for k in missing), (missing, unexpected)
    if "salvo" in new.heads:
        with torch.no_grad():
            new.heads["salvo"].weight.mul_(0.01)
            new.heads["salvo"].bias.copy_(torch.tensor([salvo_bias, -salvo_bias]))
    return new


class PhasedPolicy(nn.Module):
    """An opening policy while the ship is small, then the main policy (like an opening book).

    Loaded by `load` from a JSON recipe: {"opening": ..., "main": ..., "until_level": L}, with
    paths relative to the recipe. Rows whose own ship is at most level L use the opening policy."""

    MAX_LEVEL = 10  # observation feature 8 is level / EntityData::MAX_BOAT_LEVEL

    def __init__(self, opening: Policy, main: Policy, until_level: int):
        super().__init__()
        self.opening, self.main, self.until_level = opening, main, until_level
        self.layout = main.layout

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        early = (obs[:, 8] * self.MAX_LEVEL).round() <= self.until_level
        if early.all():
            return self.opening(obs)
        if not early.any():
            return self.main(obs)
        # Each network only computes its own rows.
        a, b = self.opening(obs[early]), self.main(obs[~early])
        out = {}
        for k in b:
            merged = b[k].new_empty((len(obs),) + b[k].shape[1:])
            merged[early], merged[~early] = a[k], b[k]
            out[k] = merged
        return out


def load(path: Path | str, device: str = "cpu") -> tuple[Policy, Value | None, dict]:
    if str(path).endswith(".json"):  # a PhasedPolicy recipe
        recipe = json.loads(Path(path).read_text())
        here = Path(path).parent
        opening = load(here / recipe["opening"], device)[0]
        main = load(here / recipe["main"], device)[0]
        return PhasedPolicy(opening, main, int(recipe["until_level"])).to(device), None, recipe
    state = torch.load(path, map_location=device, weights_only=False)
    layout = Layout(**state["layout"]) if "layout" in state else V5  # V5 checkpoints predate the field
    policy = Policy(layout).to(device)
    policy.load_state_dict(state["policy"])
    value = None
    if "value" in state:
        value = Value(layout).to(device)
        value.load_state_dict(state["value"])
    return policy, value, state
