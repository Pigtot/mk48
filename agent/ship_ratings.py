"""Ship ratings for NN ship personas: the score/min a policy gets in each ship.

    python ship_ratings.py runs/ppo_elite/policy.pt             # -> ship_ratings.tsv
    python ship_ratings.py runs/ppo_elite2/policy.pt --pressure "elite v2" --out ship_ratings_v2.tsv

Reads the policy's evaluations with ship personas (`evaluate.py --ship-style --save ...`) from
runs/evals.json, or with --pressure the `skills` scenario of a pressure-test label (personas
without ratings, so every ship gets played), and writes `name<TAB>score_per_min<TAB>level<TAB>minutes`
per ship. Ships with little play time are pulled toward their level's average, and ships never
played get 80% of it, so one lucky minute doesn't make a ship a favourite. The game server reads the file through
`MK48_NN_SHIP_RATINGS` (set by run_game.sh) and personas then choose only among ships rated close
to the best option at each level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import TYPES_FILE

PRIOR_MINUTES = 10.0  # play time at which a ship's own result and its level average count equally
UNPLAYED = 0.8  # share of the level average given to ships never played


def boat_levels() -> dict[str, int]:
    rows = (line.split("\t") for line in TYPES_FILE.read_text().splitlines())
    return {r[1]: int(r[4]) for r in rows if r[2] == "Boat"}


def ratings(evals: list[dict], policy: str) -> dict[str, tuple[float, int, float]]:
    """name -> (rating, level, minutes played)."""
    levels = boat_levels()
    minutes: dict[str, float] = {}
    points: dict[str, float] = {}
    for e in evals:
        if e.get("policy") != policy or not e.get("ship_style") or "by_type" not in e:
            continue
        total = e["agents"] * e["minutes"]
        for name, t in e["by_type"].items():
            m = t["share"] * total
            minutes[name] = minutes.get(name, 0.0) + m
            points[name] = points.get(name, 0.0) + m * t["score_per_min"]
    if not minutes:
        raise SystemExit(f"no evaluations with --ship-style for {policy} in runs/evals.json")

    by_level: dict[int, list[str]] = {}
    for name in minutes:
        if name in levels:
            by_level.setdefault(levels[name], []).append(name)
    level_mean = {
        level: sum(points[n] for n in names) / sum(minutes[n] for n in names)
        for level, names in by_level.items()
    }
    out = {}
    for name, level in levels.items():
        prior = level_mean.get(level)
        if prior is None:
            continue
        m = minutes.get(name, 0.0)
        if m == 0.0:
            out[name] = (UNPLAYED * prior, level, 0.0)
        else:
            out[name] = ((points[name] + PRIOR_MINUTES * prior) / (m + PRIOR_MINUTES), level, m)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="policy path as saved in runs/evals.json")
    p.add_argument("--evals", type=Path, default=Path("runs/evals.json"))
    p.add_argument("--pressure", metavar="LABEL", help="use this pressure-test label's skills scenario")
    p.add_argument("--out", type=Path, default=Path("ship_ratings.tsv"))
    a = p.parse_args()
    if a.pressure:
        from pressure_test import label_dir

        d = json.loads((label_dir(a.pressure) / "skills.json").read_text())
        evals = [{"policy": d["policy"], "ship_style": True, "agents": d["agents"], "minutes": d["minutes"],
                  "by_type": d["result"]["by_type"]}]
    else:
        evals = json.loads(a.evals.read_text())
    table = ratings(evals, a.policy)
    lines = [f"# score/min with {a.policy} (evaluate.py --ship-style); name, rating, level, minutes played"]
    for name, (rating, level, minutes) in sorted(table.items(), key=lambda kv: (kv[1][1], -kv[1][0])):
        lines.append(f"{name}\t{rating:.1f}\t{level}\t{minutes:.0f}")
    a.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(table)} ship ratings to {a.out}")


if __name__ == "__main__":
    main()
