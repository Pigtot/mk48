"""Pressure tests: a policy against hard scenarios, each with a pass/fail target.

    python pressure_test.py runs/ppo_elite2/policy.pt --ratings ship_ratings.tsv --label "elite v2"

Scenarios (fresh worlds, NN ships play as in the game: player rules, lead aim, ship personas):

  top10     the elite among 11 imitation NN bots and 40 built-in bots (the game's setup):
            average scoreboard position in the top 10%
  crowd     12 copies of the policy and 40 built-in bots: copies of itself are the toughest
            opponents; average position in the top 20% (12 ships can't all be in the top 10%)
  hostile   built-in bots at double aggression: dodging and evading, few deaths
  rich      starts with 1,500 points (a level-9 ship) among fresh bots: protecting a lead
  skills    every ship family (personas without ratings): submarines dive, defensive weapons
            (SAMs, decoys, depth charges) answer the threats they exist for
  opening   the top10 setup for its first 20 minutes, in 16 worlds: the climb from zero

Each scenario's result goes to runs/pressure/<label>/<scenario>.json as soon as it finishes, so
scenarios can run as parallel processes (--only); --report prints the scorecards of saved labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluate import evaluate
from mk48env import SERVER_BUILDS

# The imitation policy that drives the other NN bots in the game (working tree or published repo).
IMITATION = next((p for p in ("runs/bc_v6/policy.pt", "models/bc_v6.pt") if Path(p).exists()), "models/bc_v6.pt")


def scenarios(ratings: Path | None) -> dict[str, dict]:
    game = dict(agent_kind="nn-bot", ship_style=True, ship_ratings=ratings)
    return {
        "top10": dict(game, agents=1, opponents=11, opponent_policy=IMITATION, bots=40, procs=8, minutes=60),
        "crowd": dict(game, agents=12, bots=40, procs=4, minutes=30),
        "hostile": dict(game, agents=2, bots=40, procs=8, minutes=30, bot_aggression=2.0),
        "rich": dict(game, agents=2, bots=32, procs=8, minutes=30, start_score=1500),
        "skills": dict(agent_kind="nn-bot", ship_style=True, agents=4, bots=32, procs=8, minutes=30, skills=True),
        "opening": dict(game, agents=1, opponents=11, opponent_policy=IMITATION, bots=40, procs=16, minutes=20),
    }


def checks(name: str, r: dict) -> list[tuple[str, float | None, str, bool | None]]:
    """(metric, value, target, passed) for a scenario's result; passed is None if there was too
    little to judge."""
    rank, deaths = r.get("board_rank", (None,))[0], r["deaths"][0]
    if name == "top10":
        return [("board rank", rank, ">= 90%", rank is not None and rank >= 0.90)]
    if name == "opening":
        blocks = r.get("board_rank_by_10min", []) + [None, None]
        first, second = blocks[:2]
        return [("board rank, first 10 minutes", first, ">= 75%", None if first is None else first >= 0.75),
                ("board rank, minutes 10-20", second, ">= 90%", None if second is None else second >= 0.90)]
    if name == "crowd":
        return [("board rank", rank, ">= 80%", rank is not None and rank >= 0.80)]
    if name == "hostile":
        return [("deaths/min", deaths, "<= 0.10", deaths <= 0.10),
                ("board rank", rank, ">= 85%", rank is not None and rank >= 0.85)]
    if name == "rich":
        return [("board rank", rank, ">= 95%", rank is not None and rank >= 0.95),
                ("deaths/min", deaths, "<= 0.05", deaths <= 0.05)]
    s = r["skills"]
    out = [("subs submerged", s["sub_submerged_share"], ">= 50%", (s["sub_submerged_share"] or 0) >= 0.5),
           ("subs submerged when threatened", s["sub_submerged_under_threat"], ">= 50%",
            (s["sub_submerged_under_threat"] or 0) >= 0.5)]
    for weapon, target in (("SAM", 0.3), ("decoy", 0.2), ("depth charge/mine", 0.3)):
        n = s["situations"].get(weapon, 0)
        value = s["response_rate"].get(weapon)
        out.append((f"{weapon} response ({n} situations)", value, f">= {target:.0%}",
                    None if n < 20 else (value is not None and value >= target)))
    return out


RESULTS = Path("runs/pressure")


def label_dir(label: str) -> Path:
    return RESULTS / re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def report(labels: list[str]) -> None:
    for label in labels:
        rows = []
        for f in sorted(label_dir(label).glob("*.json")):
            rows += json.loads(f.read_text())["scorecard"]
        passed = sum(r[-1] == "PASS" for r in rows)
        judged = sum(r[-1] != "SKIP" for r in rows)
        print(f"== {label}: {passed}/{judged} checks passed")
        for name, metric, shown, target, status in rows:
            print(f"{name:8s} {metric:38s} {shown:>7s}  target {target:8s} {status}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy", nargs="?")
    p.add_argument("--label", required=True, nargs="+", help="one label; several with --report")
    p.add_argument("--report", action="store_true", help="print the saved scorecards of the labels")
    p.add_argument("--ratings", type=Path, default=None, help="ship ratings for this policy (ship_ratings.py)")
    p.add_argument("--only", nargs="*", help="run only these scenarios")
    p.add_argument("--device", default="mps")
    p.add_argument("--server", type=Path, default=SERVER_BUILDS["v7"])
    p.add_argument("--scale", type=float, default=1.0, help="multiply every scenario's minutes (quick check: 0.05)")
    a = p.parse_args()
    if a.report:
        report(a.label)
        return
    label = a.label[0]
    out_dir = label_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)

    card = []
    for name, cfg in scenarios(a.ratings).items():
        if a.only and name not in a.only:
            continue
        cfg = dict(cfg)
        r = evaluate(a.policy, cfg.pop("minutes") * a.scale, cfg.pop("agents"), cfg.pop("bots"), cfg.pop("procs"),
                     a.device, server_path=a.server, **cfg)
        measured = (scenarios(a.ratings)[name]["agents"] * scenarios(a.ratings)[name]["procs"],
                    scenarios(a.ratings)[name]["minutes"] * a.scale)
        rows = []
        for metric, value, target, passed in checks(name, r):
            shown = "n/a" if value is None else (f"{value:.0%}" if "%" in target else f"{value:.3f}")
            rows.append((name, metric, shown, target, "SKIP" if passed is None else "PASS" if passed else "FAIL"))
            print(f"{name:8s} {metric:38s} {shown:>7s}  target {target:8s} {rows[-1][-1]}", flush=True)
        card += rows
        (out_dir / f"{name}.json").write_text(json.dumps(
            {"policy": a.policy, "ratings": str(a.ratings) if a.ratings else None, "scale": a.scale,
             "agents": measured[0], "minutes": measured[1], "ship_style": True, "scorecard": rows,
             "result": r}, indent=1))
    passed = sum(row[-1] == "PASS" for row in card)
    judged = sum(row[-1] != "SKIP" for row in card)
    print(f"{label}: {passed}/{judged} checks passed ({len(card) - judged} skipped: too few situations)")


if __name__ == "__main__":
    main()
