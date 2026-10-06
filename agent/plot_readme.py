"""Figures for the top-level README.

    python plot_readme.py

Writes to figures/ (each SVG in a light and a -dark variant, for GitHub's two themes):

  overview.svg            what the network receives and what it controls
  training_pipeline.svg   teacher -> imitation -> PPO -> tougher opponents and new skills
  results.svg             the standard test, from results/evals.json
  gameplay_annotated.png  results/game_ai_elite.png with a few labels (needs Pillow)

Every number comes from a file in this repository: the results from results/evals.json and
results/pressure/, the input size from ../server/src/train.rs, the weight count from the elite
checkpoint (checked when PyTorch is installed).
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "figures"
EVALS = HERE / "results" / "evals.json"
PRESSURE = HERE / "results" / "pressure"
TRAIN_RS = HERE.parent / "server" / "src" / "train.rs"
ELITE = HERE / "models" / "elite_v2b_3M.pt"
SCREENSHOT = HERE / "results" / "game_ai_elite.png"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

# GitHub's own light and dark page colours, so the figures sit on the page without a frame.
THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", line="#d1d9e0", box="#f6f8fa",
                  accent="#2b62a8", accent_box="#eef4fb", bar="#8c959f", bar_2="#aab9cc",
                  grid="#eaeef2", strike="#b5473f"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", line="#3d444d", box="#151b23",
                 accent="#78aef0", accent_box="#14243a", bar="#6e7681", bar_2="#4b6585",
                 grid="#21262d", strike="#e5807a"),
}


# --- numbers -------------------------------------------------------------------------------------

def evals() -> dict[str, dict]:
    return {e["label"]: e for e in json.loads(EVALS.read_text())}


# The standard test (2 test ships + 32 built-in bots per world, 8 worlds, 60 game-minutes).
STANDARD = [  # (label in evals.json, label in the chart)
    ("random actions", "Random actions"),
    ("built-in bot", "Built-in bot"),
    ("imitation, transformer + 7 weapons", "Imitation (step 1)"),
    ("expert: bot via action space", "Built-in bot, NN controls*"),
    ("elite 15M + guard (final)", "Elite 15M (step 2)"),
]


def board_rank(label: str) -> float:
    return json.loads((PRESSURE / label / "top10.json").read_text())["result"]["board_rank"][0]


def obs_dim() -> int:
    """OBS_DIM = SELF_FEATURES + MAX_CONTACTS * CONTACT_FEATURES + TERRAIN_GRID^2 (train.rs)."""
    src = TRAIN_RS.read_text()
    c = {k: int(re.search(rf"const {k}: usize = (\d+);", src).group(1))
         for k in ("SELF_FEATURES", "MAX_CONTACTS", "CONTACT_FEATURES", "TERRAIN_GRID")}
    return c["SELF_FEATURES"] + c["MAX_CONTACTS"] * c["CONTACT_FEATURES"] + c["TERRAIN_GRID"] ** 2


def check_numbers() -> None:
    assert obs_dim() == 1201, obs_dim()
    e = evals()
    for label, _ in STANDARD:
        assert e[label]["agents"] == 16 and e[label]["bots_per_world"] == 32 and e[label]["minutes"] == 60, label
    assert round(board_rank("elite_15M_before_"), 2) == 0.78
    assert round(board_rank("elite_v2b_at_3M_confirm_"), 2) == 0.85
    try:
        import torch
    except ImportError:
        return
    weights = torch.load(ELITE, map_location="cpu", weights_only=False)["policy"]
    n = sum(v.numel() for v in weights.values())
    assert 870_000 < n < 890_000, n


# --- SVG helpers ---------------------------------------------------------------------------------

class Svg:
    def __init__(self, w: int, h: int, theme: dict[str, str], title: str):
        self.w, self.h, self.t = w, h, theme
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" role="img">',
            f"<title>{escape(title)}</title>",
            "<defs>"
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{theme["muted"]}"/></marker>'
            "</defs>",
            f'<rect width="{w}" height="{h}" fill="{theme["bg"]}"/>',
        ]

    def c(self, key: str) -> str:
        return self.t.get(key, key)

    def text(self, x, y, s, size=13, weight=400, fill="fg", anchor="start", **kw) -> None:
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items())
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{self.c(fill)}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def rect(self, x, y, w, h, fill="box", stroke="line", r=6, width=1) -> None:
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{self.c(fill)}" '
                          f'stroke="{self.c(stroke)}" stroke-width="{width}"/>')

    def line(self, x1, y1, x2, y2, stroke="line", width=1, dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.c(stroke)}" '
                          f'stroke-width="{width}"{d}/>')

    def arrow(self, points: list[tuple[float, float]]) -> None:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.c("muted")}" stroke-width="1.5" '
                          f'marker-end="url(#arrow)"/>')

    def raw(self, s: str) -> None:
        self.parts.append(s)

    def save(self, name: str) -> None:
        (OUT / name).write_text("\n".join(self.parts + ["</svg>"]) + "\n")


def save_both(name: str, draw) -> None:
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        svg = draw(THEMES[theme])
        svg.save(f"{name}{suffix}.svg")


# --- overview.svg --------------------------------------------------------------------------------

def overview(t: dict[str, str]) -> Svg:
    W, H = 760, 548
    s = Svg(W, H, t, "Game state (1,201 numbers) goes into a transformer, which chooses the ship's controls")
    cx = W / 2

    # The game server.
    s.rect(cx - 150, 14, 300, 54)
    s.text(cx, 37, "Mk48 game server", 15, 600, anchor="middle")
    s.text(cx, 56, "local copy · real rules and physics", 12.5, fill="muted", anchor="middle")

    # State goes down; screenshots don't.
    s.arrow([(cx, 68), (cx, 110)])
    s.text(cx - 12, 88, "game state:", 12.5, fill="muted", anchor="end")
    s.text(cx - 12, 103, "what a player's client is sent", 12.5, fill="muted", anchor="end")
    camera_x, camera_y = cx + 16, 84
    s.raw(f'<g fill="none" stroke="{t["muted"]}" stroke-width="1.4">'
          f'<rect x="{camera_x}" y="{camera_y}" width="20" height="14" rx="2"/>'
          f'<circle cx="{camera_x + 10}" cy="{camera_y + 7}" r="4"/>'
          f'<path d="M{camera_x + 6},{camera_y} l2,-3 h4 l2,3"/></g>'
          f'<line x1="{camera_x - 2}" y1="{camera_y + 17}" x2="{camera_x + 22}" y2="{camera_y - 5}" '
          f'stroke="{t["strike"]}" stroke-width="1.8"/>')
    s.text(camera_x + 30, 96, "no screenshots, no computer vision", 12.5, fill="muted")

    # What the network receives.
    x0, y0, bw, bh = 40, 112, 680, 182
    s.rect(x0, y0, bw, bh)
    s.text(x0 + 18, y0 + 25, "What the network receives at each decision", 14, 600)
    cols = [
        ("Own ship", "40 values", ["health, speed, heading", "position, map edge", "weapons ready", "ship type, level"]),
        ("24 nearest objects", "24 × 39 values", ["enemy ships, torpedoes", "missiles, aircraft", "barrels, coins, platforms",
                                                 "can a weapon hit it now?"]),
        ("Local map", "15 × 15 grid", ["land and map edge", "around the ship,", "turned to its heading"]),
    ]
    cw = bw / 3
    for i, (head, count, items) in enumerate(cols):
        x = x0 + i * cw + 18
        if i:
            s.line(x0 + i * cw, y0 + 42, x0 + i * cw, y0 + 140)
        s.text(x, y0 + 54, head, 13.5, 600)
        s.text(x, y0 + 72, count, 12.5, fill="accent")
        for j, item in enumerate(items):
            s.text(x, y0 + 94 + j * 17, item, 13)
    s.line(x0 + 18, y0 + 150, x0 + bw - 18, y0 + 150)
    s.text(cx, y0 + 172, f"40 + 24 × 39 + 15 × 15 = {obs_dim():,} numbers", 13.5, 600, anchor="middle")

    # The network.
    s.arrow([(cx, y0 + bh), (cx, 334)])
    s.rect(cx - 130, 334, 260, 72, fill="accent_box", stroke="accent", width=1.5)
    s.text(cx, 359, "Transformer", 15, 600, anchor="middle")
    s.text(cx, 378, "2 attention layers · ~880k weights", 12.5, anchor="middle")
    s.text(cx, 395, "each object is one token", 12.5, fill="muted", anchor="middle")

    # What it controls.
    s.arrow([(cx, 406), (cx, 444)])
    ay = 444
    s.rect(x0, ay, bw, 90)
    s.text(x0 + 18, ay + 25, "Actions, 5 times per second", 14, 600)
    chips = ["steer", "throttle", "target", "fire", "weapon", "salvo", "dive", "sonar/radar", "upgrade"]
    widths = [len(c) * 7.3 + 20 for c in chips]
    gap = (bw - 36 - sum(widths)) / (len(chips) - 1)
    x = x0 + 18
    for chip, w in zip(chips, widths):
        s.rect(x, ay + 38, w, 26, fill="bg", stroke="line", r=13)
        s.text(x + w / 2, ay + 55.5, chip, 13, anchor="middle")
        x += w + gap
    s.text(x0 + 18, ay + 80, "sent to the server as ordinary player commands", 12.5, fill="muted")

    # Back to the game.
    s.arrow([(x0 + bw, ay + 51), (W - 14, ay + 51), (W - 14, 41), (cx + 152, 41)])
    s.text(cx + 166, 34, "commands back to the game", 12.5, fill="muted")
    return s


# --- training_pipeline.svg -----------------------------------------------------------------------

def training_pipeline(t: dict[str, str]) -> Svg:
    e = evals()
    score = {k: e[k]["score"][0] for k, _ in STANDARD}
    W, H = 940, 360
    s = Svg(W, H, t, "Training: built-in bot as teacher, imitation learning, PPO, tougher opponents and new skills")
    bw, gap, y0, bh = 200, 36, 14, 246
    stages = [
        dict(kicker="TEACHER", title="Built-in Mk48 bot", method="hand-written rules",
             lines=["ships with the game", "(server/src/bot.rs)", "steers, aims and fires", "by fixed rules"],
             value=f"{score['built-in bot']:.1f} pts/min", value_note="standard test",
             becomes="the normal game bots"),
        dict(kicker="STEP 1", title="Imitation learning", method="DAgger",
             lines=["the network drives;", "the bot labels what it", "would do in each moment", "10 rounds · 640k labels"],
             value=f"{score['imitation, transformer + 7 weapons']:.1f} pts/min", value_note="standard test",
             becomes="NN 1, NN 2, … in the game"),
        dict(kicker="STEP 2", title="Reinforcement learning", method="PPO + rewards",
             lines=["+ score, damage, kills", "− hits taken, sinking", "kept close to step 1 at first",
                    "~17M decisions"],
             value=f"{score['elite 15M + guard (final)']:.1f} pts/min", value_note="standard test",
             becomes="Elite 15M (results chart)"),
        dict(kicker="STEP 3", title="Tougher opponents", method="and new skills (PPO)",
             lines=["copies of itself as rivals", "every ship type", "reward: protect the lead",
                    "lessons: dive, SAM, decoy", "+3M decisions"],
             value=f"{board_rank('elite_v2b_at_3M_confirm_'):.0%} avg. place",
             value_note=f"scoreboard (was {board_rank('elite_15M_before_'):.0%})",
             becomes="NN Elite in the game"),
    ]
    x0 = (W - 4 * bw - 3 * gap) / 2
    for i, st in enumerate(stages):
        x = x0 + i * (bw + gap)
        learned = i > 0
        s.rect(x, y0, bw, bh, fill="accent_box" if i == 3 else "box", stroke="accent" if i == 3 else "line",
               width=1.5 if i == 3 else 1)
        s.text(x + 16, y0 + 24, st["kicker"], 11.5, 600, fill="accent" if learned else "muted", letter_spacing="0.8")
        s.text(x + 16, y0 + 46, st["title"], 15, 600)
        s.text(x + 16, y0 + 65, st["method"], 13, fill="accent" if learned else "muted")
        for j, line in enumerate(st["lines"]):
            s.text(x + 16, y0 + 92 + j * 18, line, 12.5)
        s.line(x + 16, y0 + 190, x + bw - 16, y0 + 190)
        s.text(x + 16, y0 + 214, st["value"], 16, 600)
        s.text(x + 16, y0 + 233, st["value_note"], 12, fill="muted")
        if i < 3:
            ym = y0 + 112
            s.arrow([(x + bw + 4, ym), (x + bw + gap - 4, ym)])
        # What each stage is in the game.
        s.arrow([(x + bw / 2, y0 + bh + 4), (x + bw / 2, y0 + bh + 26)])
        s.text(x + bw / 2, y0 + bh + 44, st["becomes"], 13, 600 if i == 3 else 400,
               fill="accent" if i == 3 else "fg", anchor="middle")
    s.text(x0, H - 30, "pts/min: game score per minute in the standard test (2 test ships + 32 built-in bots, "
                       "60 game-minutes, 8 games).", 12, fill="muted")
    s.text(x0, H - 13, "avg. place: average scoreboard position over an hour, as NN Elite among 11 NN bots and "
                       "40 built-in bots (100% = first).", 12, fill="muted")
    return s


# --- results.svg ---------------------------------------------------------------------------------

def results(t: dict[str, str]) -> Svg:
    e = evals()
    rows = [(name, e[label]) for label, name in STANDARD]
    W = 760
    label_w, bar_x, row_h = 236, 248, 34
    top = 64
    scale = 440 / 60  # px per point
    fighters = rows[1:]  # kills and deaths for the bots that fight
    panel_top = top + len(rows) * row_h + 56
    foot = panel_top + 30 + len(fighters) * 28 + 22
    H = foot + 62
    s = Svg(W, H, t, "Game score per minute in the standard test, with kills and deaths per minute")

    s.text(16, 26, "Game score per minute", 15, 600)
    s.text(16, 46, "standard test against the game's built-in bots · higher is better", 12.5, fill="muted")
    for v in range(0, 61, 10):
        x = bar_x + v * scale
        s.line(x, top - 6, x, top + len(rows) * row_h, stroke="grid")
        s.text(x, top + len(rows) * row_h + 16, str(v), 12, fill="muted", anchor="middle")
    for i, (name, r) in enumerate(rows):
        y = top + i * row_h
        mean, ci = r["score"]
        elite = name.startswith("Elite")
        fill = "accent" if elite else "bar_2" if name.startswith("Imitation") else "bar"
        s.text(label_w, y + 19, name, 13.5, 600 if elite else 400, anchor="end")
        s.raw(f'<rect x="{bar_x}" y="{y + 5}" width="{mean * scale:.1f}" height="20" fill="{t[fill]}"/>')
        lo, hi = bar_x + (mean - ci) * scale, bar_x + (mean + ci) * scale
        s.line(lo, y + 15, hi, y + 15, stroke="fg", width=1.2)
        s.line(lo, y + 10, lo, y + 20, stroke="fg", width=1.2)
        s.line(hi, y + 10, hi, y + 20, stroke="fg", width=1.2)
        s.text(hi + 8, y + 19.5, f"{mean:.1f}", 13, 600 if elite else 400)
    bot = e["built-in bot"]["score"][0]
    elite_score = e["elite 15M + guard (final)"]["score"][0]
    y_elite = top + (len(rows) - 1) * row_h  # the ratio sits inside the elite's bar
    s.text(bar_x + 10, y_elite + 19.5, f"{elite_score / bot:.1f}× the built-in bot", 12.5, 600, fill="bg")

    for p, (key, title, note, vmax) in enumerate([
        ("kills", "Ships sunk per minute", "higher is better", 0.9),
        ("deaths", "Times sunk per minute", "lower is better", 0.2),
    ]):
        px = 16 + p * 372
        s.text(px, panel_top, title, 14, 600)
        s.text(px, panel_top + 18, note, 12, fill="muted")
        lw, bx, bmax = 178, px + 188, 130
        for i, (name, r) in enumerate(fighters):
            y = panel_top + 30 + i * 28
            elite = name.startswith("Elite")
            fill = "accent" if elite else "bar_2" if name.startswith("Imitation") else "bar"
            v = r[key][0]
            short = name.replace(" (step 1)", "").replace(" (step 2)", "")
            s.text(px + lw, y + 15, short, 12.5, 600 if elite else 400, anchor="end")
            s.raw(f'<rect x="{bx}" y="{y + 3}" width="{v / vmax * bmax:.1f}" height="16" fill="{t[fill]}"/>')
            s.text(bx + v / vmax * bmax + 6, y + 15.5, f"{v:.2f}", 12.5, 600 if elite else 400)

    for i, line in enumerate([
        "Standard test: 2 test ships + 32 built-in bots per game, 60 game-minutes, 8 independent games, fresh worlds.",
        "Whiskers: 95% interval within one run. Repeat runs of the same model have differed by up to 6 pts/min.",
        "* The built-in bot's own decisions sent through the network's controls: aim snaps onto ships, and it fires",
        "whenever a weapon can hit. Data: agent/results/evals.json",
    ]):
        s.text(16 + 8 * (i == 3), foot + i * 17, line, 12, fill="muted")
    return s


# --- gameplay_annotated.png ----------------------------------------------------------------------

def gameplay_annotated() -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(SCREENSHOT).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    def font(size: int, bold: bool = False):
        for path, index in (("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
                            ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0)):
            if Path(path).exists():
                return ImageFont.truetype(path, size, index=index)
        import matplotlib
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)

    ink = (255, 255, 255, 255)
    shade = (10, 20, 35, 170)
    lead = (255, 255, 255, 230)

    blue = (110, 200, 255, 255)

    def label(xy, lines, anchor="la", dot_line=None):
        """Text on a thin dark backing, so it reads on sea and on land. `dot_line` gets a blue dot."""
        f_bold, f = font(24, True), font(20)
        fonts = [f_bold] + [f] * (len(lines) - 1)
        indent = [26 if i == dot_line else 0 for i in range(len(lines))]
        x, y = xy
        boxes = [draw.textbbox((0, 0), line, font=fnt) for line, fnt in zip(lines, fonts)]
        w = max(b[2] + d for b, d in zip(boxes, indent))
        h = sum(b[3] for b in boxes) + 6 * (len(lines) - 1)
        if anchor == "ra":
            x -= w
        draw.rounded_rectangle((x - 12, y - 8, x + w + 12, y + h + 10), 8, fill=shade)
        for line, fnt, box, d in zip(lines, fonts, boxes, indent):
            if d:
                draw.ellipse((x + 2, y + box[3] / 2 - 4, x + 14, y + box[3] / 2 + 8), fill=blue)
            draw.text((x + d, y), line, font=fnt, fill=ink)
            y += box[3] + 6

    # The ship the network is driving (centre of the screen).
    ship = (719, 452)
    draw.ellipse((ship[0] - 32, ship[1] - 38, ship[0] + 32, ship[1] + 38), outline=lead, width=3)
    draw.line((ship[0] + 22, ship[1] + 30, 800, 648), fill=lead, width=2)
    label((780, 660), ["AI Elite", "the neural network is controlling this ship"])

    # A built-in bot on the map.
    draw.line((392, 556, 330, 616), fill=lead, width=2)
    label((150, 626), ["HK-47", "a normal game bot (hand-written rules)"])

    # The scoreboard: who is who.
    nn_rows = [18, 81, 102, 165, 207, 270, 375]  # NN 8, NN 6, NN Elite, NN 3, NN 5, NN 4, NN 10
    for y in nn_rows:
        draw.ellipse((1234, y - 5, 1244, y + 5), fill=blue)
    label((1206, 130), ["Scoreboard: every ship in the game", "NN bot, driven by the network",
                        "other names: built-in game bots"], anchor="ra", dot_line=1)
    img.save(OUT / "gameplay_annotated.png", optimize=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    check_numbers()
    save_both("overview", overview)
    save_both("training_pipeline", training_pipeline)
    save_both("results", results)
    gameplay_annotated()
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
