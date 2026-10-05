#!/usr/bin/env bash
# Runs your local mk48 server with neural-network bots.
#
#   ./run_game.sh [policy.pt] [nn_bots] [builtin_bots] [elite_policy.pt]
#
#   nn_bots       bots driven by the network (default 8)
#   builtin_bots  bots running the game's own logic (default 8)
#   elite_policy  optional: the last NN bot ("NN Elite") uses this policy
#
# Open https://localhost:8443 (accept the self-signed certificate warning; it's your own server).
# NN bots are named "NN 1", "NN 2", ... To see through an NN bot's eyes, enter a name starting
# with "AI" (e.g. "AI"; "AI Elite" uses the elite policy) before playing: the network then drives
# your ship and respawns it. Ctrl+C stops the server and the policy process.
set -euo pipefail
cd "$(dirname "$0")"
abs() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
policy="${1:-models/bc_v6.pt}"
nn_bots="${2:-8}"
builtin_bots="${3:-8}"
elite="${4:-}"

export MK48_NN_BOTS="$nn_bots"
export MK48_NN_POLICY="$(abs "$policy")"
export MK48_NN_PYTHON="$(pwd)/.venv/bin/python"
export MK48_NN_SCRIPT="$(pwd)/serve_policy.py"
if [ -n "$elite" ]; then export MK48_NN_ELITE_POLICY="$(abs "$elite")"; fi
# Ship personas choose only among ships the network plays well (python ship_ratings.py <policy>).
if [ -f ship_ratings.tsv ]; then export MK48_NN_SHIP_RATINGS="$(pwd)/ship_ratings.tsv"; fi

if lsof -nP -iTCP:8443 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "A server is already running on port 8443; stop it first." >&2
  exit 1
fi

mkdir -p runs/game && cd runs/game  # the server writes playtime.json to its working directory
# The engine's first bots (0..nn_bots-1) are the NN-driven ones.
exec ../../../server/target/release/server --bots "$((nn_bots + builtin_bots))" --http-port 8080 --https-port 8443
