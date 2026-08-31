#!/usr/bin/env bash
# Offline gate: verify a run directory proves the DECLARED protocol before
# publication-shaped artifacts are derived. Fails unless exact planned
# counts all succeeded AND the soak window covers the declared duration.
# Usage: REBENCH_OUTDIR=results/run-<id> bash bench/verify-rows.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${REBENCH_OUTDIR:?set REBENCH_OUTDIR to the run directory}"
for f in warmups.jsonl speed-c1.jsonl ladder.jsonl soak.jsonl soak-window.json; do
  [ -f "$OUT/$f" ] || { echo "missing $OUT/$f"; exit 1; }
done

python3 - "$OUT" <<'PY'
import json, sys

# Declared protocol (must match rebench.sh manifest and phase63-driver.sh).
PLANNED = {
    "warmup_rows": 3,
    "speed_rows": 5,
    "ladder_levels": [1, 2, 4],
    "ladder_reps_per_level": 2,
    "soak_window_s": 1200,
}
# The soak window is stamped around the loop, so allow small stamping
# jitter only; this is not a duration discount.
SOAK_WINDOW_MIN_S = 1180

out = sys.argv[1]
def rows(name):
    with open(out + "/" + name) as f:
        return [json.loads(l) for l in f if l.strip().startswith("{")]

failed = 0
def fail(msg):
    global failed
    print("FAIL:", msg)
    failed += 1

warmups = rows("warmups.jsonl")
speed = rows("speed-c1.jsonl")
ladder = rows("ladder.jsonl")
soak = rows("soak.jsonl")
window = json.load(open(out + "/soak-window.json"))

if len(warmups) != PLANNED["warmup_rows"]:
    fail(f"warmups: {len(warmups)} rows != planned {PLANNED['warmup_rows']}")
bad_warm = [r for r in warmups if not r.get("ok")]
if bad_warm:
    fail(f"{len(bad_warm)} warmup rows with failed requests")

if len(speed) != PLANNED["speed_rows"]:
    fail(f"speed c1: {len(speed)} rows != planned {PLANNED['speed_rows']}")
bad_speed = [r for r in speed if not r.get("ok")]
if bad_speed:
    fail(f"{len(bad_speed)} speed rows with failed requests")

levels = sorted({r["concurrency"] for r in ladder})
if levels != PLANNED["ladder_levels"]:
    fail(f"ladder levels {levels} != planned {PLANNED['ladder_levels']}")
reps = {}
for r in ladder:
    reps.setdefault(r["concurrency"], []).append(r)
for lv in PLANNED["ladder_levels"]:
    got = reps.get(lv, [])
    if len(got) != PLANNED["ladder_reps_per_level"]:
        fail(f"ladder c={lv}: {len(got)} reps != planned {PLANNED['ladder_reps_per_level']}")
bad_ladder = [r for r in ladder if not r.get("ok")]
if bad_ladder:
    fail(f"{len(bad_ladder)} ladder rows with failed requests")

if not soak:
    fail("soak produced no rows")
bad_soak = [r for r in soak if not r.get("ok")]
if bad_soak:
    fail(f"{len(bad_soak)} soak rows with failed requests")
window_s = window.get("window_s")
if not isinstance(window_s, int) or window_s < SOAK_WINDOW_MIN_S:
    fail(f"soak window {window_s}s < required {SOAK_WINDOW_MIN_S}s "
         f"(planned {PLANNED['soak_window_s']}s)")

counts = {
    "warmup_rows": len(warmups),
    "warmup_failed": len(bad_warm),
    "speed_rows": len(speed),
    "speed_failed": len(bad_speed),
    "ladder_levels": levels,
    "ladder_rows": len(ladder),
    "ladder_failed_rows": len(bad_ladder),
    "soak_rows": len(soak),
    "soak_failed_rows": len(bad_soak),
    "soak_window_s": window_s,
    "planned": PLANNED,
    "protocol_matches_plan": failed == 0,
}
with open(out + "/actual-counts.json", "w") as f:
    json.dump(counts, f, indent=2)
    f.write("\n")
if failed:
    sys.exit(1)
print("row verification PASS:", counts)
PY
