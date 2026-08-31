#!/usr/bin/env python3
"""Offline test for verify-rows.sh: pass and failure-mode fixtures."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_run(d, speed=5, ladder_reps=2, soak=41, window=1200, ok=True):
    def row(c, r, o):
        return {"concurrency": c, "run": r, "ok": o}
    ladder = []
    for c in (1, 2, 4):
        for r in range(ladder_reps):
            ladder.append(row(c, r, ok))
    soak_rows = [row(2, i, ok) for i in range(soak)]
    speed_rows = [row(1, i, ok) for i in range(speed)]
    warmups = [row(1, i, ok) for i in range(3)]
    (Path(d) / "ladder.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ladder))
    (Path(d) / "soak.jsonl").write_text("".join(json.dumps(r) + "\n" for r in soak_rows))
    (Path(d) / "speed-c1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in speed_rows))
    (Path(d) / "warmups.jsonl").write_text("".join(json.dumps(r) + "\n" for r in warmups))
    (Path(d) / "soak-window.json").write_text(json.dumps({
        "start_utc": "2026-08-31T00:00:00Z",
        "end_utc": "2026-08-31T00:20:00Z",
        "window_s": window}) + "\n")

def run_gate(d):
    return subprocess.run(["bash", "bench/verify-rows.sh"], check=False, env={
        "PATH": "/usr/bin:/bin", "REBENCH_OUTDIR": str(d)}, capture_output=True, text=True)

tmp = tempfile.mkdtemp()
fails = 0
cases = (
    ("all-ok", {"window": 1200}, 0),
    ("ladder-fail", {"ok": False}, 1),
    ("soak-fail", {"ok": False}, 1),
    ("short-soak-window", {"window": 600}, 1),
    ("boundary-soak-window", {"window": 1199}, 1),
    ("exact-soak-window", {"window": 1200}, 0),
    ("short-speed", {"speed": 1}, 1),
    ("short-ladder", {"ladder_reps": 1}, 1),
)
for name, kw, expect_ok in cases:
    d = Path(tmp) / name
    d.mkdir()
    write_run(d, **kw)
    out = run_gate(d)
    rc = out.returncode
    status = "PASS" if rc == expect_ok else "FAIL"
    if rc != expect_ok:
        fails += 1
        print(name, "stderr:", out.stdout[-400:])
    print(f"{name}: rc={rc} expected={expect_ok} {status}")
sys.exit(1 if fails else 0)
