#!/usr/bin/env python3
"""Offline test for verify-rows.sh: synthetic pass and fail fixtures."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_run(d, ladder_ok, soak_ok, missing_rep=False):
    def row(c, r, ok):
        return {"concurrency": c, "run": r, "ok": ok}
    ladder = []
    for c in (1, 2, 4):
        reps = [0] if missing_rep else [0, 1]
        for r in reps:
            ladder.append(row(c, r, ladder_ok))
    soak = [row(2, i, soak_ok) for i in range(41)]
    speed = [row(1, i, True) for i in range(5)]
    (Path(d) / "ladder.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ladder))
    (Path(d) / "soak.jsonl").write_text("".join(json.dumps(r) + "\n" for r in soak))
    (Path(d) / "speed-c1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in speed))

def run_gate(d):
    return subprocess.run(["bash", "bench/verify-rows.sh"], check=False, env={
        "PATH": "/usr/bin:/bin", "REBENCH_OUTDIR": str(d)}, capture_output=True, text=True)

tmp = tempfile.mkdtemp()
fails = 0
for name, kw, expect_ok in (
    ("all-ok", {"ladder_ok": True, "soak_ok": True}, 0),
    ("ladder-fail", {"ladder_ok": False, "soak_ok": True}, 1),
    ("soak-fail", {"ladder_ok": True, "soak_ok": False}, 1),
    ("short-ladder", {"ladder_ok": True, "soak_ok": True, "missing_rep": True}, 1),
):
    d = Path(tmp) / name
    d.mkdir()
    write_run(d, **kw)
    rc = run_gate(d).returncode
    status = "PASS" if rc == expect_ok else "FAIL"
    if rc != expect_ok:
        fails += 1
    print(f"{name}: rc={rc} expected={expect_ok} {status}")
sys.exit(1 if fails else 0)
