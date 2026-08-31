# Phase C (#63) evidence — GLM-5.3-Flash UD-IQ4_XS on CMP 170HX

Baseline for the CMP card assigned to GLM. Runtime: llama.cpp fork at
commit 00699716 (CUDA, sm_80). Checkpoint: unsloth UD-IQ4_XS 5-shard GGUF
at pinned revision 2975ab41, SHA-256 verified at gate.

## Files

- `warmups.jsonl` — 3 discarded warm-up reps (c=1)
- `speed-c1.jsonl` — 5 measured single-stream reps, 400-tok prompt /
  256-tok completion: median 17.73 tok/s aggregate, 14.44 s e2e;
  streaming TTFT field is a harness artifact (0.0), prefill proxy is
  59.7 ms/tok from the phase-B smoke
- `ladder.jsonl` — concurrency ladder 1/2/4 x 2 reps; aggregate ~flat
  (17.5-17.7 tok/s, compute-bound), per-req 17.3 / 13.1 / 7.4
- `soak.jsonl` — 41 reps at c=2 across 20 min, all ok, aggregate stable
  17.5 -> 17.7 tok/s
- `quality.jsonl` — first quality pack (known harness defects: coding
  sandbox dropped the candidate file; needle budget starved by
  reasoning tokens)
- `quality-final2.jsonl` — corrected quality pack after harness repair
  (see below)
- `gpu-final.csv` — end-of-run per-card memory/thermal/power snapshot

## Harness defects found and fixed (bench/eval.py, bench/sandboxed_python.sh)

1. The sandbox wrapper forwarded the harness path to the inner shell but
   dropped the candidate solution argument, so every coding task ran
   against an empty file. Fix: pass the argument through.
2. The sandbox ran Python in isolated mode, which removes the script
   directory from module search even when the file was present. Fix:
   rely on the chroot for isolation and drop the flag.
3. Reasoning models spend part of the completion budget on
   reasoning_content before answer content. The needle (64) and coding
   (512) budgets starved before any answer was produced. Fix: uniform
   512 + REASONING_MARGIN (2048) ceiling on reasoning-affected buckets.

## Corrected quality (quality-final2.jsonl, 26 tasks)

math 8/8, instruction 4/5, longctx 3/3, heldout-math 4/4,
heldout-instr 0/2, coding 1/3, heldout-code 1/1 -> 21/26 (80.8%).

Remaining misses are genuine failures at the evaluated budget. The
committed receipts do not include finish/reasoning/output fields, so no
per-miss diagnosis (e.g. reasoning-length exhaustion or specific format
deviations) is claimed here; confirming any such mechanism would require
new receipts with full output metadata.
