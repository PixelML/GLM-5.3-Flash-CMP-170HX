# Attempt — EXL3/TR3 4 bpw on ExLlamaV3

Status: compatibility-only, failed
Date: 2026-08-30

## Checkpoint

- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) @ `25a44fdbf16862a46b7cc9921142c6c81350af2f`
  (byte-identical mirror of brandonmusic snapshot `5ab363a8`)
- ~171.8 GiB download / ~176 GB installed, 120 shards (community-reported, from the EXL3 recipe documentation)
- EXL3/TR3 4 bpw, uniform K4 group codebooks, routed experts only
- License: ShapleyMCG License 1.0 (source-available) — check redistribution terms before mirroring

## Runtime

- ExLlamaV3 (pin `c5d9c657`, 0.0.43) via the SM121 fork's Docker overlay
- SM80 support: **untested on SM80; distribution is SM121-only** — the overlay
  builds `sm_121a` cubins on arm64, and the image docs explicitly disallow
  x86_64/QEMU use. The quant-method capability gate in the overlay returns 80
  (Ampere), which is why this path is the most plausible future candidate, but a
  capability gate is not a working SM80 stack (inferred, primary-source: overlay
  source in the DGX repository, file `exl3/overlay/exl3.py` around line 480).

## Static fit calculation

- 176 GB installed / 3 cards = ~58.7 GB/card = ~54.6 GiB/card if perfectly
  weight-balanced at TP=3 (community-reported size, arithmetic exact).
- On paper this leaves ~9 GiB/card for CUDA context + KV — the only candidate
  that is even arguable on memory. BUT: routed experts dominate the byte count,
  and TP=3 weight balance is not guaranteed for expert layouts; treat the fit as
  unverified until a real attempt loads the model.

## Execution status and outcome

Not executed. Two independent blockers, either one fatal on its own:

1. No SM80 build of the ExLlamaV3 extension exists (would require rebuilding
   `exllamav3_ext` for `sm_80` — untested, undocumented, plausibly blocked
   by the NoPE-MLA attention architecture).
2. The attention path is sparse-MLA targeting SM12x backends; SM80 has no
   equivalent. An SM80 port needs a non-sparse or differently-sparse attention
   implementation that no runtime provides today.

## Blocker

SM121-only binary distribution plus a missing SM80 attention backend. The
checkpoint itself is the closest thing to viable; the software around it is not.

## Evidence

- Quant capability gate: DGX repository, `exl3/overlay/exl3.py` (commit `bd7f55e` audited pin)
- Size and shard layout: EXL3 recipe documentation in the DGX repository
- Upstream ExLlamaV3 targets Ampere+ (community-reported)

## Re-run instructions

Blocked until both items above exist. Once an SM80 build is available: stage the
checkpoint in shared model storage, TP=3 across the three-card node, verify
per-card weight placement before accepting the fit estimate, 180 W policy,
forced airflow, stop at 80 C core / 85 C memory, abort on Xid.
