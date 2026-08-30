# Candidate — a checkpoint that would actually fit

Status: not-attempted (no qualifying checkpoint existed as of 2026-08-30)
Date: 2026-08-30

## What would qualify

A W4A16/AWQ/GPTQ-class quant whose **exact blob total is <= ~165 GiB**
(55 GiB x 3 cards), leaving per-card room for CUDA context, activations, and a
usable KV pool at TP=3. Practically: a true 3-bit quant of GLM-5.3-Flash, or a
4-bit quant with enough expert-layer exclusions to shave ~17% off the current
198.1 GiB AWQ total.

## Search performed

HF search for `GLM-5.3-Flash` on 2026-08-30 (community-visible quants
enumerated): NVFP4, EXL3/TR3 4bpw, AWQ INT4, FP8, BF16, GGUF, MLX. **None met
the byte budget.** Re-run this search before assuming the conclusion still
holds — new quants appear frequently.

## Also required

Upstream vLLM support for `glm5_next` — currently absent (see
[NVFP4 attempt](../nvfp4-vllm-sm121/README.md)). Both blockers must clear
independently.

## Re-run trigger

When a candidate appears: verify exact blob bytes from the HF API (not the
model card), apply the [static-fit method](../awq-int4-vllm/README.md#static-fit-calculation),
then the [benchmark methodology](../awq-int4-vllm/README.md#re-run-instructions).
