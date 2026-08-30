# Attempt — official FP8 and BF16 checkpoints

Status: static-fit-only, failed
Date: 2026-08-30

## Checkpoint

- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8) and
  [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
- Sizes: >= ~328 GB FP8; BF16 larger (community-reported sizes; the FP8 release
  notes reference this magnitude — not re-measured here)
- Official zai-org releases

## Runtime / SM80 / fit

Moot on memory grounds: at TP=3 these need ~165+ GiB per card against a 64 GiB
physical budget. No fit calculation beyond that is meaningful.

## Execution status and outcome

Not executed. Size alone rules the node out.

## Blocker

Static fit, by a factor of ~2.6x per card even before overhead.

## Evidence

- HF model cards (community-reported sizes)
- This conclusion does not depend on runtime compatibility, which is
  separately blocked per the [NVFP4 attempt](../nvfp4-vllm-sm121/README.md).
