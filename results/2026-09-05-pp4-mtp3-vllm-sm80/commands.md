# Redacted commands

Every command that produced the receipts in this directory. Private endpoints,
filesystem paths, container names and ports are replaced by `<...>` placeholders
using the same convention as the receipt JSONs.

## 1. Preflight (read-only, before any launch)

```bash
# device health and, critically, the current PCIe link state of every card
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,pcie.link.gen.current,pcie.link.width.current,pcie.link.width.max \
  --format=csv

# no other workload owns the GPUs
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
docker ps

# clean fault-log window
dmesg --ctime | grep -iE 'xid|nvrm' | tail -20
```

The link-state query is not optional. A card trained narrow silently invalidates
every aggregate, prefill and TTFT number, and invalidates any tensor-parallel
comparison entirely.

## 2. Weights

```bash
pip install -U huggingface_hub
hf download wtdcode/GLM-5.3-Flash-AWQ-W4A16 \
  --revision abd7b07719111f137e1de8a0c1b7e01c11b74d1a \
  --local-dir <weights>
```

Verify 24 files and 190,843,146,533 bytes before spending GPU time. Stage on
local NVMe; over a network mount the shard load dominates boot.

## 3. Image

```bash
docker pull ghcr.io/pixelml/club-170hx@sha256:62f612b49614523e6a46e1493d35d3efd1f363917129d38cc923a31053693bfb
```

## 4. Serve — the recipe of record

```bash
docker run -d --name <container> --gpus '"device=0,1,2,3"' \
  --shm-size 16g --ipc=host -p 127.0.0.1:<port>:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PP_LAYER_PARTITION=14,12,12,7 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TORCH_CUDA_ARCH_LIST=8.0 \
  -e VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2 \
  -e VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256 \
  -v <weights>:/weights:ro \
  ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905 \
  --model /weights --served-model-name GLM-5.3-Flash \
  --pipeline-parallel-size 4 \
  --max-model-len 393216 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

Wait for `Application startup complete`; about 1,029 s from locally staged
weights. For the draft-depth sweep, the only field changed between boots is
`num_speculative_tokens` (2, 3, 5, 7). The DFlash2 negative cell replaces the
speculative config with the block drafter at depth 7 and changes nothing else.

## 5. Gates → `receipts/cells.json`, `receipts/k3/gate.json`

```bash
curl -s http://127.0.0.1:<port>/v1/models

# deterministic greedy repeat: three identical single-token completions
for i in 1 2 3; do
  curl -s http://127.0.0.1:<port>/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"GLM-5.3-Flash","prompt":"The capital of France is","temperature":0,"max_tokens":1}'
done
```

## 6. P1 — the upstream harness → `receipts/<k>/p1.json`

```bash
python3 scripts/bench.py --url http://127.0.0.1:<port>/v1 --model GLM-5.3-Flash
```

Temperature 0, 512 output tokens, three repetitions, five workloads plus a
repetition diagnostic, with the author's repeat guard active. Used unmodified;
see the attribution table in the README.

## 7. P2 — this repository's protocol → `receipts/<k>/decode_c1.json`

```bash
python3 bench_glm53.py --url http://127.0.0.1:<port>/v1 --model GLM-5.3-Flash
```

Temperature 0.7, `ignore_eos`, 512 output tokens, five repetitions, first
repetition cold.

## 8. Acceptance → `receipts/[<k>/]accept_{before,after}.json`

```bash
curl -s http://127.0.0.1:<port>/metrics \
  | grep -E 'vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total'
```

Captured immediately before and immediately after each measurement window and
differenced. These are exact cumulative counters, not gauges; do not read a
single sample as a rate.

## 9. Context sweep → `receipts/k3/sweep/context_sweep.json`

`context_sweep.py`, one boot, arms alternated per length. Prompts are built from
a seeded nonce-word payload and calibrated to the target token count by
rescaling the word count against the tokenizer, accepting within 5%, so every
prompt is a guaranteed cache miss. Per point: one cold request, one immediate
warm repeat, then three measured repetitions, 512 output tokens each. Lengths
327, 860, 2k, 4k, 8k, 16k, 33k, 66k, 131k, 258k tokens; two arms, thinking on
and thinking off, sent as `chat_template_kwargs`.

Every metric comes from the final SSE `usage` object: prompt tok/s, generation
tok/s, cold and warm TTFT, ms per output token, prompt KB/s and generation B/s
from the actual UTF-8 byte counts, and the cold-over-warm TTFT ratio.

The 258k point failed: the calibration fallback overshot and the server rejected
the request against its 393,216-token limit. Recorded as untested with the
reason rather than retried.

## 10. Concurrency → `receipts/k3/conc_sweep.json`

c = 1, 2, 4, 8, 16 at a 4,096-token uncached prompt with a per-request nonce,
256 output tokens, P2 sampling. Aggregate and per-stream throughput, e2e p50 and
p95, and success rate are recorded per level.

## 11. Prefill and TTFT → `receipts/k3/prefill_{4096,16384}/`

```bash
# uncached prefill: one output token, three reps, prompt length verified
# against the tokenizer before sending
# warm streaming TTFT: 32 output tokens, three reps, wall time to the first
# content-bearing chunk
```

## 12. Sustained stability → `receipts/k3/c8_stability.json`

Three rounds of c=8 back to back, 2,900-token uncached prompts with a per-request
nonce, 256 output tokens, P2 sampling. After **each** round, device health is
sampled — per-card temperature, instantaneous power, enforced power limit, PCIe
link width and generation — and the fault log is checked for new entries. The
verdict is rounds passed and requests succeeded; the aggregate throughput column
is link-bound and does not carry the verdict.

## 13. Quality battery → `receipts/k3/quality.json`

Run at the **checkpoint's own sampling defaults**, read from its
`generation_config.json` rather than chosen: temperature 1.0, top_p 0.95, seed
1234, 3,072 max tokens, thinking left at the model default. A quality battery
run at the greedy throughput settings would measure a configuration nobody
serves.

Buckets: GSM8K (50 items, exact match on the final number), HumanEval (20 items,
pass@1 by executing the reference tests in a subprocess), structured output (10
prompts, strict JSON parse plus required top-level fields). The receipt carries
every item, not only the failures, so any score can be recomputed from it.

## 14. Lossless check → `receipts/lossless/`

Run the controls **before** the comparison, or the comparison cannot be read.

```
# 1. noise floor with speculation ON: the same 20 prompts, twice,
#    against one server, one boot, temperature 0, fixed seed, 256 output tokens
#    -> spec_on_a.json, spec_on_b.json, self_consistency.json

# 2. noise floor with speculation OFF: same 20 prompts, twice, same server
#    -> spec_off_a.json, spec_off_b.json, self_consistency_nospec.json

# 3. only then, on versus off
#    -> lossless_on_vs_off.json
```

Each summary records identical completions out of 20 and a token-for-token match
rate, defined as positions matching over positions compared under greedy
decoding.

Both control runs disagree with themselves on this stack, so step 3 is reported
beside them and never on its own. A comparison that cannot clear its own noise
floor is not a lossless verdict.

## 15. Speculation-off baseline → `receipts/nospec/`

The same P1 and P2 commands from sections 6 and 7, against a boot with
`--speculative-config` removed and nothing else changed. This is what makes the
drafter's uplift a measurement rather than an assumption.

## 16. Teardown

```bash
docker rm -f <container>
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.width.current --format=csv
dmesg --ctime | grep -iE 'xid|nvrm' | tail -20
```

Health and link state are re-checked after every boot. A depth-sweep row
measured across a link-state change is not comparable to the rows around it.

A caution learned here: the management interface can report every card healthy
while the driver cannot initialise at all. After a fault, check driver
initialisation directly rather than trusting the management interface's summary.

## 17. Chart

```bash
python3 assets/charts/2026-09-05-glm-5.3-flash-pp4-context-sweep.py
```

In the club repository; regenerates the 3x3 figure from the committed
`context_sweep.json` with no GPU and no network.
