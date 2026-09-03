# API — TabbyAPI OpenAI-compatible endpoint

This page documents how to call the TabbyAPI server used in
[attempts/exl3-4.05bpw-exllamav3](../attempts/exl3-4.05bpw-exllamav3/README.md).
It applies to that recipe specifically; other runtimes in this repository
(llama.cpp's `llama-server`, vLLM) expose their own OpenAI-compatible or
native endpoints and are documented in their own attempt records.

## Endpoint

TabbyAPI exposes a standard OpenAI-compatible surface. The default port is
**5000**. Generic example (substitute your own host):

```bash
curl http://<host>:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "glm-5.3-flash-exl3-4.05bpw",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer with just the city name."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

- `GET /v1/models` is the liveness/readiness gate used in this recipe's
  attempt record: it returns 200 OK once the model has finished loading.

## `reasoning: true` is required for this model

GLM-5.3-Flash defaults to `reasoning_effort: max` and wraps chain-of-thought
in `<think>...</think>` tags. **If TabbyAPI's server-side `reasoning` config
is left `false`, the chain-of-thought leaks into the visible `content`
field unparsed, and burns the entire completion token budget with no final
answer.** This was diagnosed and fixed during the 2026-09-03 attempt (see
the attempt record).

Set `reasoning: true` in the TabbyAPI server config for this model. With it
set:

- Chain-of-thought is returned in a separate `reasoning_content` field on
  the response message.
- The visible `content` field carries only the final answer.

### Token budget consequence

Because reasoning tokens are drawn from the same `max_tokens` budget as the
final answer, a short `max_tokens` value can be entirely consumed by
reasoning, leaving `content: null` and `finish_reason: "length"`. Measured
on this recipe: `max_tokens=32` reproducibly (3/3) returns no answer,
while `max_tokens=128` reliably returns a complete short answer. **Use
`max_tokens >= 128` for short factual answers whenever `reasoning: true` is
set**; scale it up further for tasks that require longer reasoning chains.

## Usage accounting: `stream_options.include_usage`

This repository's evidence rules require deriving generated-token counts
from the response's `usage` object, never from counting stream events. For
**streaming** requests, TabbyAPI (like upstream OpenAI-compatible servers)
only includes a `usage` object in the final stream chunk when the request
sets:

```json
{
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

For **non-streaming** requests, set `stream_options.include_usage: true` in
the same way to guarantee the `usage` object (`prompt_tokens`,
`completion_tokens`, `total_tokens`) is populated on the single returned
response. Omitting this option is a common source of missing or
zero-valued usage fields — always set it explicitly rather than relying on
a runtime default.

## Prefill / prompt-time reporting

The response `usage` object on this runtime also carries a `prompt_time`
field (seconds spent on prefill for that request). This recipe's prefill
measurement (see the attempt record's "Prefill / TTFT" section) relies on
this field, not on client-side wall-clock timing, to separate prefill from
decode time.

## Known limitation: long-context requests under a quantized KV cache

Any single request whose context exceeds approximately 2,048 tokens will
fail (HTTP 503 on that request only, not a server crash) if the server is
running with `cache_mode: Q8`. See
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) for the full explanation and
the required `cache_mode: FP16` workaround for long-context workloads.
