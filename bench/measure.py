#!/usr/bin/env python3
"""Single-stream and concurrent measurement harness.

Captures cold/warm TTFT, inter-token latency, end-to-end time,
per-request and aggregate output throughput, GPU memory/temps/power.
Token counts come from the final usage object in streaming responses.
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class Sample:
    t_first: float          # seconds to first token
    t_last: float           # seconds to last token (e2e)
    itl_ms: list = field(default_factory=list)  # inter-token gaps, ms
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    err: str | None = None


async def one_request(session, base, payload, sample_out):
    t0 = time.perf_counter()
    itl = []
    prev = None
    usage = None
    try:
        async with session.post(f"{base}/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                sample_out.err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                sample_out.ok = False
                return
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if obj.get("usage"):
                    usage = obj["usage"]
                choice = ((obj.get("choices") or [{}])[0].get("delta") or {})
                if choice.get("content"):
                    now = time.perf_counter()
                    if prev is None:
                        sample_out.t_first = now - t0
                    else:
                        itl.append((now - prev) * 1000)
                    prev = now
        sample_out.t_last = time.perf_counter() - t0
        sample_out.itl_ms = itl
        if usage:
            sample_out.prompt_tokens = usage.get("prompt_tokens", 0)
            sample_out.completion_tokens = usage.get("completion_tokens", 0)
        else:
            sample_out.ok = False
            sample_out.err = "missing final usage object"
    except Exception as e:
        sample_out.ok = False
        sample_out.err = f"{type(e).__name__}: {e}"


async def sweep(base, prompt, max_tokens, concurrencies, runs, out_path, seed):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rows = []
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for conc in concurrencies:
            for run in range(runs):
                samples = [Sample(0.0, 0.0) for _ in range(conc)]
                t0 = time.perf_counter()
                await asyncio.gather(*(one_request(session, base, payload, s) for s in samples))
                wall = time.perf_counter() - t0
                all_itl = [x for s in samples for x in s.itl_ms]
                total_ct = sum(s.completion_tokens for s in samples)
                row = {
                    "concurrency": conc, "run": run, "wall_s": round(wall, 3),
                    "ok": all(s.ok for s in samples),
                    "ttft_s": round(statistics.median([s.t_first for s in samples if s.ok]), 4),
                    "e2e_s_median": round(statistics.median([s.t_last for s in samples if s.ok]), 3),
                    "itl_ms_p50": round(statistics.median(all_itl), 2) if all_itl else None,
                    "itl_ms_p95": round(sorted(all_itl)[int(len(all_itl)*0.95)-1], 2) if len(all_itl) >= 20 else None,
                    "tok_per_s_per_req": round(statistics.median([s.completion_tokens / s.t_last for s in samples if s.ok and s.t_last > 0]), 2),
                    "tok_per_s_aggregate": round(total_ct / wall, 2) if wall > 0 else 0,
                    "prompt_tokens": sum(s.prompt_tokens for s in samples),
                    "completion_tokens": total_ct,
                    "errors": [s.err for s in samples if not s.ok],
                    "raw_log": out_path,
                }
                rows.append(row)
                print(json.dumps(row))
    def _append():
        with open(out_path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    await asyncio.to_thread(_append)


def gpu_snapshot():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw,clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits"], text=True)
        return [dict(zip(["index", "mem_mib", "mem_total_mib", "temp_core_c", "temp_mem_c", "power_w", "throttle_mask"], line.split(", "))) for line in out.strip().split("\n")]
    except Exception:
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8199")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conc = [int(x) for x in a.concurrency.split(",")]
    print(json.dumps({"gpu_before": gpu_snapshot()}))
    asyncio.run(sweep(a.base, a.prompt, a.max_tokens, conc, a.runs, a.out, a.seed))
    print(json.dumps({"gpu_after": gpu_snapshot()}))
