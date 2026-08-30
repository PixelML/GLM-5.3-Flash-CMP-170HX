#!/usr/bin/env python3
"""Deterministic task evaluation: math, code, instruction-following, long-context.
Public/licensed tasks only. No LLM judge. Scores are exact-match / pass-based.
Usage: BASE=http://127.0.0.1:8199 python3 eval.py --out results.jsonl
"""
import argparse, json, re, time, urllib.request, urllib.error


TASKS = []
def task(fn):
    TASKS.append(fn)
    return fn


def call(base, messages, max_tokens=512, temperature=0.0, seed=42):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "seed": seed,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read())
    dt = time.perf_counter() - t0
    choice = obj["choices"][0]["message"]["content"]
    usage = obj.get("usage", {})
    return choice, usage, dt


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return m[-1].strip() if m else None


def extract_code_fences(text):
    return re.findall(r"```(?:python)?\n(.*?)```", text, re.S)


# ---------- MATH (GSM8K-style public subset; deterministic exact match) ------
MATH_TASKS = [
    # (question, gold_answer) — verified public-domain arithmetic/reasoning
    ("A bakery sells 240 muffins per day. Each muffin uses 12 grams of flour. How many kilograms of flour does the bakery use in one week? Give your answer as a number only.", "20.16"),
    ("A train travels 420 km in 3.5 hours. A bus travels the same distance in 5 hours. How many km/h faster is the train? Answer with the number only.", "36"),
    ("A shirt costs $60 after a 25% discount. What was the original price in dollars? Answer with the number only.", "80"),
    ("Compute 17 * 23 - 145. Answer with the number only.", "246"),
    ("If 3 machines produce 270 parts in 2 hours, how many parts do 5 machines produce in 4 hours at the same rate? Answer with the number only.", "900"),
    ("A rectangle has perimeter 64 and length 20. What is its area? Answer with the number only.", "240"),
    ("A water tank fills at 12 L/min but drains at 3 L/min simultaneously. How long to fill an empty 540 L tank? Answer in minutes, number only.", "60"),
    ("Sum the integers from 1 to 100 inclusive. Answer with the number only.", "5050"),
]


@task
def math_eval(base):
    results = []
    for q, gold in MATH_TASKS:
        try:
            out, usage, dt = call(base, [{"role": "user", "content": q}])
            pred = extract_boxed(out) or re.search(r"-?\d+(?:\.\d+)?", out.split("\n")[-1])
            pred_val = pred if isinstance(pred, str) else (pred.group() if pred else None)
            ok = pred_val is not None and abs(float(pred_val.replace(",", "")) - float(gold)) < 1e-4
            results.append({"bucket": "math", "task": q[:40], "ok": bool(ok), "gold": gold, "pred": pred_val, "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2), "raw": out[:300]})
        except Exception as e:
            results.append({"bucket": "math", "task": q[:40], "ok": False, "err": str(e)[:200]})
    return results


# ---------- INSTRUCTION ADHERENCE (deterministic constraints) ---------------
def _json_color_count(t):
    """Accept a bare JSON object or one fenced in a ```json block; require
    exactly the keys 'color' (string) and 'count' (integer)."""
    s = t.strip()
    if "```json" in s:
        s = s.split("```json")[-1].split("```")[0].strip()
    try:
        obj = json.loads(s)
    except Exception:
        return False
    return (isinstance(obj, dict)
            and set(obj.keys()) == {"color", "count"}
            and isinstance(obj["color"], str)
            and isinstance(obj["count"], int)
            and not isinstance(obj["count"], bool))


INSTR_TASKS = [
    ("Write exactly 3 sentences about the ocean. Separate them with newlines.", lambda t: len([l for l in t.strip().split("\n") if l.strip()]) == 3),
    ("List exactly 5 prime numbers below 20, one per line, nothing else.", lambda t: (nums := re.findall(r"\d+", t)) and len(nums) == 5 and all(int(n) in {2,3,5,7,11,13,17,19} for n in nums)),
    ("Reply with only the word YES if 7 is prime, or NO otherwise.", lambda t: t.strip().upper() == "YES"),
    ("Produce a JSON object with exactly two keys: 'color' (string) and 'count' (integer). Do not add any other keys or text.", _json_color_count),
    ("Say 'alpha' then 'beta' then 'gamma', one word per line, in that order, nothing else.", lambda t: [w.lower() for w in t.strip().split("\n") if w.strip()] == ["alpha", "beta", "gamma"]),
]


@task
def instruction_eval(base):
    results = []
    for q, check in INSTR_TASKS:
        try:
            out, usage, dt = call(base, [{"role": "user", "content": q}], max_tokens=256)
            ok = check(out)
            results.append({"bucket": "instruction", "task": q[:40], "ok": bool(ok), "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2), "raw": out[:300]})
        except Exception as e:
            results.append({"bucket": "instruction", "task": q[:40], "ok": False, "err": str(e)[:200]})
    return results


# ---------- LONG-CONTEXT RETRIEVAL (synthetic needle, deterministic) --------
def needle_task(base, ctx_tokens_approx=2000):
    needle = f"The access code for vault {ctx_tokens_approx} is ZX-Q7-1943."
    filler = ("The warehouse ledger records daily grain deliveries. Rows repeat with minor variations in tonnage and dock number. " * 80)
    q = f"{filler}\n\n{needle}\n\n{filler}\n\nWhat is the access code for vault {ctx_tokens_approx}? Answer with the code only."
    try:
        out, usage, dt = call(base, [{"role": "user", "content": q}], max_tokens=64)
        ok = "ZX-Q7-1943" in out
        return [{"bucket": "longctx", "task": f"needle@{ctx_tokens_approx}", "ok": bool(ok), "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2)}]
    except Exception as e:
        return [{"bucket": "longctx", "task": f"needle@{ctx_tokens_approx}", "ok": False, "err": str(e)[:200]}]


@task
def longctx_eval(base):
    return needle_task(base, 2000) + needle_task(base, 4000) + needle_task(base, 8000)


BUCKETS = {"math": math_eval, "instruction": instruction_eval, "longctx": longctx_eval}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8199")
    ap.add_argument("--buckets", default="math,instruction,longctx")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    all_results = []
    for b in a.buckets.split(","):
        all_results.extend(BUCKETS[b](a.base))
    with open(a.out, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    ok_count = sum(1 for r in all_results if r.get("ok"))
    print(json.dumps({"total": len(all_results), "passed": ok_count, "rate": round(ok_count/len(all_results), 3) if all_results else 0}))
    for r in all_results:
        print(("PASS" if r.get("ok") else "FAIL"), r["bucket"], r["task"], r.get("err", ""))
