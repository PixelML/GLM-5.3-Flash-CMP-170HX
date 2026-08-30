#!/usr/bin/env python3
"""Deterministic task evaluation: math, code, instruction-following, long-context.
Public/licensed tasks only. No LLM judge. Scores are exact-match / pass-based.
Usage: BASE=http://127.0.0.1:8199 python3 eval.py --out results.jsonl
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request

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
    msg = obj["choices"][0]["message"]
    choice = msg.get("content") or ""
    usage = obj.get("usage", {})
    usage = dict(usage)
    usage["finish_reason"] = obj["choices"][0].get("finish_reason")
    usage["answer_bytes"] = bool(choice.strip())
    usage["reasoning_bytes"] = bool((msg.get("reasoning_content") or "").strip())
    return choice, usage, dt


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return m[-1].strip() if m else None


def extract_code_fences(text):
    return re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)


# Completion-budget margin for models that emit reasoning_content before the
# final answer channel. Applied on top of the 512-token ceiling for buckets
# whose outputs are long enough that reasoning overhead can starve them.
REASONING_MARGIN = 1536


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
    ("Write exactly 3 sentences about the ocean. Separate them with newlines.", lambda t: len([line for line in t.strip().split("\n") if line.strip()]) == 3),
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


# ---------- CODING (deterministic: model output is executed, pass/fail) -----
# Each task: (prompt, harness_code). The harness defines test(harness) and is
# run in a subprocess with a hard timeout; only its exit code and stdout are
# inspected. No network, no file access outside a fresh temp dir (Python
# subprocess runs with cwd=tmpdir; the code under test is passed as a string).
CODING_TASKS = [
    (
        ("Write a Python function is_balanced(s) that returns True if the "
        "parentheses in the string s are balanced, False otherwise. Only "
        " '(' and ')' matter. Answer with a single Python code block only."),
        (
            "from solution import is_balanced\n"
            "assert is_balanced('') == True\n"
            "assert is_balanced('()') == True\n"
            "assert is_balanced('(())') == True\n"
            "assert is_balanced('(()') == False\n"
            "assert is_balanced('())') == False\n"
            "assert is_balanced('a(b)c(d)e') == True\n"
            "assert is_balanced(')(') == False\n"
            "print('PASS')\n"
        ),
    ),
    (
        ("Write a Python function second_max(nums) that returns the second "
        "largest distinct value in the list nums, or None if fewer than two "
        "distinct values exist. Do not sort the full list with sorted() or "
        ".sort(); use a single pass. Answer with a single Python code block only."),
        (
            "from solution import second_max\n"
            "assert second_max([1, 3, 2]) == 2\n"
            "assert second_max([5, 5, 4]) == 4\n"
            "assert second_max([7]) is None\n"
            "assert second_max([2, 2]) is None\n"
            "assert second_max([-1, -5, -3]) == -3\n"
            "assert second_max([10, 9, 8, 9]) == 9\n"
            "print('PASS')\n"
        ),
    ),
    (
        ("Write a Python function flatten(d) that takes a dict whose values "
        "may be nested dicts (arbitrary depth, string keys) and returns a "
        "flat dict where nested keys are joined with dots, e.g. "
        "{'a': {'b': 1}} -> {'a.b': 1}. Answer with a single Python code block only."),
        (
            "from solution import flatten\n"
            "assert flatten({'a': 1}) == {'a': 1}\n"
            "assert flatten({'a': {'b': 1}}) == {'a.b': 1}\n"
            "assert flatten({'a': {'b': {'c': 3}}}) == {'a.b.c': 3}\n"
            "assert flatten({'x': 1, 'y': {'z': 2}}) == {'x': 1, 'y.z': 2}\n"
            "assert flatten({}) == {}\n"
            "print('PASS')\n"
        ),
    ),
]


def _run_coding_task(model_output, harness_code):
    """Extract code from the model output, run the fixed test harness in a
    subprocess. Returns (ok, detail). 25 s hard timeout; the harness only
    asserts on pure functions and imports only the candidate solution."""
    import os
    import re
    import subprocess
    import tempfile
    # 'harness' here is the raw model output; extract the first fenced block
    # if present, else use the whole output as the code.
    fence = chr(96) * 3
    m = re.search(fence + r"(?:python)?\s*\n(.*?)" + fence, model_output, re.DOTALL)
    code = m.group(1) if m else model_output
    with tempfile.TemporaryDirectory() as td:
        sol = os.path.join(td, "solution.py")
        test = os.path.join(td, "test_harness.py")
        sandbox = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sandboxed_python.sh")
        with open(sol, "w") as f:
            f.write(code + "\n")
        with open(test, "w") as f:
            f.write(harness_code)
        try:
            r = subprocess.run([sandbox, test, sol], cwd=td,
                               capture_output=True, text=True, timeout=25,
                               check=False)
            return (r.returncode == 0 and "PASS" in r.stdout,
                    (r.stdout + r.stderr)[-200:])
        except subprocess.TimeoutExpired:
            return False, "timeout"


@task
def coding_eval(base):
    results = []
    for prompt, harness in CODING_TASKS:
        try:
            # This model family spends part of the budget on reasoning_content
            # before answer content; coding needs headroom for the full program.
            out, usage, dt = call(base, [{"role": "user", "content": prompt}], max_tokens=512 + REASONING_MARGIN)
            ok, detail = _run_coding_task(out, harness)
            results.append({"bucket": "coding", "task": harness.split("\n")[0][:40], "ok": bool(ok), "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2), "detail": detail, "finish_reason": usage.get("finish_reason"), "answer_bytes": usage.get("answer_bytes"), "reasoning_bytes": usage.get("reasoning_bytes")})
        except Exception as e:
            results.append({"bucket": "coding", "task": prompt[:40], "ok": False, "err": str(e)[:200]})
    return results


# ---------- HELD-OUT SET (never used for config tuning) ----------------------
# Same task types as the tuning set above, disjoint instances. Per the
# experiment plan, config keep/discard decisions use only the tuning set;
# the final recommended config is scored once here.
HELDOUT_MATH = [
    ("A printer produces 45 pages per minute. How many pages does it produce in 2 hours? Answer with the number only.", "5400"),
    ("A cyclist rides 96 km in 4 hours. What is the average speed in km/h? Answer with the number only.", "24"),
    ("A jacket costs $150 and is discounted by 30%. What is the sale price in dollars? Answer with the number only.", "105"),
    ("Compute 13 * 17 - 121. Answer with the number only.", "100"),
]

HELDOUT_INSTR = [
    ("Write exactly 2 sentences about rivers. Separate them with newlines.", lambda t: len([line for line in t.strip().split("\n") if line.strip()]) == 2),
    ("Reply with only the word NO if 9 is prime, or YES otherwise.", lambda t: t.strip().upper() == "NO"),
]

HELDOUT_CODE = [
    (
        ("Write a Python function count_vowels(s) that returns the number of "
        "vowels (a, e, i, o, u, case-insensitive) in the string s. Answer with "
        "a single Python code block only."),
        (
            "from solution import count_vowels\n"
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('WORLD') == 1\n"
            "assert count_vowels('xyz') == 0\n"
            "assert count_vowels('') == 0\n"
            "assert count_vowels('AeiOu') == 5\n"
            "print('PASS')\n"
        ),
    ),
]


@task
def heldout_eval(base):
    results = []
    for q, gold in HELDOUT_MATH:
        try:
            out, usage, dt = call(base, [{"role": "user", "content": q}])
            pred = extract_boxed(out) or re.search(r"-?\d+(?:\.\d+)?", out.split("\n")[-1])
            pred_val = pred if isinstance(pred, str) else (pred.group() if pred else None)
            ok = pred_val is not None and abs(float(pred_val.replace(",", "")) - float(gold)) < 1e-4
            results.append({"bucket": "heldout-math", "task": q[:40], "ok": bool(ok), "gold": gold, "pred": pred_val, "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2)})
        except Exception as e:
            results.append({"bucket": "heldout-math", "task": q[:40], "ok": False, "err": str(e)[:200]})
    for q, check in HELDOUT_INSTR:
        try:
            out, usage, dt = call(base, [{"role": "user", "content": q}], max_tokens=256)
            results.append({"bucket": "heldout-instr", "task": q[:40], "ok": bool(check(out)), "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2)})
        except Exception as e:
            results.append({"bucket": "heldout-instr", "task": q[:40], "ok": False, "err": str(e)[:200]})
    for prompt, harness in HELDOUT_CODE:
        try:
            out, usage, dt = call(base, [{"role": "user", "content": prompt}], max_tokens=512 + REASONING_MARGIN)
            ok, detail = _run_coding_task(out, harness)
            results.append({"bucket": "heldout-code", "task": harness.split("\n")[0][:40], "ok": bool(ok), "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2), "detail": detail, "finish_reason": usage.get("finish_reason"), "answer_bytes": usage.get("answer_bytes"), "reasoning_bytes": usage.get("reasoning_bytes")})
        except Exception as e:
            results.append({"bucket": "heldout-code", "task": prompt[:40], "ok": False, "err": str(e)[:200]})
    return results


# ---------- LONG-CONTEXT RETRIEVAL (synthetic needle, deterministic) --------
SENTENCE = ("The warehouse ledger records daily grain deliveries. Rows repeat "
            "with minor variations in tonnage and dock number.")
SENTENCE_APPROX_TOKENS = 27  # ~1.35 tokens/word x 20 words; refined by usage


def _average_prompt_tokens(samples):
    known = [s for s in samples if isinstance(s.get("prompt_tokens"), int)]
    if not known:
        return None
    return sum(s["prompt_tokens"] for s in known) // len(known)


def needle_task(base, ctx_tokens_approx=2000, position_fraction=0.5):
    needle = f"The access code for vault {ctx_tokens_approx} is ZX-Q7-1943."
    total_reps = max(1, round(ctx_tokens_approx / SENTENCE_APPROX_TOKENS))
    pre_reps = max(0, round(total_reps * position_fraction))
    post_reps = max(0, total_reps - pre_reps)
    pre = "\n".join(SENTENCE for _ in range(pre_reps))
    post = "\n".join(SENTENCE for _ in range(post_reps))
    q = f"{pre}\n\n{needle}\n\n{post}\n\nWhat is the access code for vault {ctx_tokens_approx}? Answer with the code only."
    try:
        # Reasoning tokens precede the needle answer (see REASONING_MARGIN).
        out, usage, dt = call(base, [{"role": "user", "content": q}], max_tokens=512 + REASONING_MARGIN)
        ok = "ZX-Q7-1943" in out
        return [{"bucket": "longctx", "task": f"needle@{ctx_tokens_approx}:pos{position_fraction}", "ok": bool(ok), "prompt_tokens": usage.get("prompt_tokens"), "tokens": usage.get("completion_tokens"), "latency_s": round(dt, 2), "answer": out[-120:], "finish_reason": usage.get("finish_reason")}]
    except Exception as e:
        return [{"bucket": "longctx", "task": f"needle@{ctx_tokens_approx}:pos{position_fraction}", "ok": False, "err": str(e)[:200]}]


@task
def longctx_eval(base):
    results = (needle_task(base, 2000, 0.25) + needle_task(base, 4000, 0.5)
               + needle_task(base, 8000, 0.75))
    avg = _average_prompt_tokens(results)
    for r in results:
        r["prompt_tokens_target_approx"] = avg
        r["label"] = f"needle@{r['task'].split('@')[1].split(':')[0]}:avg{avg}"
    return results


BUCKETS = {"math": math_eval, "instruction": instruction_eval,
           "coding": coding_eval, "heldout": heldout_eval,
           "longctx": longctx_eval}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8199")
    ap.add_argument("--buckets", default="math,instruction,coding,longctx")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    all_results = []
    for b in a.buckets.split(","):
        all_results.extend(BUCKETS[b](a.base))
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in all_results)
    ok_count = sum(1 for r in all_results if r.get("ok"))
    print(json.dumps({"total": len(all_results), "passed": ok_count, "rate": round(ok_count/len(all_results), 3) if all_results else 0}))
    for r in all_results:
        print(("PASS" if r.get("ok") else "FAIL"), r["bucket"], r["task"], r.get("err", ""))
