#!/usr/bin/env python3
"""
Ollama vs vLLM load generator: one client, same API, three traffic shapes.

  batch : fire C identical requests at once, wait for all (classic concurrency sweep)
  users : C virtual users in a closed loop: request -> think time -> request, for --duration s
  rate  : open loop, Poisson arrivals at R req/s for --duration s (a real traffic spike;
          shows the queueing cliff because arrivals do not slow down when the server does)

Every request goes to an OpenAI-compatible /v1/chat/completions endpoint with streaming.
Per-request records are written to  results/<label>/requests-<mode>-<level>-run<N>.jsonl
Per-run summaries are appended to   results/<label>/summary.jsonl

Examples:
  python loadgen.py --url http://gpu:11434/v1/chat/completions --model qwen3:8b-fp16 --label ollama-default --mode batch --levels 1,4,8,16,32 --runs 3
  python loadgen.py --url http://gpu:8000/v1/chat/completions  --model qwen3-8b --label vllm-bf16 --mode users --levels 4,8,16,32 --duration 120
  python loadgen.py --url http://gpu:8000/v1/chat/completions  --model qwen3-8b --label vllm-bf16 --mode rate  --levels 0.5,1,2,4,8 --duration 90
"""
import argparse
import asyncio
import json
import random
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = HERE / "prompts.json"


# ----------------------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def med(xs):
    return st.median(xs) if xs else None


def r(x, n=1):
    return None if x is None else round(x, n)


def detect_engine(base: str):
    """Return (engine, version). Ollama exposes /api/version, vLLM exposes /version."""
    try:
        j = httpx.get(f"{base}/api/version", timeout=5).json()
        if "version" in j:
            return "ollama", j["version"]
    except Exception:
        pass
    try:
        j = httpx.get(f"{base}/version", timeout=5).json()
        if "version" in j:
            return "vllm", j["version"]
    except Exception:
        pass
    return "unknown", "?"


def load_prompts(path: Path, mix: str):
    pool = json.loads(path.read_text(encoding="utf-8"))
    if mix == "fixed":
        fixed = [p for p in pool if p["id"] == "fixed-500"]
        if not fixed:
            sys.exit("prompts.json has no 'fixed-500' prompt")
        return fixed
    weights = {"short": 0.3, "medium": 0.5, "long": 0.2}
    return [(p, weights.get(p.get("size"), 0.0)) for p in pool if p["id"] != "fixed-500"]


def pick_prompt(prompts, rng: random.Random):
    if isinstance(prompts[0], dict):
        return prompts[0]
    items, weights = zip(*prompts)
    return rng.choices(items, weights=weights, k=1)[0]


# ----------------------------------------------------------------------------- one request
async def one_request(client: httpx.AsyncClient, cfg, prompt, user_id: int, seq: int):
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": prompt.get("system", "You are a helpful assistant.")},
            {"role": "user", "content": prompt["user"]},
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Qwen3 "thinking" must be off in BOTH engines or TTFT / throughput are not comparable.
    # Verified on Ollama 0.40 / vLLM 0.33: Ollama's OpenAI endpoint honours ONLY reasoning_effort="none"
    # (think=false and /no_think are ignored there); vLLM honours chat_template_kwargs.enable_thinking=false.
    if cfg.no_think == "template":
        if getattr(cfg, "engine", "") == "ollama":
            body["reasoning_effort"] = "none"
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
    elif cfg.no_think == "soft":
        body["messages"][0]["content"] += " /no_think"              # Qwen3 soft switch, works everywhere

    rec = {
        "user": user_id, "seq": seq, "prompt_id": prompt["id"], "size": prompt.get("size"),
        "t_start": time.time(), "ttft": None, "total": None, "out_tokens": None,
        "in_tokens": None, "chunks": 0, "error": None, "status": None, "saw_think": False,
    }
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", cfg.url, json=body, timeout=cfg.timeout) as resp:
            rec["status"] = resp.status_code
            if resp.status_code != 200:
                body_bytes = await resp.aread()
                rec["error"] = f"http {resp.status_code}: {body_bytes[:200]}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if j.get("usage"):
                        u = j["usage"]
                        rec["out_tokens"] = u.get("completion_tokens", rec["out_tokens"])
                        rec["in_tokens"] = u.get("prompt_tokens", rec["in_tokens"])
                    ch = j.get("choices") or []
                    if not ch:
                        continue
                    d = ch[0].get("delta") or {}
                    reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
                    txt = d.get("content") or reasoning
                    if txt:
                        if rec["ttft"] is None:
                            rec["ttft"] = time.perf_counter() - t0
                        if reasoning or "<think>" in txt:
                            rec["saw_think"] = True
                        rec["chunks"] += 1
    except Exception as e:  # timeouts, connection resets, OOM-kills land here
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["total"] = time.perf_counter() - t0
    if rec["out_tokens"] is None and rec["chunks"]:
        rec["out_tokens"] = rec["chunks"]          # Ollama fallback: ~1 token per chunk
    return rec


# ----------------------------------------------------------------------------- traffic shapes
async def shape_batch(client, cfg, level, prompts, rng):
    return await asyncio.gather(*[
        one_request(client, cfg, pick_prompt(prompts, rng), i, 0) for i in range(int(level))
    ])


async def shape_users(client, cfg, level, prompts, rng):
    deadline = time.perf_counter() + cfg.duration
    recs = []

    async def user(uid):
        urng = random.Random(rng.random())
        seq = 0
        await asyncio.sleep(urng.uniform(0, min(2.0, cfg.think)))     # stagger the start
        while time.perf_counter() < deadline:
            recs.append(await one_request(client, cfg, pick_prompt(prompts, urng), uid, seq))
            seq += 1
            if cfg.think > 0:
                await asyncio.sleep(urng.expovariate(1.0 / cfg.think))

    await asyncio.gather(*[user(i) for i in range(int(level))])
    return recs


async def shape_rate(client, cfg, level, prompts, rng):
    deadline = time.perf_counter() + cfg.duration
    tasks, seq = [], 0
    while time.perf_counter() < deadline:
        tasks.append(asyncio.create_task(
            one_request(client, cfg, pick_prompt(prompts, rng), seq % 1000, seq)))
        seq += 1
        await asyncio.sleep(rng.expovariate(float(level)))   # Poisson arrivals at `level` req/s
    return await asyncio.gather(*tasks)


SHAPES = {"batch": shape_batch, "users": shape_users, "rate": shape_rate}


# ----------------------------------------------------------------------------- one run
async def run_level(cfg, level, run_idx, prompts, engine, version):
    rng = random.Random(f"{cfg.seed}-{cfg.mode}-{level}-{run_idx}")   # identical traffic on both engines
    limits = httpx.Limits(max_connections=max(64, int(level) * 4 + 8), max_keepalive_connections=64)
    started_at, t0_epoch = now_iso(), time.time()
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        recs = await SHAPES[cfg.mode](client, cfg, level, prompts, rng)
        wall = time.perf_counter() - t0
    ended_at, t1_epoch = now_iso(), time.time()

    ok = [x for x in recs if x["error"] is None and x["ttft"] is not None]
    n_err = len(recs) - len(ok)
    ttft = [x["ttft"] for x in ok]
    tot = [x["total"] for x in ok]
    outt = [x["out_tokens"] or 0 for x in ok]
    tpot = [(x["total"] - x["ttft"]) / (x["out_tokens"] - 1) for x in ok if (x["out_tokens"] or 0) > 1]
    per_req = [x["out_tokens"] / (x["total"] - x["ttft"]) for x in ok if (x["total"] - x["ttft"]) > 0]

    summary = {
        "label": cfg.label, "engine": engine, "engine_version": version, "model": cfg.model,
        "mode": cfg.mode, "level": level, "run": run_idx,
        "started_at": started_at, "ended_at": ended_at,
        "t0_epoch": round(t0_epoch, 3), "t1_epoch": round(t1_epoch, 3), "wall_s": r(wall, 2),
        "n_req": len(recs), "n_ok": len(ok), "n_err": n_err,
        "ttft_med_ms": r(med(ttft) * 1000 if ttft else None, 0),
        "ttft_p95_ms": r(pct(ttft, 95) * 1000 if ttft else None, 0),
        "ttft_p99_ms": r(pct(ttft, 99) * 1000 if ttft else None, 0),
        "tpot_med_ms": r(med(tpot) * 1000 if tpot else None, 1),
        "lat_med_s": r(med(tot), 2), "lat_p95_s": r(pct(tot, 95), 2), "lat_p99_s": r(pct(tot, 99), 2),
        "agg_tok_s": r(sum(outt) / wall if wall else None, 1),
        "req_s": r(len(ok) / wall if wall else None, 2),
        "per_req_tok_s": r(med(per_req), 1),
        "in_tok_med": med([x["in_tokens"] for x in ok if x["in_tokens"]]),
        "out_tok_med": med(outt),
        "saw_think": any(x["saw_think"] for x in ok),
        "max_tokens": cfg.max_tokens, "prompt_mix": cfg.prompt_mix,
        "think_s": cfg.think if cfg.mode == "users" else None,
        "duration_s": cfg.duration if cfg.mode != "batch" else None,
    }
    out_dir = Path(cfg.out) / cfg.label
    out_dir.mkdir(parents=True, exist_ok=True)
    lvl = str(level).replace(".", "p")
    with open(out_dir / f"requests-{cfg.mode}-{lvl}-run{run_idx}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in recs))
    with open(out_dir / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="http://HOST:PORT/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="e.g. ollama-default | ollama-tuned | vllm-bf16")
    ap.add_argument("--mode", choices=list(SHAPES), default="batch")
    ap.add_argument("--levels", default="1,4,8,16", help="batch: concurrency | users: virtual users | rate: req/s")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--duration", type=float, default=120, help="seconds per level (users / rate)")
    ap.add_argument("--think", type=float, default=3.0, help="mean think time between a user's requests (users)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-mix", choices=["fixed", "realistic"], default="fixed")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--no-think", choices=["template", "soft", "none"], default="template")
    ap.add_argument("--seed", default="simutalk")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    ap.add_argument("--no-warmup", action="store_true")
    cfg = ap.parse_args()

    base = cfg.url.split("/v1/")[0]
    engine, version = detect_engine(base)
    cfg.engine = engine
    prompts = load_prompts(Path(cfg.prompts), cfg.prompt_mix)
    print(f"# engine={engine} version={version} model={cfg.model} mode={cfg.mode} label={cfg.label}",
          file=sys.stderr)

    if not cfg.no_warmup:
        async with httpx.AsyncClient() as client:
            w = await one_request(client, cfg, pick_prompt(prompts, random.Random(0)), 0, -1)
        if w["error"]:
            sys.exit(f"warm-up failed: {w['error']}")
        print(f"# warm-up ok: ttft={w['ttft']*1000:.0f}ms tokens={w['out_tokens']} saw_think={w['saw_think']}",
              file=sys.stderr)
        if w["saw_think"]:
            print("# WARNING: model emitted thinking tokens; results are not comparable. Re-run with --no-think soft",
                  file=sys.stderr)

    levels = [float(x) if "." in x else int(x) for x in cfg.levels.split(",")]
    for lvl in levels:
        for i in range(1, cfg.runs + 1):
            s = await run_level(cfg, lvl, i, prompts, engine, version)
            print(json.dumps(s), flush=True)
            await asyncio.sleep(5)       # let the server queue drain between runs


if __name__ == "__main__":
    asyncio.run(main())
