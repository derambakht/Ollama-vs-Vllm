#!/usr/bin/env python3
"""
Sanity check one server before benchmarking it. Verifies the things that silently
ruin a comparison:

  1. the server answers and reports its version
  2. Qwen3 "thinking" is OFF (no <think> block, no reasoning field)
  3. the fixed prompt is ~500 input tokens on THIS engine (tokenizer agreement)
  4. streaming chunk count vs usage.completion_tokens (Ollama fallback accuracy)
  5. the answer is sane (non-empty, deterministic at temperature 0)

Usage:
  python sanity.py --url http://gpu:11434/v1/chat/completions --model qwen3:8b-fp16
  python sanity.py --url http://gpu:8000/v1/chat/completions  --model qwen3-8b
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-think", choices=["template", "soft", "none"], default="template")
    a = ap.parse_args()

    base = a.url.split("/v1/")[0]
    engine, version = "unknown", "?"
    for path, name in (("/api/version", "ollama"), ("/version", "vllm")):
        try:
            j = httpx.get(base + path, timeout=5).json()
            if "version" in j:
                engine, version = name, j["version"]
                break
        except Exception:
            pass
    print(f"[1] engine={engine} version={version}")

    pool = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    p = next(x for x in pool if x["id"] == "fixed-500")
    msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
    body = {"model": a.model, "messages": msgs, "max_tokens": 64, "temperature": 0, "stream": False}
    if a.no_think == "template":
        if engine == "ollama":
            body["reasoning_effort"] = "none"                     # the only switch Ollama's OpenAI endpoint honours
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
    elif a.no_think == "soft":
        msgs[0]["content"] += " /no_think"

    ok = True
    r1 = httpx.post(a.url, json=body, timeout=300)
    if r1.status_code != 200:
        sys.exit(f"[!] HTTP {r1.status_code}: {r1.text[:300]}")
    j = r1.json()
    msg = j["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = j.get("usage") or {}
    thinks = ("<think>" in text and "</think>" not in text[:20]) or bool(reasoning.strip())
    print(f"[2] thinking_off={'YES' if not thinks else 'NO  <-- fix this (try --no-think soft)'}")
    if thinks:
        ok = False
    print(f"[3] prompt_tokens={usage.get('prompt_tokens')}  (expect roughly 450-550 for fixed-500)")
    print(f"    completion_tokens={usage.get('completion_tokens')}  first 120 chars: {text[:120]!r}")
    if not text.strip():
        print("    [!] empty answer")
        ok = False

    # streaming: chunks vs usage
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    chunks, comp = 0, None
    with httpx.stream("POST", a.url, json=body, timeout=300) as r:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            jj = json.loads(d)
            if jj.get("usage"):
                comp = jj["usage"].get("completion_tokens")
            ch = jj.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                chunks += 1
    if comp is None:
        print(f"[4] stream usage: MISSING -> loadgen will count chunks ({chunks}); verify vs non-stream count above")
    else:
        diff = abs(chunks - comp) / max(comp, 1) * 100
        print(f"[4] stream usage: completion_tokens={comp} chunks={chunks} diff={diff:.1f}% "
              f"({'ok' if diff < 5 else 'LARGE, report it'})")

    r2 = httpx.post(a.url, json={**body, "stream": False, "stream_options": None}, timeout=300).json()
    same = (r2["choices"][0]["message"].get("content") or "") == text
    print(f"[5] deterministic at temperature 0: {'yes' if same else 'no (acceptable, but note it)'}")
    print("RESULT:", "OK" if ok else "FIX BEFORE BENCHMARKING")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
