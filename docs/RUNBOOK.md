# Ollama vs vLLM Benchmark Runbook

> **About this document.** This is the original manual plan for the benchmark, dated 2026-11-03. The repository automates it with [`run-all.sh`](../run-all.sh) and extends it. Where the executed run differs from this plan, the run is authoritative:
>
> - `bench.py` below became [`bench/loadgen.py`](../bench/loadgen.py). It adds closed-loop virtual users and open-loop Poisson arrivals, and sweeps concurrency up to 32.
> - The run used a realistic mix of short, medium and long prompts. The single fixed ~500-token prompt is still available as `fixed-500` in [`bench/prompts.json`](../bench/prompts.json) and is used by `sanity.py`.
> - vLLM 0.33.0 ran as a native pip install, not in Docker, because RunPod pods have no Docker daemon. See [RUNPOD.md](RUNPOD.md).
> - Qwen3 thinking must be disabled explicitly. Ollama's OpenAI endpoint needs `reasoning_effort: "none"`; vLLM needs `chat_template_kwargs.enable_thinking=false`.
> - Ollama 0.40's default context is 40,960 tokens, not the 4,096 assumed in section 3.
> - Two items in section 7 were not run: a second workload on the freed half of the card (7b), and the optional llama.cpp server.
>
> Results: [REPORT-2026-11-03.md](REPORT-2026-11-03.md).

**Goal:** repeat the comparison from the earlier vLLM post on your own RTX PRO 6000 Blackwell, and get real numbers for Part 2. The numbers needed are single-user speed, TTFT, and throughput at 1, 4, 8 and 16 concurrent requests, plus VRAM use and behaviour under load.

## 1. Test rules (before anything else)

The earlier post was credible because it used the same card and the same model. This test can only be published if the following stay identical between the two servers:

| Variable | Fixed value | Why |
|---|---|---|
| Model | Qwen3-8B (or whichever Qwen you serve) | One model, two engines |
| Weights | BF16 on both (Ollama: tag `:8b-fp16`; vLLM: the original Hugging Face repo) | In the earlier post Ollama ran 4-bit and vLLM ran BF16. Make the comparison fair this time. The 4-bit test is separate, in section 7. |
| Prompt | One fixed text of ~500 input tokens | The real SimuTalk workload: medium-length context |
| max_tokens | 256 output tokens | Enough to measure decode |
| temperature | 0 | Stable output |
| Concurrency | 1, 4, 8, 16 | 1 is the single-user case; the rest are the real-world case |
| Repeats | 3 runs per level; report the median | One run is noise |
| GPU | Only the server under test on the GPU, no other model | vLLM at 0.9 memory utilization starves everything else |

Three numbers come out of every run:

1. **TTFT** (time to first token): median and p95, in milliseconds.
2. **Decode throughput**: output tokens per second, summed over all concurrent requests.
3. **Latency per request**: total time of each request, median and p95.

In addition, record VRAM use and GPU temperature during each run (section 6).

One important rule: every settings change (for example `OLLAMA_NUM_PARALLEL`) is a separate run with its own name. Never save a number without the name of its configuration.

## 2. Preparing the environment

Assumptions: Ubuntu 22.04 or 24.04, NVIDIA driver 570 or newer (required for Blackwell, SM120), and Docker with the NVIDIA Container Toolkit. On Windows with WSL2 all commands are the same, but label the numbers "WSL2".

Initial check:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv
# expected: NVIDIA RTX PRO 6000 Blackwell ..., 97887 MiB
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Client-side tools can run on the same machine or on another machine in the LAN. Another machine is preferable, so the client's CPU does not load the server.

```bash
python3 -m venv ~/bench && source ~/bench/bin/activate
pip install httpx numpy
```

Model: get the **BF16 weights of one model** for both servers.

```bash
# Ollama (fp16 weights, not the default 4-bit tag)
ollama pull qwen3:8b-fp16

# vLLM (downloads from Hugging Face on first start)
# repo name: Qwen/Qwen3-8B
export HF_HOME=/data/hf   # somewhere with at least 40 GB free
```

If the model you actually serve is a different Qwen (for example 14B or 32B), use that one. Make sure it is the **same** on both servers, and write it down exactly in the report. BF16 weights take about 2 bytes per parameter: 8B is about 16 GB and 32B is about 65 GB.

## 3. Setting up Ollama

Test two configurations separately: **default** (what you started with) and **tuned** (what the earlier post called "tuned Ollama"). Without the second, critics will say the comparison was unfair.

**A — default:**

```bash
systemctl stop ollama 2>/dev/null
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
sleep 3
# warm up, and check that the model runs on the GPU
curl -s localhost:11434/api/generate -d '{"model":"qwen3:8b-fp16","prompt":"hi","stream":false}' | head -c 300
ollama ps   # the PROCESSOR column must show 100% GPU
```

**B — tuned (for concurrency):**

```bash
pkill ollama
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_NUM_PARALLEL=16 \
OLLAMA_KEEP_ALIVE=-1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=f16 \
ollama serve &
```

What each variable does:

| Variable | Effect |
|---|---|
| `OLLAMA_NUM_PARALLEL=16` | 16 concurrent slots. Each slot reserves its own context; with a default `num_ctx` of 4,096 that is 65k tokens of KV cache in total. |
| `OLLAMA_KEEP_ALIVE=-1` | The model is never unloaded. Otherwise the TTFT of the first request includes loading the model. |
| `OLLAMA_FLASH_ATTENTION=1` | Less KV memory, and more speed at long context |
| `OLLAMA_KV_CACHE_TYPE=f16` | Matches vLLM, which keeps the KV cache in FP16/BF16 |

After every change of an environment variable, check `ollama ps` and keep the `nvidia-smi` output for the report. A sentence such as "with 16 slots, Ollama took X GB at start-up" is itself a number for the post.

Endpoint for the benchmark: Ollama serves an OpenAI-compatible API at `http://HOST:11434/v1/chat/completions`. The script in section 5 calls this same API on both servers, so the client code stays identical.

## 4. Setting up vLLM

Stop Ollama completely (`pkill ollama`) and confirm with `nvidia-smi` that the VRAM is empty. vLLM competes with anything else that is on the card.

```bash
docker run -d --name vllm --gpus all --ipc=host \
  -p 8000:8000 \
  -v $HF_HOME:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 \
  --served-model-name qwen3-8b

docker logs -f vllm   # wait for "Application startup complete"
```

On Blackwell (SM120), if the `latest` image fails with a kernel error, try a specific recent tag built with CUDA 12.8 (for example `v0.11.x` or newer). **Write the exact version in the report**; a commenter asked for exactly this.

What each flag does, and why this value:

| Flag | Value | Why |
|---|---|---|
| `--max-model-len` | 8192 | What you actually serve. The 128k on the model card means a huge KV cache and possibly a failed start. |
| `--gpu-memory-utilization` | 0.85 | First test at 0.85; section 7 repeats it at 0.5 to leave room for evaluation jobs |
| `--max-num-seqs` | 64 | Upper limit of concurrent requests in one batch; above our 16-request test |
| `--dtype bfloat16` | — | Matches Ollama's fp16 |

After start-up, find these log lines and keep them. They go straight into the post:

```
INFO ... Model loading took X.XX GiB
INFO ... # GPU blocks: NNNNN   (or "Available KV cache memory: X GiB")
INFO ... Maximum concurrency for 8192 tokens per request: NN.Nx
```

The last line says how many full-length requests fit at the same time with this VRAM and this context. It is the best number for the "where does max-model-len go" section of Part 2.

Health check:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
curl -s localhost:8000/metrics | grep -E 'vllm:(num_requests_running|gpu_cache_usage_perc)'
```

## 5. Benchmark script

One script, both servers, the same streaming OpenAI-compatible API. TTFT is taken from the first chunk that has content. Output tokens are taken from the `usage` field, or from the number of chunks if `usage` is missing. Save it as `bench.py`:

```python
#!/usr/bin/env python3
"""Ollama vs vLLM concurrency benchmark. Same prompt, same client, N parallel requests."""
import argparse, asyncio, json, time, statistics as st
import httpx

PROMPT = (
    "You are a technical writer. Explain, in about 300 words, how a streaming "
    "speech-to-speech translation pipeline (ASR -> LLM -> TTS) keeps end-to-end "
    "latency under 1.5 seconds. Cover chunking, early commit, and KV cache reuse."
)

async def one(client, url, model, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter(); ttft = None; chunks = 0; out_tokens = None
    async with client.stream("POST", url, json=body, timeout=600) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                out_tokens = j["usage"].get("completion_tokens")
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    total = time.perf_counter() - t0
    return {"ttft": ttft, "total": total, "tokens": out_tokens or chunks}

async def run(url, model, conc, max_tokens):
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(client, url, model, max_tokens) for _ in range(conc)])
        wall = time.perf_counter() - t0
    ttft = [r["ttft"] for r in res]; tot = [r["total"] for r in res]
    tokens = sum(r["tokens"] for r in res)
    p95 = lambda xs: sorted(xs)[max(0, int(len(xs) * 0.95) - 1)]
    return {
        "conc": conc,
        "ttft_med_ms": round(st.median(ttft) * 1000),
        "ttft_p95_ms": round(p95(ttft) * 1000),
        "lat_med_s": round(st.median(tot), 2),
        "lat_p95_s": round(p95(tot), 2),
        "agg_tok_s": round(tokens / wall, 1),
        "per_req_tok_s": round(st.median([r["tokens"] / (r["total"] - (r["ttft"] or 0)) for r in res]), 1),
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)      # http://host:11434/v1/chat/completions  or :8000
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)    # e.g. ollama-default, ollama-tuned, vllm-bf16
    ap.add_argument("--conc", default="1,4,8,16")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    await run(a.url, a.model, 1, 32)            # warm-up
    rows = []
    for c in [int(x) for x in a.conc.split(",")]:
        for i in range(a.runs):
            r = await run(a.url, a.model, c, a.max_tokens)
            r.update(label=a.label, run=i + 1)
            rows.append(r); print(json.dumps(r), flush=True)
            await asyncio.sleep(5)              # let the queue drain
    with open(f"results-{a.label}.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))

asyncio.run(main())
```

Running it, with only one server on the GPU at a time:

```bash
# Ollama default (section 3-A)
python bench.py --url http://GPU_HOST:11434/v1/chat/completions --model qwen3:8b-fp16 --label ollama-default
# Ollama tuned (section 3-B)
python bench.py --url http://GPU_HOST:11434/v1/chat/completions --model qwen3:8b-fp16 --label ollama-tuned
# vLLM (section 4)
python bench.py --url http://GPU_HOST:8000/v1/chat/completions --model qwen3-8b --label vllm-bf16
```

Output columns:

| Column | Meaning | What it becomes in the post |
|---|---|---|
| `ttft_med_ms` / `ttft_p95_ms` | Time to first token | "Time to first token: X vs Y" |
| `agg_tok_s` | Output tokens per second, summed over all requests | The main concurrency number (the equivalent of "342 vs 124 tok/s") |
| `per_req_tok_s` | Decode speed of each request on its own | At conc=1, the equivalent of "134 vs 89 tok/s" |
| `lat_p95_s` | How long the worst-off user waited | The sentence "latency climbed with every extra person", with a number |

Note: if Ollama does not include `usage` in the stream, the script counts chunks, which is roughly the number of tokens. For full accuracy, send one non-streaming request and compare its `eval_count` with the chunk count. A difference under 5% can be ignored.

## 6. Running the tests

In a second terminal on the GPU server, start the monitor before each label and leave it running until the end:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw \
  --format=csv -l 1 > gpu-$LABEL.csv
```

Order of the runs, each about 10 to 15 minutes:

1. Ollama default: `bench.py --label ollama-default`
2. Ollama tuned, restarted with the variables from section 3-B: `--label ollama-tuned`
3. `pkill ollama`, then check with `nvidia-smi` that the VRAM is free
4. vLLM from section 4: `--label vllm-bf16`
5. After each label, take from its `gpu-*.csv` file the maximum `memory.used` and `temperature.gpu`, and the minimum `clocks.sm` under the 16-request load. If the clock dropped under load while the temperature was above about 83 °C, the card was thermally throttling. That is the promised throttling post, but measured on the server.

Build the raw table from the JSONL files, taking the median of 3 runs:

```bash
pip install pandas
python - <<'EOF'
import pandas as pd, glob
df = pd.concat(pd.read_json(f, lines=True) for f in glob.glob("results-*.jsonl"))
print(df.groupby(["label","conc"])[["ttft_med_ms","per_req_tok_s","agg_tok_s","lat_p95_s"]].median().round(1).to_markdown())
EOF
```

The table to fill in, which is also the basis for the post image:

| label | conc | TTFT med (ms) | per-req tok/s | aggregate tok/s | p95 latency (s) | VRAM peak (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| ollama-default | 1 |  |  |  |  |  |
| ollama-default | 4 |  |  |  |  |  |
| ollama-default | 16 |  |  |  |  |  |
| ollama-tuned | 1 |  |  |  |  |  |
| ollama-tuned | 4 |  |  |  |  |  |
| ollama-tuned | 16 |  |  |  |  |  |
| vllm-bf16 | 1 |  |  |  |  |  |
| vllm-bf16 | 4 |  |  |  |  |  |
| vllm-bf16 | 16 |  |  |  |  |  |

If `ollama-default` at conc=16 shows strange numbers, such as a p95 of several tens of seconds, they are correct: that is the queue. That number is the pain point of the post, so do not remove it.

## 7. Additional tests for Part 2

The comments raised three open questions, and each needs one extra run. All of them use the same `bench.py`; only the server changes.

**7a — AWQ vs BF16, and the Marlin kernel** (a commenter's question)

```bash
# without --quantization, so vLLM picks awq_marlin itself
docker run -d --name vllm-awq --gpus all --ipc=host -p 8000:8000 -v $HF_HOME:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B-AWQ --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --max-num-seqs 64 --served-model-name qwen3-8b
docker logs vllm-awq 2>&1 | grep -i -E 'quantization|marlin|awq'   # which kernel was chosen; this line goes into the post
python bench.py --url ... --model qwen3-8b --label vllm-awq-marlin

# the same with an explicit --quantization awq (the plain kernel)
... --quantization awq ...   → --label vllm-awq-plain
```

Expected result: Marlin should beat BF16 at conc=1, because there is less weight data to read. The plain kernel may fall behind. Whatever the result is, write it down.

**7b — evaluation and serving on one card** (another commenter's question)

Start vLLM with `--gpu-memory-utilization 0.5`, which leaves half the card free for evaluation, Whisper or TTS. Record two things:

- The log line `Maximum concurrency for 8192 tokens per request` at 0.85 and at 0.5. The difference is exactly the price of sharing the card.
- `bench.py --label vllm-bf16-u050` at conc=16. Does it still fit, or do requests get preempted? Check with `curl localhost:8000/metrics | grep preempt`.

Then, during the same run, start a real evaluation job, or just Whisper, on the same card, and see how much TTFT degrades. That number answers the question.

**7c — out-of-memory and preemption at peak**

```bash
python bench.py --url ... --model qwen3-8b --label vllm-peak --conc 32,64 --runs 1 --max-tokens 512
watch -n1 'curl -s localhost:8000/metrics | grep -E "num_requests_(running|waiting)|gpu_cache_usage|preemptions"'
```

Look for the first point where `num_requests_waiting` rises above zero or `preemptions_total` increases. That concurrency is the real ceiling of the card at this context length. If the container crashes, keep the `docker logs` output: that is the post's "OOM on Friday".

**Optional — llama.cpp server** (two people asked): if there is time, run `llama-server -m qwen3-8b-f16.gguf -ngl 99 -c 8192 -np 16 -fa --port 8000` and the same `bench.py --label llamacpp`. Note that `-np 16` with `-c 8192` gives each slot only 512 tokens. To be fair, use `-c 131072` (16 × 8,192). That detail alone is a paragraph of the post.

## 8. From the table to the post

Every `[…]` blank in the Part 2 draft is filled from one of these sources:

| Section of the post | Source of the number |
|---|---|
| "gpu-memory-utilization is a decision" | The `Model loading took` log line from section 4, plus the two `Maximum concurrency` numbers from section 7b (0.85 vs 0.5) |
| "max-model-len is where KV cache goes" | From section 4: `# GPU blocks` or `Available KV cache memory` at 8,192. Start vLLM once more with 32,768 and take the second number. |
| "OOM on Friday" | From section 7c: the first concurrency at which waiting > 0, and the p95 at that point |
| "AWQ mystery resolved" | From section 7a: `per_req_tok_s` at conc=1 for bf16, awq-marlin and awq-plain, plus the kernel log line |
| The answer on sharing the card | From section 7b: TTFT with and without an evaluation job on the same card |
| Post image | From section 6: `agg_tok_s` at conc 1, 4 and 16 for the three labels, in the same style as the first post's chart |

In the post, always state the exact model, the vLLM and Ollama versions, the driver, the OS, and that BF16 was used on both engines. Last time, the missing details produced the best comments. This time the numbers are your own, and they must be reproducible.

Total time: about 1 hour of preparation (mostly downloads), about 45 minutes for section 6, and about 1 hour for section 7. One afternoon.
