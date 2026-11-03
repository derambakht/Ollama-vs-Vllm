# Test plan

Everything below is run by `./run-all.sh`; this page is the "why" so the post can defend each number.

## Fixed across every run

| Variable | Value | Reason |
|---|---|---|
| Model | Qwen3-8B, 16-bit on both engines (`qwen3:8b-fp16` / `Qwen/Qwen3-8B` BF16) | one model, two engines; Part 1 compared 4-bit with BF16 and the comments noticed |
| Thinking | OFF on both (Ollama: `reasoning_effort: "none"`; vLLM: `chat_template_kwargs.enable_thinking=false`; verified by `sanity.py` and by `saw_think` on every run) | Qwen3 thinking tokens would make TTFT and output length incomparable |
| Context budget | 8192 on both (`--max-model-len 8192` / `OLLAMA_CONTEXT_LENGTH=8192` in tuned) | KV cache size is the thing being compared |
| Prompts | same seeded sequence on both engines (`--seed simutalk`) | identical traffic, request for request |
| max_tokens | 256, temperature 0 (512 for the peak probe) | long enough to measure decode, deterministic |
| Client | one script, OpenAI-compatible streaming API on both | same code path, same TTFT definition (first content chunk) |
| GPU | only the engine under test on the card | vLLM reserves `gpu-memory-utilization` and will fight anything else |
| Repeats | batch: 3 runs, median reported; users: one 120 s window per level; rate: one 60 s window per level | single runs are noise; the long windows of users and rate average over hundreds of requests |

## Configurations (labels)

| Label | Engine | What changes | Answers |
|---|---|---|---|
| `ollama-default` | Ollama | fresh-install defaults (+ keep-alive so model load is not measured) | "what you get out of the box" |
| `ollama-tuned` | Ollama | `NUM_PARALLEL=16`, `CONTEXT_LENGTH=8192`, flash attention, f16 KV | "was it a fair fight?" (the Part 1 criticism) |
| `vllm-bf16` | vLLM | `--max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 64` | the headline comparison |
| `vllm-awq-marlin` | vLLM | Qwen3-8B-AWQ, kernel auto (awq_marlin) | AWQ question from the comments |
| `vllm-awq-plain` | vLLM | same, `--quantization awq` forced | whether the plain kernel is still selectable (in vLLM 0.33 it is not: Marlin is used anyway) |
| `vllm-bf16-u050` | vLLM | `--gpu-memory-utilization 0.5` | "can eval and serving share one card?" |
| `vllm-peak` | vLLM | conc 32/64/96, 512 tokens, scrape `/metrics` | where waiting > 0 and preemptions start |
| `vllm-ctx32k` | vLLM | start-up log only, `--max-model-len 32768` | "max-model-len is where the KV cache goes" |

## Traffic shapes (per label)

1. **batch** (`--mode batch --levels 1,4,8,16,32 --runs 3`): C requests fired together. Classic,
   comparable with Part 1's "16 concurrent" number. Produces per-request tok/s at conc 1 and aggregate tok/s.
2. **users** (`--mode users --levels 4,8,16,32 --duration 120 --think 3`): closed loop. Each virtual
   user sends, waits for the full answer, "thinks" ~3 s (exponential), and sends again.
   This is what a workshop with N people looks like.
   Metric that matters: TTFT p95 (the person is waiting with headphones on).
3. **rate** (`--mode rate --levels 0.5,1,2,4,8 --duration 60 --timeout 120`): open loop, Poisson arrivals.
   Arrivals do not slow down when the server does, so latency explodes past capacity. A request that
   receives no data for 120 s is abandoned and counted as an error ("dropped"). The chart of served
   req/s vs offered req/s shows each engine's real ceiling; the p95 chart shows the cliff.

Prompt mix is `realistic` by default: 30% short (60 to 71 tokens), 50% medium (286 to 391 tokens),
20% long (756 to 1,021 tokens), token counts as measured on vLLM. Set `PROMPT_MIX=fixed` in
`config.env` for the runbook's single ~500-token prompt (`fixed-500` in `bench/prompts.json`).

## Metrics (per run, in `summary.jsonl`)

| Field | Meaning |
|---|---|
| `ttft_med_ms`, `ttft_p95_ms`, `ttft_p99_ms` | time to first content token |
| `tpot_med_ms` | time per output token after the first (inter-token latency) |
| `per_req_tok_s` | median decode speed of one request (tokens / (total - ttft)) |
| `agg_tok_s` | sum of output tokens of all requests / wall time of the run |
| `req_s` | completed requests per second |
| `lat_med_s`, `lat_p95_s`, `lat_p99_s` | full request latency |
| `n_err` | failed requests (timeouts, HTTP errors, resets) |
| `vram_peak_gb`, `temp_max_c`, `clock_min_mhz`, `power_avg_w` | from `gpu.csv` inside the run's window (added by `analyze.py`) |
| `saw_think` | true if any response contained thinking tokens; must be false |

## Time budget (measured on an RTX PRO 6000, 2026-11-03)

| Step | Minutes |
|---|---|
| Setup and downloads (`runpod-setup.sh`: vLLM, kernels, Ollama, 8B in two formats, AWQ) | about 45 |
| Core: ollama-default 47 + ollama-tuned 28 + vllm-bf16 18 | 93 |
| Extras: vllm-awq-marlin 19 + vllm-awq-plain 17 + vllm-bf16-u050 18 + peak and ctx32k 3 | 57 |
| `analyze.py` | under 1 |
| **Total** | **about 3.5 h** |

Ollama default takes longest because it serves requests one at a time, so each queue drains slowly.

## Interpretation rules (write these in the post)

- Report the median of 3 for batch; show p95, never only the mean.
- If `n_err` > 0 at a level, say so; an engine that drops requests is not "faster".
- If `clock_min_mhz` falls > 15% below the conc=1 value, note thermal throttling and which chassis.
- If Ollama's `usage` field was missing in streams, the token count is chunk-based; `sanity.py` prints the error margin.
- Always print engine versions, driver, image digest, OS, and that thinking was off. `meta.json` has all of it.
