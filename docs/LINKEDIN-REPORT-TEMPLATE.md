# LinkedIn Part 2: fill-in template

Every `[...]` comes from `results/report/REPORT.md` (tables) or `results/report/charts/*.png`.
Attach 2-3 charts max: `batch-throughput.png`, `users-ttft-p95.png`, `rate-latency-p95.png`.

---

**Ollama vs vLLM, Part 2: same card, same model, real users this time.**

Last time you asked for three things: BF16 on both, a tuned Ollama, and the exact versions. Done.

Setup: [GPU] · Qwen3-8B BF16 on both · Ollama [ver] · vLLM [ver] ([image tag]) · driver [ver] · Ubuntu [ver] · thinking off · 8k context on both · one engine on the card at a time.

**Single user (conc = 1)**
Ollama [per_req_tok_s] tok/s, TTFT [ttft_med_ms] ms · vLLM [per_req_tok_s] tok/s, TTFT [ttft_med_ms] ms.
[One sentence: who wins alone and by how much. If Ollama is close or ahead, say it.]

**16 concurrent requests (the Part 1 number, redone fairly)**
| | Ollama default | Ollama tuned | vLLM |
|---|---|---|---|
| aggregate tok/s | [ ] | [ ] | [ ] |
| TTFT p95 | [ ] ms | [ ] ms | [ ] ms |
| p95 latency | [ ] s | [ ] s | [ ] s |
| VRAM peak | [ ] GB | [ ] GB | [ ] GB |

[One sentence on what tuning Ollama bought and what it did not.]

**Real users, not a batch** (16 people, each talking, reading, thinking ~3 s, talking again, 2 minutes)
TTFT p95: Ollama tuned [ ] ms vs vLLM [ ] ms. Requests served: [ ] vs [ ].
[The feeling: "with headphones on, X ms is where people start talking over the translation."]

**Where it breaks** (open loop, requests keep arriving whether you are ready or not)
Ollama tuned flattens at ~[req_s] req/s; vLLM at ~[req_s] req/s. Past that, p95 goes from [ ] s to [ ] s within one step.
[Chart: rate-latency-p95.png]

**The answers you asked for**
- gpu-memory-utilization is a decision: 0.85 → "[Maximum concurrency line]"; 0.5 → "[line]". That is the price of sharing the card with Whisper/eval: [TTFT delta] ms at 16 users.
- max-model-len is where the KV cache goes: 8k → [GPU KV cache size]; 32k → [size]. Same weights, [N]x fewer concurrent slots.
- AWQ: Marlin kernel [per_req_tok_s] tok/s vs plain [ ] vs BF16 [ ] at conc 1. Log line: "[quantization line]".
- OOM on Friday: first `num_requests_waiting > 0` at conc [ ]; preemptions at [ ]; p95 there [ ] s.

**What I would do differently from Part 1**
[2-3 honest lines. Include anything that went against the Part 1 narrative.]

Full tables, scripts and raw jsonl: [repo link]. Reproduce it on your card; I will add your numbers.

---

### Disclosure block (paste under the post or as first comment)

```
Model: Qwen/Qwen3-8B (BF16) | Ollama tag qwen3:8b-fp16
Ollama [ver] | vLLM [ver] image [tag@digest] | driver [ver] | CUDA [ver] | [OS]
Client: bench/loadgen.py, OpenAI-compatible streaming API on both, same seeded prompts, temperature 0, max_tokens 256
Thinking disabled on both (verified). Context 8192 on both. Median of 3 runs (batch), 120 s windows (users), 60 s windows (rate).
One engine on the GPU at a time. GPU sampled at 1 Hz; min SM clock during the run: [ ] MHz (no throttling / throttled).
```
