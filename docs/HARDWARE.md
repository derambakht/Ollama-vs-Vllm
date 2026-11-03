# Hardware to prepare

Goal: one GPU server that can run Ollama and vLLM one after the other on the **same card**, serve
Qwen3-8B in BF16 to 32 concurrent requests at 8k context, and stay thermally stable for ~3 hours
of sustained load so the numbers are publishable. Plus one small client machine.

The published run used a rented RunPod pod with this GPU instead of owned hardware; see [RUNPOD.md](RUNPOD.md).

## Recommended server (matches the previous post: "same card, same model")

| Component | Spec | Why |
|---|---|---|
| GPU | **1x NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7** (Workstation, Max-Q or Server Edition) | Same card as Part 1, so Part 2 is a direct continuation. 96 GB lets you hold 16 GB of BF16 weights plus a KV cache for 32+ concurrent 8k requests, and still leaves room for the 50% VRAM experiment and the Qwen3-32B optional run. |
| CPU | 16+ physical cores (AMD EPYC 9004/Threadripper PRO, or Xeon w5/w7), 3+ GHz | vLLM's scheduler, tokenizer and HTTP layer are CPU-bound at 32 streams. Ollama's runner also uses several threads. 8 cores works but you will see CPU in the TTFT at high concurrency. |
| RAM | **128 GB** (64 GB minimum) | Weights are read through page cache on load; 2x the largest model you load keeps restarts fast. 32B BF16 = 65 GB, so 128 GB if you run the optional 32B comparison. |
| Disk | **2 TB NVMe** (1 TB minimum, 512 GB is too small once you add the 32B run) | Breakdown below. Sustained read matters for load time; any PCIe 4.0 NVMe is fine. |
| PSU / cooling | 1000 W+ PSU; **rack or open-air chassis with front-to-back airflow** | The Workstation and Server editions are rated at 600 W (Max-Q at 300 W). In the published run the average draw per run reached 390 W. A desktop with poor airflow can throttle under sustained load, and then your p95 numbers become a cooling benchmark. |
| Network | 1 GbE is enough (traffic is < 5 Mbit/s); 2.5/10 GbE nice to have | Client-to-server latency must be stable, not fast. Use a wired LAN, not Wi-Fi. |
| OS | Ubuntu 24.04 LTS (22.04 ok), NVIDIA driver **>= 570** (R580 preferred), CUDA 12.8+, Docker 27+ with nvidia-container-toolkit | Blackwell (SM 12.0) needs 570+. Pin the vLLM image tag; `latest` is not reproducible. |

## Disk budget

| Item | Size |
|---|---|
| Qwen3-8B BF16 (Hugging Face, for vLLM) | 16.4 GB |
| qwen3:8b-fp16 (Ollama blob, same weights, different format) | 16.4 GB |
| Qwen3-8B-AWQ (AWQ experiment) | 5.9 GB |
| vLLM Docker image (one pinned tag; add 20-25 GB per extra tag you try) | 20-25 GB |
| Ollama install + CUDA runtime libs | 3-5 GB |
| torch / triton / vLLM compile caches | 5-10 GB |
| Optional: Qwen3-8B GGUF f16 for llama.cpp | 16 GB |
| Optional: Qwen3-32B BF16 on both engines | 2 x 65 GB |
| OS, Docker overhead, logs, results (gpu.csv at 1 Hz is < 1 MB/h) | 30-40 GB |
| **Core total** | **~100 GB** |
| **With optional 32B + llama.cpp + a second vLLM tag** | **~280 GB** |

A 1 TB NVMe leaves comfortable room for a few false starts (a wrong image tag, a re-download).
2 TB if this box will also hold the Whisper / TTS models for the shared-card experiment.

## Minimum alternative if the RTX PRO 6000 is not available

| Card | VRAM | Works for this benchmark? |
|---|---|---|
| RTX 6000 Ada / A6000 | 48 GB | Yes, all core runs. 32B BF16 does not fit; skip it. |
| H100 / A100 80 GB (cloud) | 80 GB | Yes, everything. But the post must say "H100", it is not "your card". |
| RTX 5090 / 4090 | 32 / 24 GB | 8B BF16 fits (16 GB weights), but 32 concurrent x 8k KV does not. Reduce to conc 16 and `--max-model-len 4096`; label it clearly. |

## Cloud rental option (if you only need the afternoon)

RunPod, Lambda, Vast.ai and DataCrunch list the RTX PRO 6000 Blackwell and the H100 80 GB at roughly 1.5 to 3 USD/h.
The published run took about 3.5 pod-hours including setup, under 10 USD, with a 150 GB persistent volume.
Pick a host whose driver is already 570 or newer (check `nvidia-smi` before paying for the hour), and
follow [RUNPOD.md](RUNPOD.md).

## Client machine

Any 8-core laptop or desktop on the same wired LAN, Python 3.11+. The load generator is async and
idles at 32 streams. Running the client on the GPU server itself is acceptable for the batch sweep,
but the users/rate runs at 32 streams steal CPU from vLLM's scheduler; prefer a second machine and
set `BENCH_HOST` in `config.env` to the server's IP.

## Before buying anything: check thermals

The one measurement that invalidates everything is thermal throttling. The monitor records SM clock
and temperature every second; `analyze.py` reports the minimum clock inside each run. If the clock
drops more than ~15% between conc=1 and conc=32, the chassis is the bottleneck, not the engine.
