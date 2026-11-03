# Running the benchmark on RunPod

This page describes how the published run of 2026-11-03 was set up, so it can be repeated exactly. RunPod prices change often, so check the live price before you deploy.

## What to rent

| GPU | VRAM | Secure Cloud price seen on 2026-11-03 | Verdict |
|---|---|---|---|
| **RTX PRO 6000** | 96 GB | about 2.49 to 2.53 USD/h | Used for the published run |
| H100 PCIe | 80 GB | about 2.89 USD/h | Works; report it as "H100" |
| RTX 6000 Ada | 48 GB | about 0.99 USD/h | Works for the 8B model |
| Pro 6000 MIG 48GB | 48 GB | about 1.29 USD/h | Do not use: it is a slice of a card, not a whole card |

The full run took about 3.5 pod-hours including setup, which cost under 10 USD.

## Deploy settings

1. **Cloud type: Secure Cloud.** Community hosts share CPU and PCIe with other tenants, so results vary more between runs.
2. **GPU: 1x RTX PRO 6000 (96 GB).** Do not pick the MIG variant. The published run got the Server Edition, which has a 600 W limit; write down which edition you get.
3. **Template image:** `runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404`, which provides Ubuntu 24.04 and CUDA 12.8.1. The setup script installs vLLM into its own virtual environment, so the image's PyTorch version does not matter.
4. **Container disk: at least 50 GB.** It holds the vLLM virtual environment in `/opt/venv`, with the torch and vLLM wheels. The container disk is wiped when the pod is stopped.
5. **Volume disk: 150 GB, mounted at `/workspace`.** It holds the models (36 GB in the published run), the repository and the results, and it survives a stop.
6. **Ports:** TCP 22 for SSH is enough. The load generator runs inside the pod against localhost on purpose, because RunPod's HTTPS proxy can buffer streaming responses and distort TTFT.
7. **SSH key:** add your public key under Settings, SSH Public Keys, before you deploy.

After the pod shows "Running", copy the SSH command from the pod card. Check the driver before you go further:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# the published run: NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.91.07, 97887 MiB
```

The driver must be 570 or newer. `runpod-setup.sh` stops if it is older.

## Setup inside the pod

Copy the repository to the volume, either with `git clone` or from your own machine:

```bash
# from your own machine
tar czf - --exclude=results --exclude=logs --exclude=.git . | ssh -p <port> root@<pod-ip> \
  'mkdir -p /workspace/Ollama-vs-Vllm && tar xzf - --no-same-owner -C /workspace/Ollama-vs-Vllm'
```

Then, in the pod:

```bash
cd /workspace/Ollama-vs-Vllm
chmod +x run-all.sh server/*.sh
bash server/runpod-setup.sh
bash server/preflight.sh          # must end with PREFLIGHT OK
```

`runpod-setup.sh` does the following. It should take about 20 minutes, mostly downloads. This is an estimate: the published run's setup took about 45 minutes, because the two fixes the script now contains had to be found by hand.

1. It checks that the driver is 570 or newer.
2. It points `config.env` at the volume: `HF_HOME=/workspace/hf` and `OLLAMA_MODELS=/workspace/ollama`. It also sets `VLLM_NATIVE=1` and points `PY` at `/opt/venv/bin/python`.
3. It creates `/opt/venv` and installs `vllm==0.33.0` and the client dependencies. Ubuntu 24.04 blocks system-wide `pip install`, so a virtual environment is required.
4. It runs `vllm download-kernels`. Without the prebuilt FlashInfer kernels, vLLM crashes at start-up on Blackwell, because the image has no `nvcc` to compile them.
5. It installs Ollama with the official script and pulls `qwen3:8b-fp16`.
6. It downloads `Qwen/Qwen3-8B` and `Qwen/Qwen3-8B-AWQ` from Hugging Face.

## Running

Start the run detached from the SSH session, so that a dropped connection does not stop it:

```bash
setsid nohup ./run-all.sh all > logs/all.out 2>&1 < /dev/null &
tail -f logs/phase-status.txt
```

Every step writes one line to `logs/phase-status.txt` with its result and duration. A failed step is recorded and skipped, and the next configuration still runs. To repeat a single configuration, run `./run-all.sh one <label>`.

Measured durations in the published run:

| Configuration | Minutes | Why |
|---|---|---|
| ollama-default | 47 | Requests are served one at a time, so every queue drains slowly |
| ollama-tuned | 28 | |
| vllm-bf16 | 18 | |
| vllm-awq-marlin | 19 | The first start compiles kernels for the AWQ model |
| vllm-awq-plain | 17 | |
| vllm-bf16-u050 | 18 | |
| vllm-peak and vllm-ctx32k | 3 | |
| **Total** | **150** | |

## Copying the results and shutting down

```bash
# from your own machine
ssh -p <port> root@<pod-ip> 'cd /workspace/Ollama-vs-Vllm && tar czf - results logs' | tar xzf - --no-same-owner
python bench/analyze.py --results results --dark
```

Then shut the pod down in the RunPod console:

- **Stop** ends GPU billing. The volume keeps the models and results at about 0.04 USD/h, and the `/opt/venv` environment is lost.
- **Terminate** deletes the pod and its volume and ends all billing. Copy the results first.

## Differences from a bare-metal server

- **vLLM runs as a native process instead of a Docker container,** because a pod is itself a container without a Docker daemon. `meta.json` records the vLLM and torch versions instead of an image digest.
- **Ollama runs as a plain process,** because pods have no systemd.
- **The pod gets a share of the host CPU.** The published run had 32 vCPU. The client and vLLM's scheduler share it at 32 streams; vLLM's TTFT stayed flat, so the effect is small.
- **Thermals come from a datacenter chassis.** The published run never exceeded 53 °C, and the SM clock never dropped below 2,280 MHz. A desktop under a desk may throttle.
