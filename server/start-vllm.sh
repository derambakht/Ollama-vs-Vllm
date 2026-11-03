#!/usr/bin/env bash
# Start vLLM in a named configuration. Usage: server/start-vllm.sh bf16|awq-marlin|awq-plain|u050|ctx32k
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env
variant="${1:-bf16}"
mkdir -p logs "$HF_HOME"
server/stop-all.sh >/dev/null

model="$MODEL_HF"; util="$GPU_UTIL"; ctx="$MAX_MODEL_LEN"; extra=()
case "$variant" in
  bf16)       extra=(--dtype bfloat16) ;;
  awq-marlin) model="$MODEL_HF_AWQ" ;;                         # let vLLM pick awq_marlin itself
  awq-plain)  model="$MODEL_HF_AWQ"; extra=(--quantization awq) ;;
  u050)       extra=(--dtype bfloat16); util=0.5 ;;             # half the card left for eval / Whisper / TTS
  ctx32k)     extra=(--dtype bfloat16); ctx=32768 ;;            # only to read the KV-cache budget line
  *) echo "unknown variant: $variant"; exit 1 ;;
esac

vllm_args=(--model "$model" --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port 8000
           --max-model-len "$ctx" --gpu-memory-utilization "$util" --max-num-seqs "$MAX_NUM_SEQS" "${extra[@]}")

# Two ways to run vLLM: Docker on a bare-metal box, or native `vllm serve` inside a container that has no
# Docker daemon (RunPod / Vast / any cloud pod). VLLM_NATIVE=1 in config.env forces native.
if [ "${VLLM_NATIVE:-0}" = 1 ] || ! command -v docker >/dev/null 2>&1; then
  export HF_HOME
  [ -d /usr/local/cuda ] && export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # lets FlashInfer JIT-build if a cubin is missing
  VLLM_BIN="$(dirname "$PY")/vllm"; [ -x "$VLLM_BIN" ] || VLLM_BIN=vllm            # the venv's vllm, same interpreter as $PY
  cmd=("$VLLM_BIN" serve "${vllm_args[@]}")
  printf '%q ' "${cmd[@]}" | tee "logs/vllm-${variant}.cmd"; echo
  nohup "${cmd[@]}" > "logs/vllm-${variant}.log" 2>&1 &
  echo $! > logs/vllm.pid
  alive() { kill -0 "$(cat logs/vllm.pid)" 2>/dev/null; }
  dump()  { tail -40 "logs/vllm-${variant}.log"; }
else
  cmd=(docker run -d --name vllm --gpus all --ipc=host -p 8000:8000
       -v "$HF_HOME:/root/.cache/huggingface" -e HF_TOKEN="${HF_TOKEN:-}"
       "$VLLM_IMAGE" "${vllm_args[@]}")
  printf '%q ' "${cmd[@]}" | tee "logs/vllm-${variant}.cmd"; echo
  "${cmd[@]}"
  alive() { docker ps -q -f name=vllm | grep -q .; }
  dump()  { docker logs vllm 2>&1 | tail -40; }
fi

echo "waiting for vLLM (first start downloads the model; can take 10+ min)..."
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
  if ! alive; then echo "vLLM died:"; dump; exit 1; fi
  sleep 5
done
[ -f logs/vllm.pid ] || docker logs vllm > "logs/vllm-${variant}.log" 2>&1
echo "--- lines for the post:"
grep -E "Model loading took|GPU KV cache size|Available KV cache memory|# GPU blocks|Maximum concurrency|quantization|marlin|awq|vLLM API server version" \
  "logs/vllm-${variant}.log" | sed 's/^/  /' || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/vllm-${variant} VRAM after load: /"
