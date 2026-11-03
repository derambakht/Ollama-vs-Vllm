#!/usr/bin/env bash
# Start Ollama in one of two named configurations. Usage: server/start-ollama.sh default|tuned
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env
variant="${1:-default}"
mkdir -p logs
server/stop-all.sh >/dev/null

export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
case "$variant" in
  default)
    # Exactly what a fresh install does: default parallel slots, default context, model unloads after 5 min.
    # KEEP_ALIVE is the one exception: without it a run after a pause would measure model load, not inference.
    export OLLAMA_KEEP_ALIVE=-1
    ;;
  tuned)
    export OLLAMA_KEEP_ALIVE=-1
    export OLLAMA_NUM_PARALLEL="$OLLAMA_PARALLEL"
    export OLLAMA_CONTEXT_LENGTH="$OLLAMA_CTX"      # same context budget as vLLM --max-model-len
    export OLLAMA_FLASH_ATTENTION=1
    export OLLAMA_KV_CACHE_TYPE=f16
    ;;
  *) echo "unknown variant: $variant (default|tuned)"; exit 1 ;;
esac

env | grep ^OLLAMA_ | sort | tee "logs/ollama-${variant}.env"
nohup ollama serve > "logs/ollama-${variant}.log" 2>&1 &
echo $! > logs/ollama.pid
sleep 3
# force-load the model so VRAM reservation happens before the monitor starts
curl -sf "http://127.0.0.1:${OLLAMA_PORT}/api/generate" -d "{\"model\":\"${OLLAMA_TAG}\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":-1}" >/dev/null
ollama ps
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/ollama-${variant} VRAM after load: /"
