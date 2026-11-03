#!/usr/bin/env bash
# Stop every LLM server and wait until the GPU is actually empty. Only one engine on the card at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
systemctl stop ollama 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true
pkill -f "ollama runner" 2>/dev/null || true
command -v docker >/dev/null 2>&1 && docker rm -f vllm vllm-awq llamacpp 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true; rm -f logs/vllm.pid
pkill -f llama-server 2>/dev/null || true
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${used:-0}" -lt 1024 ] && { echo "GPU empty (${used} MiB used)"; exit 0; }
  sleep 1
done
echo "WARNING: GPU still has ${used} MiB in use after 30 s:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
exit 1
