#!/usr/bin/env bash
# Record everything a reader needs to reproduce a label. Usage: server/capture-meta.sh <label> <ollama|vllm> [notes]
# Writes results/<label>/meta.json (used by analyze.py for the Environment table in the report).
set -uo pipefail
cd "$(dirname "$0")/.."
source config.env
label="${1:?label}"; engine="${2:?ollama|vllm}"; notes="${3:-}"
mkdir -p "$RESULTS_DIR/$label"

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"); kernel=$(uname -r)
cpu=$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')
ram=$(free -g 2>/dev/null | awk '/Mem:/{print $2" GB"}')
wsl=$(grep -qi microsoft /proc/version 2>/dev/null && echo "WSL2" || echo "")

if [ "$engine" = ollama ]; then
  ver=$(curl -s "http://127.0.0.1:${OLLAMA_PORT}/api/version" | sed 's/.*"version":"\([^"]*\)".*/\1/')
  model="$OLLAMA_TAG"; image="ollama binary $(ollama --version 2>/dev/null | head -1)"
  envs=$(cat logs/ollama-*.env 2>/dev/null | tail -8 | tr '\n' ' ')
  startup=""
else
  ver=$(curl -s http://127.0.0.1:8000/version | sed 's/.*"version": *"\([^"]*\)".*/\1/')
  model="$MODEL_HF"
  if [ -f logs/vllm.pid ]; then
    image="native: pip vllm $($PY -c 'import vllm;print(vllm.__version__)' 2>/dev/null) torch $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null)"
    logsrc=$(ls -t logs/vllm-*.log | head -1)
  else
    image="$VLLM_IMAGE @ $(docker inspect --format='{{index .RepoDigests 0}}' "$VLLM_IMAGE" 2>/dev/null | cut -d@ -f2 | cut -c1-19)"
    docker logs vllm > logs/vllm-current.log 2>&1; logsrc=logs/vllm-current.log
  fi
  envs=$(cat logs/vllm-*.cmd 2>/dev/null | tail -1)
  startup=$(grep -E "Model loading took|GPU KV cache size|Available KV cache memory|# GPU blocks|Maximum concurrency|quantization|marlin" "$logsrc" | head -8)
fi

$PY - "$label" "$engine" "$ver" "$model" "$gpu" "$driver" "$vram" "$os $wsl" "$kernel" "$cpu" "$ram" "$image" "$envs" "$startup" "$notes" <<'PYEOF'
import json, sys, datetime
k = ["label","engine","engine_version","model","gpu","driver","vram","os","kernel","cpu","ram","image","launch","vllm_startup_lines","notes"]
d = dict(zip(k, sys.argv[1:])); d["captured_at"] = datetime.datetime.now().isoformat(timespec="seconds")
json.dump(d, open(f"results/{d['label']}/meta.json", "w"), indent=2)
print(json.dumps(d, indent=2))
PYEOF
