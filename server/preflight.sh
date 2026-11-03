#!/usr/bin/env bash
# Run once on the GPU server before anything else. Fails loudly on the things that waste an afternoon.
set -uo pipefail
cd "$(dirname "$0")/.."
source config.env
fail=0
ok()   { echo "  [ok]   $*"; }
bad()  { echo "  [FAIL] $*"; fail=1; }
warn() { echo "  [warn] $*"; }

echo "== GPU"
if nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
  [ "$drv" -ge 570 ] && ok "driver $drv >= 570 (Blackwell / SM120 ok)" || bad "driver $drv < 570: Blackwell needs 570+"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 1024 ] && ok "VRAM idle (${used} MiB used)" || warn "VRAM already in use (${used} MiB): something else is on the card"
else
  bad "nvidia-smi not working"
fi

echo "== vLLM runtime"
if [ "${VLLM_NATIVE:-0}" = 1 ] || ! command -v docker >/dev/null 2>&1; then
  if $PY -c "import vllm" 2>/dev/null; then
    ok "native vllm $($PY -c 'import vllm;print(vllm.__version__)') (docker not needed: native mode)"
  else
    bad "VLLM_NATIVE=1 but 'import vllm' fails for $PY (use the vllm/vllm-openai image as the pod image, or pip install vllm)"
  fi
elif docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
  ok "docker --gpus all works"
  docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 && ok "vLLM image present: $VLLM_IMAGE" || warn "vLLM image not pulled yet: docker pull $VLLM_IMAGE (~20 GB)"
else
  bad "docker cannot see the GPU (install nvidia-container-toolkit, restart docker) or set VLLM_NATIVE=1"
fi

echo "== Ollama"
if command -v ollama >/dev/null; then
  ok "ollama $(ollama --version 2>/dev/null | head -1)"
  ollama list 2>/dev/null | grep -q "$OLLAMA_TAG" && ok "model present: $OLLAMA_TAG" || warn "pull it: ollama pull $OLLAMA_TAG (~16 GB)"
else
  bad "ollama not installed (curl -fsSL https://ollama.com/install.sh | sh)"
fi

echo "== Disk"
mkdir -p "$HF_HOME" 2>/dev/null || true
for p in "$HF_HOME" /var/lib/docker "${OLLAMA_MODELS:-$HOME/.ollama}"; do
  [ -d "$p" ] || continue
  free_gb=$(df -BG --output=avail "$p" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge 60 ] && ok "$p: ${free_gb} GB free" || bad "$p: only ${free_gb} GB free (need >= 60 GB, see docs/HARDWARE.md)"
done

echo "== Python client"
if $PY -c "import httpx, pandas, matplotlib" 2>/dev/null; then
  ok "python deps present ($($PY --version))"
else
  bad "pip install -r requirements.txt   (python: $PY)"
fi

echo "== Thermal / power sanity"
nvidia-smi --query-gpu=power.limit,temperature.gpu,clocks.max.sm,pstate --format=csv,noheader 2>/dev/null | sed 's/^/  /'
warn "if the box is a desktop under a desk, expect throttling after ~10 min; label results with the machine"

echo
[ $fail -eq 0 ] && echo "PREFLIGHT OK" || { echo "PREFLIGHT FAILED: fix the [FAIL] lines"; exit 1; }
