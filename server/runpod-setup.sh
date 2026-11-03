#!/usr/bin/env bash
# One-shot setup inside a RunPod pod. Tested image: runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404 (Ubuntu 24.04).
# Creates a virtual environment (Ubuntu 24.04 blocks system-wide pip), installs vLLM and the client dependencies,
# installs prebuilt FlashInfer kernels, installs Ollama, and downloads both model formats to the persistent volume.
# Takes about 20 minutes, mostly downloads.
#   bash server/runpod-setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env
VENV=${VENV:-/opt/venv}
VOL=${VOL:-/workspace}
mkdir -p logs

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 570 ] || { echo "driver $drv < 570: this host cannot run Blackwell; pick another pod"; exit 1; }

echo "== persistent volume"
df -h "$VOL"
HF_DIR="$VOL/hf"; OLLAMA_DIR="$VOL/ollama"
mkdir -p "$HF_DIR" "$OLLAMA_DIR"
sed -i "s#^HF_HOME=.*#HF_HOME=\"$HF_DIR\"#; s#^VLLM_NATIVE=.*#VLLM_NATIVE=1#; s#^PY=.*#PY=\"$VENV/bin/python\"#" config.env
grep -q '^export OLLAMA_MODELS=' config.env || echo "export OLLAMA_MODELS=\"$OLLAMA_DIR\"" >> config.env
export HF_HOME="$HF_DIR" OLLAMA_MODELS="$OLLAMA_DIR"

echo "== system packages"
apt-get update -qq
apt-get install -y -qq curl lsof pciutils zstd python3-venv >/dev/null

echo "== python environment ($VENV)"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -r requirements.txt huggingface_hub
"$VENV/bin/python" -c "import vllm" 2>/dev/null || "$VENV/bin/pip" install -q "${VLLM_PIP:-vllm}"
# Prebuilt FlashInfer kernels. Without them vLLM tries to JIT-compile its sampling kernel at start-up,
# which fails on images without nvcc.
"$VENV/bin/vllm" download-kernels

echo "== ollama"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh -o /tmp/ollama-install.sh
  sh /tmp/ollama-install.sh        # no systemd in a pod: the benchmark starts `ollama serve` itself
fi
ollama --version || true

echo "== models"
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" nohup ollama serve > logs/ollama-pull.log 2>&1 &
ollama_pid=$!
sleep 3
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" ollama pull "$OLLAMA_TAG"
kill "$ollama_pid" 2>/dev/null || true
"$VENV/bin/python" - <<EOF
from huggingface_hub import snapshot_download
for repo in ["$MODEL_HF", "$MODEL_HF_AWQ"]:
    print(repo, snapshot_download(repo), flush=True)
EOF

echo "== versions"
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| compute capability', torch.cuda.get_device_capability())"
echo
echo "SETUP DONE. Next:  bash server/preflight.sh  &&  ./run-all.sh all"
