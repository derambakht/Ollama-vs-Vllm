#!/usr/bin/env bash
# The whole benchmark, start to finish, on the GPU server. Measured on an RTX PRO 6000: core 93 min, extras 57 min.
#
#   ./run-all.sh                 core: ollama-default, ollama-tuned, vllm-bf16   (batch + users + rate)
#   ./run-all.sh extras          extra experiments: vllm-awq-marlin, vllm-awq-plain, vllm-bf16-u050, vllm-peak, ctx32k log
#   ./run-all.sh all             core, then extras
#   ./run-all.sh one <label>     a single label, e.g.  ./run-all.sh one vllm-bf16
#   ./run-all.sh report          only rebuild results/report (tables + charts)
#
# Client can run on another machine: set BENCH_HOST in config.env and run `bench/loadgen.py` there instead
# of the `bench` function below; everything else stays the same.
# No `set -e`: one failed phase must never abort the others. Every phase reports to logs/phase-status.txt.
set -uo pipefail
cd "$(dirname "$0")"
source config.env
chmod +x server/*.sh
mkdir -p "$RESULTS_DIR" logs
LOG="logs/run-all-$(date +%Y%m%d-%H%M).log"; exec > >(tee -a "$LOG") 2>&1
STATUS=logs/phase-status.txt
NO_THINK="${NO_THINK:-template}"; RATE_TIMEOUT="${RATE_TIMEOUT:-120}"
status() { echo "$(date +%H:%M:%S) $*" | tee -a "$STATUS"; }

url_ollama="http://${BENCH_HOST}:${OLLAMA_PORT}/v1/chat/completions"
url_vllm="http://${BENCH_HOST}:8000/v1/chat/completions"

step() {   # step <label> <what> <cmd...>: run, time it, record pass/fail, never abort
  local label=$1 what=$2; shift 2
  local t0=$(date +%s)
  if "$@"; then status "$label $what OK ($(( $(date +%s) - t0 ))s)"; return 0
  else status "$label $what FAILED rc=$? ($(( $(date +%s) - t0 ))s)"; return 1; fi
}

bench() {   # bench <label> <url> <model>: the three traffic shapes, same seed on every engine
  local label=$1 url=$2 model=$3
  step "$label" sanity $PY bench/sanity.py --url "$url" --model "$model" --no-think "$NO_THINK" \
    || status "$label WARNING: sanity failed, benchmarking anyway; check saw_think in summary.jsonl"
  step "$label" batch $PY bench/loadgen.py --url "$url" --model "$model" --label "$label" --mode batch \
      --levels "$BATCH_LEVELS" --runs "$BATCH_RUNS" --max-tokens "$MAX_TOKENS" --prompt-mix "$PROMPT_MIX" \
      --no-think "$NO_THINK" --out "$RESULTS_DIR"
  step "$label" users $PY bench/loadgen.py --url "$url" --model "$model" --label "$label" --mode users \
      --levels "$USER_LEVELS" --runs 1 --duration "$USER_DURATION" --think "$USER_THINK" \
      --max-tokens "$MAX_TOKENS" --prompt-mix "$PROMPT_MIX" --no-think "$NO_THINK" --out "$RESULTS_DIR"
  step "$label" rate $PY bench/loadgen.py --url "$url" --model "$model" --label "$label" --mode rate \
      --levels "$RATE_LEVELS" --runs 1 --duration "$RATE_DURATION" --timeout "$RATE_TIMEOUT" \
      --max-tokens "$MAX_TOKENS" --prompt-mix "$PROMPT_MIX" --no-think "$NO_THINK" --out "$RESULTS_DIR"
}

phase() {   # phase <label> <engine> <variant> [notes]
  local label=$1 engine=$2 variant=$3 notes=${4:-}
  echo; echo "================ $label  ($(date)) ================"
  if [ "$engine" = ollama ]; then url=$url_ollama; model=$OLLAMA_TAG;   starter=server/start-ollama.sh
  else                            url=$url_vllm;   model=$SERVED_NAME; starter=server/start-vllm.sh; fi
  if ! step "$label" start "$starter" "$variant"; then
    status "$label SKIPPED (server did not start); see logs/"; server/stop-all.sh || true; return 0
  fi
  server/capture-meta.sh "$label" "$engine" "$notes" >/dev/null || status "$label meta capture failed"
  server/gpu-monitor.sh start "$label"
  bench "$label" "$url" "$model"
  server/gpu-monitor.sh stop
  step "$label" stop server/stop-all.sh || true
}

core() {
  phase ollama-default ollama default "fresh-install defaults + KEEP_ALIVE=-1"
  phase ollama-tuned   ollama tuned   "NUM_PARALLEL=$OLLAMA_PARALLEL CONTEXT_LENGTH=$OLLAMA_CTX FLASH_ATTENTION=1 KV=f16"
  phase vllm-bf16      vllm   bf16    "max-model-len=$MAX_MODEL_LEN util=$GPU_UTIL max-num-seqs=$MAX_NUM_SEQS"
}

extras() {
  phase vllm-awq-marlin vllm awq-marlin "AWQ, kernel auto-selected (expect awq_marlin)"
  phase vllm-awq-plain  vllm awq-plain  "AWQ, --quantization awq forced"
  phase vllm-bf16-u050  vllm u050       "gpu-memory-utilization=0.5: half the card for eval/Whisper/TTS"
  # peak: find the cliff with big batches and longer answers, one run each
  echo "================ vllm-peak ================"
  if step vllm-peak start server/start-vllm.sh bf16; then
    mkdir -p "$RESULTS_DIR/vllm-peak"
    server/capture-meta.sh vllm-peak vllm "peak probe: conc 32/64/96, 512 tokens" >/dev/null
    server/gpu-monitor.sh start vllm-peak
    ( while true; do echo "$(date +%s) $(curl -s localhost:8000/metrics | grep -E '^vllm:(num_requests_(running|waiting)|gpu_cache_usage_perc|num_preemptions_total)' | tr '\n' ' ')"; sleep 2; done ) >> "$RESULTS_DIR/vllm-peak/vllm-metrics.log" &
    mpid=$!
    step vllm-peak batch $PY bench/loadgen.py --url "$url_vllm" --model "$SERVED_NAME" --label vllm-peak --mode batch \
      --levels 32,64,96 --runs 1 --max-tokens 512 --prompt-mix "$PROMPT_MIX" --no-think "$NO_THINK" --out "$RESULTS_DIR"
    kill $mpid 2>/dev/null; server/gpu-monitor.sh stop
  fi
  server/stop-all.sh || true
  # ctx32k: only the start-up log matters ("max-model-len is where KV cache goes")
  if step vllm-ctx32k start server/start-vllm.sh ctx32k; then
    server/capture-meta.sh vllm-ctx32k vllm "start-up only, KV budget at 32k" >/dev/null
  fi
  server/stop-all.sh || true
}

report() { step report analyze $PY bench/analyze.py --results "$RESULTS_DIR" --dark; }
pre()    { server/preflight.sh || { status "PREFLIGHT FAILED, aborting"; exit 1; }; status "START $1"; }

case "${1:-core}" in
  core)   pre core;   core; report ;;
  extras) pre extras; extras; report ;;
  all)    pre all;    core; extras; report ;;
  one)    case "$2" in
            ollama-default) phase ollama-default ollama default ;;
            ollama-tuned)   phase ollama-tuned ollama tuned ;;
            vllm-bf16)      phase vllm-bf16 vllm bf16 ;;
            vllm-awq-marlin) phase vllm-awq-marlin vllm awq-marlin ;;
            vllm-awq-plain) phase vllm-awq-plain vllm awq-plain ;;
            vllm-bf16-u050) phase vllm-bf16-u050 vllm u050 ;;
            *) echo "unknown label $2"; exit 1 ;;
          esac; report ;;
  report) report ;;
  *) echo "usage: $0 [core|extras|all|one <label>|report]"; exit 1 ;;
esac
status "ALL DONE"
echo; echo "DONE. Report: $RESULTS_DIR/report/REPORT.md   charts: $RESULTS_DIR/report/charts/   log: $LOG"
echo "phase status:"; cat "$STATUS"
