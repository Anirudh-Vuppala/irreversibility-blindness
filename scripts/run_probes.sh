#!/usr/bin/env bash
# Usage: nohup scripts/run_probes.sh Qwen_Qwen2.5-7B-Instruct meta-llama_Llama-3.1-8B-Instruct >/dev/null 2>&1 &
# Runs probes, AUROC refits, then the report for each model directory under results/, one after the other.
# Logs go to results/analysis/run_<short>.log; rerunning resumes finished probe files.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
mkdir -p results/analysis
for model in "$@"; do
  case "$model" in
    Qwen*) short=qwen ;;
    meta-llama*) short=llama ;;
    *) short="$model" ;;
  esac
  log="results/analysis/run_${short}.log"
  {
    echo "=== $model probes start $(date)"
    "$PY" -m experiments.analyze --model-dir "results/$model" --stage probes --jobs "${JOBS:-4}" ${EXTRA_ARGS:-} \
      && "$PY" -m experiments.analyze --model-dir "results/$model" --stage auroc ${EXTRA_ARGS:-} \
      && "$PY" -m experiments.analyze --model-dir "results/$model" --stage report ${EXTRA_ARGS:-}
    echo "=== $model finished with exit $? at $(date)"
  } >>"$log" 2>&1
done
