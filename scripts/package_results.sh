#!/usr/bin/env bash
# Build submission/irreversibility_results_bundle.zip: the small result files, without hidden-state arrays.
# Usage: scripts/package_results.sh [output.zip]   (run from anywhere; reads the repo's results/, labeling/, data/)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-submission/irreversibility_results_bundle.zip}"
PYTHON="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
mkdir -p "$(dirname "$OUT")"

"$PYTHON" - "$OUT" <<'EOF'
import hashlib
from pathlib import Path
import re
import sys
import zipfile

out = Path(sys.argv[1])
root = Path(".")
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{30,}")


def small_result_files(model_dir):
    """Everything a model run saved except the hidden-state arrays."""
    return [p for p in model_dir.rglob("*") if p.is_file() and "arrays" not in p.parts and p.suffix != ".npy"]


models = sorted(p.parent.parent for p in root.glob("results/*/extraction/index.jsonl"))
if not models:
    sys.exit("no results/<model>/extraction/index.jsonl found; unzip the Kaggle results first")
files = [f for m in models for f in small_result_files(m)]
files += [p for p in (root / "results" / "analysis").rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
files += [p for p in (root / "labeling").glob("*") if p.is_file()]
files += [root / "data" / "sandbox.jsonl", root / "data" / "scienceworld.jsonl"]
files = sorted({p.as_posix(): p for p in files}.values(), key=lambda p: p.as_posix())

leaks = [p.as_posix() for p in files if p.name == ".env" or TOKEN.search(p.read_bytes())]
if leaks:
    sys.exit(f"refusing to package files that look like secrets: {leaks}")

manifest = ["# path\tbytes\tsha256"]
for p in files:
    data = p.read_bytes()
    manifest.append(f"{p.as_posix()}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")

readme = f"""Irreversibility blindness: result bundle
=========================================

Unzip at the repository root; paths match the repo layout. MANIFEST.txt lists every file with its size and sha256.

Models: {', '.join(m.name for m in models)}

results/<model>/                 Kaggle outputs (sandbox) for one model
  extraction/index.jsonl         one row per data row: id, label, groups, array path and shape
  extraction/prompts.jsonl       exact chat messages used for hidden-state extraction
  extraction/summary.json        extraction settings and timing
  rq2/<run>/                     RQ2 stated answers: prompts, responses, failures, summary
                                 (three-class-h3-neutral-v2 is the reported run; -neutral is the preliminary v1)
  rq3/h3, rq3/h3-extra/          RQ3 agent: tasks, prompts, every turn, tool calls, failures, summary (30 + 30 pairs)
  runner_log.jsonl, pilot.json   runner step log and pilot
  sw/                            the same for ScienceWorld (extraction and RQ2 only)
results/analysis/<model>/        probes_summary.csv, probes/ (per-split, per-row test predictions for every layer,
                                 position and control), rq2/rq3 summaries, headline*.json, posthoc_*.json,
                                 figures/, sw/ (ScienceWorld probes, transfer, horizons)
results/analysis/final/          report figures, tables (CSV, LaTeX) and headline_numbers.json (every quoted number
                                 with its source file)
results/analysis/human_agreement.json, labeling/   human label check: blind sheets, answer key, retest sheet, guide
data/sandbox.jsonl, data/scienceworld.jsonl         the labelled datasets

Not included: the hidden-state arrays (results/<model>/extraction/arrays/*.npy, about 0.8-1 GB per model).
Regenerate them on Kaggle (GPU T4 x2, HF_TOKEN in Kaggle Secrets, one model per session), from the repo root:
  python kaggle/run_full.py --model Qwen/Qwen2.5-7B-Instruct --only extract
  python kaggle/run_full.py --model meta-llama/Llama-3.1-8B-Instruct --only extract
  python kaggle/run_full.py --model Qwen/Qwen2.5-7B-Instruct --env scienceworld --only extract   # and Llama
Each takes about 7-12 min per model and environment; unzip the produced archive under results/. The probe analysis
(experiments/analyze.py via scripts/run_probes.sh, experiments/analyze_sw.py) needs the arrays; the report tables
(python -m experiments.make_final_tables) need only the files in this bundle.
"""

out.unlink(missing_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for p in files:
        bundle.write(p, p.as_posix())
    bundle.writestr("MANIFEST.txt", "\n".join(manifest) + "\n")
    bundle.writestr("README_BUNDLE.txt", readme)
raw = sum(p.stat().st_size for p in files)
print(f"{out}: {len(files)} files, {raw / 1e6:.1f} MB uncompressed, {out.stat().st_size / 1e6:.1f} MB zipped")
EOF
