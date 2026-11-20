"""Resumable one-model Kaggle run: pilot, disk check, extraction, RQ2 main config, RQ3, zip."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.extract_hidden import completed_ids, model_slug
from experiments.model_runtime import append_jsonl, read_jsonl
from experiments.rq2_stated import SW_PROMPT_VERSION, candidate_pairs, select_stated_pairs
from experiments.rq3_agent import MAX_TURNS

STEPS = ("pilot", "disk", "extract", "rq2", "rq3", "zip")
SW_STEPS = ("extract", "rq2", "zip")
INPUTS = {"sandbox": "data/sandbox.jsonl", "scienceworld": "data/scienceworld.jsonl"}
RQ2_LABEL_MODE, RQ2_HORIZON, RQ2_CONDITION, RQ2_SEED = "three-class", 3, "neutral", 17
RQ2_PROMPT_VERSION = "rq2-stated-v2"
RQ3_CONDITIONS = ("neutral", "explicit-carefulness")
KAGGLE_WORKING = Path("/kaggle/working")


class Paths:
    def __init__(self, results_root, model, rq3_batch="main", env="sandbox"):
        self.results_root = results_root
        self.root = results_root / model_slug(model) / ("sw" if env == "scienceworld" else "")
        self.zip_name = f"{model_slug(model)}{'_sw' if env == 'scienceworld' else ''}_results.zip"
        self.rq2_prompt_version = SW_PROMPT_VERSION if env == "scienceworld" else RQ2_PROMPT_VERSION
        self.extraction = self.root / "extraction"
        self.rq2 = self.root / "rq2" / f"{RQ2_LABEL_MODE}-h{RQ2_HORIZON}-{RQ2_CONDITION}-v2"
        self.rq3 = self.root / "rq3" / ("h3" if rq3_batch == "main" else "h3-extra")
        self.pilot = self.root / "pilot.json"
        self.log = self.root / "runner_log.jsonl"


def run_module(paths, step, module, args):
    command = [sys.executable, "-m", module, *args]
    print(f"\n=== {step}: {' '.join(command)}", flush=True)
    started = time.perf_counter()
    code = subprocess.run(command, cwd=REPO_ROOT).returncode
    elapsed = time.perf_counter() - started
    append_jsonl(paths.log, {"step": step, "command": command, "returncode": code,
                             "elapsed_seconds": elapsed, "finished_at": time.time()})
    print(f"=== {step}: exit {code} after {elapsed / 60:.1f} min", flush=True)
    return code


def pilot_projection(paths, total_rows):
    index_rows, _ = read_jsonl(paths.extraction / "index.jsonl")
    timed = [row["extract_seconds"] for row in index_rows if "extract_seconds" in row]
    if not timed:
        return None
    array_bytes = [(paths.extraction / row["array_path"]).stat().st_size for row in index_rows]
    prompt_bytes = (paths.extraction / "prompts.jsonl").stat().st_size / len(index_rows)
    sec_per_row = statistics.median(timed)
    gb_per_row = (statistics.mean(array_bytes) + prompt_bytes) / 1e9
    remaining = total_rows - len(index_rows)
    return {
        "pilot_rows": len(index_rows),
        "sec_per_row_median": sec_per_row,
        "sec_per_row_mean": statistics.mean(timed),
        "total_rows": total_rows,
        "remaining_rows": remaining,
        "projected_remaining_extraction_hours": remaining * sec_per_row / 3600,
        "projected_total_extraction_gb": total_rows * gb_per_row,
        "projected_remaining_extraction_gb": remaining * gb_per_row,
        "note": "Forward-pass time only; excludes model loading, RQ2 and RQ3.",
    }


def gpu_report():
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU visible"
    gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    total = sum(gpu.total_memory for gpu in gpus) / 1e9
    names = ", ".join(f"{gpu.name} {gpu.total_memory / 1e9:.1f} GB" for gpu in gpus)
    warning = "" if total >= 20 else "  WARNING: under 20 GB total; a 7B/8B fp16 model will offload or fail. Use GPU T4 x2."
    return f"{len(gpus)} GPU(s): {names}{warning}"


def rq2_done(paths, expected_ids):
    responses, _ = read_jsonl(paths.rq2 / "responses.jsonl")
    done = {str(row.get("id")) for row in responses}
    summary = paths.rq2 / "summary.json"
    return (
        summary.exists()
        and json.loads(summary.read_text()).get("prompt_version") == paths.rq2_prompt_version
        and expected_ids <= done
    )


def rq3_done(paths, rq3_rows):
    tasks, _ = read_jsonl(paths.rq3 / "tasks.jsonl")
    summary = paths.rq3 / "summary.json"
    if len(tasks) != rq3_rows or not summary.exists() or json.loads(summary.read_text()).get("max_turns") != MAX_TURNS:
        return False
    responses, _ = read_jsonl(paths.rq3 / "responses.jsonl")
    done = {(str(r.get("id")), r.get("condition")) for r in responses if "turn" not in r}
    return all((str(t["id"]), c) in done for t in tasks for c in RQ3_CONDITIONS)


def zip_results(paths, zip_dir):
    zip_dir.mkdir(parents=True, exist_ok=True)
    target = zip_dir / paths.zip_name
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths.root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(paths.results_root))
    temporary.replace(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", choices=tuple(INPUTS), default="sandbox",
                        help="scienceworld: extraction and RQ2 only, under results/<model>/sw, zipped separately")
    parser.add_argument("--input", help="default: data/sandbox.jsonl or data/scienceworld.jsonl by --env")
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--zip-dir", default=str(KAGGLE_WORKING if KAGGLE_WORKING.exists() else REPO_ROOT))
    parser.add_argument("--hf-cache", default="/tmp/hf_cache" if KAGGLE_WORKING.exists() else None)
    parser.add_argument("--device", choices=("cpu", "auto"), default="auto")
    parser.add_argument("--pilot-rows", type=int, default=50)
    parser.add_argument("--rq2-pairs", type=int, help="default: every eligible minimal pair")
    parser.add_argument("--rq3-rows", type=int, default=60)
    parser.add_argument("--rq3-batch", choices=("main", "extra"), default="main",
                        help="extra: the next --rq3-rows/2 pairs of the same selection, written to rq3/h3-extra")
    order = parser.add_mutually_exclusive_group()
    order.add_argument("--stop-after", choices=STEPS, default="zip")
    order.add_argument("--only", choices=STEPS, help="run one step (resuming it), then re-zip")
    args = parser.parse_args(argv)
    steps = tuple(dict.fromkeys((args.only, "zip"))) if args.only else STEPS[:STEPS.index(args.stop_after) + 1]
    if args.env == "scienceworld":
        if args.only and args.only not in SW_STEPS:
            parser.error(f"--env scienceworld runs only {', '.join(SW_STEPS)} (no ScienceWorld agent runs)")
        steps = tuple(step for step in steps if step in SW_STEPS)

    os.environ["PYTHONUNBUFFERED"] = "1"
    if args.hf_cache:
        os.environ["IRREV_HF_CACHE"] = args.hf_cache
    input_arg = args.input or INPUTS[args.env]
    input_path = Path(input_arg) if Path(input_arg).is_absolute() else REPO_ROOT / input_arg
    paths = Paths(Path(args.results_root).resolve(), args.model, args.rq3_batch, args.env)
    paths.root.mkdir(parents=True, exist_ok=True)
    rows, errors = read_jsonl(input_path)
    if errors:
        raise SystemExit(f"{input_path} has {len(errors)} malformed rows")
    all_ids = {str(row["id"]) for row in rows}
    available_pairs = len(candidate_pairs(rows, RQ2_HORIZON, RQ2_LABEL_MODE))
    rq2_pairs = args.rq2_pairs or available_pairs
    common = ["--model", args.model, "--input", str(input_path), "--device", args.device]
    print(f"model={args.model} results={paths.root} input rows={len(rows)}")
    print(gpu_report())
    print(f"env={args.env} RQ2 {paths.rq2_prompt_version} {RQ2_LABEL_MODE} h{RQ2_HORIZON} {RQ2_CONDITION}: {rq2_pairs} of {available_pairs} eligible pairs ({2 * rq2_pairs} rows)")
    if args.env == "sandbox":
        print(f"RQ3 {args.rq3_batch} batch: {args.rq3_rows} rows, conditions {', '.join(RQ3_CONDITIONS)}, output {paths.rq3}")
    print(f"steps: {', '.join(steps)}")
    incomplete = []

    for step in steps:
        if step == "pilot":
            if paths.pilot.exists():
                print(f"\n=== pilot: done, {paths.pilot.read_text()}")
                continue
            run_module(paths, step, "experiments.extract_hidden",
                       [*common, "--limit", str(args.pilot_rows), "--output-dir", str(paths.extraction)])
            projection = pilot_projection(paths, len(rows))
            if projection is None:
                raise SystemExit(f"Pilot produced no rows; see {paths.extraction / 'failures.jsonl'}")
            paths.pilot.write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(projection, indent=2))
        elif step == "disk":
            projection = json.loads(paths.pilot.read_text())
            for location in dict.fromkeys(str(p) for p in (paths.root, args.zip_dir, args.hf_cache) if p):
                Path(location).mkdir(parents=True, exist_ok=True)
                subprocess.run(["df", "-h", location])
            needed = projection["projected_remaining_extraction_gb"] + projection["projected_total_extraction_gb"] + 1
            free = shutil.disk_usage(paths.root).free / 1e9
            print(f"free {free:.1f} GB at results; need about {needed:.1f} GB (remaining arrays + zip + 1 GB margin)")
            if free < needed:
                raise SystemExit("Not enough disk space for extraction and zip")
        elif step == "extract":
            if all_ids <= completed_ids(paths.extraction / "index.jsonl"):
                print("\n=== extract: done")
                continue
            run_module(paths, step, "experiments.extract_hidden",
                       [*common, "--limit", str(len(rows)), "--output-dir", str(paths.extraction)])
            missing = len(all_ids - completed_ids(paths.extraction / "index.jsonl"))
            if missing:
                incomplete.append(f"extract: {missing} rows missing (see failures.jsonl; rerun retries them)")
        elif step == "rq2":
            if not rq2_pairs:
                incomplete.append(f"rq2: no eligible reversible/irreversible pairs in {input_path}")
                continue
            expected = {str(r["id"]) for r in select_stated_pairs(rows, 2 * rq2_pairs, RQ2_HORIZON, RQ2_LABEL_MODE, RQ2_SEED)}
            if rq2_done(paths, expected):
                print("\n=== rq2: done")
                continue
            run_module(paths, step, "experiments.rq2_stated", [
                *common, "--limit", str(2 * rq2_pairs), "--label-mode", RQ2_LABEL_MODE,
                "--horizon", str(RQ2_HORIZON), "--condition", RQ2_CONDITION,
                "--seed", str(RQ2_SEED), "--prompt-version", RQ2_PROMPT_VERSION,
                "--output-dir", str(paths.rq2),
            ])
            if not rq2_done(paths, expected):
                incomplete.append("rq2: not all rows answered")
        elif step == "rq3":
            if rq3_done(paths, args.rq3_rows):
                print("\n=== rq3: done")
                continue
            run_module(paths, step, "experiments.rq3_agent", [
                *common, "--limit", str(args.rq3_rows), "--condition", "both",
                "--horizon", "3", "--batch", args.rq3_batch, "--output-dir", str(paths.rq3),
            ])
            if not rq3_done(paths, args.rq3_rows):
                incomplete.append("rq3: not all task/condition runs finished")
        elif step == "zip":
            target = zip_results(paths, Path(args.zip_dir))
            print(f"\n=== zip: {target} ({target.stat().st_size / 1e9:.2f} GB)")

    print("\nINCOMPLETE:\n  " + "\n  ".join(incomplete) if incomplete else f"\nAll steps complete: {', '.join(steps)}.")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
