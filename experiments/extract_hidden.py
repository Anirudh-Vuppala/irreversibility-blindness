import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def model_slug(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def read_jsonl(path):
    rows, errors = [], []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc), "raw": line.rstrip("\n")[:500]})
    return rows, errors


def stratified_limit(rows, limit, label_key="label"):
    if limit is None or len(rows) <= limit:
        return list(rows)
    groups = defaultdict(lambda: defaultdict(deque))
    label_order = []
    scenario_order = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get("scenario_id") or row.get("group_id") or row.get("id") or index)
        label = str(row.get(label_key, "unknown"))
        if label not in groups:
            label_order.append(label)
        if group not in groups[label]:
            scenario_order[label].append(group)
        groups[label][group].append(row)
    selected = []
    used_scenarios = set()
    while len(selected) < limit:
        added = False
        for label in label_order:
            unseen = [group for group in scenario_order[label] if group not in used_scenarios and groups[label][group]]
            candidates = unseen or [group for group in scenario_order[label] if groups[label][group]]
            if candidates and len(selected) < limit:
                group = candidates[0]
                selected.append(groups[label][group].popleft())
                used_scenarios.add(group)
                added = True
        if not added:
            break
    return selected


def action_char_span(rendered, action_text):
    start = rendered.rfind(str(action_text))
    if start < 0:
        raise ValueError("Candidate action is not present in the rendered chat prompt")
    return start, start + len(str(action_text))


def token_span_from_offsets(offsets, char_span):
    char_start, char_end = char_span
    indices = [
        i for i, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        raise ValueError(f"No tokenizer offsets overlap candidate action span {char_span}")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Candidate action tokens are not a contiguous span")
    return indices[0], indices[-1] + 1


def make_messages(row):
    task_text = row.get("task_text", "")
    state_text = row.get("state_text", "")
    action_text = row.get("action_text", "")
    return [
        {
            "role": "system",
            "content": "Represent the candidate action in context. Do not add commentary.",
        },
        {
            "role": "user",
            "content": f"Task:\n{task_text}\n\nCurrent state:\n{state_text}\n\nConsider the candidate action below.",
        },
        {"role": "assistant", "content": str(action_text)},
    ]


def save_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def atomic_numpy_save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed_ids(index_path):
    rows, errors = read_jsonl(index_path) if index_path.exists() else ([], [])
    if errors:
        raise ValueError(f"Malformed extraction index {index_path}: {errors[0]}")
    return {str(row["id"]) for row in rows if row.get("id")}


def extract_one(model, tokenizer, row, output_dir, max_tokens, prompt_callback=None):
    import torch

    messages = make_messages(row)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    char_span = action_char_span(rendered, row["action_text"])
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    start, end = token_span_from_offsets(offsets, char_span)
    if encoded["input_ids"].shape[1] > max_tokens:
        raise ValueError(
            f"Prompt has {encoded['input_ids'].shape[1]} tokens, above --max-tokens={max_tokens}"
        )
    if end <= start:
        raise ValueError("Candidate action token span is empty")

    prompt_record = {
        "id": str(row["id"]),
        "model": row.get("model") or "",
        "messages": messages,
        "rendered_prompt": rendered,
        "action_char_span": list(char_span),
        "action_token_span": [start, end],
        "input_token_count": int(encoded["input_ids"].shape[1]),
    }
    if prompt_callback is not None:
        prompt_callback(prompt_record)

    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    layers = []
    for hidden in output.hidden_states:
        last = hidden[0, end - 1, :].float().cpu()
        mean = hidden[0, start:end, :].float().mean(dim=0).cpu()
        layers.append(torch.stack((last, mean), dim=0).numpy().astype(np.float16))
    array = np.stack(layers, axis=0)
    array_path = output_dir / "arrays" / f"{hashlib.sha256(str(row['id']).encode()).hexdigest()[:24]}.npy"
    atomic_numpy_save(array_path, array)

    index_record = {
        "id": str(row["id"]),
        "environment": row.get("environment"),
        "label": row.get("label"),
        "labels_by_horizon": row.get("labels_by_horizon"),
        "horizon": row.get("horizon"),
        "scenario_id": row.get("scenario_id"),
        "group_id": row.get("group_id"),
        "variant": row.get("variant"),
        "is_minimal_pair": row.get("is_minimal_pair", False),
        "action_text": row["action_text"],
        "verb": row.get("verb") or str(row["action_text"]).split(maxsplit=1)[0],
        "model": row.get("model"),
        "array_path": str(array_path.relative_to(output_dir)),
        "shape": list(array.shape),
        "positions": ["last_action_token", "mean_action_tokens"],
        "action_token_span": [start, end],
        "input_token_count": int(encoded["input_ids"].shape[1]),
    }
    return prompt_record, index_record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract all-layer action representations in one pass")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--input", default="data/sandbox.jsonl")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_tokens < 1 or args.threads < 1:
        parser.error("limit, max-tokens, and threads must be positive")

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "extractions" / model_slug(args.model)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompts.jsonl"
    index_path = output_dir / "index.jsonl"
    failures_path = output_dir / "failures.jsonl"
    summary_path = output_dir / "summary.json"

    rows, parse_errors = read_jsonl(Path(args.input))
    if parse_errors:
        for item in parse_errors:
            save_jsonl(failures_path, {"stage": "input-parse", **item})
        raise SystemExit(f"Input has {len(parse_errors)} malformed JSONL rows; see {failures_path}")
    selected = stratified_limit(rows, args.limit)
    if not selected:
        raise SystemExit("No input rows selected")
    existing = completed_ids(index_path)
    prompts_existing, prompt_errors = read_jsonl(prompt_path) if prompt_path.exists() else ([], [])
    if prompt_errors:
        for item in prompt_errors:
            save_jsonl(failures_path, {"stage": "prompt-log-parse", **item})
        raise SystemExit(f"Prompt log has {len(prompt_errors)} malformed rows; see {failures_path}")
    prompt_ids = {str(row.get("id")) for row in prompts_existing}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    cache_dir = REPO_ROOT / ".cache" / "huggingface"
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, token=False, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=False,
            cache_dir=cache_dir,
        ).to("cpu")
        model.eval()
    except Exception as exc:
        save_jsonl(failures_path, {
            "stage": "model-load",
            "model": args.model,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise

    completed = 0
    failed = 0
    for row in selected:
        row_id = str(row.get("id", ""))
        if not row_id or not row.get("action_text"):
            failed += 1
            save_jsonl(failures_path, {
                "stage": "input-row",
                "id": row_id or None,
                "error": "Missing required id or action_text",
            })
            continue
        if row_id in existing:
            continue
        row["model"] = args.model
        try:
            def save_prompt(prompt_record):
                if row_id not in prompt_ids:
                    save_jsonl(prompt_path, prompt_record)
                    prompt_ids.add(row_id)

            prompt_record, index_record = extract_one(
                model, tokenizer, row, output_dir, args.max_tokens,
                prompt_callback=save_prompt,
            )
            save_jsonl(index_path, index_record)
            existing.add(row_id)
            completed += 1
        except Exception as exc:
            failed += 1
            save_jsonl(failures_path, {
                "stage": "extract",
                "id": row_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"failed row {row_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary = {
        "model": args.model,
        "input": str(Path(args.input)),
        "selected": len(selected),
        "already_completed": len(selected) - completed - failed,
        "completed_this_run": completed,
        "failed_this_run": failed,
        "array_layout": "layers x [last_action_token, mean_action_tokens] x hidden_size; float16",
        "device": "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "prompts": str(prompt_path),
        "index": str(index_path),
        "failures": str(failures_path),
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
