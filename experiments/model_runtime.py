"""Small local CPU runtime shared by stated-answer and sandbox-agent experiments."""
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    rows, errors = [], []
    path = Path(path)
    if not path.exists():
        return rows, errors
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"line": number, "error": str(exc), "raw": line.rstrip()[:500]})
    return rows, errors


def load_cpu_model(model_name, threads=2):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    cache_dir = REPO_ROOT / ".cache" / "huggingface"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    ).to("cpu")
    model.eval()
    return tokenizer, model


def render_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_chat(tokenizer, model, messages, max_new_tokens=160):
    import torch

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")
    input_len = int(input_ids.shape[-1])
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )
    return tokenizer.decode(output[0, input_len:], skip_special_tokens=True)


def parse_json_object(text):
    """Parse one JSON object, permitting a surrounding Markdown code fence only."""
    raw = str(text).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    decoder = json.JSONDecoder()
    start = raw.find("{")
    if start < 0:
        raise ValueError("Model output did not contain a JSON object")
    value, end = decoder.raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError("Model output JSON must be an object")
    if raw[start + end:].strip():
        raise ValueError("Unexpected text after model JSON object")
    return value
