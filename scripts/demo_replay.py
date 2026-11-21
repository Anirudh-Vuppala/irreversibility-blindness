"""Replay one saved RQ3 pair in the terminal: the reversible version, then the irreversible version.

Reads only results/<model>/rq3/h3 and h3-extra (tasks, responses, tool calls); no model, GPU or network.

    python scripts/demo_replay.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/demo_replay.py --model meta-llama/Llama-3.1-8B-Instruct --condition explicit-carefulness \
        --pair-id sb-0001-p2 --delay 0
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCHES = ("h3", "h3-extra")
COMPLETE = ("ok", "max_turns")
CONDITIONS = ("neutral", "explicit-carefulness")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def model_dir(results_root, model):
    return Path(results_root) / model.replace("/", "_")


def load_episodes(directory):
    """{pair_id: {"batch", "tasks": {label: task}, "outcomes": {(id, condition): ...}, "calls", "turns"}}."""
    pairs = {}
    for batch in BATCHES:
        base = directory / "rq3" / batch
        tasks = read_jsonl(base / "tasks.jsonl")
        if not tasks:
            continue
        responses = read_jsonl(base / "responses.jsonl")
        calls, turns = defaultdict(list), defaultdict(list)
        for call in read_jsonl(base / "tool_calls.jsonl"):
            calls[(str(call["id"]), call["condition"])].append(call)
        for response in responses:
            if "turn" in response:
                turns[(str(response["id"]), response["condition"])].append(response)
        outcomes = {(str(r["id"]), r["condition"]): r for r in responses if "turn" not in r}
        for task in tasks:
            entry = pairs.setdefault(str(task["pair_id"]), {"batch": batch, "tasks": {}, "outcomes": outcomes,
                                                           "calls": calls, "turns": turns})
            entry["tasks"][task["label"]] = task
    return pairs


def complete(entry, condition):
    outcomes = [entry["outcomes"].get((str(entry["tasks"][label]["id"]), condition))
                for label in ("reversible", "irreversible")]
    return all(o and o.get("status") in COMPLETE for o in outcomes), outcomes


def default_pair(pairs, condition):
    """A pair complete in both versions; prefer both finished, then one showing a cautious step, then fewest turns."""
    candidates = []
    for pair_id, entry in pairs.items():
        if {"reversible", "irreversible"} - set(entry["tasks"]):
            continue
        ok, outcomes = complete(entry, condition)
        if not ok:
            continue
        finished = all(o["status"] == "ok" for o in outcomes)
        cautious = any(o.get("cautious_before_target") for o in outcomes)
        turns = sum(int(o.get("turns") or 0) for o in outcomes)
        candidates.append(((not finished, not cautious, turns, pair_id), pair_id))
    if not candidates:
        raise SystemExit(f"no pair completed in both versions under condition {condition}")
    return min(candidates)[1]


def format_call(call):
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (call.get("args") or {}).items())
    return f"{call.get('tool')}({args})"


def replay_episode(entry, label, condition, delay, out):
    task = entry["tasks"][label]
    key = (str(task["id"]), condition)
    outcome = entry["outcomes"][key]
    calls = {int(c["turn"]): c for c in entry["calls"].get(key, [])}
    turns = sorted(entry["turns"].get(key, []), key=lambda t: int(t["turn"]))
    out(f"=== {label.upper()} version ({task['id']}), condition: {condition} ===")
    out(f"Task: {task.get('task_text', '')}")
    out("Workspace:")
    for line in task.get("state_text", "").splitlines():
        out(f"  {line}")
    out(f"Requested action: {task['action_text']}")
    out("")
    time.sleep(delay)
    for turn in turns:
        number = int(turn["turn"])
        call = calls.get(number)
        if call:
            status = "" if call["ok"] else "  [tool error]"
            out(f"step {number + 1}: {format_call(call['tool_call'])}{status}")
            out(f"         -> {call['observation']}")
        else:
            out(f"step {number + 1}: (no valid tool call) {turn.get('raw_output', '').strip()[:160]}")
        time.sleep(delay)
    cautious = outcome.get("cautious_before_target")
    tools = ", ".join(outcome.get("caution_tools_before_target") or [])
    out(f"Status: {outcome.get('status')}, {outcome.get('turns')} turns, target action executed: "
        f"{'yes' if outcome.get('target_action_executed') else 'no'}")
    out(f"Cautious step before the target action: {'yes (' + tools + ')' if cautious else 'no'}")
    out("")


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--condition", choices=CONDITIONS, default="neutral")
    parser.add_argument("--pair-id")
    parser.add_argument("--delay", type=float, default=0.8, help="seconds between steps")
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    args = parser.parse_args(argv)

    pairs = load_episodes(model_dir(args.results_root, args.model))
    if not pairs:
        raise SystemExit(f"no saved RQ3 tasks under {model_dir(args.results_root, args.model)}/rq3")
    pair_id = args.pair_id or default_pair(pairs, args.condition)
    if pair_id not in pairs:
        raise SystemExit(f"pair {pair_id} not found in {', '.join(BATCHES)}")
    ok, _ = complete(pairs[pair_id], args.condition)
    if not ok:
        raise SystemExit(f"pair {pair_id} did not complete in both versions under {args.condition}")
    note = "" if args.pair_id else " (default: completed in both versions)"
    out(f"{args.model}, RQ3 pair {pair_id}{note}, batch {pairs[pair_id]['batch']}")
    out("")
    for label in ("reversible", "irreversible"):
        replay_episode(pairs[pair_id], label, args.condition, args.delay, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
