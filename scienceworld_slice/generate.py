import argparse
import atexit
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

from scienceworld_slice.core import (
    FOCUS_VERBS,
    action_verb,
    build_parent_index,
    canonicalize_tree,
    expand_with_parents,
    focus_candidates,
    labels_from_depths,
    restoration_candidates,
    state_digest,
    state_key,
)


DEFAULT_TASKS = ("chemistry-mix", "change-the-state-of-matter-of", "find-plant")
_WORKER_ENV = None


def _stable_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _write_jsonl_append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _append_failure(path, failure):
    _write_jsonl_append(path, failure)


def _init_worker():
    global _WORKER_ENV
    from scienceworld import ScienceWorldEnv

    _WORKER_ENV = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    atexit.register(_close_worker)


def _close_worker():
    global _WORKER_ENV
    if _WORKER_ENV is not None:
        try:
            _WORKER_ENV.close()
        finally:
            _WORKER_ENV = None


def _rebuild(job, extra_actions=()):
    env = _WORKER_ENV
    env.load(job["task_name"], job["variation"], "", generateGoldPath=False)
    observation, info = env.reset()
    for action in [*job["history"], *extra_actions]:
        observation, _, _, info = env.step(action)
        if not _step_action_valid(info, action):
            raise RuntimeError(f"Replay action became invalid: {action!r}")
    return observation, info


def _action_rows(env):
    strings = env.get_valid_action_object_combinations()
    templates = env.get_valid_action_object_combinations_with_templates()
    return strings, templates


def _step_action_valid(info, action):
    value = info.get("valid")
    if not isinstance(value, bool):
        raise RuntimeError(
            "Unsupported ScienceWorld step() validity field: expected a boolean "
            f"for action {action!r}, got {type(value).__name__}. "
            "ScienceWorld 1.2.3 returns the full valid-action list in info['valid']; "
            "the action-level restoration check must be resolved before generating labels."
        )
    return value


def _run_candidate(job, max_depth):
    started = time.perf_counter()
    rebuild_count = 0
    evaluated_edges = 0
    observation, _ = _rebuild(job)
    rebuild_count += 1
    target_raw = _WORKER_ENV.getObjectTree()
    target = canonicalize_tree(target_raw)
    parent_index = build_parent_index(target)
    initial_ids = {int(value) for value in job["object_ids"]}
    allowed_ids = expand_with_parents(initial_ids, parent_index)

    observation, _, _, info = _WORKER_ENV.step(job["action_text"])
    if not _step_action_valid(info, job["action_text"]):
        return {
            "status": "invalid",
            "id": job["id"],
            "reason": "candidate rejected by ScienceWorld step()",
            "seconds": time.perf_counter() - started,
            "rebuild_count": rebuild_count,
            "evaluated_edges": evaluated_edges,
        }
    post = canonicalize_tree(_WORKER_ENV.getObjectTree())

    exact_depth = 0 if state_key(post) == state_key(target) else None
    partial_depth = 0 if state_key(post, drop_temperatures=True) == state_key(target, drop_temperatures=True) and exact_depth is None else None
    restore_sequence = [] if exact_depth == 0 or partial_depth == 0 else None
    visited = {state_digest(post)}
    frontier = [()]

    for depth in range(1, max_depth + 1):
        next_frontier = []
        found_exact = False
        for path in frontier:
            _rebuild(job, [job["action_text"], *path])
            rebuild_count += 1
            node_tree = canonicalize_tree(_WORKER_ENV.getObjectTree())
            action_strings, template_rows = _action_rows(_WORKER_ENV)
            candidates = restoration_candidates(
                action_strings, template_rows, allowed_ids, job["action_text"]
            )

            for action in candidates:
                _rebuild(job, [job["action_text"], *path])
                rebuild_count += 1
                observation, _, _, action_info = _WORKER_ENV.step(action)
                evaluated_edges += 1
                if not _step_action_valid(action_info, action):
                    continue
                next_state = canonicalize_tree(_WORKER_ENV.getObjectTree())
                new_path = (*path, action)

                if state_key(next_state) == state_key(target):
                    exact_depth = depth
                    restore_sequence = list(new_path)
                    found_exact = True
                    break

                if (
                    partial_depth is None
                    and state_key(next_state, drop_temperatures=True)
                    == state_key(target, drop_temperatures=True)
                ):
                    partial_depth = depth
                    if restore_sequence is None:
                        restore_sequence = list(new_path)

                digest = state_digest(next_state)
                if depth < max_depth and digest not in visited:
                    visited.add(digest)
                    next_frontier.append(new_path)
            if found_exact:
                break
        if found_exact or depth == max_depth:
            break
        frontier = next_frontier
        if not frontier:
            break

    labels = labels_from_depths(exact_depth, partial_depth, max_depth)
    horizon = max_depth
    row = {
        "id": job["id"],
        "environment": "scienceworld",
        "state_text": job["state_text"],
        "action_text": job["action_text"],
        "label": labels[str(horizon)],
        "horizon": horizon,
        "source": "scienceworld-gold-replay-v1",
        "labels_by_horizon": labels,
        "scenario_id": job["scenario_id"],
        "group_id": job["group_id"],
        "variant": job["variant"],
        "is_minimal_pair": False,
        "task_text": job["task_text"],
        "verb": action_verb(job["action_text"]),
        "state_json": target,
        "action_json": {
            "action": job["action_text"],
            "obj_ids": job["object_ids"],
            "template_id": job["template_id"],
        },
        "restore_sequence": restore_sequence or [],
        "restore_depth": exact_depth if exact_depth is not None else partial_depth,
        "history_length": len(job["history"]),
        "task_name": job["task_name"],
        "variation": job["variation"],
    }
    return {
        "status": "ok",
        "row": row,
        "seconds": time.perf_counter() - started,
        "rebuild_count": rebuild_count,
        "evaluated_edges": evaluated_edges,
    }


def _worker_run(job):
    try:
        return _run_candidate(job, int(job["max_depth"]))
    except Exception as exc:
        return {
            "status": "error",
            "id": job.get("id"),
            "task_name": job.get("task_name"),
            "action_text": job.get("action_text"),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def collect_proposals(tasks, variations_per_task, max_history, actions_per_verb, seed, depth3_fraction):
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    proposals = []
    installed_names = env.get_task_names()
    missing = [task for task in tasks if task not in installed_names]
    if missing:
        env.close()
        raise ValueError(f"Tasks unavailable in installed ScienceWorld: {missing}")

    try:
        for task in tasks:
            max_variations = env.get_max_variations(task)
            for variation in range(min(max_variations, variations_per_task)):
                env.load(task, variation, "", generateGoldPath=True)
                observation, _ = env.reset()
                gold = env.get_gold_action_sequence()
                if not gold or gold[0].startswith("ERROR:"):
                    raise RuntimeError(f"No gold sequence for {task} variation {variation}")
                history = []
                history_object_ids = []
                stop_prefix = min(max_history, len(gold))

                for prefix in range(stop_prefix + 1):
                    tree = canonicalize_tree(env.getObjectTree())
                    action_strings, template_rows = _action_rows(env)
                    metadata = {}
                    for item in template_rows:
                        if item.get("action"):
                            metadata.setdefault(str(item["action"]), item)

                    context_ids = set()
                    for obj_ids in history_object_ids[-2:]:
                        context_ids.update(int(value) for value in obj_ids)
                    next_gold = gold[prefix] if prefix < len(gold) else None
                    if next_gold in metadata:
                        context_ids.update(int(value) for value in metadata[next_gold].get("obj_ids", []))
                    context_ids = {
                        int(value)
                        for value in expand_with_parents(context_ids, build_parent_index(tree))
                    }

                    state_rng = random.Random(f"{seed}:{task}:{variation}:{prefix}")
                    selected = focus_candidates(
                        action_strings, template_rows, context_ids
                    )
                    if next_gold and action_verb(next_gold) in FOCUS_VERBS:
                        next_meta = metadata.get(next_gold)
                        if next_meta and next_gold in action_strings:
                            next_ids = {int(value) for value in next_meta.get("obj_ids", [])}
                            if next_ids and next_ids.intersection(context_ids):
                                selected = [(next_gold, next_meta)] + [
                                    item for item in selected if item[0] != next_gold
                                ]
                    if selected:
                        selected = sorted(
                            selected,
                            key=lambda pair: (
                                FOCUS_VERBS.index(action_verb(pair[0])),
                                pair[0],
                            ),
                        )
                        by_verb = defaultdict(list)
                        for candidate in selected:
                            by_verb[action_verb(candidate[0])].append(candidate)
                        balanced = []
                        for verb in FOCUS_VERBS:
                            candidates = by_verb.get(verb, [])
                            state_rng.shuffle(candidates)
                            balanced.extend(candidates[:actions_per_verb])
                        if next_gold:
                            next_item = metadata.get(next_gold)
                            if next_item and any(item[0] == next_gold for item in selected):
                                balanced = [(next_gold, next_item)] + [
                                    item for item in balanced if item[0] != next_gold
                                ]
                        seen_actions = set()
                        for action_text, item in balanced:
                            if action_text in seen_actions:
                                continue
                            seen_actions.add(action_text)
                            object_ids = [int(value) for value in item.get("obj_ids", [])]
                            if not object_ids:
                                continue
                            pair_hash = _stable_hash(action_text)[:10]
                            row_id = f"sw-{task}-v{variation:03d}-p{prefix:02d}-{pair_hash}"
                            depth3 = int(_stable_hash(row_id)[:8], 16) / 0xFFFFFFFF < depth3_fraction
                            proposals.append({
                                "id": row_id,
                                "task_name": task,
                                "variation": variation,
                                "history": list(history),
                                "history_length": len(history),
                                "state_text": observation,
                                "state_json": tree,
                                "task_text": env.get_task_description(),
                                "action_text": action_text,
                                "object_ids": object_ids,
                                "template_id": item.get("template_id"),
                                "scenario_id": f"sw-{task}-v{variation:03d}",
                                "group_id": f"sw-{task}-v{variation:03d}-{pair_hash}",
                                "variant": f"prefix-{prefix:02d}",
                                "depth3": depth3,
                                "max_depth": 3 if depth3 else 2,
                            })

                    if prefix < stop_prefix:
                        action = gold[prefix]
                        gold_item = metadata.get(action)
                        observation, _, _, info = env.step(action)
                        if not _step_action_valid(info, action):
                            raise RuntimeError(
                                f"Gold action invalid at {task} variation {variation} step {prefix}: {action}"
                            )
                        history.append(action)
                        history_object_ids.append(
                            [int(value) for value in (gold_item or {}).get("obj_ids", [])]
                        )
    finally:
        env.close()
    return proposals, installed_names


def round_robin(proposals, tasks, limit):
    per_task = {task: [p for p in proposals if p["task_name"] == task] for task in tasks}
    selected = []
    while len(selected) < limit:
        found = False
        for task in tasks:
            if per_task[task]:
                selected.append(per_task[task].pop(0))
                found = True
                if len(selected) >= limit:
                    break
        if not found:
            break
    return selected


def mark_minimal_pairs(path):
    rows = _read_jsonl(path)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("group_id")].append(row)
    for group in groups.values():
        labels_differ = any(
            first.get("variant") != second.get("variant")
            and any(
                first.get("labels_by_horizon", {}).get(str(h)) is not None
                and second.get("labels_by_horizon", {}).get(str(h)) is not None
                and first["labels_by_horizon"][str(h)] != second["labels_by_horizon"][str(h)]
                for h in (1, 2, 3)
            )
            for index, first in enumerate(group)
            for second in group[index + 1:]
        )
        for row in group:
            row["is_minimal_pair"] = labels_differ
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a replay-labeled ScienceWorld slice")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--variations-per-task", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=14)
    parser.add_argument("--min-history", type=int, default=0)
    parser.add_argument("--actions-per-verb", type=int, default=1)
    parser.add_argument("--limit", type=int, default=300, help="Maximum successfully labeled pairs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--depth3-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/scienceworld.jsonl")
    parser.add_argument("--failures", default="results/scienceworld_generation_failures.jsonl")
    parser.add_argument("--metadata", default="results/scienceworld_generation.json")
    args = parser.parse_args(argv)

    if args.limit < 1 or args.variations_per_task < 1 or args.actions_per_verb < 1:
        parser.error("limit, variations-per-task, and actions-per-verb must be positive")
    if not 0 <= args.max_history < 15 or not 0 <= args.min_history <= args.max_history:
        parser.error("--max-history must be from 0 to 14 so recorded histories stay shorter than 15")
    if args.workers < 1 or not 0 <= args.depth3_fraction <= 1:
        parser.error("workers must be positive and depth3-fraction must be in [0, 1]")

    output = Path(args.output)
    failures = Path(args.failures)
    metadata_path = Path(args.metadata)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    started = time.perf_counter()
    try:
        proposals, installed_names = collect_proposals(
            tasks, args.variations_per_task, args.max_history, args.actions_per_verb,
            args.seed, args.depth3_fraction,
        )
    except Exception as exc:
        failure = {
            "stage": "collect-proposals",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _append_failure(failures, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    proposals = [proposal for proposal in proposals if proposal["history_length"] >= args.min_history]

    existing_rows = _read_jsonl(output)
    completed_ids = {row["id"] for row in existing_rows if "id" in row}
    remaining = max(0, args.limit - len(completed_ids))
    pending_proposals = [p for p in proposals if p["id"] not in completed_ids]
    selected = round_robin(pending_proposals, tasks, max(remaining * 3, remaining))

    counts = {"completed": 0, "invalid_candidates": 0, "errors": 0, "rebuilds": 0, "edges": 0}
    durations = []
    job_iter = iter(selected)
    pending = set()
    inflight_limit = min(max(remaining, 1), args.workers * 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
        def submit_next():
            try:
                job = next(job_iter)
            except StopIteration:
                return False
            future = executor.submit(_worker_run, job)
            pending.add(future)
            return True

        if remaining > 0:
            while len(pending) < inflight_limit and submit_next():
                pass
        while pending and counts["completed"] < remaining:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                counts["rebuilds"] += int(result.get("rebuild_count", 0))
                counts["edges"] += int(result.get("evaluated_edges", 0))
                if result["status"] == "ok":
                    if counts["completed"] < remaining:
                        _write_jsonl_append(output, result["row"])
                        counts["completed"] += 1
                        durations.append(float(result["seconds"]))
                elif result["status"] == "invalid":
                    counts["invalid_candidates"] += 1
                    _append_failure(failures, result)
                else:
                    counts["errors"] += 1
                    _append_failure(failures, result)
            while len(pending) < inflight_limit and counts["completed"] < remaining and submit_next():
                pass

    total_rows = mark_minimal_pairs(output) if output.exists() else 0
    metadata = {
        "scienceworld_version": "1.2.3",
        "installed_task_names": installed_names,
        "selected_tasks": tasks,
        "limit": args.limit,
        "variations_per_task": args.variations_per_task,
        "max_history": args.max_history,
        "min_history": args.min_history,
        "actions_per_verb": args.actions_per_verb,
        "depth3_fraction": args.depth3_fraction,
        "workers": args.workers,
        "proposals": len(proposals),
        "already_completed": len(completed_ids),
        "run_counts": counts,
        "total_output_rows": total_rows,
        "mean_seconds_per_completed_pair": sum(durations) / len(durations) if durations else None,
        "elapsed_seconds": time.perf_counter() - started,
        "state_comparison": "canonical tree; move counters removed; temperature fields rounded to 0.1 C; temperature-only matches labeled partially reversible",
        "state_reconstruction": "reload task and variation, reset, replay recorded action history",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if counts["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
