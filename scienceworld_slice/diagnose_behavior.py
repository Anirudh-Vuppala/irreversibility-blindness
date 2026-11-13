"""Small replay, action-list, and world-drift diagnostic for ScienceWorld."""

import argparse
import json
from pathlib import Path
import time

from scienceworld import ScienceWorldEnv

from scienceworld_slice.core import FOCUS_VERBS, action_verb, canonicalize_tree, state_key


DEFAULT_TASKS = ("chemistry-mix", "change-the-state-of-matter-of", "find-plant")
FIXED_ACTIONS = {
    "chemistry-mix": ("mix door to kitchen", "mix kitchen", "open door to kitchen"),
    "change-the-state-of-matter-of": ("mix kitchen", "open door to kitchen"),
    "find-plant": ("open door to greenhouse", "move picture to greenhouse"),
}
WAIT_COMPARISON_ACTIONS = {
    "chemistry-mix": "mix door to kitchen",
    "change-the-state-of-matter-of": "open door to kitchen",
    "find-plant": "open door to greenhouse",
}


def flatten(tree, prefix=""):
    result = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            result.update(flatten(value, f"{prefix}/{key}"))
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            result.update(flatten(value, f"{prefix}/{index}"))
    else:
        result[prefix or "/"] = tree
    return result


def diff_fields(before, after):
    left, right = flatten(before), flatten(after)
    changes = []
    for path in sorted(set(left) | set(right)):
        first, second = left.get(path, "<absent>"), right.get(path, "<absent>")
        if first != second:
            changes.append({"path": path, "before": first, "after": second})
    return changes


def state_comparison(before, after):
    raw = diff_fields(before, after)
    canonical_before = canonicalize_tree(before)
    canonical_after = canonicalize_tree(after)
    canonical = diff_fields(canonical_before, canonical_after)
    approximate_equal = state_key(before, drop_temperatures=True) == state_key(
        after, drop_temperatures=True
    )
    return {
        "raw_diff_count": len(raw),
        "canonical_diff_count": len(canonical),
        "canonical_equal": not canonical,
        "equal_after_dropping_temperatures": approximate_equal,
        "temperature_changes": [item for item in raw if "temperature" in item["path"].casefold()],
        "outside_water_changes": [
            item for item in raw
            if any(word in item["path"].casefold() for word in ("outside", "fountain", "ground", "water"))
        ][:80],
        "raw_changes": raw[:120],
        "canonical_changes": canonical[:120],
    }


def reset_task(env, task):
    env.load(task, 0, "", generateGoldPath=True)
    observation, info = env.reset()
    return observation, info, list(env.get_gold_action_sequence())


def observation_only_control(env, task, sleep_seconds):
    reset_task(env, task)
    before = env.getObjectTree()
    calls = {
        "action_candidates": len(env.get_valid_action_object_combinations()),
        "possible_actions": len(env.get_possible_actions()),
        "look_chars": len(env.look()),
        "inventory_chars": len(env.inventory()),
        "task_description_chars": len(env.taskdescription()),
    }
    immediate = env.getObjectTree()
    time.sleep(sleep_seconds)
    after_sleep = env.getObjectTree()
    return {
        "task": task,
        "control": "read_only_calls_and_wall_clock_wait",
        "sleep_seconds": sleep_seconds,
        "read_only_call_counts": calls,
        "before_to_after_calls": state_comparison(before, immediate),
        "after_calls_to_after_sleep": state_comparison(immediate, after_sleep),
    }


def replay_state(env, task, history):
    reset_task(env, task)
    memberships = []
    for action in history:
        candidates = env.get_valid_action_object_combinations()
        memberships.append({"action": action, "in_pre_step_candidates": action in candidates})
        env.step(action)
    return env.getObjectTree(), memberships


def replay_control(env, task, history):
    first, first_memberships = replay_state(env, task, history)
    second, second_memberships = replay_state(env, task, history)
    return {
        "task": task,
        "control": "same_task_variation_reset_and_replay",
        "history": history,
        "first_replay_membership": first_memberships,
        "second_replay_membership": second_memberships,
        "comparison": state_comparison(first, second),
    }


def action_trial(env, task, action):
    reset_task(env, task)
    before = env.getObjectTree()
    candidates = env.get_valid_action_object_combinations()
    present = action in candidates
    result = {
        "task": task,
        "control": "candidate_action",
        "action": action,
        "pre_step_candidate_count": len(candidates),
        "in_pre_step_candidates": present,
    }
    if not present:
        result["executed"] = False
        return result

    observation, reward, done, info = env.step(action)
    after = env.getObjectTree()
    post_candidates = info.get("valid")
    result.update({
        "executed": True,
        "observation": observation,
        "reward": reward,
        "done": done,
        "post_step_valid_type": type(post_candidates).__name__,
        "post_step_valid_count": len(post_candidates) if isinstance(post_candidates, list) else None,
        "action_still_in_post_step_candidates": action in post_candidates if isinstance(post_candidates, list) else None,
        "state_difference": state_comparison(before, after),
    })
    return result


def run_one_step(env, task, action):
    reset_task(env, task)
    before = env.getObjectTree()
    candidates = env.get_valid_action_object_combinations()
    if action not in candidates:
        return before, before, {"action": action, "in_pre_step_candidates": False}
    observation, reward, done, info = env.step(action)
    after = env.getObjectTree()
    post_candidates = info.get("valid")
    return before, after, {
        "action": action,
        "in_pre_step_candidates": True,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info_valid_type": type(post_candidates).__name__,
        "info_valid_count": len(post_candidates) if isinstance(post_candidates, list) else None,
    }


def action_vs_wait_control(env, task, action):
    action_before, action_after, action_result = run_one_step(env, task, action)
    wait_before, wait_after, wait_result = run_one_step(env, task, "wait1")
    action_delta = diff_fields(canonicalize_tree(action_before), canonicalize_tree(action_after))
    wait_delta = diff_fields(canonicalize_tree(wait_before), canonicalize_tree(wait_after))
    action_paths = {item["path"]: item for item in action_delta}
    wait_paths = {item["path"]: item for item in wait_delta}
    shared = set(action_paths) & set(wait_paths)
    return {
        "task": task,
        "control": "candidate_action_vs_wait1",
        "candidate_result": action_result,
        "wait_result": wait_result,
        "initial_states": state_comparison(action_before, wait_before),
        "final_states": state_comparison(action_after, wait_after),
        "canonical_delta_counts": {"candidate": len(action_paths), "wait1": len(wait_paths)},
        "shared_canonical_paths": len(shared),
        "same_shared_transitions": sum(action_paths[path] == wait_paths[path] for path in shared),
        "candidate_only_paths": sorted(set(action_paths) - set(wait_paths))[:80],
        "wait_only_paths": sorted(set(wait_paths) - set(action_paths))[:80],
        "shared_transition_differences": [
            {"candidate": action_paths[path], "wait1": wait_paths[path]}
            for path in sorted(shared)
            if action_paths[path] != wait_paths[path]
        ][:80],
    }


def task_records(env, task, sleep_seconds, history_steps):
    _, _, gold = reset_task(env, task)
    initial_candidates = env.get_valid_action_object_combinations()
    preview = [
        action for action in initial_candidates
        if action_verb(action) in FOCUS_VERBS or action in ("wait", "wait1")
    ]
    gold_focus = next(
        (action for action in gold[:10] if action_verb(action) in FOCUS_VERBS), None
    )
    actions = [*FIXED_ACTIONS.get(task, ()), "wait1"]
    if gold_focus:
        actions.append(gold_focus)
    unique_actions = list(dict.fromkeys(actions))
    history = gold[:history_steps]
    return [
        {
            "record_type": "task_api_summary",
            "task": task,
            "variation": 0,
            "initial_candidate_count": len(initial_candidates),
            "initial_candidate_preview": preview[:100],
            "gold_action_prefix": gold[:10],
            "history_for_replay_control": history,
        },
        observation_only_control(env, task, sleep_seconds),
        replay_control(env, task, history),
        action_vs_wait_control(env, task, WAIT_COMPARISON_ACTIONS[task]),
        *[action_trial(env, task, action) for action in unique_actions],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--history-steps", type=int, default=2)
    parser.add_argument("--output", default="results/scienceworld_behavior_diagnostic.jsonl")
    args = parser.parse_args(argv)
    if args.sleep_seconds < 0 or args.history_steps < 0:
        parser.error("sleep-seconds and history-steps must be non-negative")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Refusing to overwrite existing diagnostic: {output}")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    try:
        installed = env.get_task_names()
        missing = [task for task in tasks if task not in installed]
        if missing:
            parser.error(f"Tasks unavailable in installed ScienceWorld: {missing}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            header = {
                "record_type": "diagnostic_header",
                "scienceworld_task_count": len(installed),
                "selected_tasks": tasks,
                "info_valid_semantics": "step() populates info['valid'] after the step by calling get_valid_action_object_combinations(); it is a list, not a boolean success flag",
                "candidate_semantics": "candidate strings are produced from visible objects and action-specific possible-valid-action generators; list membership is not evidence of a state change or task success",
            }
            stream.write(json.dumps(header, sort_keys=True) + "\n")
            for task in tasks:
                for record in task_records(env, task, args.sleep_seconds, args.history_steps):
                    record["record_type"] = record.get("record_type", "action_trial")
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
        print(f"Wrote diagnostic JSONL: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
