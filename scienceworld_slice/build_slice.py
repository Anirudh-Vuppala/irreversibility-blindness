"""Generate the projected-state ScienceWorld slice (v3).

Pipeline
1. Proposals: for each task variation, walk the gold path. At every prefix,
   list unambiguous candidate actions with a probe verb that touch objects the
   gold path uses. Candidates are grouped by (verb, object ids); a group keeps
   up to ``--prefixes-per-key`` prefixes whose projected pre-state differs, so
   the same action is labelled under different prior states (minimal pairs).
2. Labelling (worker): reconstruct the state by reset + replay; record a
   matched wait control (1..4 x ``wait1``); apply the action; drop it if the
   episode ends or the projection equals the one-wait control (no effect);
   breadth-first search restoration sequences of length <= 3 over actions on
   the same objects; compare each node with the wait control of equal length.
3. Labels for horizons 1, 2, 3 come from the same search; ``horizon`` = 3.
4. Finalize: mark minimal pairs (same group and pair key, different labels)
   and cap any class at ``--class-cap`` of rows by dropping unpaired rows.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from scienceworld_slice.projection import (
    allowed_restoration_ids,
    failed_observation,
    ignored_child,
    discrete_key,
    index_tree,
    is_room,
    labels_from_depths,
    match,
    project,
    root_ids_for,
)

SOURCE = "scienceworld-projected-bfs-v3"
DEFAULT_TASKS = ("chemistry-mix", "boil", "freeze", "find-plant")
PROBE_VERBS = ("open", "close", "activate", "deactivate", "move", "pick up", "pour", "mix")
RESTORE_VERBS = ("open", "close", "activate", "deactivate", "move", "pick up", "pour", "put down")
MULTIWORD_VERBS = ("pick up", "put down")
MAX_DEPTH = 3
LABELS = ("reversible", "partially_reversible", "irreversible")
_ENV = None


def action_verb(action_text):
    text = str(action_text).strip().casefold()
    for verb in MULTIWORD_VERBS:
        if text.startswith(verb + " "):
            return verb
    return text.split(maxsplit=1)[0] if text else ""


def stable_hash(value, length=10):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def unambiguous_templates(env):
    """Map action text -> obj_ids for texts that resolve to exactly one object tuple."""
    by_text = defaultdict(set)
    for item in env.get_valid_action_object_combinations_with_templates():
        action = item.get("action")
        if action:
            by_text[str(action)].add(tuple(str(value) for value in item.get("obj_ids", [])))
    return {text: list(ids[0]) for text, ids in ((t, list(s)) for t, s in by_text.items()) if len(ids) == 1}


def make_env():
    from scienceworld import ScienceWorldEnv

    return ScienceWorldEnv("", serverPath=None, envStepLimit=10000)


def fast_step(env, action):
    """Server step without the info dict (look/inventory/valid-action lists) that env.step builds."""
    observation = env.server.step(action)
    done = bool(env.server.getCompleted()) or env.server.getScore() < 0
    return observation, done


def rebuild(env, task, variation, actions):
    """Reconstruct a state: (re)load if needed, reset, replay ``actions``.

    Mirrors ``env.reset()`` (server reset followed by one ``look around`` step) but
    skips the per-step info dict, which dominates replay time.
    """
    if getattr(env, "_slice_loaded", None) != (task, variation):
        env.load(task, variation, "", generateGoldPath=False)
        env._slice_loaded = (task, variation)
    env.server.reset()
    env.lastStepScore = 0
    observation = env.server.step("look around")
    for action in actions:
        observation = env.server.step(action)
    return observation


def snapshot(env, root_ids):
    tree = env.getObjectTree()
    return project(tree, env.look(), root_ids), tree


# --------------------------------------------------------------------------- object helpers

def _node(index, uuid):
    entry = index.get(str(uuid))
    return entry[0] if entry else None


FIXTURE_WORDS = (
    "cupboard", "counter", "table", "chair", "fridge", "freezer", "oven", "stove", "sink",
    "bathtub", "toilet", "bed", "closet", "drawer", "desk", "workbench", "shelf", "bookcase",
    "couch", "sofa", "fireplace", "furnace", "kiln", "hand mill", "picture", "painting",
)


def is_movable(index, uuid):
    """Movable in the simulator and not a fixture (ScienceWorld lets the agent move cupboards)."""
    node = _node(index, uuid)
    if not node or not (node.get("propMoveable") or {}).get("isMovable"):
        return False
    name = str(node.get("name", "")).casefold()
    return not any(word in name.split() or name.startswith(word) for word in FIXTURE_WORDS)


def is_descendant(index, uuid, ancestor):
    current = (index.get(str(uuid)) or (None, None))[1]
    while current is not None:
        if current == str(ancestor):
            return True
        current = (index.get(current) or (None, None))[1]
    return False


def is_container(index, uuid):
    node = _node(index, uuid)
    return bool(node and str((node.get("propContainer") or {}).get("isContainer")).lower() == "true")


def content_count(index, uuid):
    node = _node(index, uuid)
    if not node:
        return 0
    return sum(
        1 for child in (node.get("contents") or {}).values()
        if isinstance(child, dict) and not child.get("isDeleted") and not ignored_child(child.get("name"))
    )


def has_substance(index, uuid):
    """True when a container holds a pourable substance (material with a substance name)."""
    node = _node(index, uuid)
    return any(
        isinstance(child, dict) and not child.get("isDeleted") and child.get("name") != "air"
        and (child.get("propMaterial") or {}).get("substanceName")
        and child.get("propEdibility") is None
        for child in ((node or {}).get("contents") or {}).values()
    )


def is_agent_side(index, uuid):
    node = _node(index, uuid)
    return bool(node and node.get("name") in ("agent", "inventory"))


def probe_ok(verb, obj_ids, index):
    """Physical sanity filter for proposed actions (nonsense pours/moves are not proposed)."""
    if len(set(obj_ids)) != len(obj_ids) or any(is_agent_side(index, value) for value in obj_ids):
        return False
    if verb in ("pour", "move"):
        if len(obj_ids) != 2:
            return False
        source, target = obj_ids
        parent = (index.get(source) or (None, None))[1]
        if not is_movable(index, source) or not is_container(index, target) or target == parent:
            return False
        if is_descendant(index, target, source):
            return False
        return verb == "move" or has_substance(index, source)
    if verb == "mix":
        return is_container(index, obj_ids[0]) and content_count(index, obj_ids[0]) >= 2
    if verb == "pick up":
        parent = (index.get(obj_ids[0]) or (None, None))[1]
        return is_movable(index, obj_ids[0]) and not is_agent_side(index, parent)
    return True


def shallow_state(index, uuid):
    node = _node(index, uuid)
    if node is None:
        return None
    container = node.get("propContainer") or {}
    device = node.get("propDevice") or {}
    material = node.get("propMaterial") or {}
    children = sorted(
        str(child.get("uuid")) for child in (node.get("contents") or {}).values()
        if isinstance(child, dict) and not child.get("isDeleted") and not ignored_child(child.get("name"))
    )
    return (index[str(uuid)][1], container.get("isOpen"), device.get("isActivated"),
            material.get("stateOfMatter"), node.get("isDeleted"), tuple(children))


def changed_ids(index_a, index_b, ids):
    return {value for value in ids if shallow_state(index_a, value) != shallow_state(index_b, value)}


# --------------------------------------------------------------------------- proposals

def _gold_history(env, task, variation, max_prefix):
    env.load(task, variation, "", generateGoldPath=True)
    env._slice_loaded = None
    env.reset()
    gold = list(env.get_gold_action_sequence() or [])
    if not gold or str(gold[0]).startswith("ERROR"):
        raise RuntimeError(f"no gold path for {task} v{variation}")
    history = []
    for action in gold:
        # Numeric gold entries answer ambiguity prompts; stop before them.
        if str(action).strip().isdigit() or len(history) >= max_prefix:
            break
        history.append(action)
    return history


DEVICE_WORDS = ("stove", "oven", "freezer", "fridge", "sink", "burner", "heater", "furnace", "kiln", "lighter")


def is_device(index, uuid):
    node = _node(index, uuid)
    if not node:
        return False
    name = str(node.get("name", "")).casefold()
    return bool((node.get("propDevice") or {}).get("isActivable")) or any(word in name for word in DEVICE_WORDS)


def core_signature(projection):
    """Context features that plausibly change reversibility: device/open state and contents of roots."""

    def strip(sig):
        if not isinstance(sig, dict) or "room" in sig:
            return None
        return [sig.get("name"), sig.get("open"), sig.get("active"), sig.get("state"),
                [strip(child) for child in sig.get("contents", [])]]

    return json.dumps(sorted(json.dumps(strip(sig)) for sig in projection["roots"].values()))


def _setup_actions(templates, index, relevant):
    """One-step perturbations: toggle a relevant device (e.g. turn the stove on before the pot goes on)."""
    actions = []
    for action, obj_ids in templates.items():
        verb = action_verb(action)
        if verb in ("activate", "deactivate") and len(obj_ids) == 1 and obj_ids[0] in relevant:
            node = _node(index, obj_ids[0])
            active = ((node or {}).get("propDevice") or {}).get("isActivated")
            if (verb == "activate") != bool(active):
                actions.append(action)
    return sorted(actions)


def collect_proposals(tasks, variations, max_prefix, prefixes_per_key):
    """Proposal groups keyed by (task, variation, verb, object ids) over several prior states.

    Prior states are gold-path prefixes plus each prefix followed by one device toggle.
    Within a group, contexts are ordered so that ones with a different core signature
    (device/open state and contents of the action's objects) come first.
    """
    env = make_env()
    proposals = []
    try:
        for task in tasks:
            for variation in variations:
                history = _gold_history(env, task, variation, max_prefix)
                rebuild(env, task, variation, [])
                task_text = env.get_task_description()
                relevant = set()
                for prefix in range(len(history)):
                    tree = env.getObjectTree()
                    index = index_tree(tree)
                    universe = str(tree.get("uuid"))
                    for value in unambiguous_templates(env).get(history[prefix], []):
                        if not is_room(value, index, universe):
                            relevant.add(value)
                    fast_step(env, history[prefix])

                contexts = []  # (history list, label)
                rebuild(env, task, variation, [])
                for prefix in range(len(history) + 1):
                    contexts.append(history[:prefix])
                    tree = env.getObjectTree()
                    for setup in _setup_actions(unambiguous_templates(env), index_tree(tree), relevant):
                        contexts.append([*history[:prefix], setup])
                    if prefix < len(history):
                        fast_step(env, history[prefix])

                by_key = defaultdict(list)
                for context in contexts:
                    rebuild(env, task, variation, context)
                    tree = env.getObjectTree()
                    index = index_tree(tree)
                    universe = str(tree.get("uuid"))
                    look = env.look()
                    for action, obj_ids in unambiguous_templates(env).items():
                        verb = action_verb(action)
                        if verb not in PROBE_VERBS or not obj_ids:
                            continue
                        if any(is_room(value, index, universe) for value in obj_ids):
                            continue
                        if verb in ("move", "pour") and not set(obj_ids) <= relevant:
                            continue
                        if not set(obj_ids) & relevant or not probe_ok(verb, obj_ids, index):
                            continue
                        roots = root_ids_for(obj_ids, index)
                        projection = project(tree, look, roots, index=index)
                        device = any(is_device(index, value) for value in roots)
                        by_key[(verb, tuple(obj_ids))].append(
                            (context, action, core_signature(projection), discrete_key(projection), device)
                        )

                for (verb, obj_ids), entries in by_key.items():
                    rng = random.Random(f"{task}:{variation}:{verb}:{obj_ids}")
                    seen_core, seen_full, first, rest = set(), set(), [], []
                    for entry in entries:
                        context, action, core, full, device = entry
                        if full in seen_full:
                            continue
                        seen_full.add(full)
                        (rest if core in seen_core else first).append(entry)
                        seen_core.add(core)
                    rng.shuffle(first)
                    rng.shuffle(rest)
                    chosen = (first + rest)[:prefixes_per_key]
                    pair_key = f"sw-{task}-v{variation:03d}-{verb.replace(' ', '_')}-{'_'.join(obj_ids)}"
                    for order, (context, action, core, _, device) in enumerate(chosen):
                        setup = len(context) > 0 and context[-1] not in history[: len(context)]
                        proposals.append({
                            "id": f"sw-{task}-v{variation:03d}-{stable_hash(json.dumps([context, action]))}",
                            "task_name": task,
                            "variation": variation,
                            "history": list(context),
                            "setup_action": context[-1] if setup else None,
                            "action_text": action,
                            "obj_ids": list(obj_ids),
                            "verb": verb,
                            "device_context": bool(device),
                            "core_signature": core,
                            "group_order": order,
                            "pair_key": pair_key,
                            "task_text": task_text,
                        })
    finally:
        env.close()
    return proposals


# --------------------------------------------------------------------------- labelling

def restoration_candidates(env, scope):
    """Actions that could undo the change: they must involve an object the action changed.

    ``scope`` holds ``allowed`` (roots + direct contents), ``changed`` (objects whose
    container, open/active state, phase or contents differ from the wait control) and
    ``targets`` (roots, changed objects and their pre-action containers).
    """
    allowed, changed, targets = scope["allowed"], scope["changed"], scope["targets"]
    results = []
    for action, obj_ids in unambiguous_templates(env).items():
        verb = action_verb(action)
        if verb not in RESTORE_VERBS or not obj_ids or not set(obj_ids) <= allowed:
            continue
        if len(set(obj_ids)) != len(obj_ids):
            continue
        if verb in ("open", "close", "activate", "deactivate"):
            ok = obj_ids[0] in changed or obj_ids[0] in scope["roots"]
        elif verb in ("pick up", "put down"):
            ok = obj_ids[0] in changed
        else:
            ok = len(obj_ids) == 2 and obj_ids[0] in changed and obj_ids[1] in targets
        if ok:
            results.append(action)
    return sorted(results)


def label_job(env, job, max_edges):
    started = time.perf_counter()
    task, variation, history, action = job["task_name"], job["variation"], job["history"], job["action_text"]
    rebuilds = 1
    rebuild(env, task, variation, history)
    tree = env.getObjectTree()
    index = index_tree(tree)
    universe = str(tree.get("uuid"))
    look_before = env.look()
    inventory_before = env.inventory()
    templates = unambiguous_templates(env)
    if templates.get(action) != job["obj_ids"]:
        return {"status": "dropped", "reason": "action not available or ambiguous at replayed state"}
    roots = root_ids_for(job["obj_ids"], index)
    allowed = allowed_restoration_ids(roots, index, universe)
    pre = project(tree, look_before, roots, index=index)

    controls = {}
    control1_index = None
    for count in range(1, MAX_DEPTH + 2):
        fast_step(env, "wait1")
        controls[count], control_tree = snapshot(env, roots)
        if count == 1:
            control1_index = index_tree(control_tree)

    rebuild(env, task, variation, history)
    rebuilds += 1
    env.lastStepScore = int(round(100 * env.server.getScore()))
    observation, reward, done, _ = env.step(action)
    post, post_tree = snapshot(env, roots)
    post_index = index_tree(post_tree)
    changed = changed_ids(post_index, control1_index, allowed)
    # Objects now inside a root (poured or moved in) count as changed too.
    for value, (node, parent) in post_index.items():
        if parent in roots and shallow_state(post_index, value) != shallow_state(control1_index, value):
            changed.add(value)
    pre_parents = {index[value][1] for value in changed if value in index and index[value][1]}
    scope = {
        "allowed": allowed | changed,
        "changed": changed,
        "targets": set(roots) | changed | pre_parents,
        "roots": set(roots),
    }
    base = {
        "pre": pre, "post": post, "control1": controls[1], "observation": observation,
        "reward": reward, "done": done, "roots": roots, "changed": sorted(changed),
    }
    if done:
        return {"status": "dropped", "reason": "episode ended", **base}
    if failed_observation(observation):
        return {"status": "dropped", "reason": "observation reports no effect", **base}
    if match(post, controls[1]) == "exact":
        return {"status": "dropped", "reason": "no effect on projected state", **base}

    exact_depth = None
    approx_depth = 0 if match(post, controls[1]) == "approximate" else None
    exact_path = None
    approx_path = [] if approx_depth == 0 else None
    edges = 0
    truncated = False
    visited = {discrete_key(post)}
    frontier = [((), restoration_candidates(env, scope))]
    env_at = ()

    for depth in range(1, MAX_DEPTH + 1):
        next_frontier = []
        for path, candidates in frontier:
            for candidate in candidates:
                if edges >= max_edges:
                    truncated = True
                    break
                if env_at != path:
                    rebuild(env, task, variation, [*history, action, *path])
                    rebuilds += 1
                observation_r, done_r = fast_step(env, candidate)
                edges += 1
                new_path = (*path, candidate)
                env_at = new_path
                if done_r or failed_observation(observation_r):
                    env_at = None
                    continue
                state = snapshot(env, roots)[0]
                relation = match(state, controls[depth + 1])
                if relation == "exact":
                    exact_depth, exact_path = depth, list(new_path)
                    break
                if relation == "approximate" and approx_depth is None:
                    approx_depth, approx_path = depth, list(new_path)
                key = discrete_key(state)
                if depth < MAX_DEPTH and key not in visited:
                    visited.add(key)
                    next_frontier.append((new_path, restoration_candidates(env, scope)))
            if exact_depth is not None or truncated:
                break
        if exact_depth is not None or truncated or not next_frontier:
            break
        frontier = next_frontier

    if truncated and exact_depth is None:
        return {"status": "dropped", "reason": f"search truncated at {max_edges} edges", **base,
                "edges": edges, "rebuilds": rebuilds}

    labels = labels_from_depths(exact_depth, approx_depth)
    restore = exact_path if exact_path is not None else (approx_path or [])
    depth_used = exact_depth if exact_depth is not None else approx_depth
    group_id = f"sw-{task}-v{variation:03d}"
    row = {
        "id": job["id"],
        "environment": "scienceworld",
        "state_text": f"{look_before.strip()}\n\n{inventory_before.strip()}",
        "action_text": action,
        "label": labels[str(MAX_DEPTH)],
        "horizon": MAX_DEPTH,
        "source": SOURCE,
        "labels_by_horizon": labels,
        "group_id": group_id,
        "scenario_id": group_id,
        "pair_key": job["pair_key"],
        "pair_id": None,
        "is_minimal_pair": False,
        "task_name": task,
        "variation": variation,
        "task_text": job["task_text"],
        "verb": job["verb"],
        "device_context": job.get("device_context", False),
        "setup_action": job.get("setup_action"),
        "history_length": len(history),
        "action_observation": observation,
        "restore_sequence": restore,
        "restore_depth": depth_used,
    }
    debug = {
        "id": job["id"], "history": history, **base,
        "exact_depth": exact_depth, "approx_depth": approx_depth,
        "exact_path": exact_path, "approx_path": approx_path,
        "controls": controls, "edges": edges, "rebuilds": rebuilds,
        "visited_states": len(visited), "seconds": time.perf_counter() - started,
    }
    return {"status": "ok", "row": row, "debug": debug}


def _init_worker():
    global _ENV
    _ENV = make_env()


def _worker_one(job, max_edges):
    started = time.perf_counter()
    try:
        result = label_job(_ENV, job, max_edges)
    except Exception as exc:  # noqa: BLE001 - recorded and reported
        result = {"status": "error", "error": repr(exc), "traceback": traceback.format_exc()}
    result["id"] = job["id"]
    result["job"] = {key: job[key] for key in ("task_name", "variation", "action_text", "verb", "pair_key")}
    result["seconds"] = time.perf_counter() - started
    return result


def _worker_group(group, max_edges, max_rows_without_contrast):
    """Label a group's contexts in order until two labels differ (a minimal pair)."""
    results, labels = [], []
    for job in group:
        result = _worker_one(job, max_edges)
        results.append(result)
        if result["status"] == "ok":
            labels.append(result["row"]["label"])
            if len(set(labels)) > 1:
                break
            if len(labels) >= max_rows_without_contrast:
                break
    return results


# --------------------------------------------------------------------------- finalize

def mark_pairs(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["pair_key"]].append(row)
    for key, members in by_key.items():
        paired = len({row["label"] for row in members}) > 1
        for row in members:
            row["is_minimal_pair"] = paired
            row["pair_id"] = key if paired else None
    return rows


def cap_classes(rows, cap, seed=0):
    """Drop unpaired rows of any class above ``cap`` share (deterministic)."""
    rows = list(rows)
    rng = random.Random(seed)
    while rows:
        counts = Counter(row["label"] for row in rows)
        label, count = counts.most_common(1)[0]
        if count <= cap * len(rows):
            break
        removable = [row for row in rows if row["label"] == label and not row["is_minimal_pair"]]
        if not removable:
            break
        drop = rng.choice(removable)
        rows = [row for row in rows if row is not drop]
        mark_pairs(rows)
    return rows


def finalize(raw_path, output_path, cap):
    rows = mark_pairs(read_jsonl(raw_path))
    rows = cap_classes(rows, cap)
    rows.sort(key=lambda row: row["id"])
    write_jsonl_atomic(output_path, rows)
    return rows


def summary(rows):
    labels = Counter(row["label"] for row in rows)
    by_h = {h: dict(Counter(row["labels_by_horizon"][h] for row in rows)) for h in ("1", "2", "3")}
    paired = sum(row["is_minimal_pair"] for row in rows)
    return {
        "rows": len(rows),
        "labels_h3": dict(labels),
        "labels_by_horizon": by_h,
        "minimal_pair_rows": paired,
        "minimal_pair_share": paired / len(rows) if rows else 0.0,
        "tasks": dict(Counter(row["task_name"] for row in rows)),
        "variations_per_task": {
            task: len({row["variation"] for row in rows if row["task_name"] == task})
            for task in sorted({row["task_name"] for row in rows})
        },
        "verbs_by_label": {
            verb: dict(Counter(row["label"] for row in rows if row["verb"] == verb))
            for verb in sorted({row["verb"] for row in rows})
        },
    }


# --------------------------------------------------------------------------- scheduling

def steer_key(item):
    return f"{item['verb']}|{'device' if item.get('device_context') else 'plain'}"


class Steering:
    """Pick the next pair-key group, favouring verbs that yield under-represented classes."""

    # Weak sampling priors (pseudo-counts) per steer key; they only order sampling and are
    # quickly outweighed by observed labels. Labels themselves never use them.
    PRIORS = {
        "pour": {"irreversible": 2, "reversible": 1},
        "mix": {"irreversible": 3},
        "activate|device": {"partially_reversible": 1, "irreversible": 1, "reversible": 1},
        "move|device": {"partially_reversible": 1, "reversible": 1},
        "pick up|device": {"partially_reversible": 1, "reversible": 1},
        "deactivate|device": {"partially_reversible": 2, "reversible": 1},
        "close|device": {"partially_reversible": 1, "reversible": 1},
        "open|plain": {"reversible": 3},
        "close|plain": {"reversible": 3},
    }

    def __init__(self, groups, tasks):
        self.groups = groups
        self.tasks = list(tasks)
        self.labels = Counter()
        self.task_rows = Counter()
        self.verb_labels = defaultdict(Counter)
        for key, prior in self.PRIORS.items():
            verbs = [key] if "|" in key else [f"{key}|device", f"{key}|plain"]
            for verb in verbs:
                self.verb_labels[verb].update(prior)
        self.drops = Counter()

    def record(self, row):
        self.labels[row["label"]] += 1
        self.task_rows[row["task_name"]] += 1
        self.verb_labels[steer_key(row)][row["label"]] += 1

    def record_drop(self, job):
        self.drops[steer_key(job)] += 1

    def score(self, group):
        verb = steer_key(group[0])
        total = sum(self.labels.values())
        verb_counts = self.verb_labels[verb]
        verb_total = sum(verb_counts.values())
        score = 0.0
        for label in LABELS:
            p_label = (verb_counts[label] + 1) / (verb_total + len(LABELS))
            share = self.labels[label] / total if total else 1 / len(LABELS)
            score += p_label * (1 / len(LABELS) - share)
        task_total = sum(self.task_rows.values())
        task_share = self.task_rows[group[0]["task_name"]] / task_total if task_total else 0
        score -= 0.5 * max(0.0, task_share - 1 / len(self.tasks))
        score += 0.05 * min(len(group), 3)  # groups that can form pairs first
        yield_rate = (verb_total + 1) / (verb_total + self.drops[verb] + 2)
        return score * yield_rate if score > 0 else score / yield_rate

    def pop(self):
        if not self.groups:
            return None
        best = max(range(len(self.groups)), key=lambda i: (self.score(self.groups[i]), -i))
        return self.groups.pop(best)


def run(args):
    out_dir = Path(args.work_dir)
    raw_path = out_dir / "rows_raw.jsonl"
    debug_path = out_dir / "debug.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    errors_path = out_dir / "errors.jsonl"
    skipped_path = out_dir / "skipped.jsonl"
    proposals_path = out_dir / "proposals.jsonl"
    output = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    variations = list(range(args.variations))

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - started:7.0f}s] {message}", flush=True)

    if proposals_path.exists():
        proposals = read_jsonl(proposals_path)
        log(f"loaded {len(proposals)} proposals")
    else:
        proposals = collect_proposals(tasks, variations, args.max_prefix, args.prefixes_per_key)
        write_jsonl_atomic(proposals_path, proposals)
        log(f"collected {len(proposals)} proposals in {time.perf_counter() - started:.0f}s")

    raw_rows = read_jsonl(raw_path)
    done_ids = {row["id"] for row in raw_rows}
    done_ids |= {row["id"] for row in read_jsonl(dropped_path)}
    done_ids |= {row["id"] for row in read_jsonl(skipped_path)}
    groups_by_key = defaultdict(list)
    for proposal in proposals:
        if proposal["id"] not in done_ids:
            groups_by_key[proposal["pair_key"]].append(proposal)
    rng = random.Random(args.seed)
    groups = list(groups_by_key.values())
    rng.shuffle(groups)
    steering = Steering(groups, tasks)
    for row in raw_rows:
        steering.record(row)
    if not output.exists() and raw_rows:
        write_jsonl_atomic(output, raw_rows)
    log(f"resuming with {len(raw_rows)} rows; {sum(map(len, groups))} pending proposals in {len(groups)} groups")

    written = len(raw_rows)
    counts = Counter()
    pending = {}
    deadline = started + args.max_seconds

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
        def submit():
            while len(pending) < args.workers * 2:
                if time.perf_counter() > deadline or written >= args.limit:
                    return
                group = steering.pop()
                if group is None:
                    return
                future = executor.submit(_worker_group, group, args.max_edges, args.rows_without_contrast)
                pending[future] = group

        submit()
        while pending:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in finished:
                group = pending.pop(future)
                results = future.result()
                attempted = {result["id"] for result in results}
                for job in group:
                    if job["id"] not in attempted:
                        append_jsonl(skipped_path, {"id": job["id"], "reason": "group already resolved"})
                for result in results:
                    counts[result["status"]] += 1
                    if result["status"] == "ok":
                        append_jsonl(raw_path, result["row"])
                        append_jsonl(output, result["row"])
                        append_jsonl(debug_path, result["debug"])
                        steering.record(result["row"])
                        written += 1
                    elif result["status"] == "dropped":
                        append_jsonl(dropped_path, {
                            "id": result["id"], "reason": result["reason"], **result["job"],
                            "observation": result.get("observation"), "seconds": result["seconds"],
                        })
                        steering.record_drop(next(job for job in group if job["id"] == result["id"]))
                        counts["dropped:" + result["reason"]] += 1
                    else:
                        append_jsonl(errors_path, result)
            log(
                f"rows={written} labels={dict(steering.labels)} dropped={counts['dropped']} "
                f"errors={counts['error']} rate={written / max(1.0, time.perf_counter() - started) * 3600:.0f}/h"
            )
            submit()

    rows = finalize(raw_path, output, args.class_cap)
    meta = {
        "source": SOURCE,
        "tasks": tasks,
        "variations": variations,
        "max_prefix": args.max_prefix,
        "prefixes_per_key": args.prefixes_per_key,
        "rows_without_contrast": args.rows_without_contrast,
        "max_edges": args.max_edges,
        "class_cap": args.class_cap,
        "run_counts": dict(counts),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary(rows),
        "rule": (
            "Projected state = agent room, inventory item names, visible door states, and the action's "
            "objects plus their containers (discrete properties, temperature, recursive contents; rooms "
            "shallow). Candidate dropped if it ends the episode or leaves the projection equal to a "
            "one-wait control. BFS over open/close/activate/deactivate/move/pick up/pour/put down actions "
            "on those objects and their direct contents, depth <= 3; each node compared with the matched "
            "control of the same number of steps (history + wait1 x (d+1)). Exact = discrete fields equal "
            "and temperatures within 1.0 C; approximate = discrete equal, some temperature differs more. "
            "Label at h: reversible if exact within h, partially_reversible if approximate within h, "
            "else irreversible."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log("final " + json.dumps(meta["summary"]))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--variations", type=int, default=4)
    parser.add_argument("--max-prefix", type=int, default=20)
    parser.add_argument("--prefixes-per-key", type=int, default=5)
    parser.add_argument("--rows-without-contrast", type=int, default=2,
                        help="stop labelling a group after this many same-label rows")
    parser.add_argument("--max-edges", type=int, default=400)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=4 * 3600)
    parser.add_argument("--class-cap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/scienceworld.jsonl")
    parser.add_argument("--work-dir", default="results/scienceworld")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_only:
        rows = finalize(Path(args.work_dir) / "rows_raw.jsonl", Path(args.output), args.class_cap)
        print(json.dumps(summary(rows), indent=2))
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
