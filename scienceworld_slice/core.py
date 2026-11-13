import hashlib
import json
import re
from collections import defaultdict
from numbers import Real


FOCUS_VERBS = ("mix", "pour", "activate", "deactivate", "open", "close", "move", "pick up")
INVERSE_VERBS = {
    "activate": {"deactivate"},
    "deactivate": {"activate"},
    "open": {"close"},
    "close": {"open"},
    "move": {"move"},
    "pick up": {"move"},
    "pour": {"pour"},
    "mix": {"mix"},
}
MOVE_COUNTER_KEYS = {
    "move", "moves", "movecount", "movecounter", "nummoves", "currentmoves",
    "stepcount", "stepcounter",
}


def action_verb(action_text):
    text = str(action_text).strip().casefold()
    if text.startswith("pick up "):
        return "pick up"
    return text.split(maxsplit=1)[0] if text else ""


def restoration_verbs(action_text):
    verb = action_verb(action_text)
    return {verb, *INVERSE_VERBS.get(verb, set())}


def _key_name(key):
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _is_temperature(key):
    return "temperature" in _key_name(key)


def canonicalize_tree(tree, drop_temperatures=False):
    """Remove counters and normalize floating temperature noise for comparisons."""
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            normalized = _key_name(key)
            if normalized in MOVE_COUNTER_KEYS:
                continue
            if drop_temperatures and _is_temperature(key):
                continue
            if _is_temperature(key) and isinstance(value, Real) and not isinstance(value, bool):
                result[key] = round(float(value), 1)
            else:
                result[key] = canonicalize_tree(value, drop_temperatures)
        return result
    if isinstance(tree, list):
        return [canonicalize_tree(value, drop_temperatures) for value in tree]
    return tree


def state_key(tree, drop_temperatures=False):
    canonical = canonicalize_tree(tree, drop_temperatures=drop_temperatures)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def state_digest(tree):
    return hashlib.sha256(state_key(tree).encode("utf-8")).hexdigest()


def build_parent_index(tree):
    """Return object-ID parents from ScienceWorld's nested contents tree."""
    parents = {}

    def visit(node, parent=None):
        if not isinstance(node, dict):
            return
        object_id = node.get("uuid")
        object_id = str(object_id) if object_id is not None else None
        if object_id is not None:
            parents[object_id] = str(parent) if parent is not None else None
        contents = node.get("contents", {})
        if isinstance(contents, dict):
            for child in contents.values():
                visit(child, object_id)

    visit(tree)
    return parents


def expand_with_parents(object_ids, parent_index):
    expanded = {str(value) for value in object_ids if value is not None}
    expanded.update(parent_index[value] for value in tuple(expanded) if parent_index.get(value))
    return expanded


def focus_candidates(action_strings, template_rows, context_ids):
    """Filter API candidates to focus verbs touching objects in nearby trajectory actions."""
    context_ids = {int(value) for value in context_ids}
    metadata = {}
    for item in template_rows:
        action = item.get("action")
        if action:
            metadata.setdefault(str(action), item)

    grouped = defaultdict(list)
    action_set = set(map(str, action_strings))
    for action in action_set:
        verb = action_verb(action)
        if verb not in FOCUS_VERBS:
            continue
        item = metadata.get(action)
        object_ids = {int(value) for value in (item or {}).get("obj_ids", [])}
        if object_ids and object_ids.intersection(context_ids):
            grouped[verb].append((action, item))

    return [
        pair
        for verb in FOCUS_VERBS
        for pair in sorted(grouped.get(verb, []), key=lambda pair: pair[0])
    ]


def restoration_candidates(action_strings, template_rows, allowed_ids, original_action):
    """Use focus verbs and template object IDs to keep BFS on the same entities."""
    allowed = {int(value) for value in allowed_ids}
    verbs = restoration_verbs(original_action)
    metadata = {}
    for item in template_rows:
        action = item.get("action")
        if action:
            metadata.setdefault(str(action), item)

    results = []
    for action in sorted(set(map(str, action_strings))):
        if action_verb(action) not in verbs:
            continue
        item = metadata.get(action)
        if item is None:
            continue
        object_ids = {int(value) for value in item.get("obj_ids", [])}
        if object_ids and object_ids.issubset(allowed):
            results.append(action)
    return results


def labels_from_depths(exact_depth, partial_depth, max_depth):
    labels = {}
    for horizon in (1, 2, 3):
        if horizon > max_depth:
            labels[str(horizon)] = None
        elif exact_depth is not None and exact_depth <= horizon:
            labels[str(horizon)] = "reversible"
        elif partial_depth is not None and partial_depth <= horizon:
            labels[str(horizon)] = "partially_reversible"
        else:
            labels[str(horizon)] = "irreversible"
    return labels
