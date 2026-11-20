"""Projected-state labelling rule for the ScienceWorld slice.

A labelled pair is compared on a *projection* of the world instead of the full
object tree, so that drift elsewhere in the world (fountain water, distant
temperatures, growing plants) cannot change a label. The projection keeps:

* the agent's room and the names of inventory items;
* door states visible in ``look()`` (doors are not in the object tree);
* the "root" objects of the action: its object ids plus their containers,
  each with its discrete properties and recursive contents. Rooms are kept
  shallow (names of top-level items only).

Two projections match *exactly* when every discrete field is equal and every
aligned temperature differs by at most ``TEMP_TOLERANCE_C``. They match
*approximately* when the discrete fields are equal but some temperature
differs by more than that. Anything else is no match.
"""

import json
import re

FAILURE_PATTERNS = (
    "no known action", "does not appear to produce anything new", "no items to mix",
    "there is only one thing", "can't", "cannot", "is not movable", "is not moveable",
    "not open", "is closed", "already", "nothing happens", "doesn't", "isn't", "not possible",
    "it's not clear", "unknown", "ambiguous",
)


def failed_observation(observation):
    """True when the simulator's reply says the action did nothing."""
    text = str(observation or "").casefold()
    return any(pattern in text for pattern in FAILURE_PATTERNS)

TEMP_TOLERANCE_C = 1.0
MAX_CONTENT_DEPTH = 4
IGNORED_CONTENT_NAMES = {"air", "agent"}


def ignored_child(name):
    """Air, the agent and electrical terminals (parts of every object) are not compared."""
    name = str(name)
    return name in IGNORED_CONTENT_NAMES or name.startswith("terminal ")
DOOR_LINE = re.compile(r"^\s*A door to (?:the )?(.+?) \(that is (open|closed)\)\s*$", re.MULTILINE)
MISSING = "MISSING"


def index_tree(tree):
    """Map uuid -> (node, parent_uuid) for the nested ScienceWorld object tree."""
    index = {}

    def visit(node, parent):
        if not isinstance(node, dict):
            return
        uuid = node.get("uuid")
        if uuid is not None:
            index[str(uuid)] = (node, None if parent is None else str(parent))
        for child in (node.get("contents") or {}).values():
            visit(child, uuid)

    visit(tree, None)
    return index


def _root_uuid(tree):
    return str(tree.get("uuid")) if isinstance(tree, dict) else None


def is_room(uuid, index, universe_uuid):
    entry = index.get(str(uuid))
    return entry is not None and entry[1] == universe_uuid


def _names_of_contents(node):
    return sorted(
        str(child.get("name"))
        for child in (node.get("contents") or {}).values()
        if isinstance(child, dict)
        and not child.get("isDeleted")
        and not ignored_child(child.get("name"))
    )


def node_signature(node, parent_name, depth=0):
    """Discrete properties plus temperature and recursive contents of one object."""
    container = node.get("propContainer") or {}
    device = node.get("propDevice") or {}
    material = node.get("propMaterial") or {}
    electrical = node.get("propElectricalConnection")
    contents = []
    if depth < MAX_CONTENT_DEPTH:
        for child in (node.get("contents") or {}).values():
            if not isinstance(child, dict) or child.get("isDeleted"):
                continue
            if ignored_child(child.get("name")):
                continue
            contents.append(node_signature(child, node.get("name"), depth + 1))
    contents.sort(key=discrete_key)
    return {
        "name": node.get("name"),
        "parent": parent_name,
        "deleted": bool(node.get("isDeleted")),
        "open": container.get("isOpen") if container.get("isClosable") else None,
        "active": device.get("isActivated") if device.get("isActivable") else None,
        "broken": device.get("isBroken"),
        "state": material.get("stateOfMatter"),
        "substance": material.get("substanceName"),
        "burning": material.get("isCombusting"),
        "burnt": material.get("hasCombusted"),
        "electrical": json.dumps(electrical, sort_keys=True) if electrical else None,
        "temp": material.get("temperatureC"),
        "contents": contents,
    }


def door_states(look_text):
    return sorted(f"{name}:{state}" for name, state in DOOR_LINE.findall(str(look_text or "")))


def root_ids_for(action_obj_ids, index):
    """Action objects plus their current containers."""
    roots = {str(value) for value in action_obj_ids}
    for value in list(roots):
        entry = index.get(value)
        if entry is not None and entry[1] is not None:
            roots.add(entry[1])
    return sorted(roots, key=lambda value: (len(value), value))


def allowed_restoration_ids(root_ids, index, universe_uuid):
    """Roots plus direct contents of non-room roots (objects a restorer may touch)."""
    allowed = {str(value) for value in root_ids}
    for value in root_ids:
        entry = index.get(str(value))
        if entry is None or is_room(value, index, universe_uuid) or str(value) == universe_uuid:
            continue
        for child in (entry[0].get("contents") or {}).values():
            if isinstance(child, dict) and child.get("uuid") is not None:
                allowed.add(str(child["uuid"]))
    return allowed


def project(tree, look_text, root_ids, index=None):
    index = index_tree(tree) if index is None else index
    universe = _root_uuid(tree)
    agent_room = None
    inventory = []
    for node, parent in index.values():
        if node.get("name") == "agent" and parent is not None:
            agent_room = index[parent][0].get("name") if parent in index else None
            for child in (node.get("contents") or {}).values():
                if isinstance(child, dict) and child.get("name") == "inventory":
                    inventory = _names_of_contents(child)
            break
    roots = {}
    for value in root_ids:
        entry = index.get(str(value))
        if entry is None:
            roots[str(value)] = MISSING
            continue
        node, parent = entry
        if str(value) == universe:
            continue
        if is_room(value, index, universe):
            roots[str(value)] = {"room": node.get("name"), "items": _names_of_contents(node)}
            continue
        parent_name = index[parent][0].get("name") if parent in index else None
        roots[str(value)] = node_signature(node, parent_name)
    return {
        "agent_room": agent_room,
        "inventory": inventory,
        "doors": door_states(look_text),
        "roots": roots,
    }


def _strip_temps(value):
    if isinstance(value, dict):
        return {
            key: ("T" if value[key] is not None else None) if key == "temp" else _strip_temps(value[key])
            for key in value
        }
    if isinstance(value, list):
        return [_strip_temps(item) for item in value]
    return value


def discrete_key(projection):
    return json.dumps(_strip_temps(projection), sort_keys=True, separators=(",", ":"))


def _temps(value, out):
    if isinstance(value, dict):
        for key in sorted(value):
            if key == "temp":
                out.append(value[key])
            else:
                _temps(value[key], out)
    elif isinstance(value, list):
        for item in value:
            _temps(item, out)
    return out


def max_temperature_gap(first, second):
    gaps = [
        abs(float(a) - float(b))
        for a, b in zip(_temps(first, []), _temps(second, []))
        if a is not None and b is not None
    ]
    return max(gaps) if gaps else 0.0


def match(candidate, target, tolerance=TEMP_TOLERANCE_C):
    """Return 'exact', 'approximate' or None."""
    if discrete_key(candidate) != discrete_key(target):
        return None
    return "exact" if max_temperature_gap(candidate, target) <= tolerance else "approximate"


def labels_from_depths(exact_depth, approx_depth, horizons=(1, 2, 3)):
    """Label each horizon from the shallowest exact / approximate restoration depth."""
    labels = {}
    for horizon in horizons:
        if exact_depth is not None and exact_depth <= horizon:
            labels[str(horizon)] = "reversible"
        elif approx_depth is not None and approx_depth <= horizon:
            labels[str(horizon)] = "partially_reversible"
        else:
            labels[str(horizon)] = "irreversible"
    return labels


def compact_projection(projection):
    """Readable one-line-per-root summary for evidence printing (temps to 0.1 C)."""

    def describe(sig):
        if sig == MISSING:
            return MISSING
        if "room" in sig:
            return f"room {sig['room']}"
        bits = [str(sig["name"])]
        if sig.get("parent"):
            bits.append(f"in {sig['parent']}")
        if sig.get("open") is not None:
            bits.append("open" if sig["open"] else "closed")
        if sig.get("active") is not None:
            bits.append("on" if sig["active"] else "off")
        if sig.get("state") and sig.get("temp") is not None:
            bits.append(f"{sig['state']} {float(sig['temp']):.1f}C")
        if sig.get("contents"):
            bits.append("[" + ", ".join(describe(child) for child in sig["contents"]) + "]")
        return " ".join(bits)

    roots = "; ".join(describe(sig) for sig in projection["roots"].values())
    return f"agent@{projection['agent_room']} inv={projection['inventory']} | {roots}"
