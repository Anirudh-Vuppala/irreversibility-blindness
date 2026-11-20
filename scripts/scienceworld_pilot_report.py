"""Print labelled ScienceWorld rows with their evidence (projection before/after, search result).

Usage: python scripts/scienceworld_pilot_report.py results/scienceworld/pilot3 [--n 15] [--raw]
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scienceworld_slice.build_slice import mark_pairs, read_jsonl, summary  # noqa: E402
from scienceworld_slice.projection import compact_projection, max_temperature_gap  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", help="row file (default: <work_dir>/rows_raw.jsonl)")
    args = parser.parse_args(argv)
    work = Path(args.work_dir)
    rows = mark_pairs(read_jsonl(args.rows or work / "rows_raw.jsonl"))
    debug = {item["id"]: item for item in read_jsonl(work / "debug.jsonl")}
    dropped = read_jsonl(work / "dropped.jsonl")
    print(json.dumps(summary(rows), indent=1))
    print("dropped:", dict(Counter(item["reason"] for item in dropped)))
    rng = random.Random(args.seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    chosen = []
    while len(chosen) < min(args.n, len(rows)):
        for label in sorted(by_label):
            if by_label[label] and len(chosen) < args.n:
                chosen.append(by_label[label].pop(rng.randrange(len(by_label[label]))))
    for number, row in enumerate(chosen, 1):
        item = debug[row["id"]]
        print(f"\n#{number} [{row['label']}] {row['labels_by_horizon']} pair={row['is_minimal_pair']} "
              f"{row['task_name']} v{row['variation']} hist={row['history_length']} setup={row.get('setup_action')}")
        print(f"  action: {row['action_text']}  ->  {row['action_observation']!r}")
        print(f"  before:  {compact_projection(item['pre'])}")
        print(f"  after:   {compact_projection(item['post'])}")
        print(f"  wait1:   {compact_projection(item['control1'])}")
        print(f"  changed objects: {len(item.get('changed', []))}; edges={item['edges']} visited={item['visited_states']}")
        print(f"  exact_depth={item['exact_depth']} path={item['exact_path']}  "
              f"approx_depth={item['approx_depth']} path={item['approx_path']}")
        if item["approx_depth"] is not None and item["exact_depth"] is None:
            depth = item["approx_depth"]
            control = item["controls"][str(depth + 1)]
            print(f"  max temp gap of best approx vs control (post if depth 0): "
                  f"{max_temperature_gap(item['post'], item['controls']['1']):.1f}C at depth0; control@{depth + 1}: "
                  f"{compact_projection(control)}")


if __name__ == "__main__":
    main()
