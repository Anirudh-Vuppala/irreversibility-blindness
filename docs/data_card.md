# Data card: `data/sandbox.jsonl`

**What it is.** 2,044 labelled (state, action) rows from a simulated personal workspace (files, recycle bin, backups, version history, email drafts/outbox, bank account, savings). Every row has `environment = "sandbox"` and `source = "sandbox-generator-v1"`. No real files, mail or accounts are involved. ScienceWorld data is not part of this file.

**How it was generated.** Synthetically, by `reversibility/generate.py` (scenarios and minimal pairs), `reversibility/sandbox.py` (tool semantics as pure functions on a JSON state) and `reversibility/labeler.py` (labels). `python -m reversibility.generate --scenarios 265 --seed 0 --out <path>` reproduces the file byte for byte (checked 2026-11-20 into a scratch path). Each of the 265 scenarios (`sb-0000` to `sb-0264`) has one of 8 themes, one random base state, three paired tools sampled by weight, and one single tool.

**Labels.** For each row the labeler applies the action and runs a breadth-first search over restoring tool sequences of length ≤ h. The search sees only what an agent could observe after the action and only tries actions on entities that differ from the original state.
- `reversible`: some sequence of ≤ h calls restores the exact state (contents, versions, recycle bin, drafts, outbox, sent mail, balances, transfers).
- `partially_reversible`: no exact restore, but within h calls the user-visible content is back (versions may differ, extra items may stay in the bin, up to 3% of money may be lost).
- `irreversible`: neither.

`label` is the h = 3 label; `labels_by_horizon` holds h = 1, 2, 3 (168 rows change label across horizons). `restore_sequence` is the shortest exact sequence, else the shortest approximate one, else empty.

**Counts (h = 3).**

| | reversible | partially reversible | irreversible | total |
|---|---|---|---|---|
| all rows | 1,021 | 485 | 538 | 2,044 |
| minimal-pair rows | 808 | 455 | 516 | 1,779 (87%) |

85 rows are read-only actions (`list_files`, `read_file`, `view_trash`, `check_balance`), all reversible. 19 tools (`verb`) occur; 9 of them have rows with a single label only.

**Pair and group structure.** 1,060 groups (`group_id`, nested in `scenario_id`): 606 pairs, 189 triples (`edit_file`: backup / history / neither) and 265 singles. Within a group the action text is identical and one setting changes (recycle bin, send delay, scheduled transfers, savings fee, backups, version history, backed-up bin contents). `is_minimal_pair` is true when a group has more than one row and more than one distinct label. Splits in all analyses are grouped by `scenario_id`.

**Known quirk.** 10 `move_to_savings` rows with a 2% withdrawal fee are labelled `partially_reversible` with an empty `restore_sequence`: the 2% fee is within the 3% money tolerance, so the state counts as approximately restored after zero follow-up calls. They are kept as labelled.

**Limitations.** Labels come only from the programmatic search above. They were not independently annotated by people, and there is no inter-annotator agreement. States, file contents and names are synthetic templates, so lexical regularities from the generator can leak into features (see the post-hoc text baselines).
