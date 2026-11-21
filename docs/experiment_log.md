# Experiment log

- Day 2 (2026-11-13): first ScienceWorld slice attempt (smoke, pilots, a stopped partial generation; diagnostics in `results/scienceworld_*`). It failed because labels compared full object trees across steps: a plain `wait1` already changed many fields (temperatures, outside fountain water, plant entries) and door states are not in the tree, so exact/approximate restore labels were not reliable. Those rows were never used.
- Local CPU smoke runs with Qwen2.5-0.5B (`results/rq2_cpu_smoke_v*`, `rq3_cpu_smoke_v*`) and 2-row Kaggle smoke runs; the Kaggle smoke outputs were not archived.
- 2026-11-19, Kaggle T4 x2, Qwen2.5-7B via `kaggle/run_full.py`: pilot 3.9 min, extraction 11.7 min (2,044 rows), RQ2 prompt v1 81.8 min, RQ3 (8-turn version) 32.3 min.
- RQ2 v1 (Qwen only) is preliminary: accuracy 0.340, K = 0.064 on 516 pairs. Kept in `rq2/three-class-h3-neutral/`, never in headline numbers.
- RQ3 was then aligned to the spec (15 turns, spec caution definition) and RQ2 prompt v2 added the agent tool list. Qwen reruns: RQ2 v2 96.4 min (accuracy 0.360, K = 0.025), RQ3 31.8 min. v2 does not change the conclusion (K near zero in both).
- Llama-3.1-8B on Kaggle: a first run was lost when the Kaggle session was recycled; the rerun gave identical numbers. Kept rerun: extraction 11.1 min, RQ2 v2 127.8 min (accuracy 0.314, K = 0, 7 parse failures), RQ3 47.2 min.
- 2026-11-20, local Apple M2 (8 cores, 8 GB, no GPU): `scripts/run_probes.sh`, all layers, 4 workers. Qwen probes 190.6 min + report 1.6 min (3.2 h); Llama 367.0 min + 1.8 min (6.1 h). All recomputation checks passed for both models.
- 2026-11-20, post-hoc additions (not in the original plan): full-prompt BoW/TF-IDF baselines, leave-one-tool-out probe, RQ2 directional metric (`experiments/posthoc.py`, `posthoc_*.json`).
- 2026-11-20: balanced-accuracy CIs and AUROC refits added (brief requirement); refits reproduce the saved predictions exactly.
- 2026-11-20: ScienceWorld labelling rebuilt (commit 4623867): compare a projection of the touched objects, agent room, inventory and `look()` door states against a matched `wait1` control of equal length, with a breadth-first search over ≤ 3 restoring actions on the touched objects; temperatures within 1 °C count as equal.
- 2026-11-21: first slice of 242 rows (ac7ce6c); second generation pass to 512 rows (f10600f) from 1,113 raw labelled rows, with any class capped at 50%. 72 exact duplicates are removed in the analyses (440 rows, 33 groups).
- 2026-11-21, RQ3 extended to 60 pairs with a disjoint second batch of 30 (`--rq3-batch extra`): Qwen 28.7 min, Llama 53.8 min on Kaggle. The first-batch files are unchanged; merged results are in `headline_h3_plus_extra.json`.
- 2026-11-21, ScienceWorld on Kaggle (`--env scienceworld`, 512 rows, 111 RQ2 pairs = 222 rows): Qwen extraction 7.1 min, RQ2 12.5 min; Llama extraction 8.0 min, RQ2 14.7 min (3 parse failures, so the runner marked the step incomplete). Only 10 (Qwen) and 9 (Llama) pairs have one reversible and one irreversible member, so K rests on very few pairs.
- 2026-11-21, local ScienceWorld analysis (`experiments.analyze_sw`, every second layer): Qwen 37.9 min, Llama 55.4 min.
- 2026-11-21, human label check: 100 blind items labelled by one annotator; kappa vs automatic labels 0.26 overall; reversible-vs-irreversible-only kappa 0.68 (post-hoc). No second rater or retest yet.
- 2026-11-21: `experiments/make_final_tables.py` builds the report figures and tables from saved files only (about 4 s).
