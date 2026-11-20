# Experiment log

- Early (Day 2): ScienceWorld slice attempted (smoke, pilots, a stopped partial generation; diagnostics in `results/scienceworld_*`). Labels were not reliable under the object-tree comparison; the slice is deferred and used for no claim.
- Local CPU smoke runs with Qwen2.5-0.5B (`results/rq2_cpu_smoke_v*`, `rq3_cpu_smoke_v*`) and 2-row Kaggle smoke runs; the Kaggle smoke outputs were not archived.
- 2026-11-19, Kaggle T4 x2, Qwen2.5-7B via `kaggle/run_full.py`: pilot 3.9 min, extraction 11.7 min (2,044 rows), RQ2 prompt v1 81.8 min, RQ3 (8-turn version) 32.3 min.
- RQ2 v1 (Qwen only) is preliminary: accuracy 0.340, K = 0.064 on 516 pairs. Kept in `rq2/three-class-h3-neutral/`, never in headline numbers.
- RQ3 was then aligned to the spec (15 turns, spec caution definition) and RQ2 prompt v2 added the agent tool list. Qwen reruns: RQ2 v2 96.4 min (accuracy 0.360, K = 0.025), RQ3 31.8 min. v2 does not change the conclusion (K near zero in both).
- Llama-3.1-8B on Kaggle: a first run was lost when the Kaggle session was recycled; the rerun gave identical numbers. Kept rerun: extraction 11.1 min, RQ2 v2 127.8 min (accuracy 0.314, K = 0, 7 parse failures), RQ3 47.2 min.
- 2026-11-20, local Apple M2 (8 cores, 8 GB, no GPU): `scripts/run_probes.sh`, all layers, 4 workers. Qwen probes 190.6 min + report 1.6 min (3.2 h); Llama 367.0 min + 1.8 min (6.1 h). All recomputation checks passed for both models.
- 2026-11-20, post-hoc additions (not in the original plan): full-prompt BoW/TF-IDF baselines, leave-one-tool-out probe, RQ2 directional metric (`experiments/posthoc.py`, `posthoc_*.json`).
