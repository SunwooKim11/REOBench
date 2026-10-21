# Archived launchers

These shell scripts launched experiments that have already finished. They are kept as a record of the exact commands
and orderings, but they predate the `repro/` restructuring (2026-10-21), so their relative paths (`D_tta/`,
`C_ablation/`, `E_verify/`, `../C_ablation/outputs`) no longer exist. To rerun something, use the current entry points
in `scripts/` with the same arguments. The argument lists are in these files and in `scripts/cls/ablation_jobs*.sh`.

- `cls/`: classification ablation launchers (`one.sh`, `run_all.sh`, `run_sched.sh`)
- `tta/`: AID TTA launchers (head training, main runs, lr sweep, new baselines, CoTTA reruns and diagnostics, SAR reset 0.2)
- `verify/`: ImageNet-C verification launcher
- `run_queue*.sh`: GPU queues used after the CoTTA OOM
