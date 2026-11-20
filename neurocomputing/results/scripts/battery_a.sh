#!/bin/zsh
# Fast battery, part A (RUN_PLAN.md Step 3, R1 to R3, commands as written there; one process at a time).
# Stops at the first step that fails, except that R3 is expected to stop after sweep:dinov2 (see DEVIATIONS.md);
# its outcome is inspected before part B.
RE="$(cd "$(dirname "$0")/../.." && pwd)"                      # this re-analysis folder (neurocomputing/ in the repository)
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}              # project tree that results/ mirrors (as gpu/run_gpu_re.sh)
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}   # interpreter: set PY to override (as gpu/run_gpu_re.sh)
RS=$RE/results/scripts/run_step.sh
$RS R1_local_nulls $RE nice -n 10 env OMP_NUM_THREADS=4 $PY -B -I scoring/local_nulls_rigid.py || exit 1
$RS R2_iie $RE nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py iie || exit 2
$RS R3_sweep_all $RE nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py sweep:dinov2 sweep:clip sweep:siglip || exit 3
exit 0
