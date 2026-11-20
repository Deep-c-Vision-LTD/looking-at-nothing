#!/bin/zsh
# Fast battery, part B (RUN_PLAN.md Step 3): R3 remainder (sweep:clip, sweep:siglip as separate processes; DEVIATIONS.md
# D1), R4a (CPU check of the heads, rule Section 3.2), R4 (60 swap heads), R5 (6 registers heads). Commands as in
# RUN_PLAN.md; one process at a time; stops at the first failure. eval_re.py logs a failure on a non-stim_v3 set and
# still exits 0, so its log is also searched for "FAILED".
RE="$(cd "$(dirname "$0")/../.." && pwd)"                      # this re-analysis folder (neurocomputing/ in the repository)
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}              # project tree that results/ mirrors (as gpu/run_gpu_re.sh)
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}   # interpreter: set PY to override (as gpu/run_gpu_re.sh)
RS=$RE/results/scripts/run_step.sh
L=$RE/results/run_logs
$RS R3b_sweep_clip $RE nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py sweep:clip || exit 31
$RS R3c_sweep_siglip $RE nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py sweep:siglip || exit 32
# R4a: one plain head pair per backbone, then the registers head
for spec in "dinov2 dinov2_L12_R4_s0,dinov2_L6_R1n_s0" "siglip siglip_L24_R4_s0,siglip_L24_R1n_s0" "iie iie_shapenetc_R4_s0,iie_shapenetc_R1n_s0"; do
  set -- ${=spec}
  $RS R4a_eval_$1 $RE/gpu nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb $1 --out ../check_cpu_$1 --ckpt-dir $ROOT/v3/gpu/full/ckpt --tags $2 --stimsets stim_v3 --stim-limit 18 --skip-mit || exit 41
  $RS R4a_check_$1 $RE/gpu $PY -B -I check_eval_re.py ../check_cpu_$1 --tol 1e-4 || exit 42
done
$RS R4a_eval_dinov2reg $RE/gpu nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb dinov2reg --out ../check_cpu_dinov2reg --ckpt-dir $ROOT/nn/registers/full/ckpt --tags dinov2reg_L12_R4_s0 --stimsets stim_v3 --stim-limit 18 --skip-mit || exit 43
$RS R4a_check_dinov2reg $RE/gpu $PY -B -I check_eval_re.py ../check_cpu_dinov2reg --tol 1e-4 --pub $ROOT/nn/registers/full/scores || exit 44
# R4: the 60 swap heads on stim_rigid
for bb in dinov2 siglip iie; do
  $RS R4_eval_$bb $RE/gpu nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb $bb --out ../scores_rigid/heads --ckpt-dir $ROOT/v3/gpu/full/ckpt --stimsets stim_rigid --skip-mit || exit 51
  grep -q "FAILED" $L/R4_eval_$bb.log && exit 52
done
# R5: the 6 registers R4 heads on stim_rigid
$RS R5_eval_dinov2reg $RE/gpu nice -n 10 env V3_DEVICE=cpu V3_MAPS=none OMP_NUM_THREADS=4 $PY -B eval_re.py --bb dinov2reg --out ../scores_rigid/heads --ckpt-dir $ROOT/nn/registers/full/ckpt --stimsets stim_rigid --skip-mit || exit 61
grep -q "FAILED" $L/R5_eval_dinov2reg.log && exit 62
exit 0
