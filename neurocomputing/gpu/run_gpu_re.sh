#!/bin/bash
# neuro/reanalysis/gpu/run_gpu_re.sh: GPU runs G3 and A14 of REANALYSIS_RULE.md Section 5.6 (decision D6).
#   G3:  readouts R1n, R1, R2, R3 on facebook/dinov2-with-registers-base blocks 12 and 6, seeds 0, 1, 2
#        (R4 exists: nn/registers/full, 17 Nov 2026); all 8 heads of a seed share one frozen forward pass per batch.
#   A14: readouts R1n, R1, R2, R3, R4 on facebook/dinov2-base hidden_states[7]+[11] (1,536 channels) and
#        hidden_states[11] alone, seeds 0, 1, 2; all 10 heads of a seed share one forward pass.
#   Recipe: the published readout-swap training and evaluation (copied: train_re.py, eval_re.py, stats_re.py);
#   features on the fly in float32 on MPS, as the published run; no feature cache; no maps saved (V3_MAPS=none).
# Gates, in this order (REANALYSIS_RULE.md Sections 3.3 and 8; tightened 18 November 2026 after the design review):
#   1. the rule is fixed and public and nothing fixed with it has changed: REANALYSIS_RULE.sha256 and PUBLIC_PUSH.txt
#      exist, every hashed file matches its SHA-256 and every stimulus file its md5 (scoring/common_rigid.require_public);
#   2. ./GO_AHEAD exists (the author creates it once the ablation runner that held the GPU on 18 November 2026,
#      ablation/run_all.sh, has exited and its log ends with its analyze step, and no other GPU job runs);
#   3. no ablation runner (ablation/run_all.sh) is alive at start (refuse, do not wait);
#   4. gpu_idle.sh reports no other heavy job twice, 60 s apart, before every step (a Python process holding a Metal
#      client counts as heavy whatever its CPU share);
#   5. at least 10 GB free on the disk (re-checked before every step);
#   6. sanity gate 3: the copied evaluation reproduces the published scores of dinov2_L12_R4_s0 (plain path) and of
#      dinov2reg_L12_R4_s0 (registers path) on MIT1003, stim_v3 and stim_alt to 1e-6.
# Usage: bash run_gpu_re.sh [g3|a14|both]   (default both, G3 first; resumable; one instance; holds caffeinate -i)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RE="$(dirname "$HERE")"
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}
WHICH=${1:-both}
L=$HERE/logs; QL=$L/run_gpu_re.log
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 V3_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1
export V3_MAPS=none OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
if [ -z "${RE_CAFFEINATED:-}" ]; then
  export RE_CAFFEINATED=1
  exec caffeinate -i bash "$0" "$@"
fi
mkdir -p $L
say(){ echo "$(date '+%F %T') $*" | tee -a $QL; }
[ -f "$RE/REANALYSIS_RULE.sha256" ] && [ -f "$RE/PUBLIC_PUSH.txt" ] || { say "STOP: rule not fixed and public (REANALYSIS_RULE.sha256, PUBLIC_PUSH.txt)"; exit 2; }
$PY -B -c "import sys; sys.path.insert(0, '$RE/scoring'); import common_rigid as C; print('fixed files verified:', C.require_public())" >> $QL 2>&1 \
  || { say "STOP: a file fixed with the rule changed, or a stimulus differs from its manifest (see $QL)"; exit 2; }
[ -f "$HERE/GO_AHEAD" ] || { say "STOP: no GO_AHEAD file (the author creates it when the GPU may be used)"; exit 2; }
if pgrep -f 'ablation/run_all\.sh' >/dev/null; then say "STOP: the ablation runner (ablation/run_all.sh) is still alive"; exit 2; fi
PIDF=$L/run_gpu_re.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "run_gpu_re.sh already running (pid $(cat $PIDF))"; exit 1; fi
echo $$ > "$PIDF"
cd "$HERE"
free_gb(){ df -g "$HERE" | awk 'NR==2{print $4}'; }
wait_idle(){
  local n=0
  while true; do
    if ./gpu_idle.sh quiet; then
      sleep 60
      ./gpu_idle.sh quiet && return 0
    else
      [ $((n % 10)) = 0 ] && say "waiting: another heavy job is running ($(./gpu_idle.sh | tail -n +2 | head -3 | tr -s ' ' | cut -c1-200 | tr '\n' ';'))"
      n=$((n + 1)); sleep 60
    fi
  done
}
disk_ok(){ local f; f=$(free_gb); if [ "$f" -lt 10 ]; then say "STOP: only ${f} GB free (need >= 10 GB)"; exit 2; fi; }
FAILED=""
run(){  # run <step> <out dir> <script> <args...>: skipped if its marker exists; one retry (training resumes per epoch)
  local name=$1 out=$2; shift 2
  mkdir -p $out/done
  if [ -f $out/done/step_$name ]; then say "skip $name (done)"; return 0; fi
  for try in 1 2; do
    disk_ok; wait_idle
    say "start $name (try $try; $(free_gb) GB free)"; local t=$(date +%s)
    if nice -n 15 $PY "$@" >> $L/$name.log 2>&1; then
      date '+%F %T' > $out/done/step_$name; say "done $name $(( $(date +%s) - t ))s"; return 0
    fi
    say "error in $name (log $L/$name.log)"; sleep 60
  done
  say "FAILED $name"; FAILED="$FAILED $name"; return 1
}
say "start run_gpu_re ($WHICH; pid $$; $(free_gb) GB free)"
wait_idle
nice -n 15 $PY -c "import torch; assert torch.backends.mps.is_available(); print('mps ok', float(torch.ones(4, device='mps').sum()))" >> $QL 2>&1 \
  || { say "MPS not available: abort"; exit 1; }
# ---- sanity gate 3 (REANALYSIS_RULE.md Section 3.3)
CK=$HERE/check
run check_eval $CK eval_re.py --bb dinov2 --out $CK --ckpt-dir $ROOT/v3/gpu/full/ckpt --tags dinov2_L12_R4_s0 --stimsets stim_v3,stim_alt \
  && nice -n 15 $PY check_eval_re.py $CK >> $L/check_eval.log 2>&1 \
  || { say "STOP: sanity gate 3 failed (see $L/check_eval.log)"; exit 3; }
say "sanity gate 3 passed (plain path): $(tail -1 $L/check_eval.log)"
CK2=$HERE/check_reg
run check_eval_reg $CK2 eval_re.py --bb dinov2reg --out $CK2 --ckpt-dir $ROOT/nn/registers/full/ckpt --tags dinov2reg_L12_R4_s0 --stimsets stim_v3,stim_alt \
  && nice -n 15 $PY check_eval_re.py $CK2 --pub $ROOT/nn/registers/full/scores >> $L/check_eval_reg.log 2>&1 \
  || { say "STOP: sanity gate 3 failed for the registers path (see $L/check_eval_reg.log)"; exit 3; }
say "sanity gate 3 passed (registers path): $(tail -1 $L/check_eval_reg.log)"
# ---- G3
if [ "$WHICH" = g3 ] || [ "$WHICH" = both ]; then
  O=$HERE/g3
  run stats_dinov2reg $O stats_re.py --bb dinov2reg --out $O
  for s in 0 1 2; do
    run train_dinov2reg_s$s $O train_re.py --bb dinov2reg --seed $s --archs R1n,R1,R2,R3 --out $O && \
      run eval_dinov2reg_s$s $O eval_re.py --bb dinov2reg --out $O
  done
  run eval_dinov2reg_final $O eval_re.py --bb dinov2reg --out $O
  V3_DEVICE=cpu nice -n 15 $PY analyze_gpu.py --out $O >> $L/analyze_g3.log 2>&1 && say "G3 continuity statistics done" || say "G3 continuity statistics FAILED"
fi
# ---- A14
if [ "$WHICH" = a14 ] || [ "$WHICH" = both ]; then
  O=$HERE/a14
  run stats_dinov2ms $O stats_re.py --bb dinov2ms --out $O
  for s in 0 1 2; do
    run train_dinov2ms_s$s $O train_re.py --bb dinov2ms --seed $s --archs R1n,R1,R2,R3,R4 --out $O && \
      run eval_dinov2ms_s$s $O eval_re.py --bb dinov2ms --out $O
  done
  run eval_dinov2ms_final $O eval_re.py --bb dinov2ms --out $O
  V3_DEVICE=cpu nice -n 15 $PY analyze_gpu.py --out $O >> $L/analyze_a14.log 2>&1 && say "A14 continuity statistics done" || say "A14 continuity statistics FAILED"
fi
say "finished; failed steps:${FAILED:- none}; heads: G3 $(ls $HERE/g3/ckpt/*.pt 2>/dev/null | wc -l | tr -d ' ') of 24, A14 $(ls $HERE/a14/ckpt/*.pt 2>/dev/null | wc -l | tr -d ' ') of 30"
date '+%F %T' > $HERE/DONE_$WHICH
