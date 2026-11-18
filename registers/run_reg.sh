#!/bin/bash
# nn/registers/run_reg.sh: registers control for the DINOv2-B readout swap.
#   R4 readouts on facebook/dinov2-with-registers-base blocks 12 and 6 (one forward pass), seeds 0, 1, 2, trained with
#   the published readout-swap recipe (copied scripts, v3/gpu/train_full.py), each seed scored after training on the
#   held-out MIT1003 200, v2/stim_v3 and v3/stim_alt (copied v3/gpu/eval_full.py), then the published statistics
#   (copied v3/gpu/analyze_full.py) and the side-by-side report (report_reg.py -> REGISTERS_RESULTS.md).
# Features on the fly in float32 with the backbone in the loop (as the published run; no feature cache).
# Before every GPU step: waits until gpu_idle.sh reports no other heavy job twice in a row, 60 s apart; stops if
# less than 15 GB would be free. Resumable (step markers in full/done/, training resumes per epoch).
# Usage: bash nn/registers/run_reg.sh      (holds caffeinate -i internally; one instance)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/Users/hamitsoyel/Downloads/saliency_research/.venv/bin/python
OUT=$HERE/full; L=$HERE/logs; QL=$L/run_reg.log
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 V3_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
if [ -z "${REG_CAFFEINATED:-}" ]; then
  export REG_CAFFEINATED=1
  exec caffeinate -i bash "$0" "$@"
fi
mkdir -p $OUT/done $L
PIDF=$L/run_reg.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "run_reg.sh already running (pid $(cat $PIDF))"; exit 1; fi
echo $$ > "$PIDF"
cd "$HERE"
say(){ echo "$(date '+%F %T') $*" | tee -a $QL; }
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
disk_ok(){
  local f; f=$(free_gb)
  if [ "$f" -lt 17 ]; then say "STOP: only ${f} GB free (need >= 15 GB after this step's ~1 GB of maps and state)"; exit 2; fi
}
FAILED=""
run(){  # run <step name> <script> <args...>: skipped if its marker exists; one retry
  local name=$1; shift
  if [ -f $OUT/done/step_$name ]; then say "skip $name (done)"; return 0; fi
  for try in 1 2; do
    disk_ok; wait_idle
    say "start $name (try $try; $(free_gb) GB free)"; local t=$(date +%s)
    if nice -n 15 $PY "$@" >> $L/$name.log 2>&1; then
      date '+%F %T' > $OUT/done/step_$name; say "done $name $(( $(date +%s) - t ))s"; return 0
    fi
    say "error in $name (log $L/$name.log)"; sleep 60
  done
  say "FAILED $name"; FAILED="$FAILED $name"; return 1
}
say "start run_reg (pid $$, out $OUT, $(free_gb) GB free)"
for s in 0 1 2; do
  run train_dinov2reg_R4_s$s train_reg.py --bb dinov2reg --seed $s --archs R4 --out $OUT && \
    run eval_dinov2reg_s$s eval_reg.py --bb dinov2reg --out $OUT
done
run eval_dinov2reg_final eval_reg.py --bb dinov2reg --out $OUT          # anything left unscored
nice -n 15 $PY analyze_reg.py --out $OUT >> $L/analyze.log 2>&1 && say "analysis done" || { say "analysis FAILED"; FAILED="$FAILED analyze"; }
nice -n 15 $PY report_reg.py >> $L/report.log 2>&1 && say "report done: $HERE/REGISTERS_RESULTS.md" || { say "report FAILED"; FAILED="$FAILED report"; }
say "finished; failed steps:${FAILED:- none}; trained heads: $(ls $OUT/ckpt/*.pt 2>/dev/null | wc -l | tr -d ' ') of 6"
date '+%F %T' > $HERE/DONE
