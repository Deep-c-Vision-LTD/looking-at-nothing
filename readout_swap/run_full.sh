#!/bin/bash
# Full-data readout swap: 4 feature sets x 5 readouts x 3 seeds = 60 readouts, then evaluation and statistics.
# Order (as in the published run, "staged"): R1n, R1, R4 for DINOv2-B (blocks 12 and 6 share one forward pass),
# SigLIP-L block 24 and the DeepGaze IIE ShapeNet-C features; then R2 and R3 for the same three backbones.
# QUEUE_MODE=folded trains all five readouts per pass instead (same models, fewer passes).
# Resumable: finished steps leave markers in $OUT/done/, and training resumes from the last finished epoch.
# Usage: V3_DEVICE=mps|cuda|cpu PY=python readout_swap/run_full.sh      (needs the training cache: models/ab_objective.py cache)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
OUT=${OUT:-$($PY -c "import sys; sys.path.insert(0, '$HERE/..'); import paths; print(paths.V3 / 'gpu' / 'full')")}
L=$OUT/logs; QL=$L/queue.log
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3} V3_DEVICE=${V3_DEVICE:-mps}
export PYTORCH_ENABLE_MPS_FALLBACK=1
MODE=${QUEUE_MODE:-staged}
mkdir -p $OUT/done $L
cd $HERE
say(){ echo "$(date '+%F %T') $*" >> $QL; }
FAILED=""
run(){  # run <step name> <script> <args...>: skipped if its marker exists; one retry (training resumes per epoch)
  local name=$1; shift
  if [ -f $OUT/done/step_$name ]; then say "skip $name (done)"; return 0; fi
  for try in 1 2; do
    say "start $name (try $try)"; local t=$(date +%s)
    if nice -n 15 $PY "$@" >> $L/$name.log 2>&1; then
      date '+%F %T' > $OUT/done/step_$name; say "done $name $(( $(date +%s) - t ))s"; return 0
    fi
    say "error in $name (log $L/$name.log)"; sleep 60
  done
  say "FAILED $name"; FAILED="$FAILED $name"; return 1
}
ev(){   # score every trained, unscored head of a backbone
  nice -n 15 $PY eval_full.py --bb $1 --out $OUT >> $L/eval_$1.log 2>&1 || { say "eval $1 failed"; FAILED="$FAILED eval_$1"; }
  say "evaluated $1"
}
job(){  # job <backbone> <archs> <label>: three seeds, evaluation after each
  run stats_$1 stats_full.py --bb $1 --out $OUT || return 1
  for s in 0 1 2; do
    run train_$1_$3_s$s train_full.py --bb $1 --seed $s --archs $2 --out $OUT && ev $1
  done
}
say "start: mode $MODE, device $V3_DEVICE, out $OUT"
if [ "$MODE" = folded ]; then
  job dinov2 R1n,R1,R4,R2,R3 all
  job siglip R1n,R1,R4,R2,R3 all
  job iie R1n,R1,R4,R2,R3 all
else
  job dinov2 R1n,R1,R4 A
  job siglip R1n,R1,R4 A
  job iie R1n,R1,R4 A
  job dinov2 R2,R3 B
  job siglip R2,R3 B
  job iie R2,R3 B
fi
for bb in dinov2 siglip iie; do ev $bb; done      # picks up stim_alt and anything left unscored
nice -n 15 $PY analyze_full.py --out $OUT >> $L/analyze.log 2>&1 && say "analysis done" || { say "analysis FAILED"; FAILED="$FAILED analyze"; }
say "finished; failed steps:${FAILED:- none}; trained heads: $(ls $OUT/ckpt/*.pt 2>/dev/null | wc -l | tr -d ' ') of 60"
