#!/bin/zsh
# Launch rescore_probe.py on the CPU (nice 10, 4 threads), check that its pid never holds a Metal client
# (first check within 5 s, then every 5 s until 30 s, then every 15 s), kill it on a Metal client or after 570 s.
# Log: neuro/verify_rescore/probe_run.log
HERE="$(cd "$(dirname "$0")" && pwd)"                          # this folder (neurocomputing/verify_rescore/ in the repository)
RE="$(dirname "$HERE")"                                        # re-analysis folder
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}              # project root (as gpu/run_gpu_re.sh)
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}   # interpreter: set PY to override (as gpu/run_gpu_re.sh)
LOG=$HERE/probe_run.log
cd $HERE
echo "### probe start $(date -u +%Y-%m-%dT%H:%M:%SZ) set ${PROBE_SET:-all}" >> $LOG
nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  V3_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1 PROBE_LIMIT_S=540 PROBE_SET=${PROBE_SET:-all} $PY -B rescore_probe.py >> $LOG 2>&1 &
PID=$!
echo "### probe pid $PID" >> $LOG
t=0
metal=0
while kill -0 $PID 2>/dev/null; do
  if ioreg -l -w0 -r -c AGXDeviceUserClient | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' | grep -q "pid $PID,"; then
    echo "### METAL CLIENT held by pid $PID at ~${t}s: killing" >> $LOG
    kill -9 $PID; metal=1; break
  fi
  if [ $t -le 30 ]; then echo "### pid $PID: no Metal client at ~${t}s" >> $LOG; fi
  if [ $t -ge 570 ]; then echo "### hard limit 570 s: killing $PID" >> $LOG; kill -9 $PID; break; fi
  if [ $t -lt 30 ]; then sleep 5; t=$((t+5)); else sleep 15; t=$((t+15)); fi
done
wait $PID 2>/dev/null; rc=$?
echo "### probe end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc metal $metal elapsed_checks ~${t}s" >> $LOG
