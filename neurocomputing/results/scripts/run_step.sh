#!/bin/zsh
# run_step.sh <label> <workdir> <command...>
# Runs one step of the CPU battery (REANALYSIS_RULE.md Section 8.3, RUN_PLAN.md Step 3) as one process, logs its
# output to results/run_logs/<label>.log and its wall time to results/run_logs/wall_times.tsv, and polls ioreg:
# every 5 s for the first 60 s, then every 30 s, the step's PID must hold no AGXDeviceUserClient (Metal). If it does,
# the step is killed at once and the exit code is 99. Written 18 November 2026 for the fast battery; nothing outside
# neuro/ is written.
label=$1; wd=$2; shift 2
RES="$(cd "$(dirname "$0")/.." && pwd)"                         # results/ of this re-analysis folder (the script is in results/scripts/)
LOGD=$RES/run_logs; mkdir -p $LOGD
LOG=$LOGD/$label.log
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 V3_DEVICE=cpu
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ); t0=$(date +%s)
echo "### $label start $start_utc cwd $wd cmd: $*" >> $LOG
cd $wd || exit 98
"$@" >> $LOG 2>&1 &
pid=$!
echo "### $label pid $pid" >> $LOG
metal=0; n=0
while kill -0 $pid 2>/dev/null; do
  if ioreg -l -w0 -r -c AGXDeviceUserClient | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' | grep -q "pid $pid,"; then
    echo "### $label METAL CLIENT HELD by pid $pid at $(date -u +%Y-%m-%dT%H:%M:%SZ): killing" >> $LOG
    kill -9 $pid; metal=1; break
  fi
  if [ $n -eq 1 ]; then echo "### $label pid $pid: no Metal client at first check (~5 s)" >> $LOG; fi
  if [ $n -eq 6 ]; then echo "### $label pid $pid: no Metal client at ~30 s" >> $LOG; fi
  n=$((n+1))
  if [ $n -le 12 ]; then sleep 5; else sleep 30; fi
done
wait $pid; rc=$?
[ $metal -eq 1 ] && rc=99
end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ); el=$(( $(date +%s) - t0 ))
echo "### $label end $end_utc exit $rc wall ${el}s" >> $LOG
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$label" "$start_utc" "$end_utc" "$el" "$rc" "$*" >> $LOGD/wall_times.tsv
exit $rc
