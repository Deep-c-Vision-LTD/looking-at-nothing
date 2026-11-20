#!/bin/zsh
# run_msdb.sh: RUN_PLAN.md R6, DeepGaze MSDB scoring of the 660 stim_rigid displays, as one CPU process.
# Written 18 November 2026. Launch detached (see results/msdb_rigid/README.txt); rerun the same way to resume.
#
# - Command, exactly as RUN_PLAN.md R6: `nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py msdb` (in RE).
#   score_rigid.py (fixed, hashed, not edited) calls require_public(), runs its MSDB pipeline check on 4 stim_v3 displays
#   (tolerance 1e-4, rule Section 3.2) before any rigid display, and writes RE/scores_rigid/DeepGaze_MSDB.json every 20
#   displays. That location is fixed by score_rigid.py and is where analyze_re.py (R7) reads it. On restart,
#   score_rigid.py loads that file and scores only the displays not yet in it: the run is resumable as written.
# - This wrapper adds: the run log results/msdb_rigid/run.log (stdout and stderr of the scoring process, plus wrapper
#   lines), the scoring pid in results/msdb_rigid/PID, a Metal-client watch (kill at once if the pid holds an
#   AGXDeviceUserClient), a backup of the score file every 5 min (results/msdb_rigid/DeepGaze_MSDB.partial.json, only
#   when the file ends with "}", i.e. was not caught mid-write), restore of that backup on restart if the score file was
#   truncated, a completeness check (660 top-level entries) and the final line "MSDB_DONE" or "MSDB_FAILED <reason>".
#   On success the final score file is also copied to results/msdb_rigid/DeepGaze_MSDB.json with its md5.
RE="$(cd "$(dirname "$0")/../.." && pwd)"                      # this re-analysis folder (neurocomputing/ in the repository)
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}              # project tree that results/ mirrors (as gpu/run_gpu_re.sh)
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}   # interpreter: set PY to override (as gpu/run_gpu_re.sh)
OUTD=$RE/results/msdb_rigid
LOG=$OUTD/run.log
SC=$RE/scores_rigid/DeepGaze_MSDB.json
BK=$OUTD/DeepGaze_MSDB.partial.json
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 V3_DEVICE=cpu
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
n_done() { [ -f "$1" ] && grep -o '": {' "$1" | wc -l | tr -d ' ' || echo 0; }
echo "=== run_msdb.sh start $(ts) wrapper pid $$ (host load: $(uptime | sed 's/.*load averages*: //'))" >> $LOG
if [ -f $SC ] && [ "$(tail -c 1 $SC)" != "}" ]; then
  mkdir -p $RE/_trash/msdb_rigid
  mv $SC $RE/_trash/msdb_rigid/DeepGaze_MSDB.truncated.$(date +%s).json
  if [ -f $BK ] && [ "$(tail -c 1 $BK)" = "}" ]; then cp $BK $SC; fi
  echo "$(ts) score file was truncated: moved to _trash/msdb_rigid; restored backup with $(n_done $SC) displays" >> $LOG
fi
echo "$(ts) displays already in $SC: $(n_done $SC) of 660" >> $LOG
cd $RE || { echo "MSDB_FAILED cannot cd to $RE" >> $LOG; exit 1; }
nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py msdb >> $LOG 2>&1 &
pid=$!
echo $pid > $OUTD/PID
echo "$(ts) scoring pid $pid (written to $OUTD/PID); command: nice -n 10 env OMP_NUM_THREADS=4 $PY -B scoring/score_rigid.py msdb" >> $LOG
metal=0; n=0; last_bk=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  if ioreg -l -w0 -r -c AGXDeviceUserClient | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' | grep -q "pid $pid,"; then
    echo "$(ts) METAL CLIENT HELD by pid $pid: killing" >> $LOG
    kill -9 $pid; metal=1; break
  fi
  [ $n -eq 2 ] && echo "$(ts) pid $pid: no Metal client at ~10 s" >> $LOG
  [ $n -eq 5 ] && echo "$(ts) pid $pid: no Metal client at ~25 s" >> $LOG
  n=$((n+1))
  now=$(date +%s)
  if [ $((now - last_bk)) -ge 300 ] && [ -f $SC ] && [ "$(tail -c 1 $SC)" = "}" ]; then
    cp $SC $BK.tmp && mv $BK.tmp $BK; last_bk=$now
  fi
  if [ $n -le 12 ]; then sleep 5; else sleep 60; fi
done
wait $pid; rc=$?
cnt=$(n_done $SC)
if [ $metal -eq 1 ]; then
  echo "MSDB_FAILED metal client held by pid $pid (killed); $cnt of 660 displays scored; fix before resuming" >> $LOG; exit 99
fi
if [ $rc -eq 0 ] && [ "$cnt" = "660" ]; then
  cp $SC $OUTD/DeepGaze_MSDB.json
  echo "$(ts) final: $cnt of 660 displays; md5 scores_rigid/DeepGaze_MSDB.json $(md5 -q $SC); copy results/msdb_rigid/DeepGaze_MSDB.json" >> $LOG
  echo "MSDB_DONE" >> $LOG; exit 0
fi
echo "MSDB_FAILED exit code $rc; $cnt of 660 displays scored (resume: relaunch run_msdb.sh; score_rigid.py skips scored displays)" >> $LOG
exit 1
