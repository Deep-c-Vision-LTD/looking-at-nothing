DeepGaze MSDB scoring of the 660 stim_rigid displays (RUN_PLAN.md R6), CPU only, one process.

Launch (detached; survives the launching shell; written 18 November 2026):
  cd $LAN_PROJECT_ROOT/neuro/reanalysis/results/msdb_rigid
  nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' zsh ./run_msdb.sh > /dev/null 2>&1 &

Files here:
  run_msdb.sh                  wrapper (comments at its top say what it adds to RUN_PLAN R6's command)
  run.log                      stdout/stderr of the scoring process plus wrapper lines; ends with "MSDB_DONE" or "MSDB_FAILED <reason>"
  PID                          pid of the scoring process (python scoring/score_rigid.py msdb)
  DeepGaze_MSDB.partial.json   backup of the score file, refreshed every 5 min (complete JSON only)
  DeepGaze_MSDB.json           copy of the final score file (written only on success)

The score file itself is RE/scores_rigid/DeepGaze_MSDB.json (location fixed by the hashed scoring/score_rigid.py; it is
where analyze_re.py R7 reads it). score_rigid.py saves it every 20 displays and, when restarted, scores only the
displays not yet in it. To resume after an interruption: relaunch with the same command (the wrapper restores the
backup first if the score file was cut mid-write). The pipeline check on 4 stim_v3 displays (tolerance 1e-4) runs again
at each start before any rigid display is scored.

After MSDB_DONE (rule order, Section 8.4): R7 `nice -n 10 $PY -B -I analyze_re.py primary --which P5` (P5 for both
models, with P1, P2 and the abstract rule), then rerun explore/a44_logmass.py, a12_kavsek.py and a13_region.py to fill
the MSDB rows of their stim_rigid sections. Do not run them while this process is running (one heavy process at a time).
