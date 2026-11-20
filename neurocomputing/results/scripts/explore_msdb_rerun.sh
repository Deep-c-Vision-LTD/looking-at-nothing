#!/bin/zsh
# explore_msdb_rerun.sh: reruns explore/a44_logmass.py, a12_kavsek.py and a13_region.py (unchanged; SHA-256 in
# results/explore_scripts.sha256) now that scores_rigid/DeepGaze_MSDB.json exists (660 displays, md5
# 51db3832b635ebc15523d3e3e8a9bc14), so that the DeepGaze MSDB rows of their stim_rigid sections are filled
# (explore/README.md: "rerun afterwards to fill it"). Same command as results/scripts/explore_battery.sh plus python -I
# (the task's process rule). One process at a time, each under run_step.sh (nice -n 10, OMP_NUM_THREADS=4, offline HF,
# V3_DEVICE=cpu, ioreg Metal-client watch that kills the step if its pid holds an AGXDeviceUserClient).
# The earlier outputs were moved first to results/explore/_previous/2026-11-18_before_msdb_rigid/ (never deleted).
# Logs: results/run_logs/EX_<script>_msdb.log; wall times appended to results/run_logs/wall_times.tsv.
# Written 18 November 2026 (DEVIATIONS.md D6).
RE="$(cd "$(dirname "$0")/../.." && pwd)"                      # this re-analysis folder (neurocomputing/ in the repository)
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}              # project tree that results/ mirrors (as gpu/run_gpu_re.sh)
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}   # interpreter: set PY to override (as gpu/run_gpu_re.sh)
RS=$RE/results/scripts/run_step.sh
DRV=$RE/results/run_logs/explore_msdb_rerun.driver.log
echo "explore MSDB rerun start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $DRV
for s in a44_logmass a12_kavsek a13_region; do
  $RS EX_${s}_msdb $RE nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 V3_DEVICE=cpu \
      PYTHONDONTWRITEBYTECODE=1 $PY -B -I explore/$s.py
  rc=$?
  echo "$s exit $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $DRV
  [ $rc -eq 99 ] && { echo "METAL CLIENT in $s: rerun stopped" >> $DRV; exit 99; }
  [ $rc -ne 0 ] && { echo "$s failed (exit $rc): rerun stopped" >> $DRV; exit $rc; }
done
echo "explore MSDB rerun end $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $DRV
exit 0
