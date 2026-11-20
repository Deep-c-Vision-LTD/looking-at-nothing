#!/bin/zsh
# explore_battery.sh: runs every exploratory script of neuro/reanalysis/explore/ (README.md there; REANALYSIS_RULE.md
# Section 6) on the real data, one process at a time, each under run_step.sh (nice -n 10, OMP_NUM_THREADS=4,
# HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, V3_DEVICE=cpu, ioreg Metal-client watch that kills the step if its pid
# holds an AGXDeviceUserClient). Exploratory: a failing script is logged and the battery goes on to the next one
# (no decision depends on any of them). Outputs: results/explore/<name>.json (written by the scripts),
# A3_tokens/registers/ (RUN_PLAN T1), logs in results/run_logs/EX_<script>.log. Written 18 November 2026.
RE="$(cd "$(dirname "$0")/../.." && pwd)"                      # this re-analysis folder (neurocomputing/ in the repository)
ROOT=${LAN_PROJECT_ROOT:-$(cd "$RE/../.." && pwd)}              # project tree that results/ mirrors (as gpu/run_gpu_re.sh)
PY=${PY:-$(cd "$ROOT/.." && pwd)/saliency_research/.venv/bin/python}   # interpreter: set PY to override (as gpu/run_gpu_re.sh)
RS=$RE/results/scripts/run_step.sh
DRV=$RE/results/run_logs/explore_battery.driver.log
echo "explore battery start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $DRV
for s in a6_margin_curve s13_block12_nulls item7_corrected a11_scale a3_reg_tokens_extract a3_reg_tokens_analyze \
         a15a_token_norms a44_logmass a12_kavsek a13_region a46_calibration; do
  $RS EX_$s $RE nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 V3_DEVICE=cpu \
      PYTHONDONTWRITEBYTECODE=1 $PY -B explore/$s.py
  rc=$?
  echo "$s exit $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $DRV
  [ $rc -eq 99 ] && { echo "METAL CLIENT in $s: battery stopped" >> $DRV; exit 99; }
done
echo "explore battery end $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $DRV
exit 0
