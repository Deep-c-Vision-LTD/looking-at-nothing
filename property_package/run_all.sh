#!/bin/bash
# nn/run_all.sh: property package (nn/ANALYSIS_RULE.md). Runs, one heavy job at a time and all on the CPU:
#   1. score the 9 unscored layer-sweep readouts on v2/stim_v3 (740) + v3/stim_alt (360), 336 and 448 px, with the
#      pipeline of v2/scores/score_v3.py + v3/scores/score_alt.py (nn/score_nn.py, 3 threads, as the stored scores);
#   2. ERF of all 18 readouts and of the blocks they read (nn/erf_nn.py, 6 threads);
#   3. the analysis fixed by nn/ANALYSIS_RULE.md (nn/analyze_nn.py) -> nn/results/summary.json, nn/results/report.md.
# Writes only inside nn/. No downloads (offline Hugging Face cache only). Resumable: rerun after an interruption.
# Usage: bash nn/run_all.sh            (wrap is internal: the whole run holds caffeinate -i)
set -euo pipefail
NN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$NN")"
PY=${PY:-$(cd "${LAN_PROJECT_ROOT:-$ROOT}/.." && pwd)/saliency_research/.venv/bin/python}   # set PY to override; default as neurocomputing/gpu/run_gpu_re.sh
LOG="$NN/logs/run_all.log"
mkdir -p "$NN/logs" "$NN/results"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "$LOG"; }

if [ -z "${NN_CAFFEINATED:-}" ]; then
  export NN_CAFFEINATED=1
  exec caffeinate -i bash "$0" "$@"
fi

# single instance
PIDF="$NN/logs/run_all.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "run_all.sh already running (pid $(cat "$PIDF")); not starting a second heavy job"; exit 1
fi
echo $$ > "$PIDF"

# 0. checks: rule unchanged, backbones cached, interpreter
cd "$NN"
want=$(awk 'NR==1{print $1}' ANALYSIS_RULE.sha256)
have=$(shasum -a 256 ANALYSIS_RULE.md | awk '{print $1}')
[ "$want" = "$have" ] || { log "ANALYSIS_RULE.md changed (recorded $want, now $have); stopping"; exit 1; }
for m in models--facebook--dinov2-base models--openai--clip-vit-large-patch14 models--google--siglip-large-patch16-256; do
  ls "$HOME/.cache/huggingface/hub/$m"/snapshots/*/model.safetensors >/dev/null 2>&1 || { log "backbone $m not in the local cache; stopping (no downloads)"; exit 1; }
done
"$PY" -c "import torch, transformers, numpy, cv2, scipy" || { log "interpreter $PY lacks a dependency"; exit 1; }
log "start run_all (rule $have)"

# 1. scoring of the 9 unscored readouts (sequential; ~45 to 60 min)
cd "$ROOT"
for spec in "dinov2 4,6,10" "clip 8,16,20" "siglip 12,16,20"; do
  set -- $spec
  log "scoring $1 layers $2"
  NN_THREADS=3 "$PY" "$NN/score_nn.py" "$1" "$2" "$NN/results" >> "$NN/logs/score_$1.out" 2>&1
  log "scoring $1 done"
done

# 2. ERF of all 18 readouts and blocks (sequential; ~45 to 60 min)
for b in dinov2 clip siglip; do
  if [ -f "$NN/results/erf/$b.json" ]; then log "ERF $b already done"; continue; fi
  log "ERF $b"
  NN_THREADS=6 "$PY" "$NN/erf_nn.py" "$b" >> "$NN/logs/erf_$b.out" 2>&1
  log "ERF $b done"
done

# 3. analysis fixed by the rule
log "analysis"
"$PY" "$NN/analyze_nn.py" >> "$NN/logs/analyze.out" 2>&1
log "analysis done: $NN/results/summary.json, $NN/results/report.md"
tail -1 "$NN/logs/analyze.out" | tee -a "$LOG"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$NN/results/DONE"
