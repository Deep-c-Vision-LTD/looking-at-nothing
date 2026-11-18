#!/bin/bash
# gpu_idle.sh [quiet]: exit 0 if no other heavy job is running, 1 otherwise (prints the offenders).
# "Heavy" (any one suffices), excluding this script's own ancestors (so the runner does not wait on itself;
# REG_SELF_PGID may name its process group):
#   1. any Python process holding a Metal (AGX) client, whatever its CPU share (added 18 November 2026 after the design
#      review, finding 2: an MPS-bound training process often sits below 20% CPU, so the CPU test alone missed it);
#   2. any Python process above 20% CPU;
#   3. any process whose command line names a known GPU job of this project (train_full / eval_full / sweep /
#      run_gpu.sh / run_full.sh / train_reg / eval_reg / train_re / eval_re / stats_re / run_gpu_re.sh) or of the
#      ablation runner that held the GPU on 18 November 2026 (ablation/run_all.sh and its steps train_arm, eval_arm,
#      cache_clip, score_arm), whatever its CPU share, so that the gaps between that runner's steps are not taken
#      for idle time.
self_pids=" $$ $PPID "
p=$PPID
while [ -n "$p" ] && [ "$p" != "1" ] && [ "$p" != "0" ]; do
  p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  [ -n "$p" ] && self_pids="$self_pids$p "
done
busy=""
while read -r pid pcpu pgid cmd; do
  case "$self_pids" in *" $pid "*) continue;; esac
  [ -n "${REG_SELF_PGID:-}" ] && [ "$pgid" = "$REG_SELF_PGID" ] && continue
  case "$cmd" in *gpu_idle.sh*|*" ps "*|ps\ *) continue;; esac
  heavy=0
  if echo "$cmd" | grep -qiE '(^|/)python[0-9.]*( |$)|/bin/python|Python\.app'; then
    awk -v c="$pcpu" 'BEGIN{exit !(c > 20)}' && heavy=1
  fi
  echo "$cmd" | grep -qE 'train_full\.py|eval_full\.py|sweep\.py|run_gpu\.sh|run_full\.sh|train_reg\.py|eval_reg\.py|train_re\.py|eval_re\.py|stats_re\.py|run_gpu_re\.sh|ablation/run_all\.sh|train_arm\.py|eval_arm\.py|cache_clip\.py|score_arm\.py' && heavy=1
  [ $heavy = 1 ] && busy="$busy\n  pid $pid cpu $pcpu: $(echo "$cmd" | cut -c1-160)"
done < <(ps -Ao pid=,pcpu=,pgid=,command=)
# 1. any other Python process holding a Metal (AGX) client is a GPU job
for mp in $(ioreg -l -w0 -r -c AGXDeviceUserClient 2>/dev/null | grep -o '"IOUserClientCreator" = "pid [0-9]*, [^"]*"' \
            | grep -i ', python' | sed -E 's/.*pid ([0-9]+),.*/\1/' | sort -u); do
  case "$self_pids" in *" $mp "*) continue;; esac
  if [ -n "${REG_SELF_PGID:-}" ] && [ "$(ps -o pgid= -p "$mp" 2>/dev/null | tr -d ' ')" = "$REG_SELF_PGID" ]; then continue; fi
  busy="$busy\n  pid $mp holds a Metal client"
done
if [ -n "$busy" ]; then
  [ "${1:-}" = quiet ] || echo -e "busy:$busy"
  exit 1
fi
exit 0
