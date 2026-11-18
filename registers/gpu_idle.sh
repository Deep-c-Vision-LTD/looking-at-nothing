#!/bin/bash
# gpu_idle.sh [quiet]: exit 0 if no other heavy job is running, 1 otherwise (prints the offenders).
# "Heavy": any python process above 20% CPU, or any process whose command line names a known GPU job of this project
# (train_full / eval_full / sweep / run_gpu.sh / run_full.sh / train_reg / eval_reg), excluding this script's own
# ancestors (so the registers runner does not wait on itself; REG_SELF_PGID may name its process group).
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
  echo "$cmd" | grep -qE 'train_full\.py|eval_full\.py|sweep\.py|run_gpu\.sh|run_full\.sh|train_reg\.py|eval_reg\.py' && heavy=1
  [ $heavy = 1 ] && busy="$busy\n  pid $pid cpu $pcpu: $(echo "$cmd" | cut -c1-160)"
done < <(ps -Ao pid=,pcpu=,pgid=,command=)
if [ -n "$busy" ]; then
  [ "${1:-}" = quiet ] || echo -e "busy:$busy"
  exit 1
fi
exit 0
