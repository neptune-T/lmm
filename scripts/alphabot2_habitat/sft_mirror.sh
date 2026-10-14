#!/bin/bash
# Mirror SFT runs from /dev/shm (volatile, fast) to /workspace (bosfs, persistent; sequential copies only).
# Every POLL s, per run: each COMPLETE checkpoint not yet mirrored (copied to NAME.tmp, renamed; mirrored
# checkpoints absent from the source are pruned), `best` once it has been unchanged for 120 s, and the small
# files (train_log, summary, DONE, run_info*, heartbeat). Restore after a container restart:
#   cp -r /workspace/runtime/sft_mirror/RUN /dev/shm/lmm_runtime/sft/RUN
#   sft_mirror.sh [SRC] [DST] [POLL]
SRC=${1:-/dev/shm/lmm_runtime/sft}
DST=${2:-/workspace/runtime/sft_mirror}
POLL=${3:-300}
exec 9> /dev/shm/lmm_sft_mirror.lock
flock -n 9 || { echo "another sft_mirror is running"; exit 0; }
while true; do
  for run in "$SRC"/*/; do
    run=${run%/}; name=$(basename "$run")
    case "$name" in *.tmp|*precontract*) continue;; esac
    d="$DST/$name"; mkdir -p "$d"
    for f in train_log.jsonl summary.json DONE heartbeat.json run_info_step*.json; do
      for g in "$run"/$f; do [ -f "$g" ] && cp "$g" "$d/" 2>/dev/null; done
    done
    for c in "$run"/ckpt_step*/; do
      c=${c%/}; cn=$(basename "$c")
      [ -f "$c/COMPLETE" ] || continue
      [ -f "$d/$cn/COMPLETE" ] && continue
      rm -rf "$d/$cn.tmp"
      if cp -r "$c" "$d/$cn.tmp" && mv "$d/$cn.tmp" "$d/$cn"; then
        echo "$(date '+%F %T') mirrored $name/$cn"
      fi
    done
    for m in "$d"/ckpt_step*/; do
      m=${m%/}; [ -d "$run/$(basename "$m")" ] || { [ -f "$m/COMPLETE" ] && ls -d "$run"/ckpt_step* > /dev/null 2>&1 && rm -rf "$m"; }
    done
    if [ -d "$run/best" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$run/best" $(ls -d "$run"/best/* 2>/dev/null) | sort -n | tail -1) ))
      new=$(stat -c %Y $(ls -d "$run"/best/* 2>/dev/null) | sort -n | tail -1)
      old=$(cat "$d/best.mtime" 2>/dev/null)
      if [ "$age" -ge 120 ] && [ "$new" != "$old" ]; then
        rm -rf "$d/best.tmp"
        cp -r "$run/best" "$d/best.tmp" && rm -rf "$d/best" && mv "$d/best.tmp" "$d/best" && echo "$new" > "$d/best.mtime" \
          && echo "$(date '+%F %T') mirrored $name/best"
      fi
    fi
  done
  sleep "$POLL"
done
