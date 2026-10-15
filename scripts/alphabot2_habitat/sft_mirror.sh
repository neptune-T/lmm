#!/bin/bash
# Mirror SFT runs from /dev/shm (volatile, fast) to /workspace (bosfs, persistent).
# bosfs is an object store: a DIRECTORY RENAME re-copies every object and silently dropped the 16.6 GB
# model.safetensors (every mirrored best / checkpoint policy lost it). So nothing is renamed: each item is
# copied straight into its final directory, every file's size is verified against the source, and only then
# a .MIRRORED marker (with the source signature) is written; a missing / stale marker means "copy again".
# Per run, every POLL s: each COMPLETE checkpoint, `best` once unchanged for 120 s, and the small files
# (train_log, summary, DONE, run_info*, heartbeat); mirrored checkpoints absent from the source are pruned.
# Restore after a container restart:
#   cp -r /workspace/runtime/sft_mirror/RUN /dev/shm/lmm_runtime/sft/RUN   (dirs with .MIRRORED only)
#   sft_mirror.sh [SRC] [DST] [POLL]
SRC=${1:-/dev/shm/lmm_runtime/sft}
DST=${2:-/workspace/runtime/sft_mirror}
POLL=${3:-300}
exec 9> /dev/shm/lmm_sft_mirror.lock
flock -n 9 || { echo "another sft_mirror is running"; exit 0; }

sig() {  # signature of a source dir: relative path + size of every file
  (cd "$1" && find . -type f ! -name .MIRRORED -printf "%P %s\n" | sort)
}

dest_sig() {  # the same signature measured on the copy, by explicit stat per path (bosfs listings lag)
  local s="$1" d="$2"
  (cd "$s" && find . -type f ! -name .MIRRORED -printf "%P\n" | sort | while read -r f; do
     echo "$f $(stat -c %s "$d/$f" 2>/dev/null || echo missing)"; done)
}

mirror_dir() {  # src dst label
  local s="$1" d="$2" want
  want=$(sig "$s")
  [ -f "$d/.MIRRORED" ] && [ "$(cat "$d/.MIRRORED")" = "$want" ] && [ "$(dest_sig "$s" "$d")" = "$want" ] && return 0
  rm -rf "$d"
  mkdir -p "$d"
  (cd "$s" && find . -type d -printf "%P\n" | while read -r sub; do [ -n "$sub" ] && mkdir -p "$d/$sub"; done)
  (cd "$s" && find . -type f -printf "%P\n" | while read -r f; do cp "$s/$f" "$d/$f" || exit 1; done) || return 1
  [ "$(dest_sig "$s" "$d")" = "$want" ] || { echo "$(date '+%F %T') size mismatch, retry later: $3"; return 1; }
  echo "$want" > "$d/.MIRRORED"
  echo "$(date '+%F %T') mirrored $3"
}

while true; do
  for run in "$SRC"/*/; do
    run=${run%/}; name=$(basename "$run")
    case "$name" in *.tmp|*precontract*) continue;; esac
    d="$DST/$name"; mkdir -p "$d"
    for f in train_log.jsonl summary.json DONE heartbeat.json run_info_step*.json SAMPLES_SEEN_CORRECTION.json; do
      for g in "$run"/$f; do [ -f "$g" ] && cp "$g" "$d/" 2>/dev/null; done
    done
    for c in "$run"/ckpt_step*/; do
      c=${c%/}; cn=$(basename "$c")
      [ -f "$c/COMPLETE" ] || continue
      mirror_dir "$c" "$d/$cn" "$name/$cn"
    done
    for m in "$d"/ckpt_step*/; do                                  # prune what the run itself pruned
      m=${m%/}
      [ -d "$run/$(basename "$m")" ] || { ls -d "$run"/ckpt_step* > /dev/null 2>&1 && rm -rf "$m"; }
    done
    if [ -d "$run/best" ]; then
      newest=$(find "$run/best" -type f -printf "%T@\n" | sort -n | tail -1)
      age=$(( $(date +%s) - ${newest%.*} ))
      [ "$age" -ge 120 ] && mirror_dir "$run/best" "$d/best" "$name/best"
    fi
  done
  sleep "$POLL"
done
