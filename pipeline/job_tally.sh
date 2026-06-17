#!/usr/bin/env bash
# job_tally.sh — summarise the caller's Slurm jobs by state and name.
#
# `squeue -u $USER` is hard to eyeball once array jobs fan out: a pending
# array shows as ONE line (e.g. 12345_[5-511]) that is really 500+ task
# slots. This expands arrays (squeue -r) so the counts reflect actual tasks,
# then breaks the total down by state and by job name — so the DCAlign shard
# array vs. its gather job are tallied separately.
#
# Usage:
#   bash pipeline/job_tally.sh                 # one snapshot of all your jobs
#   bash pipeline/job_tally.sh -p caslake      # restrict to one partition
#   bash pipeline/job_tally.sh -u someoneelse  # someone else's jobs
#   bash pipeline/job_tally.sh -w 10           # refresh every 10s (built-in loop)
#
# Prefer the built-in `-w` over the external `watch`: on Midway the loaded
# python/miniforge module puts its own libncursesw/libtinfo first on
# LD_LIBRARY_PATH, so the system `watch` (procps-ng) loads the wrong ncurses
# ABI and SEGFAULTS. `-w` uses an ANSI screen-clear (no ncurses), so it is
# immune. If you must use `watch`, strip miniforge from the path first:
#   LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH"|tr ':' '\n'|grep -v miniforge|paste -sd: -) watch ...

set -euo pipefail
IFS=$'\n\t'

user="${USER}"
partition=""
interval=0
while getopts "u:p:w:h" opt; do
  case "$opt" in
    u) user="$OPTARG" ;;
    p) partition="$OPTARG" ;;
    w) interval="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^# \?//;s/^#//'; exit 0 ;;
    *) echo "usage: $0 [-u USER] [-p PARTITION] [-w SECONDS]" >&2; exit 2 ;;
  esac
done

tally() {
  # -r expands array elements to one line each; -h drops the header so the
  # output is pure data. Fields: state|name|partition.
  local sq rows scope total running pending
  sq=(squeue -u "$user" -h -r -o "%T|%j|%P")
  [[ -n "$partition" ]] && sq+=(-p "$partition")

  rows="$("${sq[@]}")"
  scope="$user${partition:+ on $partition}"

  if [[ -z "$rows" ]]; then
    echo "No jobs for $scope."
    return 0
  fi

  total=$(wc -l <<<"$rows")
  running=$(grep -c '^RUNNING|' <<<"$rows" || true)
  pending=$(grep -c '^PENDING|' <<<"$rows" || true)

  echo "Jobs for $scope: $total task(s)  (RUNNING $running, PENDING $pending)"

  echo
  printf '%-12s %6s\n' STATE COUNT
  awk -F'|' '{c[$1]++} END {for (s in c) printf "%-12s %6d\n", s, c[s]}' <<<"$rows" \
    | sort -k2 -nr

  echo
  printf '%-30s %-10s %6s\n' NAME STATE COUNT
  awk -F'|' '{c[$2"|"$1]++} END {for (k in c){n=index(k,"|"); printf "%-30s %-10s %6d\n", substr(k,1,n-1), substr(k,n+1), c[k]}}' <<<"$rows" \
    | sort -k1,1 -k2,2
}

if (( interval > 0 )); then
  # Built-in refresh loop. ANSI clear (\033[H home, \033[2J erase) instead of
  # the `clear`/`watch` binaries, which depend on ncurses and segfault under
  # the polluted LD_LIBRARY_PATH (see header). Ctrl-C to stop.
  while true; do
    printf '\033[H\033[2J'
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  refresh every ${interval}s  (Ctrl-C to stop)"
    echo
    tally
    sleep "$interval"
  done
else
  tally
fi
