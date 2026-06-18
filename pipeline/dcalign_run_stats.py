#!/usr/bin/env python3
"""dcalign_run_stats.py — summarise the timing of a DCAlign shard fan-out.

A combine run's couplings-aware alignment is fanned out as a Slurm array
(one task per (model, shard); see pipeline/external/README.md). This script
reads the job ids the launcher recorded in ``dcalign/.shard_jids`` and prints
basic timing stats for the array: total CPU hours, the longest-running shard,
wall-clock makespan vs. the CPU time it bought (effective concurrency), and an
ASCII histogram of per-shard run times.

Source of truth is Slurm accounting (``sacct``), which gives true allocated
core-seconds, consumed CPU time, and peak RSS per task. ``sacct`` accounting is
purged after a retention window, so when it returns nothing the script falls
back to the per-shard log timestamps (durable on disk) for wall-clock elapsed
and says so loudly. ``--source`` forces one or the other.

Usage:
    python pipeline/dcalign_run_stats.py [RUN_ROOT] [--source auto|sacct|logs]
                                         [--bins N]

RUN_ROOT is the iteration dir that contains ``dcalign/`` (or that dir itself).
Default: combine/combine-CM-PPIC-dcalign/latest under the repo root.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "combine" / "combine-CM-PPIC-dcalign" / "latest"

# sacct main task lines look like "<jid>_<task>"; step lines append ".batch"/".extern".
_TASK_RE = re.compile(r"^\d+_(\d+)$")
_BATCH_RE = re.compile(r"^\d+_(\d+)\.batch$")
_LOG_NAME_RE = re.compile(r"dcalign_shard_\d+_(\d+)\.log$")
_LOG_TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")
_RSS_RE = re.compile(r"^([\d.]+)([KMGT]?)$")


def warn(msg: str) -> None:
    print(f"[dcalign_run_stats] {msg}", file=sys.stderr)


def die(msg: str) -> "None":
    warn(msg)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def parse_hms(s: str) -> float:
    """sacct duration -> seconds. Handles 'DD-HH:MM:SS', 'HH:MM:SS', 'MM:SS.mmm'."""
    s = s.strip()
    if not s or s in ("INVALID", "UNLIMITED"):
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)  # pad MM:SS -> 0:MM:SS
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def parse_dt(s: str):
    s = s.strip()
    if not s or s in ("Unknown", "None"):
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


def parse_rss(s: str) -> float:
    """sacct MaxRSS (e.g. '960900K') -> bytes. Empty -> 0."""
    s = s.strip()
    if not s:
        return 0.0
    m = _RSS_RE.match(s)
    if not m:
        return 0.0
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[m.group(2)]
    return float(m.group(1)) * mult


def human_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# --------------------------------------------------------------------------- #
# task -> (model, shard) labelling
# --------------------------------------------------------------------------- #
def load_shard_layout(dcalign_dir: Path):
    """Return (n_shards, models) from shards_manifest.json, or (None, None)."""
    import json

    mf = dcalign_dir / "shards_manifest.json"
    if not mf.exists():
        return None, None
    d = json.loads(mf.read_text())
    return d.get("n_shards"), d.get("models")


def label_task(task: int, n_shards, models) -> str:
    """task = model_index * n_shards + shard (see run_dcalign_shard.py)."""
    if not n_shards or not models:
        return f"task {task}"
    mi, shard = divmod(task, n_shards)
    model = models[mi] if 0 <= mi < len(models) else f"model[{mi}]"
    return f"{model} shard {shard}"


# --------------------------------------------------------------------------- #
# data sources
# --------------------------------------------------------------------------- #
class Task:
    __slots__ = ("task", "state", "elapsed", "cpu", "alloc_sec", "alloc_cpus",
                 "rss", "start", "end")

    def __init__(self, task):
        self.task = task
        self.state = "?"
        self.elapsed = 0.0      # wall-clock seconds
        self.cpu = 0.0          # consumed CPU seconds (TotalCPU)
        self.alloc_sec = 0.0    # allocated core-seconds (CPUTimeRAW)
        self.alloc_cpus = 1
        self.rss = 0.0          # peak resident bytes
        self.start = None
        self.end = None


def from_sacct(array_jid: str):
    """Parse `sacct` for the shard array. Returns {} if accounting is gone."""
    fmt = "JobID,State,Elapsed,TotalCPU,CPUTimeRAW,AllocCPUS,MaxRSS,Start,End"
    try:
        out = subprocess.run(
            ["sacct", "-j", array_jid, "-P", "--noheader", "--format", fmt],
            capture_output=True, text=True, check=True,
        ).stdout
    except FileNotFoundError:
        warn("sacct not found on PATH")
        return {}
    except subprocess.CalledProcessError as e:
        warn(f"sacct failed: {e.stderr.strip()}")
        return {}

    tasks: dict[int, Task] = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 9:
            continue
        jobid = f[0]
        m = _TASK_RE.match(jobid)
        if m:  # main task record: timing + state
            t = tasks.setdefault(int(m.group(1)), Task(int(m.group(1))))
            t.state = f[1]
            t.elapsed = parse_hms(f[2])
            t.cpu = parse_hms(f[3])
            t.alloc_sec = float(f[4]) if f[4].strip() else 0.0
            t.alloc_cpus = int(f[5]) if f[5].strip() else 1
            t.start = parse_dt(f[7])
            t.end = parse_dt(f[8])
            continue
        b = _BATCH_RE.match(jobid)
        if b:  # batch step carries MaxRSS
            t = tasks.setdefault(int(b.group(1)), Task(int(b.group(1))))
            t.rss = max(t.rss, parse_rss(f[6]))
    return tasks


def from_logs(dcalign_dir: Path):
    """Fallback: per-shard wall-clock from log timestamps (cpus assumed 1)."""
    log_dir = dcalign_dir / "logs"
    tasks: dict[int, Task] = {}
    for log in sorted(log_dir.glob("dcalign_shard_*.log")):
        m = _LOG_NAME_RE.search(log.name)
        if not m:
            continue
        stamps = []
        for line in log.read_text().splitlines():
            ts = _LOG_TS_RE.match(line)
            if ts:
                stamps.append(datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S"))
        if not stamps:
            continue
        task = int(m.group(1))
        t = Task(task)
        t.state = "COMPLETED"
        t.start, t.end = min(stamps), max(stamps)
        t.elapsed = (t.end - t.start).total_seconds()
        t.cpu = t.elapsed       # single-threaded; consumed ~= wall
        t.alloc_sec = t.elapsed  # cpus=1
        tasks[task] = t
    return tasks


def gather_job_summary(gather_jid: str) -> str | None:
    fmt = "JobID,State,Elapsed,TotalCPU"
    try:
        out = subprocess.run(
            ["sacct", "-j", gather_jid, "-P", "--noheader", "--format", fmt],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        f = line.split("|")
        if len(f) >= 4 and f[0] == gather_jid:
            return f"state={f[1]}  elapsed={f[2]}  cpu={f[3]}"
    return None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def ascii_hist(values_min, nbins: int) -> list[str]:
    lo, hi = min(values_min), max(values_min)
    if hi <= lo:
        return [f"  all {len(values_min)} shard(s) ~ {lo:.1f} min"]
    width = (hi - lo) / nbins
    counts = [0] * nbins
    for v in values_min:
        idx = min(int((v - lo) / width), nbins - 1)
        counts[idx] += 1
    peak = max(counts)
    bar_w = 40
    rows = []
    for i, c in enumerate(counts):
        edge_lo = lo + i * width
        edge_hi = edge_lo + width
        bar = "#" * round(c / peak * bar_w) if c else ""
        rows.append(f"  {edge_lo:6.1f}-{edge_hi:6.1f} min | {bar} {c}")
    return rows


def report(tasks: dict[int, Task], source: str, n_shards, models,
           gather_summary, bins: int) -> None:
    ts = sorted(tasks.values(), key=lambda t: t.task)
    n = len(ts)
    elapsed = [t.elapsed for t in ts]
    alloc_h = sum(t.alloc_sec for t in ts) / 3600
    cpu_h = sum(t.cpu for t in ts) / 3600

    starts = [t.start for t in ts if t.start]
    ends = [t.end for t in ts if t.end]
    makespan = (max(ends) - min(starts)).total_seconds() if starts and ends else 0.0

    longest = max(ts, key=lambda t: t.elapsed)
    rss_peak = max((t.rss for t in ts), default=0.0)

    states: dict[str, int] = {}
    for t in ts:
        states[t.state] = states.get(t.state, 0) + 1

    print("=" * 64)
    print(f"DCAlign shard-array timing  (source: {source})")
    print("=" * 64)
    print(f"shard tasks            : {n}")
    print("states                 : "
          + ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
    not_done = n - states.get("COMPLETED", 0)
    if not_done:
        warn(f"{not_done} task(s) did not COMPLETE — totals below are partial")

    print()
    print(f"total allocated CPU    : {alloc_h:9.2f} core-hours")
    if abs(cpu_h - alloc_h) > 1e-6:
        print(f"total consumed CPU     : {cpu_h:9.2f} core-hours  "
              f"({cpu_h / alloc_h * 100:.0f}% of allocation)")
    if makespan > 0:
        print(f"wall-clock makespan    : {human_dur(makespan)}  "
              f"({makespan / 3600:.2f} h)")
        print(f"effective concurrency  : {alloc_h * 3600 / makespan:6.1f}x  "
              f"(CPU-hours / makespan)")

    print()
    print("per-shard wall time")
    print(f"  min / median / mean / max : "
          f"{human_dur(min(elapsed))} / {human_dur(statistics.median(elapsed))} / "
          f"{human_dur(statistics.fmean(elapsed))} / {human_dur(max(elapsed))}")
    print(f"  longest shard             : {label_task(longest.task, n_shards, models)} "
          f"(task {longest.task}) — {human_dur(longest.elapsed)}")
    if rss_peak > 0:
        print(f"  peak RSS (any shard)      : {rss_peak / 1024**3:.2f} GiB")

    print()
    print(f"histogram of per-shard wall time ({bins} bins):")
    for row in ascii_hist([e / 60 for e in elapsed], bins):
        print(row)

    if gather_summary:
        print()
        print(f"gather job             : {gather_summary}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_root", nargs="?", default=str(DEFAULT_RUN_ROOT),
                    help="iteration dir containing dcalign/ (default: %(default)s)")
    ap.add_argument("--source", choices=["auto", "sacct", "logs"], default="auto",
                    help="timing source (default: auto = sacct, fall back to logs)")
    ap.add_argument("--bins", type=int, default=12, help="histogram bins (default: 12)")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if (run_root / "dcalign").is_dir():
        dcalign_dir = run_root / "dcalign"
    elif run_root.name == "dcalign" and run_root.is_dir():
        dcalign_dir = run_root
    else:
        die(f"no dcalign/ under {run_root} (and it is not a dcalign dir)")

    jid_file = dcalign_dir / ".shard_jids"
    if not jid_file.exists():
        die(f"missing {jid_file} — was this a fanned-out cluster run?")
    jids = jid_file.read_text().split()
    if not jids:
        die(f"{jid_file} is empty")
    array_jid = jids[0]
    gather_jid = jids[1] if len(jids) > 1 else None
    warn(f"run dir: {dcalign_dir}")
    warn(f"shard array job: {array_jid}" + (f", gather job: {gather_jid}" if gather_jid else ""))

    n_shards, models = load_shard_layout(dcalign_dir)

    tasks = {}
    used = args.source
    if args.source in ("auto", "sacct"):
        tasks = from_sacct(array_jid)
        used = "sacct"
    if not tasks and args.source in ("auto", "logs"):
        if args.source == "auto":
            warn("sacct returned no records (accounting purged?) — falling back to log timestamps")
        tasks = from_logs(dcalign_dir)
        used = "logs"

    if not tasks:
        die("no timing data found from either sacct or shard logs")

    gather_summary = None
    if gather_jid and used == "sacct":
        gather_summary = gather_job_summary(gather_jid)

    report(tasks, used, n_shards, models, gather_summary, args.bins)


if __name__ == "__main__":
    main()
