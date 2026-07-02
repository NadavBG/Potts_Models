#!/usr/bin/env python3
"""potts_align_run_stats.py — summarise the timing of a potts_align shard fan-out.

A combine run's couplings-aware alignment is fanned out as a Slurm array (one
task per shard; see pipeline/external/run_potts_align_align.sh). The layout is
*flat*: array task ``t`` IS shard ``t`` — each task loads both models and scores
the round-robin subset of in-scope ``(query_id, model)`` pairs the plan assigned
it, so a shard spans both models and its cost is set by how many cheap ``home``
(enumerate) vs. expensive ``cross`` (parallel-tempering) pairs it drew. This
script reads the job ids the launcher recorded in ``potts_align/.shard_jids`` and
prints basic timing stats for the array: total CPU hours, the longest-running
shard (annotated with its pair workload), wall-clock makespan vs. the CPU time it
bought (effective concurrency), and an ASCII histogram of per-shard run times.

Source of truth is Slurm accounting (``sacct``), which gives true allocated
core-seconds, consumed CPU time, and peak RSS per task. ``sacct`` accounting is
purged after a retention window, so when it returns nothing the script falls back
to the per-shard log timestamps for wall-clock elapsed and says so loudly. Unlike
the DCAlign predecessor, the potts_align finalizer tars the logs
(``potts_align/potts_align_logs.tar.zst``) and removes the raw ``logs/`` dir once
the run completes — so by the time accounting is purged the fallback is usually
reading the archive, not a live directory. Both are handled transparently (a live
``logs/`` wins if present). ``--source`` forces one or the other.

Usage:
    python pipeline/potts_align_run_stats.py [RUN_ROOT] [--source auto|sacct|logs]
                                             [--bins N]

RUN_ROOT is the iteration dir that contains ``potts_align/`` (or that dir itself).
Default: combine/combine-CM-PPIC-potts/latest under the repo root.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "combine" / "combine-CM-PPIC-potts" / "latest"

# sacct main task lines look like "<jid>_<task>"; step lines append ".batch"/".extern".
_TASK_RE = re.compile(r"^\d+_(\d+)$")
_BATCH_RE = re.compile(r"^\d+_(\d+)\.batch$")
# Slurm names the array logs potts_align_shard_%A_%a.log (%A = array base jid).
_LOG_NAME_RE = re.compile(r"potts_align_shard_\d+_(\d+)\.log$")
_LOG_TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")
_RSS_RE = re.compile(r"^([\d.]+)([KMGT]?)$")
# sacct AllocTRES carries the billable core-equivalents, e.g.
# "billing=1,cpu=1,mem=2G,node=1". Midway charges 1 SU per billing-core-hour;
# billing = max(cpus, mem / mem_per_core), so a fat-memory job bills > its cpus.
_BILLING_RE = re.compile(r"billing=([\d.]+)")


def warn(msg: str) -> None:
    print(f"[potts_align_run_stats] {msg}", file=sys.stderr)


def die(msg: str) -> "None":
    warn(msg)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# parsing helpers (generic sacct formats — shared with the DCAlign predecessor)
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
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)  # pad MM:SS -> 0:MM:SS
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def parse_dt(s: str):
    s = s.strip()
    if not s or s in ("Unknown", "None"):
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


def parse_billing(alloc_tres: str) -> float:
    """sacct AllocTRES -> billable core-equivalents (SU/hr), or -1.0 if absent."""
    m = _BILLING_RE.search(alloc_tres)
    return float(m.group(1)) if m else -1.0


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
# shard -> workload labelling (flat layout: task == shard)
# --------------------------------------------------------------------------- #
class ShardLayout:
    """Per-shard workload read from ``shards_manifest.json`` (or empty if absent).

    ``shards[s]`` is the list of indices into ``pairs`` assigned to shard ``s``;
    each pair carries a ``status`` (``home`` = enumerate, ``cross`` = parallel
    tempering). We precompute per-shard (n_pairs, n_home, n_cross) because the
    ``cross`` count is what explains a slow shard.
    """

    __slots__ = ("n_shards", "models", "workload", "manifest")

    def __init__(self, n_shards, models, workload, manifest):
        self.n_shards = n_shards
        self.models = models
        self.workload = workload  # {shard: (n_pairs, n_home, n_cross)}
        self.manifest = manifest  # scope/skip counts for the header, or None

    @classmethod
    def load(cls, pa_dir: Path) -> "ShardLayout":
        mf = pa_dir / "shards_manifest.json"
        if not mf.exists():
            return cls(None, None, {}, None)
        d = json.loads(mf.read_text())
        pairs = d.get("pairs", [])
        workload: dict[int, tuple[int, int, int]] = {}
        for shard, idxs in enumerate(d.get("shards", [])):
            n_home = sum(1 for i in idxs if pairs[i].get("status") == "home")
            n_cross = sum(1 for i in idxs if pairs[i].get("status") == "cross")
            workload[shard] = (len(idxs), n_home, n_cross)
        return cls(d.get("n_shards"), d.get("models"), workload, d)


def label_task(task: int, layout: ShardLayout) -> str:
    w = layout.workload.get(task)
    if w is None:
        return f"shard {task}"
    n_pairs, n_home, n_cross = w
    return f"shard {task} ({n_pairs} pairs: {n_home} home, {n_cross} cross)"


# --------------------------------------------------------------------------- #
# data sources
# --------------------------------------------------------------------------- #
class Task:
    __slots__ = ("task", "state", "elapsed", "cpu", "alloc_sec", "alloc_cpus",
                 "rss", "billing", "start", "end")

    def __init__(self, task):
        self.task = task
        self.state = "?"
        self.elapsed = 0.0      # wall-clock seconds
        self.cpu = 0.0          # consumed CPU seconds (TotalCPU)
        self.alloc_sec = 0.0    # allocated core-seconds (CPUTimeRAW)
        self.alloc_cpus = 1
        self.rss = 0.0          # peak resident bytes
        self.billing = -1.0     # billable core-equivalents (SU/hr); -1 = unknown
        self.start = None
        self.end = None


def from_sacct(array_jid: str):
    """Parse `sacct` for the shard array. Returns {} if accounting is gone."""
    fmt = "JobID,State,Elapsed,TotalCPU,CPUTimeRAW,AllocCPUS,MaxRSS,Start,End,AllocTRES"
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
        if len(f) < 10:
            continue
        jobid = f[0]
        m = _TASK_RE.match(jobid)
        if m:  # main task record: timing + state + billing (batch step has no billing=)
            t = tasks.setdefault(int(m.group(1)), Task(int(m.group(1))))
            t.state = f[1]
            t.elapsed = parse_hms(f[2])
            t.cpu = parse_hms(f[3])
            t.alloc_sec = float(f[4]) if f[4].strip() else 0.0
            t.alloc_cpus = int(f[5]) if f[5].strip() else 1
            t.start = parse_dt(f[7])
            t.end = parse_dt(f[8])
            t.billing = parse_billing(f[9])
            continue
        b = _BATCH_RE.match(jobid)
        if b:  # batch step carries MaxRSS
            t = tasks.setdefault(int(b.group(1)), Task(int(b.group(1))))
            t.rss = max(t.rss, parse_rss(f[6]))
    return tasks


def _iter_shard_logs(pa_dir: Path):
    """Yield ``(task, text)`` for each shard log, from the live ``logs/`` dir if
    present, else the finalized ``potts_align_logs.tar.zst`` archive.

    Python 3.12's ``tarfile`` cannot read zstd, and the venv has no ``zstandard``
    module, so the archive is streamed through the ``zstd`` CLI (the same tool the
    finalizer used to create it). Returns nothing (with a warning) if neither
    source is available.
    """
    log_dir = pa_dir / "logs"
    if log_dir.is_dir():
        found = False
        for log in sorted(log_dir.glob("potts_align_shard_*.log")):
            m = _LOG_NAME_RE.search(log.name)
            if m:
                found = True
                yield int(m.group(1)), log.read_text()
        if found:
            return
        warn(f"{log_dir} has no potts_align_shard_*.log files")

    tar_path = pa_dir / "potts_align_logs.tar.zst"
    if not tar_path.exists():
        warn(f"no live logs/ and no {tar_path.name}")
        return
    if not _which("zstd") and not _which("unzstd"):
        warn(f"cannot read {tar_path.name}: zstd/unzstd not on PATH")
        return
    yield from _iter_tar_logs(tar_path)


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)


def _iter_tar_logs(tar_path: Path):
    """Decompress ``<tar_path>`` via the ``zstd`` CLI and yield ``(task, text)``.

    Python 3.12 tarfile can't read zstd and the venv lacks ``zstandard``, so we
    shell out. The archive is a few MB (513 tiny text logs), so we decompress it
    whole into memory and open a *seekable* tar — streaming ``r|`` mode does not
    support ``extractfile`` cleanly (its ``_Stream`` has no ``seekable``).
    """
    decomp = _which("zstd")
    cmd = [decomp, "-dc", str(tar_path)] if decomp else [_which("unzstd"), "-c", str(tar_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        warn(f"zstd exited {proc.returncode} decompressing {tar_path.name}: "
             f"{proc.stderr.decode('utf-8', 'replace').strip()}")
        return
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            m = _LOG_NAME_RE.search(Path(member.name).name)
            if not m:
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            yield int(m.group(1)), fobj.read().decode("utf-8", "replace")


def from_logs(pa_dir: Path):
    """Fallback: per-shard wall-clock from log timestamps (cpus assumed 1).

    The shard wrapper's Python logging lines are ``YYYY-MM-DD HH:MM:SS,mmm ...``;
    the bash ``[potts_align_shard] ...`` banner lines have no timestamp and are
    skipped by the regex. Elapsed = last stamp - first stamp in the log.
    """
    tasks: dict[int, Task] = {}
    for task, text in _iter_shard_logs(pa_dir):
        stamps = []
        for line in text.splitlines():
            ts = _LOG_TS_RE.match(line)
            if ts:
                stamps.append(datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S"))
        if not stamps:
            continue
        t = Task(task)
        t.state = "COMPLETED"
        t.start, t.end = min(stamps), max(stamps)
        t.elapsed = (t.end - t.start).total_seconds()
        t.cpu = t.elapsed       # single-threaded; consumed ~= wall
        t.alloc_sec = t.elapsed  # cpus=1
        tasks[task] = t
    return tasks


def gather_job_summary(gather_jid: str) -> str | None:
    fmt = "JobID,State,Elapsed,TotalCPU,AllocTRES"
    try:
        out = subprocess.run(
            ["sacct", "-j", gather_jid, "-P", "--noheader", "--format", fmt],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        f = line.split("|")
        if len(f) >= 5 and f[0] == gather_jid:
            billing = parse_billing(f[4])
            su = f"  su={billing * parse_hms(f[2]) / 3600:.2f}" if billing >= 0 else ""
            return f"state={f[1]}  elapsed={f[2]}  cpu={f[3]}{su}"
    return None


def gather_completeness(pa_dir: Path) -> str | None:
    """One-line scored/in-scope summary from gather_status.json, if present.

    Independent of sacct (durable on disk), this reads out whether the gather
    actually merged every in-scope pair — the completeness cross-check the
    timing numbers alone can't give.
    """
    status = pa_dir / "gather_status.json"
    if not status.exists():
        return None
    try:
        d = json.loads(status.read_text())
    except (OSError, json.JSONDecodeError) as e:
        warn(f"could not read gather_status.json: {e}")
        return None
    n_scored = d.get("n_scored")
    n_scope = d.get("n_pairs_in_scope")
    partial = d.get("partial")
    per_model = ", ".join(
        f"{m.get('model')}={m.get('meta', {}).get('n_scored')}" for m in d.get("models", [])
    )
    return f"{n_scored}/{n_scope} pairs scored (partial={partial})" + (f"  [{per_model}]" if per_model else "")


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


def report(tasks: dict[int, Task], source: str, layout: ShardLayout,
           gather_summary, gather_done, bins: int) -> None:
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
    print(f"potts_align shard-array timing  (source: {source})")
    print("=" * 64)

    mf = layout.manifest
    if mf:
        print(f"models                 : {', '.join(layout.models or [])}")
        print(f"in-scope pairs         : {mf.get('n_pairs_in_scope')} of {mf.get('n_pairs_total')} "
              f"(skip N>L={mf.get('n_skip_NgtL')}, skip_subsample={mf.get('n_skip_subsample')})")
    print(f"shard tasks            : {n}"
          + (f" of {layout.n_shards} planned" if layout.n_shards and layout.n_shards != n else ""))
    print("states                 : "
          + ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
    not_done = n - states.get("COMPLETED", 0)
    if not_done:
        warn(f"{not_done} task(s) did not COMPLETE — totals below are partial")
    if layout.n_shards and n < layout.n_shards:
        warn(f"{layout.n_shards - n} shard(s) have no timing record (missing from sacct/logs)")

    print()
    print(f"total allocated CPU    : {alloc_h:9.2f} core-hours")
    if abs(cpu_h - alloc_h) > 1e-6:
        print(f"total consumed CPU     : {cpu_h:9.2f} core-hours  "
              f"({cpu_h / alloc_h * 100:.0f}% of allocation)")

    # SU = billing-core-hours (Midway charges 1 SU per billing-core-hour). When
    # every task's billing == its allocated cpus (memory within the per-core
    # share), SU == core-hours; a fat-memory job bills above core-hours.
    billed = [t for t in ts if t.billing >= 0]
    if n and len(billed) == n:
        su = sum(t.billing * t.elapsed / 3600 for t in ts)
        note = ("= core-hours; memory within the per-core share"
                if su <= alloc_h * 1.001
                else f"memory inflates the charge to {su / alloc_h:.2f}x core-hours")
        print(f"SU billed (billing×hr) : {su:9.2f} SU  ({note})")
    else:
        print(f"SU billed (estimate)   : {alloc_h:9.2f} SU  "
              f"(sacct billing unavailable — assumes billing = allocated cpus; "
              f"a fat-memory job would bill more)")

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
    print(f"  longest shard             : {label_task(longest.task, layout)} "
          f"— {human_dur(longest.elapsed)}")
    if rss_peak > 0:
        print(f"  peak RSS (any shard)      : {rss_peak / 1024**3:.2f} GiB")

    if layout.workload:
        pair_counts = [layout.workload[t.task][0] for t in ts if t.task in layout.workload]
        cross_counts = [layout.workload[t.task][2] for t in ts if t.task in layout.workload]
        if pair_counts:
            print(f"  pairs/shard (min/med/max) : "
                  f"{min(pair_counts)} / {int(statistics.median(pair_counts))} / {max(pair_counts)}"
                  f"   (cross/shard {min(cross_counts)}/{int(statistics.median(cross_counts))}/{max(cross_counts)})")

    print()
    print(f"histogram of per-shard wall time ({bins} bins):")
    for row in ascii_hist([e / 60 for e in elapsed], bins):
        print(row)

    if gather_summary or gather_done:
        print()
        if gather_summary:
            print(f"gather job             : {gather_summary}")
        if gather_done:
            print(f"gather completeness    : {gather_done}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_root", nargs="?", default=str(DEFAULT_RUN_ROOT),
                    help="iteration dir containing potts_align/ (default: %(default)s)")
    ap.add_argument("--source", choices=["auto", "sacct", "logs"], default="auto",
                    help="timing source (default: auto = sacct, fall back to logs)")
    ap.add_argument("--bins", type=int, default=12, help="histogram bins (default: 12)")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if (run_root / "potts_align").is_dir():
        pa_dir = run_root / "potts_align"
    elif run_root.name == "potts_align" and run_root.is_dir():
        pa_dir = run_root
    else:
        die(f"no potts_align/ under {run_root} (and it is not a potts_align dir)")

    jid_file = pa_dir / ".shard_jids"
    if not jid_file.exists():
        die(f"missing {jid_file} — was this a fanned-out cluster run?")
    jids = jid_file.read_text().split()
    if not jids:
        die(f"{jid_file} is empty")
    array_jid = jids[0]
    gather_jid = jids[1] if len(jids) > 1 else None
    warn(f"run dir: {pa_dir}")
    warn(f"shard array job: {array_jid}" + (f", gather job: {gather_jid}" if gather_jid else ""))

    layout = ShardLayout.load(pa_dir)

    tasks: dict[int, Task] = {}
    used = args.source
    if args.source in ("auto", "sacct"):
        tasks = from_sacct(array_jid)
        used = "sacct"
    if not tasks and args.source in ("auto", "logs"):
        if args.source == "auto":
            warn("sacct returned no records (accounting purged?) — falling back to log timestamps")
        tasks = from_logs(pa_dir)
        used = "logs"

    if not tasks:
        die("no timing data found from either sacct or shard logs")

    gather_summary = None
    if gather_jid and used == "sacct":
        gather_summary = gather_job_summary(gather_jid)
    gather_done = gather_completeness(pa_dir)

    report(tasks, used, layout, gather_summary, gather_done, args.bins)


if __name__ == "__main__":
    main()
