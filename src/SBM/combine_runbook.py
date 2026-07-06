"""Render the per-run, copy-pasteable ``RUNBOOK.txt`` for a two-model combine run.

The single-source-of-truth for "what commands do I run to take this combine run
from two ``results/`` model dirs all the way through score → design →
characterize". It is **config-aware**: only the stages the config actually enables
are emitted, so there are no irrelevant branches to skip and nothing to comment
out (the "removing comment #s" pain).

Two consumers share this renderer so the instructions never drift:

* ``scripts/new_combine.py`` writes ``<run_root>/RUNBOOK.txt`` at scaffold time.
* the ``runbook`` rule of ``Snakefile.combine`` regenerates it on every
  ``snakemake … all`` (so it always matches the params in effect), and
  ``scripts/wf/run_design_handoff.py`` reuses :func:`design_handoff_text`.

Design notes baked into the output:

* Each ``[MAC]`` / ``[MIDWAY]`` block is contiguous, so it pastes as a unit.
* ``$RR`` / ``$CFG`` are set once per shell and a ``snake()`` shell function wraps
  the Snakemake invocation — so the long ``combine/<run>/iter-NNN-<tag>`` path is
  never re-typed (the historical source of typos that silently minted a fresh run
  dir). A function is used rather than a ``SNAKE="…"`` string var because the
  latter does not word-split under zsh (the macOS default shell); the function is
  bash- and zsh-safe.
* The Midway drivers are one-argument (``run_potts_align_align.sh $RR`` etc.); we
  orchestrate them, we do not reinvent them.
* ``sync_models.sh`` always runs **from the Mac** (it SSHes to Midway): ``push``
  before a Midway stage, ``pull`` after. The Midway blocks never call it — this
  fixes a long-standing bug in the old ``docs/RUNBOOK.md`` characterize step.
"""

from __future__ import annotations

from SBM.combine_config import CombineRunConfig

_RULE = "=" * 78
_SUB = "-" * 78

#: The Mac-side Snakemake invocation. A ``snake()`` shell function (defined in the
#: header) rather than a ``SNAKE="…"`` string var, so it word-splits correctly in
#: both bash and zsh (a plain string var does not split under zsh).
_SNAKE = "snake"
_SNAKEFILE = "Snakefile.combine"


def _header(cfg: CombineRunConfig, run_root: str, config_path: str) -> list[str]:
    a, b = cfg.models
    d = cfg.design
    design_state = "off"
    if d.enabled:
        design_state = f"on ({d.execution}, {d.n_chains} chains)"
    char_state = "on" if cfg.characterize.enabled else "off"
    return [
        _RULE,
        f" RUNBOOK — {cfg.run_name}",
        _RULE,
        "",
        f"run_root : {run_root}",
        f"config   : {config_path}",
        "",
        "models:",
        f"    A = {a.name:<24} (run_dir: {a.run_dir})",
        f"    B = {b.name:<24} (run_dir: {b.run_dir})",
        "",
        f"scoring.method = {cfg.scoring.method}     "
        f"design = {design_state}     characterize = {char_state}",
        "",
        "Written at scaffold time and refreshed from the config by the pipeline",
        "(`snakemake … all`); if you hand-edit the config, re-run `all` to update it.",
        "Copy each [MAC] / [MIDWAY] block as a unit; the [MIDWAY] blocks are a fresh",
        "SSH shell, so they re-set RR.",
        "Reference: docs/RUNBOOK.md, docs/POTTS_ALIGN.md, docs/DESIGN_TWO_MODEL.md,",
        "docs/CHARACTERIZE.md.",
        "",
        "Set once at the top of your Mac shell:",
        "",
        f"    RR={run_root}",
        f"    CFG={config_path}",
        f'    snake() {{ snakemake -s {_SNAKEFILE} --configfile "$CFG" '
        '--config run_root="$RR" --cores 8 "$@"; }',
        "",
    ]


def _midway_reset(run_root: str) -> list[str]:
    """The single line every [MIDWAY] block opens with (fresh SSH shell)."""
    return [f"    RR={run_root}"]


def _monitor() -> str:
    return "    #   monitor: pipeline/job_tally.sh -w 10   (wait for the gather END mail)"


def _stage_combine(cfg: CombineRunConfig, run_root: str) -> list[str]:
    out = [
        _SUB,
        " STAGE 1 — COMBINE  (score naturals under both models + E_tot weights)",
        _SUB,
        "",
    ]
    if cfg.scoring.method == "potts_align":
        out += [
            "potts_align pre-builds its alignment cache on a Midway Slurm array",
            "(pure numpy), then you score locally. Three blocks:",
            "",
            "[MAC]  build pre-align inputs, commit, push models + inputs to Midway",
            f"    {_SNAKE} $RR/config_snapshot.yaml $RR/models.json $RR/query/query.fasta",
            f'    git add -A && git commit -m "combine {cfg.run_name}: inputs" && git push',
            "    bash scripts/sync_models.sh push",
            "",
            "[MIDWAY]  build the potts_align cache (login node — NO snakemake)",
            *_midway_reset(run_root),
            "    bash pipeline/external/run_potts_align_align.sh $RR",
            _monitor(),
            "    bash pipeline/external/finalize_potts_align.sh $RR",
            "",
            "[MAC]  pull the cache, then score + weights + figures",
            "    bash scripts/sync_models.sh pull",
            f"    {_SNAKE} all",
        ]
    else:
        out += [
            f"method={cfg.scoring.method} scores locally — no cluster round-trip.",
            "",
            "[MAC]  score + weights + figures",
            f"    {_SNAKE} all",
        ]
    out.append("")
    return out


def _stage_design(cfg: CombineRunConfig, run_root: str, *, force_cluster: bool = False) -> list[str]:
    """Stage-2 block. ``force_cluster`` renders the cluster steps regardless of
    ``execution`` — used by :func:`design_handoff_text`, which is only ever emitted
    when the Snakefile actually routed to cluster."""
    d = cfg.design
    out = [
        _SUB,
        f" STAGE 2 — DESIGN  (joint-anneal {d.n_chains} chains over E_tot)",
        _SUB,
        "",
    ]
    if not force_cluster and d.execution == "local":
        out += [
            "design.execution: local — the anneal + the four figs/design_*.pdf are",
            f"produced by Stage 1's `{_SNAKE} all`. Nothing extra to run here.",
            "",
        ]
        return out
    if not force_cluster and d.execution == "auto":
        # The gate (Snakefile _design_runs_local) decides local-vs-cluster at DAG
        # build; the renderer can't see that verdict, so surface BOTH paths.
        out += [
            "design.execution: auto — Snakemake picks LOCAL or CLUSTER at DAG build",
            "and prints the verdict (predicted wall-time vs local_budget_minutes):",
            f"  * LOCAL   -> the anneal + figs come from Stage 1's `{_SNAKE} all`; skip this stage.",
            f"  * CLUSTER -> `{_SNAKE} all` wrote design/design_config.json and stopped; run:",
            "",
        ]
    else:
        out += [
            f"design.execution: cluster — Stage 1's `{_SNAKE} all` wrote",
            "design/design_config.json; run the anneal array on Midway.",
            "",
        ]
    # Cluster steps: emitted for both cluster and auto (local returned early above).
    out += [
        "[MAC]  push the design spec up",
        f'    git add -A && git commit -m "design {cfg.run_name}" && git push',
        "    bash scripts/sync_models.sh push",
        "",
        "[MIDWAY]  run the joint-anneal array (login node)",
        *_midway_reset(run_root),
        "    bash pipeline/external/run_design.sh $RR",
        _monitor(),
        "    bash pipeline/external/finalize_design.sh $RR",
        "",
        "[MAC]  pull the trajectories, then render the design figures",
        "    bash scripts/sync_models.sh pull",
        f"    {_SNAKE} $RR/figs/design_alignment.pdf",
        "",
    ]
    return out


def _stage_characterize(cfg: CombineRunConfig, run_root: str) -> list[str]:
    return [
        _SUB,
        " STAGE 3 — CHARACTERIZE  (ESMFold + which-fold TM-align + BLAST)",
        _SUB,
        "",
        "Compute is Midway-only (GPU + binaries); figures render on the Mac.",
        "",
        "[MIDWAY]  fold + TM-align + BLAST (login node submits the arrays)",
        *_midway_reset(run_root),
        "    # one-time per clone (skip if already built):",
        "    bash pipeline/external/build_tmalign.sh",
        "    bash pipeline/external/prefetch_esmfold.sh",
        "    # (optional) probe ESMFold cost:  "
        "bash pipeline/external/run_esmfold_probe.sh $RR 20",
        "    bash pipeline/external/run_characterize.sh $RR",
        "    #   monitor: pipeline/job_tally.sh -w 10   (wait for the merge END mail)",
        "",
        "[MAC]  pull the tables, then render the characterization figures",
        "    bash scripts/sync_models.sh pull",
        f"    {_SNAKE} $RR/figs/characterization_overview.pdf",
        "",
    ]


def _results(cfg: CombineRunConfig) -> list[str]:
    lines = [
        _SUB,
        " WHERE RESULTS LAND (under $RR)",
        _SUB,
        "",
        "    data/scores.tsv, data/energy_weights.json          # stage 1",
        "    figs/two_model_energy.pdf, figs/energy_weights.pdf # stage 1",
    ]
    if cfg.scoring.method == "potts_align":
        lines.append("    figs/potts_align_vs_inframe.pdf                     # stage 1")
    if cfg.design.enabled:
        lines += [
            "    design/designed_sequences.fasta, design/designed.tsv  # stage 2",
            "    figs/design_{trajectories,phase_space,lengths,alignment}.pdf  # stage 2",
        ]
    if cfg.characterize.enabled:
        lines += [
            "    characterize/data/summary.tsv                      # stage 3 (Midway)",
            "    figs/characterization_overview.pdf, tm_A_vs_B.pdf, "
            "fold_call_breakdown.pdf  # stage 3",
        ]
    lines.append("")
    return lines


def render_runbook(cfg: CombineRunConfig, run_root: str, config_path: str) -> str:
    """Render the complete per-run runbook text for ``cfg`` at ``run_root``.

    ``config_path`` is the config the user invokes Snakemake with (e.g.
    ``config/params_combine-<name>.yaml``); it becomes ``$CFG`` in the runbook.
    """
    lines: list[str] = []
    lines += _header(cfg, run_root, config_path)
    lines += _stage_combine(cfg, run_root)
    if cfg.design.enabled:
        lines += _stage_design(cfg, run_root)
    if cfg.characterize.enabled:
        lines += _stage_characterize(cfg, run_root)
    lines += _results(cfg)
    return "\n".join(lines) + "\n"


def design_handoff_text(cfg: CombineRunConfig, run_root: str, config_path: str) -> str:
    """The cluster-design hand-off note (``design/MIDWAY_HANDOFF.txt``).

    Reuses the Stage-2 block from :func:`render_runbook` so the hand-off can never
    go stale relative to the master runbook. Only meaningful when
    ``design.execution == "cluster"`` (the branch that emits the note)."""
    d = cfg.design
    head = [
        "Two-model design — CLUSTER execution",
        "====================================",
        "",
        "design.execution is 'cluster' (or the auto gate chose it), so the anneal is",
        f"NOT run on the Mac. The run spec is at {run_root}/design/design_config.json",
        f"({d.n_chains} chains = {d.start_random} random / {d.start_natural_a} "
        f"{cfg.models[0].name} / {d.start_natural_b} {cfg.models[1].name}, "
        f"{d.steps} steps, polish={d.polish_schedule if d.polish else 'off'}, "
        f"{d.n_shards} shards).",
        "",
        f"The full run's copy-pasteable steps are in {run_root}/RUNBOOK.txt. The",
        "design stage specifically:",
        "",
        f"    RR={run_root}",
        f"    CFG={config_path}",
        f'    snake() {{ snakemake -s {_SNAKEFILE} --configfile "$CFG" '
        '--config run_root="$RR" --cores 8 "$@"; }',
        "",
    ]
    # This note is only written when the Snakefile actually routed to cluster, so
    # force the cluster steps regardless of execution (which may be "auto").
    body = _stage_design(cfg, run_root, force_cluster=True)
    tail = [
        "To run the anneal locally instead, set design.execution: local (or raise",
        "design.local_budget_minutes for the auto gate) and re-run.",
    ]
    return "\n".join(head + body + tail) + "\n"
