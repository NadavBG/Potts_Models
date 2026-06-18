# Potts_Models single-model pipeline: one MSA -> one trained Potts model
# (+ synthetic samples, figures, optional MPNN sweep).
#
# This is a SEPARATE pipeline from the two-model `Snakefile.combine` (two
# already-trained models -> combined energy): different DAG, different validated
# config schema (workflow_config.py vs combine_config.py), and a different output
# tree (results/ vs combine/). They are intentionally not merged. The end-to-end
# story spanning both — and the Mac-primary / Midway-for-DCAlign split — is the
# runbook docs/PIPELINE.md.
#
# One validated YAML config = one run. Every output lands under a single
# run directory so it is obvious which parameters produced which figure.
#
#   snakemake --configfile config/params_<name>.yaml --cores 8 all
#
# By default the run dir is results/<run_name>/; pass an explicit
# `--config run_root=...` to place it elsewhere (the iteration helper
# scripts/iter.py mints results/<run_name>/iter-NNN-<tag>/ for you).
#
# Runs inside the project's uv venv (NOT conda — conda-forge Python
# segfaults on the compiled MCMC kernel), so there is no `conda:` directive.
# `threads`/`resources` are declared on every rule so a Slurm/Midway
# profile can be added later without touching the rules.

import sys

from SBM import workflow_config as wc

# ── Config: validate, reject unknown keys ────────────────────────────────
if "run_name" not in config or "msa_fasta" not in config:
    sys.exit(
        "ERROR: missing config. Invoke with a config file, e.g.\n"
        "  snakemake --configfile config/params_CM-bm-dense.yaml --cores 8 all"
    )

cfg = wc.from_dict({k: v for k, v in config.items() if k not in {"run_root"}})

RUN_NAME = cfg.run_name
RUN_ROOT = config.get("run_root") or f"results/{RUN_NAME}"

# ── Path scheme (all rooted under RUN_ROOT) ──────────────────────────────
# The MSA enters as an aligned FASTA (raw, immutable input) and is encoded
# into a run-local integer .npy by the `encode_msa` rule; every downstream
# rule consumes that derived array (MSA), never the FASTA directly.
MSA_FASTA      = cfg.msa_fasta  # input; relative to the dir Snakemake is run from
MSA            = f"{RUN_ROOT}/inputs/msa.npy"            # derived: encode_msa output
ENCODE_MANIFEST = f"{RUN_ROOT}/inputs/msa_manifest.json"
CONFIG_SNAP    = f"{RUN_ROOT}/config_snapshot.yaml"
MSA_STATS_PDF  = f"{RUN_ROOT}/msa_stats.pdf"        # top level: render deletes figs/
MASK_J         = f"{RUN_ROOT}/masks/J_mask.npy"
MASK_H         = f"{RUN_ROOT}/masks/h_mask.npy"
MODEL          = f"{RUN_ROOT}/model.npy"
TRAIN_MANIFEST = f"{RUN_ROOT}/manifest.json"
COMMAND_SH     = f"{RUN_ROOT}/command.sh"
TRAIN_META     = f"{RUN_ROOT}/train_meta.json"
SYNTH_NPY      = f"{RUN_ROOT}/synthetic/align_T{{temp}}.npy"
SYNTH_JSON     = f"{RUN_ROOT}/synthetic/align_T{{temp}}.json"
FIGS_DIR       = f"{RUN_ROOT}/figs"
MPNN_DIR       = f"{RUN_ROOT}/synthetic/mpnn_sweep_seed{cfg.mpnn_seed}"
RUN_MANIFEST   = f"{RUN_ROOT}/run_manifest.json"

# Sampler formats temperatures with "%.10g" (0.75 -> "0.75", 1.0 -> "1").
TEMPS = [f"{t:.10g}" for t in cfg.sample.temperatures]

_PRUNE_J = cfg.pruning.enabled and cfg.pruning.couplings is not None
_PRUNE_H = cfg.pruning.enabled and cfg.pruning.fields is not None

wildcard_constraints:
    temp=r"[0-9]+(\.[0-9]+)?",


# ── Aggregate targets ────────────────────────────────────────────────────
rule all:
    default_target: True
    input:
        CONFIG_SNAP,
        MODEL,
        TRAIN_MANIFEST,
        COMMAND_SH,
        *([MSA_STATS_PDF] if cfg.msa_stats.enabled else []),
        expand(SYNTH_NPY, temp=TEMPS),
        FIGS_DIR,
        *([MPNN_DIR] if cfg.mpnn.enabled else []),
        RUN_MANIFEST,


# Render just the MSA-only figure with no training (proves decoupling).
rule msa_stats_only:
    input:
        MSA_STATS_PDF,


# ── Provenance: freeze the validated config into the run dir ─────────────
rule snapshot_config:
    output:
        CONFIG_SNAP,
    log:
        f"{RUN_ROOT}/logs/snapshot_config.log"
    threads: 1
    resources:
        mem_mb=256,
        runtime=2,
    script:
        "scripts/wf/run_snapshot_config.py"


# ── Encode the aligned FASTA into a run-local integer MSA (.npy) ─────────
# Raw FASTA in, derived array out (+ a manifest recording the input hash
# and any sequences dropped for non-canonical residues). Every rule below
# depends on this output, not on the FASTA.
rule encode_msa:
    input:
        fasta=MSA_FASTA,
    output:
        npy=MSA,
        manifest=ENCODE_MANIFEST,
    log:
        f"{RUN_ROOT}/logs/encode_msa.log"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=10,
    script:
        "scripts/wf/run_encode_msa.py"


# ── MSA-only statistics figure (independent of any model) ────────────────
rule msa_stats:
    input:
        msa=MSA,
    output:
        MSA_STATS_PDF,
    log:
        f"{RUN_ROOT}/logs/msa_stats.log"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=15,
    script:
        "scripts/wf/run_msa_stats.py"


# ── Optional pruning masks (only defined when pruning is enabled) ────────
if _PRUNE_J:

    rule build_mask_J:
        input:
            msa=MSA,
        output:
            MASK_J,
        log:
            f"{RUN_ROOT}/logs/build_mask_J.log"
        threads: 1
        resources:
            mem_mb=4000,
            runtime=20,
        script:
            "scripts/wf/run_build_mask_J.py"


if _PRUNE_H:

    rule build_mask_h:
        input:
            msa=MSA,
        output:
            MASK_H,
        log:
            f"{RUN_ROOT}/logs/build_mask_h.log"
        threads: 1
        resources:
            mem_mb=4000,
            runtime=20,
        script:
            "scripts/wf/run_build_mask_h.py"


# ── Training ─────────────────────────────────────────────────────────────
def _train_inputs(wildcards):
    inputs = {"msa": MSA}
    if _PRUNE_J:
        inputs["prune_J"] = MASK_J
    if _PRUNE_H:
        inputs["prune_h"] = MASK_H
    return inputs


rule train:
    input:
        unpack(_train_inputs),
    output:
        model=MODEL,
        manifest=TRAIN_MANIFEST,
        command=COMMAND_SH,
        meta=TRAIN_META,
    params:
        run_root=RUN_ROOT,
        prune_J=(MASK_J if _PRUNE_J else None),
        prune_h=(MASK_H if _PRUNE_H else None),
    log:
        f"{RUN_ROOT}/logs/train.log"
    threads: 8
    resources:
        mem_mb=8000,
        runtime=240,
    script:
        "scripts/wf/run_train.py"


# ── Synthetic sampling: one job per temperature (deterministic paths) ────
rule sample:
    input:
        model=MODEL,
        manifest=TRAIN_MANIFEST,
    output:
        align=SYNTH_NPY,
        sidecar=SYNTH_JSON,
    params:
        run_root=RUN_ROOT,
    log:
        f"{RUN_ROOT}/logs/sample_T{{temp}}.log"
    threads: 4
    resources:
        mem_mb=4000,
        runtime=60,
    script:
        "scripts/wf/run_sample.py"


# ── ProteinMPNN foldability sweep (only when enabled) ────────────────────
if cfg.mpnn.enabled:

    rule mpnn_sweep:
        input:
            model=MODEL,
            manifest=TRAIN_MANIFEST,
        output:
            directory(MPNN_DIR),
        params:
            run_root=RUN_ROOT,
        log:
            f"{RUN_ROOT}/logs/mpnn_sweep.log"
        threads: 4
        resources:
            mem_mb=8000,
            runtime=240,
        script:
            "scripts/wf/run_mpnn_sweep.py"


# ── Figures ──────────────────────────────────────────────────────────────
def _render_inputs(wildcards):
    inputs = {"model": MODEL, "synthetic": expand(SYNTH_NPY, temp=TEMPS)}
    if cfg.mpnn.enabled:
        inputs["mpnn"] = MPNN_DIR
    return inputs


rule render:
    input:
        unpack(_render_inputs),
    output:
        directory(FIGS_DIR),
    params:
        run_root=RUN_ROOT,
    log:
        f"{RUN_ROOT}/logs/render.log"
    threads: 2
    resources:
        mem_mb=8000,
        runtime=120,
    script:
        "scripts/wf/run_render.py"


# ── Aggregate run manifest (last) ────────────────────────────────────────
rule run_manifest:
    input:
        CONFIG_SNAP,
        ENCODE_MANIFEST,
        TRAIN_MANIFEST,
        FIGS_DIR,
        *([MSA_STATS_PDF] if cfg.msa_stats.enabled else []),
    output:
        RUN_MANIFEST,
    params:
        run_root=RUN_ROOT,
    log:
        f"{RUN_ROOT}/logs/run_manifest.log"
    threads: 1
    resources:
        mem_mb=512,
        runtime=10,
    script:
        "scripts/wf/run_manifest.py"
