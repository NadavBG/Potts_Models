#!/usr/bin/env bash
# Render the combine-pipeline workflow figures into docs/workflow/:
#   * combine_rulegraph.{pdf,svg}  — simplified DAG (rules + dependencies)
#   * combine_dag.{pdf,svg}        — full DAG (every job/file for one config)
#   * combine_workflow.pdf         — the conceptual Mac<->Midway lane diagram
#
# The two DAGs come straight from Snakemake (`--rulegraph` / `--dag`) so they are
# always faithful to Snakefile.combine; graphviz `dot` renders them. The lane
# diagram is a hand-drawn overview (scripts/render_workflow_diagram.py, lab_plotting).
#
# Nothing in the pipeline is executed. We only build the DAG, so we create empty
# placeholder files for the two EXTERNAL inputs no Mac rule produces — the
# potts_align cluster cache and the characterize summary — so every gated rule
# (potts_align_baseline, design_handoff, characterize_render) appears in the graph.
#
# Usage:
#   bash scripts/render_dag.sh [<config>] [<run_root>]
#     <config>    combine config to base the DAGs on (default the full production
#                 config/params_combine-CM-PPIC-potts.yaml — it exercises every
#                 gated rule: potts_align + design + characterize)
#     <run_root>  throwaway run_root for path resolution (default under the temp dir)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-config/params_combine-CM-PPIC-potts.yaml}"
_TMP="${TMPDIR:-/tmp}"; _TMP="${_TMP%/}"
RUN_ROOT="${2:-${_TMP}/combine-dag-render}"
OUT_DIR="docs/workflow"
PY="${REPO_ROOT}/.venv/bin/python"
SNAKE=("${PY}" -m snakemake -s Snakefile.combine --configfile "${CONFIG}" \
       --config "run_root=${RUN_ROOT}" --cores 1)

mkdir -p "${OUT_DIR}"

# Seed the external (no-producer) inputs so the full DAG resolves without running.
"${PY}" - "${CONFIG}" "${RUN_ROOT}" <<'PY'
import sys
from pathlib import Path
from SBM import combine_config as cc

cfg = cc.load_config(sys.argv[1])
rr = Path(sys.argv[2])
if cfg.scoring.method == "potts_align":
    for m in cfg.models:
        p = rr / "potts_align" / "cache" / m.name / "alignments.tsv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
if cfg.characterize.enabled:
    p = rr / "characterize" / "data" / "summary.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
PY

if ! command -v dot >/dev/null 2>&1; then
    echo "ERROR: graphviz 'dot' not found. Install it with:  brew install graphviz" >&2
    echo "       (writing the DOT sources to ${OUT_DIR}/*.dot only)" >&2
    "${SNAKE[@]}" --rulegraph all > "${OUT_DIR}/combine_rulegraph.dot"
    "${SNAKE[@]}" --dag all       > "${OUT_DIR}/combine_dag.dot"
    exit 1
fi

echo "Rendering simplified DAG (rulegraph) from ${CONFIG} ..."
"${SNAKE[@]}" --rulegraph all 2>/dev/null | tee "${OUT_DIR}/combine_rulegraph.dot" \
    | dot -Tpdf -o "${OUT_DIR}/combine_rulegraph.pdf"
dot -Tsvg "${OUT_DIR}/combine_rulegraph.dot" -o "${OUT_DIR}/combine_rulegraph.svg"

echo "Rendering full DAG from ${CONFIG} ..."
"${SNAKE[@]}" --dag all 2>/dev/null | tee "${OUT_DIR}/combine_dag.dot" \
    | dot -Tpdf -o "${OUT_DIR}/combine_dag.pdf"
dot -Tsvg "${OUT_DIR}/combine_dag.dot" -o "${OUT_DIR}/combine_dag.svg"

echo "Rendering the conceptual Mac<->Midway workflow diagram ..."
"${PY}" scripts/render_workflow_diagram.py --out-dir "${OUT_DIR}"

echo "Done. Figures in ${OUT_DIR}/:"
ls -1 "${OUT_DIR}"
