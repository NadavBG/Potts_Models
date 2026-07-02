"""Snakemake wrapper: derive the E_tot combining weights from the naturals.

Reads the tidy ``scores.tsv`` and ``models.json`` (for the two model names), then
writes ``data/energy_weights.json`` + ``data/energy_weight_sweep.tsv`` and, when
figures are enabled, ``figs/energy_weights.pdf``. The weights equalize each
family's median native energy so E_tot is not biased toward one family
(SBM.utils.energy_weights)."""

import json
from pathlib import Path

import pandas as pd

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM.utils import energy_weights

setup_stage_logging(snakemake, "compute_weights")  # noqa: F821
load_combine_cfg_from_snakemake(snakemake)  # noqa: F821 (validate; identity comes from models.json)

model_info = json.loads(Path(snakemake.input.models).read_text(encoding="utf-8"))["models"]  # noqa: F821
name_A, name_B = model_info[0]["name"], model_info[1]["name"]

result = energy_weights.compute_and_write(
    scores_tsv=Path(snakemake.input.scores),  # noqa: F821
    name_A=name_A,
    name_B=name_B,
    out_json=Path(snakemake.output.weights),  # noqa: F821
    out_sweep_tsv=Path(snakemake.output.sweep),  # noqa: F821
)

out = snakemake.output  # noqa: F821
if "fig" in out.keys():  # present only when figures.enabled (Snakefile.combine)
    from SBM.utils import utils_energy_plot

    sweep = pd.read_csv(out.sweep, sep="\t")
    utils_energy_plot.render_energy_weights(
        sweep,
        w_A=result["w_A"],
        m_A=result["native_median_energy"][name_A],
        m_B=result["native_median_energy"][name_B],
        model_names=(name_A, name_B),
        out_pdf=Path(out.fig),
    )
