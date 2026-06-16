"""Snakemake wrapper: render the consolidated two-model energy figure."""

import json
from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM.utils import utils_energy_plot

setup_stage_logging(snakemake, "render_combine")  # noqa: F821
load_combine_cfg_from_snakemake(snakemake)  # noqa: F821 (validate; identity comes from models.json)

model_info = json.loads(Path(snakemake.input.models).read_text(encoding="utf-8"))["models"]  # noqa: F821
names = [m["name"] for m in model_info]
utils_energy_plot.render_two_model_energy(
    scores_tsv=Path(snakemake.input.scores),  # noqa: F821
    model_names=(names[0], names[1]),
    out_pdf=Path(snakemake.output[0]),  # noqa: F821
)
