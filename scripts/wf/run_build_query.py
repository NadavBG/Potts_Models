"""Snakemake wrapper: assemble the query set into query.fasta + groups.json.

For ``query.source: model_sets`` (the default), pulls each model's natural MSA
and synthetic alignments, tags each sequence with its source group + origin
model, caps per group (seeded), and writes a mixed-length FASTA. For
``query.source: fasta``, copies the user FASTA through with no origin frame
(every model scores those by latent alignment). The drop from capping is logged.
"""

import json
import shutil
from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM.energy import datasets

log = setup_stage_logging(snakemake, "build_query")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821

fasta_out = Path(snakemake.output.fasta)  # noqa: F821
groups_out = Path(snakemake.output.groups)  # noqa: F821
fasta_out.parent.mkdir(parents=True, exist_ok=True)

if cfg.query.source == "fasta":
    shutil.copyfile(cfg.query.fasta, fasta_out)
    groups_out.write_text("{}\n", encoding="utf-8")  # no origin frames for external seqs
    log.info("query: external FASTA %s -> %s", cfg.query.fasta, fasta_out)
else:
    model_entries = [{"name": m.name, "run_dir": m.run_dir} for m in cfg.models]
    records = datasets.assemble_query_records(
        model_entries,
        include=tuple(cfg.query.include),
        cap_per_group=cfg.query.cap_per_group,
        seed=cfg.seed,
    )
    datasets.write_query_fasta(records, fasta_out, groups_out)
    by_group: dict[str, int] = {}
    for r in records:
        by_group[r.group] = by_group.get(r.group, 0) + 1
    log.info("query: %d sequences across %d groups: %s",
             len(records), len(by_group), json.dumps(by_group))
