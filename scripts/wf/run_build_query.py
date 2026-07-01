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

import numpy as np

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM.energy import datasets
from SBM.energy.datasets import QueryRecord

log = setup_stage_logging(snakemake, "build_query")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821


def _random_control_records(n: int, length: int, seed: int) -> list[QueryRecord]:
    """Random length-``length`` sequences (residues 1..20 uniform iid) as the
    ``random/N<length>`` negative control group (iter-003, docs/POTTS_ALIGN.md).

    ``origin_model=""`` ⇒ no home term; both models score them as cross pairs.
    Seeded from the run seed via an independent SeedSequence stream, so the draw
    is reproducible and does not perturb any capping RNG. This is the single
    source of the controls — the cluster/plan and the Mac canary read them back
    from query.fasta rather than regenerating.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xC02701,)))
    ints = rng.integers(1, 21, size=(n, length), dtype=np.int64)
    group = f"random/N{length}"
    return [QueryRecord(f"random|N{length}|{i}", group, "", ints[i]) for i in range(n)]

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
    if cfg.scoring.method == "potts_align" and cfg.query.n_random > 0:
        controls = _random_control_records(cfg.query.n_random, cfg.query.random_length, cfg.seed)
        records += controls
        log.info("query: appended %d random control sequence(s) (group %s, length %d, seed %d)",
                 len(controls), controls[0].group, cfg.query.random_length, cfg.seed)
    datasets.write_query_fasta(records, fasta_out, groups_out)
    by_group: dict[str, int] = {}
    for r in records:
        by_group[r.group] = by_group.get(r.group, 0) + 1
    log.info("query: %d sequences across %d groups: %s",
             len(records), len(by_group), json.dumps(by_group))
