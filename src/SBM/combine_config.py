"""Typed, validated configuration for the two-model `combine` pipeline.

The single-model pipeline (``SBM.workflow_config``) describes one run that trains
one model. A *combine* run instead consumes **two already-trained models** and
scores a set of query sequences under both — a different entity, so it gets its
own config schema and Snakefile (``Snakefile.combine``) rather than overloading
the single-model one. Same conventions: one validated YAML = one run, unknown
keys are an error, ``as_dict`` round-trips into ``config_snapshot.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Reuse the single-model validators rather than duplicate them (one package).
from SBM.workflow_config import ConfigError, _reject_unknown, _require

SCHEMA_VERSION = 1

_QUERY_SOURCES = ("model_sets", "fasta")
_INCLUDE = ("natural", "synthetic")
_METHODS = ("auto", "in_frame", "map", "marginal", "potts_align")


@dataclass(frozen=True)
class ModelRef:
    """One model to score against: a label and its run dir.

    No weight here by design: the E_tot combining weights are NOT configured, they
    are *derived post-hoc from the naturals* so each family's median native energy
    contributes equally to E_tot (the `compute_weights` stage; SBM.utils.energy_weights).
    A stray `weight:` key is therefore rejected by `_reject_unknown`.
    """

    name: str
    run_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, ctx: str) -> "ModelRef":
        _reject_unknown(cls, data, ctx)
        _require("name" in data and "run_dir" in data, f"{ctx}: needs 'name' and 'run_dir'")
        obj = cls(name=str(data["name"]), run_dir=str(data["run_dir"]))
        _require(bool(obj.name), f"{ctx}.name must be non-empty")
        return obj

    @property
    def model_path(self) -> str:
        return str(Path(self.run_dir) / "model.npy")


@dataclass(frozen=True)
class QueryConfig:
    """What to score: each model's natural+synthetic sets, or an external FASTA."""

    source: str = "model_sets"
    include: list[str] = field(default_factory=lambda: ["natural", "synthetic"])
    fasta: str | None = None
    # Per-group cap before scoring (0 = no cap). Bounds the O(samples) cost of
    # marginal scoring on large natural MSAs (e.g. the ~26k-seq PPIC alignment);
    # the subsample is seeded with the run seed and the drop is logged.
    cap_per_group: int = 500
    # potts_align negative control (iter-003): n_random random length-random_length
    # sequences (residues 1..20 uniform iid, seeded from the run seed), appended as
    # group "random/N91" with origin_model="" by build_query (scored under both
    # models). 0 = none. random_length must be <= min model L to score under both.
    n_random: int = 0
    random_length: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryConfig":
        _reject_unknown(cls, data, "query")
        rest = {k: v for k, v in data.items() if k != "include"}
        include = [str(s) for s in data.get("include", ["natural", "synthetic"])]
        obj = cls(include=include, **rest)
        _require(obj.source in _QUERY_SOURCES, f"query.source must be one of {_QUERY_SOURCES}")
        bad = set(obj.include) - set(_INCLUDE)
        _require(not bad, f"query.include has unknown value(s) {sorted(bad)}; allowed: {_INCLUDE}")
        _require(obj.cap_per_group >= 0, "query.cap_per_group must be >= 0 (0 = no cap)")
        _require(obj.n_random >= 0, "query.n_random must be >= 0 (0 = no control group)")
        _require(obj.random_length >= 0, "query.random_length must be >= 0")
        _require(
            obj.n_random == 0 or obj.random_length > 0,
            "query.n_random > 0 requires query.random_length > 0",
        )
        if obj.source == "fasta":
            _require(bool(obj.fasta), "query.source='fasta' requires query.fasta")
        return obj


@dataclass(frozen=True)
class ScoringConfig:
    # `potts_align` is the production aligner (couplings-aware gap-placement Potts
    # minimizer; docs/POTTS_ALIGN.md) — the cluster align step is pure numpy and
    # sharded over `n_shards`. `map` is the single best fields-Viterbi alignment
    # per model (same procedure for both → comparable energies); `marginal` is the
    # principled model-evidence (free energy) and the only mode that yields ESS;
    # `auto` is a speed hack that breaks A/B comparability (see the warning it emits).
    method: str = "map"
    n_samples: int = 1000
    ess_threshold: float = 100.0
    n_shards: int = 32
    # potts_align (method="potts_align"): the gap-placement aligner is pure numpy
    # and runs sharded on Slurm (n_shards above). To bound the PT cost, the cross
    # block where `pa_cross_subsample_origin` queries are scored under
    # `pa_cross_subsample_under` is restricted to a seeded subset of
    # pa_cross_subsample_n ids (0 = no subsample). The two model names must match
    # entries in `models` (checked in CombineRunConfig.from_dict, where they are
    # known). All other potts_align knobs (the g-adaptive PT schedule) are internal.
    pa_cross_subsample_origin: str | None = None
    pa_cross_subsample_under: str | None = None
    pa_cross_subsample_n: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        _reject_unknown(cls, data, "scoring")
        obj = cls(**data)
        _require(obj.method in _METHODS, f"scoring.method must be one of {_METHODS}")
        _require(obj.n_samples >= 1, "scoring.n_samples must be >= 1")
        _require(obj.ess_threshold >= 0, "scoring.ess_threshold must be >= 0")
        _require(obj.n_shards >= 1, "scoring.n_shards must be >= 1")
        _require(obj.pa_cross_subsample_n >= 0, "scoring.pa_cross_subsample_n must be >= 0")
        if obj.pa_cross_subsample_n > 0:
            _require(
                bool(obj.pa_cross_subsample_origin) and bool(obj.pa_cross_subsample_under),
                "scoring.pa_cross_subsample_n > 0 requires both pa_cross_subsample_origin "
                "and pa_cross_subsample_under (the cross block's origin and scored-under model names)",
            )
        return obj


@dataclass(frozen=True)
class CombineFiguresConfig:
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombineFiguresConfig":
        _reject_unknown(cls, data, "figures")
        return cls(**data)


@dataclass(frozen=True)
class CombineRunConfig:
    """The complete, validated configuration for one combine run."""

    run_name: str
    models: list[ModelRef]
    description: str = ""
    seed: int = 42
    omp_num_threads: int | None = None
    query: QueryConfig = field(default_factory=QueryConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    figures: CombineFiguresConfig = field(default_factory=CombineFiguresConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombineRunConfig":
        _reject_unknown(cls, data, "config")
        _require("run_name" in data, "config: 'run_name' is required")
        _require("models" in data, "config: 'models' is required (a list of two model refs)")
        _require(isinstance(data["models"], list), "config.models must be a list")
        models = [
            ModelRef.from_dict(m, ctx=f"models[{i}]") for i, m in enumerate(data["models"])
        ]
        _require(
            len(models) == 2,
            f"config.models must list exactly two models (E_tot = w_A·E_A + w_B·E_B, "
            f"weights derived post-hoc from the naturals); got {len(models)}",
        )
        names = [m.name for m in models]
        _require(len(set(names)) == len(names), f"config.models names must be unique; got {names}")
        nested = {"query": QueryConfig, "scoring": ScoringConfig, "figures": CombineFiguresConfig}
        kwargs = {k: v for k, v in data.items() if k not in nested and k != "models"}
        for key, sub_cls in nested.items():
            if data.get(key) is not None:
                kwargs[key] = sub_cls.from_dict(data[key])
        obj = cls(models=models, **kwargs)
        _require(bool(obj.run_name), "config.run_name must be non-empty")
        # potts_align cross-subsample model names must reference real models (the
        # ScoringConfig validator can't see the model list; check it here).
        if obj.scoring.method == "potts_align" and obj.scoring.pa_cross_subsample_n > 0:
            valid = {m.name for m in obj.models}
            for fld in ("pa_cross_subsample_origin", "pa_cross_subsample_under"):
                nm = getattr(obj.scoring, fld)
                _require(nm in valid, f"scoring.{fld}={nm!r} must be one of model names {sorted(valid)}")
        return obj


def from_dict(data: dict[str, Any]) -> CombineRunConfig:
    """Validate a raw config dict (as loaded from YAML) into a CombineRunConfig."""
    return CombineRunConfig.from_dict(data)


def load_config(path: Path | str) -> CombineRunConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return from_dict(raw)


def dump_config(cfg: CombineRunConfig, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.as_dict(), handle, sort_keys=False, default_flow_style=False)
    return out
