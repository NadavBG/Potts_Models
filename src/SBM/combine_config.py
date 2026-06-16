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
_METHODS = ("auto", "in_frame", "map", "marginal")


@dataclass(frozen=True)
class ModelRef:
    """One model to score against: a label, its run dir, and a weight."""

    name: str
    run_dir: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, ctx: str) -> "ModelRef":
        _reject_unknown(cls, data, ctx)
        _require("name" in data and "run_dir" in data, f"{ctx}: needs 'name' and 'run_dir'")
        obj = cls(name=str(data["name"]), run_dir=str(data["run_dir"]),
                  weight=float(data.get("weight", 1.0)))
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
        if obj.source == "fasta":
            _require(bool(obj.fasta), "query.source='fasta' requires query.fasta")
        return obj


@dataclass(frozen=True)
class ScoringConfig:
    # Operational default is `map` (the single best alignment per model, the same
    # procedure for both → comparable energies). `marginal` is the principled
    # model-evidence (free energy) and the only mode that yields ESS; `auto` is a
    # speed hack that breaks A/B comparability (see the warning it emits).
    method: str = "map"
    n_samples: int = 1000
    ess_threshold: float = 100.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        _reject_unknown(cls, data, "scoring")
        obj = cls(**data)
        _require(obj.method in _METHODS, f"scoring.method must be one of {_METHODS}")
        _require(obj.n_samples >= 1, "scoring.n_samples must be >= 1")
        _require(obj.ess_threshold >= 0, "scoring.ess_threshold must be >= 0")
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
            f"config.models must list exactly two models (E_tot = w_A·E_A + w_B·E_B); "
            f"got {len(models)}",
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
