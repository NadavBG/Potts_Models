"""Typed, validated configuration for the Snakemake pipeline.

One YAML config file describes one run end-to-end: the input MSA, the
optional pruning masks, the training regime, the synthetic sampling, and
the figures. The Snakefile loads it,
``from_dict`` validates it (unknown keys are an error), and the per-stage
wrappers in ``scripts/wf/`` read fields off the resulting frozen
dataclass.

The dataclass is the single source of truth for the schema: ``as_dict``
round-trips back to the YAML written into each run's
``config_snapshot.yaml`` so a run always carries the exact parameters that
produced it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

#: Bumped if the schema changes in a backward-incompatible way.
SCHEMA_VERSION = 1

_J_STRATEGIES = ("fij", "cij", "sca")
_H_STRATEGIES = ("fia", "dia")
_DIA_PRIORS = ("gap-corrected", "uniform")
_SECTORS = ("emily", "rama", "none")
_MODES = ("BM", "SBM")
_OPTIMIZERS = ("LBFGS", "GD")


class ConfigError(ValueError):
    """Raised when a config dict is malformed or carries unknown keys."""


def _reject_unknown(cls: type, data: dict[str, Any], ctx: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{ctx}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{ctx}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}"
        )


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class MsaStatsConfig:
    """MSA-only statistics figure (independent of any model)."""

    enabled: bool = True
    theta: float = 0.7
    lbda: float = 0.03
    Dia_prior: str = "gap-corrected"
    sector: str = "emily"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MsaStatsConfig":
        _reject_unknown(cls, data, "msa_stats")
        obj = cls(**data)
        _require(obj.Dia_prior in _DIA_PRIORS, f"msa_stats.Dia_prior must be one of {_DIA_PRIORS}")
        _require(obj.sector in _SECTORS, f"msa_stats.sector must be one of {_SECTORS}")
        _require(0.0 <= obj.theta <= 1.0, "msa_stats.theta must be in [0, 1]")
        _require(obj.lbda >= 0.0, "msa_stats.lbda must be >= 0")
        return obj


@dataclass(frozen=True)
class MaskSpec:
    """A single pruning mask: one strategy, one keep/remove percentage."""

    strategy: str
    percent: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, ctx: str, allowed: tuple[str, ...]) -> "MaskSpec":
        _reject_unknown(cls, data, ctx)
        _require("strategy" in data and "percent" in data, f"{ctx}: needs 'strategy' and 'percent'")
        obj = cls(strategy=str(data["strategy"]).lower(), percent=float(data["percent"]))
        _require(obj.strategy in allowed, f"{ctx}.strategy must be one of {allowed}")
        _require(0.0 <= obj.percent <= 100.0, f"{ctx}.percent must be in [0, 100]")
        return obj


@dataclass(frozen=True)
class PruningConfig:
    enabled: bool = False
    theta: float = 0.7
    lbda: float = 0.03
    label: str = "CM"
    Dia_prior: str = "gap-corrected"  # background for the 'dia' fields strategy
    couplings: MaskSpec | None = None
    fields: MaskSpec | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PruningConfig":
        _reject_unknown(cls, data, "pruning")
        rest = {k: v for k, v in data.items() if k not in ("couplings", "fields")}
        couplings = (
            MaskSpec.from_dict(data["couplings"], ctx="pruning.couplings", allowed=_J_STRATEGIES)
            if data.get("couplings") is not None
            else None
        )
        flds = (
            MaskSpec.from_dict(data["fields"], ctx="pruning.fields", allowed=_H_STRATEGIES)
            if data.get("fields") is not None
            else None
        )
        obj = cls(couplings=couplings, fields=flds, **rest)
        _require(obj.Dia_prior in _DIA_PRIORS, f"pruning.Dia_prior must be one of {_DIA_PRIORS}")
        if obj.enabled:
            _require(
                obj.couplings is not None or obj.fields is not None,
                "pruning.enabled is true but neither 'couplings' nor 'fields' is set",
            )
        return obj


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "SBM"
    optimizer: str = "LBFGS"
    N_iter: int = 400
    N_chains: int = 50
    m: int = 1
    lambda_J: float = 0.0
    lambda_h: float = 0.0
    theta: float = 0.3
    k_MCMC: int = 100000
    TestTrain: int = 0
    record_every: int = 5
    ignore_gaps: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        _reject_unknown(cls, data, "train")
        obj = cls(**data)
        _require(obj.mode in _MODES, f"train.mode must be one of {_MODES}")
        _require(obj.optimizer in _OPTIMIZERS, f"train.optimizer must be one of {_OPTIMIZERS}")
        _require(obj.N_iter >= 1, "train.N_iter must be >= 1")
        _require(obj.N_chains >= 1, "train.N_chains must be >= 1")
        _require(obj.m >= 1, "train.m must be >= 1")
        _require(obj.k_MCMC >= 1, "train.k_MCMC must be >= 1")
        _require(obj.TestTrain in (0, 1), "train.TestTrain must be 0 or 1")
        _require(obj.record_every >= 1, "train.record_every must be >= 1")
        # Summary Note 3 recommends BM=(m=20, lambda=0.01), SBM=(m=1, lambda=0).
        # These knobs are intentionally tunable, so mismatches warn rather than fail.
        if obj.mode == "BM" and obj.m == 1:
            log.warning("train.mode=BM with m=1 (Summary Note 3 recommends m=20 for BM)")
        if obj.mode == "SBM" and (obj.lambda_J != 0 or obj.lambda_h != 0):
            log.warning("train.mode=SBM with nonzero L2 (Summary Note 3 recommends lambda=0 for SBM)")
        return obj


@dataclass(frozen=True)
class SampleConfig:
    N: int = 2000
    temperatures: list[float] = field(default_factory=lambda: [0.75, 1.0])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleConfig":
        _reject_unknown(cls, data, "sample")
        rest = {k: v for k, v in data.items() if k != "temperatures"}
        raw_temps = data.get("temperatures", [0.75, 1.0])
        _require(isinstance(raw_temps, list), "sample.temperatures must be a list")
        temps = [float(t) for t in raw_temps]
        obj = cls(temperatures=temps, **rest)
        _require(obj.N >= 1, "sample.N must be >= 1")
        _require(len(obj.temperatures) >= 1, "sample.temperatures must be non-empty")
        _require(all(t > 0 for t in obj.temperatures), "sample.temperatures must all be > 0")
        _require(len(set(obj.temperatures)) == len(obj.temperatures), "sample.temperatures must be unique")
        return obj


@dataclass(frozen=True)
class FiguresConfig:
    which: list[str] | None = None
    sector: str = "emily"
    # Cap on sequences sampled per group before the O(N^2) all-pairs
    # similarity / diversity computations. Large natural MSAs (e.g. the
    # ~26k-sequence PPIC alignment) otherwise make these figures take
    # minutes; a few thousand sequences give faithful violins. The
    # subsample is seeded with the run's master seed. 0 = no cap.
    max_seqs_per_group: int = 2000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FiguresConfig":
        _reject_unknown(cls, data, "figures")
        rest = {k: v for k, v in data.items() if k != "which"}
        which = data.get("which")
        which = [str(w) for w in which] if which is not None else None
        obj = cls(which=which, **rest)
        _require(obj.sector in _SECTORS, f"figures.sector must be one of {_SECTORS}")
        _require(
            obj.max_seqs_per_group >= 0,
            "figures.max_seqs_per_group must be >= 0 (0 = no cap)",
        )
        return obj


@dataclass(frozen=True)
class SBMRunConfig:
    """The complete, validated configuration for one pipeline run."""

    run_name: str
    msa_fasta: str
    description: str = ""
    family: str = ""
    seed: int = 42
    omp_num_threads: int | None = None
    msa_stats: MsaStatsConfig = field(default_factory=MsaStatsConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for YAML round-trip (config_snapshot.yaml)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SBMRunConfig":
        _reject_unknown(cls, data, "config")
        _require("run_name" in data, "config: 'run_name' is required")
        _require("msa_fasta" in data, "config: 'msa_fasta' is required (path to an aligned FASTA)")
        nested = {
            "msa_stats": MsaStatsConfig,
            "pruning": PruningConfig,
            "train": TrainConfig,
            "sample": SampleConfig,
            "figures": FiguresConfig,
        }
        kwargs = {k: v for k, v in data.items() if k not in nested}
        for key, sub_cls in nested.items():
            if data.get(key) is not None:
                kwargs[key] = sub_cls.from_dict(data[key])
        obj = cls(**kwargs)
        _require(bool(obj.run_name), "config.run_name must be non-empty")
        _require(bool(obj.msa_fasta), "config.msa_fasta must be non-empty")
        return obj


def from_dict(data: dict[str, Any]) -> SBMRunConfig:
    """Validate a raw config dict (as loaded from YAML) into an SBMRunConfig."""
    return SBMRunConfig.from_dict(data)


def load_config(path: Path | str) -> SBMRunConfig:
    """Load and validate a YAML config file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return from_dict(raw)


def dump_config(cfg: SBMRunConfig, path: Path | str) -> Path:
    """Write a validated config to YAML (used for config_snapshot.yaml)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.as_dict(), handle, sort_keys=False, default_flow_style=False)
    return out
