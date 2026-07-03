"""Typed, validated configuration for the post-hoc ``derive`` pipeline.

A *derive* run takes ONE already-trained model and writes a new model that
keeps only a subset of its parameters — e.g. fields only (all couplings
zeroed), couplings only (fields zeroed), or a mask-selected subset of
couplings/fields. It lands a normal ``results/`` run dir that the combine
pipeline consumes unchanged. No training and no MSA FASTA: the source model's
already-encoded MSA is copied.

This is the no-retrain alternative to the retrain-based ``*-profile.yaml``
configs. Those *re-fit* ``h`` to the single-site statistics with ``J≡0``;
post-hoc filtering instead keeps the dense model's *already-fit* ``h`` and
zeros ``J`` — a different (and legitimate) energy function, a clean ablation of
the dense model's field component. See ``CLAUDE.md`` / ``docs``.

Same conventions as the other pipelines: one validated YAML = one run, unknown
keys are an error, ``as_dict`` round-trips into ``config_snapshot.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Reuse the single-model validators + the mask/sample/figure schemas rather
# than duplicate them (one package).
from SBM.workflow_config import (
    _DIA_PRIORS,
    _H_STRATEGIES,
    _J_STRATEGIES,
    ConfigError,
    FiguresConfig,
    MaskSpec,
    SampleConfig,
    _reject_unknown,
    _require,
)

SCHEMA_VERSION = 1

#: A per-block filter directive is one of these string sentinels or a MaskSpec.
KEEP = "keep"
ZERO = "zero"


def _parse_block(value: Any, *, ctx: str, allowed: tuple[str, ...]) -> Any:
    """Parse one filter block: ``None``/``'keep'`` → ``KEEP``; ``'zero'`` →
    ``ZERO``; a ``{strategy, percent}`` mapping → :class:`MaskSpec`."""
    if value is None:
        return KEEP
    if isinstance(value, str):
        v = value.lower()
        _require(v in (KEEP, ZERO), f"{ctx}: string must be 'keep' or 'zero' (got {value!r})")
        return v
    if isinstance(value, dict):
        return MaskSpec.from_dict(value, ctx=ctx, allowed=allowed)
    raise ConfigError(
        f"{ctx}: must be null/'keep', 'zero', or a {{strategy, percent}} mapping"
    )


@dataclass(frozen=True)
class FilterConfig:
    """Which parameters to keep from the source model.

    ``couplings`` and ``fields`` are each ``'keep'`` (as-is), ``'zero'`` (drop
    the whole block), or a :class:`MaskSpec` ``{strategy, percent}`` selecting a
    data-derived subset to keep (reusing the pruning masks; ``percent`` is the
    fraction *removed*, mask value 1 = keep). Building a mask needs the source
    MSA statistics, so ``theta``/``lbda``/``label``/``Dia_prior`` mirror
    ``PruningConfig``; they are ignored unless a block is a ``MaskSpec``.
    """

    couplings: Any = KEEP  # "keep" | "zero" | MaskSpec
    fields: Any = KEEP  # "keep" | "zero" | MaskSpec
    theta: float = 0.7
    lbda: float = 0.03
    label: str = "CM"
    Dia_prior: str = "gap-corrected"

    @property
    def couplings_mask(self) -> MaskSpec | None:
        return self.couplings if isinstance(self.couplings, MaskSpec) else None

    @property
    def fields_mask(self) -> MaskSpec | None:
        return self.fields if isinstance(self.fields, MaskSpec) else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilterConfig":
        _reject_unknown(cls, data, "filter")
        rest = {k: v for k, v in data.items() if k not in ("couplings", "fields")}
        couplings = _parse_block(
            data.get("couplings"), ctx="filter.couplings", allowed=_J_STRATEGIES
        )
        fields = _parse_block(data.get("fields"), ctx="filter.fields", allowed=_H_STRATEGIES)
        obj = cls(couplings=couplings, fields=fields, **rest)
        _require(obj.Dia_prior in _DIA_PRIORS, f"filter.Dia_prior must be one of {_DIA_PRIORS}")
        _require(0.0 <= obj.theta <= 1.0, "filter.theta must be in [0, 1]")
        _require(obj.lbda >= 0.0, "filter.lbda must be >= 0")
        return obj


@dataclass(frozen=True)
class DeriveRunConfig:
    """The complete, validated configuration for one derive run."""

    run_name: str
    source_run_dir: str
    description: str = ""
    family: str = ""
    seed: int = 42
    omp_num_threads: int | None = None
    filter: FilterConfig = field(default_factory=FilterConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)

    @property
    def source_model_path(self) -> str:
        return str(Path(self.source_run_dir) / "model.npy")

    @property
    def source_msa_path(self) -> str:
        return str(Path(self.source_run_dir) / "inputs" / "msa.npy")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeriveRunConfig":
        _reject_unknown(cls, data, "config")
        _require("run_name" in data, "config: 'run_name' is required")
        _require(
            "source_run_dir" in data,
            "config: 'source_run_dir' is required (a run dir with a trained model.npy)",
        )
        nested = {"filter": FilterConfig, "sample": SampleConfig, "figures": FiguresConfig}
        kwargs = {k: v for k, v in data.items() if k not in nested}
        for key, sub_cls in nested.items():
            if data.get(key) is not None:
                kwargs[key] = sub_cls.from_dict(data[key])
        obj = cls(**kwargs)
        _require(bool(obj.run_name), "config.run_name must be non-empty")
        _require(bool(obj.source_run_dir), "config.source_run_dir must be non-empty")
        return obj


def from_dict(data: dict[str, Any]) -> DeriveRunConfig:
    """Validate a raw config dict (as loaded from YAML) into a DeriveRunConfig."""
    return DeriveRunConfig.from_dict(data)


def load_config(path: Path | str) -> DeriveRunConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return from_dict(raw)


def dump_config(cfg: DeriveRunConfig, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.as_dict(), handle, sort_keys=False, default_flow_style=False)
    return out
