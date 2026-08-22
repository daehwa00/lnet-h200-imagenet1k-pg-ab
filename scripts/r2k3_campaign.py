#!/usr/bin/env python3
"""Shared assembly for R2K3 campaign runners derived from a parent runner.

Every campaign in this family is a small delta on an already-frozen runner: it
takes the parent's built model, swaps one operator, and republishes the run
contract.  The surrounding wiring -- variant tuples, spec lookup, the build
guard, and the contract skeleton -- was copied into each runner verbatim.  This
module holds that wiring once so a runner declares only its own delta.

The assembled contract is byte-identical to the hand-written one apart from the
runner's own ``source_sha256`` entry, which necessarily tracks the file.
"""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path
    from types import ModuleType

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


def build_derived(
    *,
    requested: str,
    variant: str,
    base_variant: str,
    config: ComplexScanConfig,
    parent_build: Callable[[str, ComplexScanConfig], ComplexScanBackbone],
    install: Callable[[nn.Module], None],
    assert_model: Callable[[nn.Module], None],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> ComplexScanBackbone:
    """Build a campaign model as a delta on its parent's frozen construction.

    The parent build, the operator swap, the runtime configure, and the
    structural assertion run in this exact order because that order fixes
    seeded initialization.
    """
    if requested != variant:
        raise ValueError(f"unsupported {variant} build request: {requested}")
    model = parent_build(base_variant, config)
    install(model)
    runtime.configure(variants, seeds)
    assert_model(model)
    return model


def campaign_contract(
    args: Namespace,
    *,
    runner_file: str | Path,
    runner_source_key: str,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    schema: str,
    evidence_status: str,
    variant: str | None = None,
    variant_config: Mapping[str, Any] | None = None,
    architecture: str | None = None,
    parameter_count: int | None = None,
    variant_configs: Mapping[str, Any] | None = None,
    architectures: Mapping[str, str] | None = None,
    parameter_counts: Mapping[str, int] | None = None,
    extra_sources: Mapping[str, str | Path] | None = None,
    references: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the immutable run contract shared by every campaign runner.

    The harness compares this payload for exact equality when resuming a run
    root, so the field set is part of the contract.

    Single-variant campaigns pass ``variant`` with the singular
    ``variant_config`` / ``architecture`` / ``parameter_count``.  Multi-variant
    campaigns pass the plural mappings keyed by variant instead.
    """
    if (variant is None) == (variant_configs is None):
        raise ValueError("pass either the singular variant form or the plural mappings")
    if variant is not None:
        if variant_config is None or architecture is None or parameter_count is None:
            raise ValueError(f"{variant} is missing its config, architecture, or parameters")
        variant_configs = {variant: dict(variant_config)}
        architectures = {variant: architecture}
        parameter_counts = {variant: parameter_count}
    if architectures is None or parameter_counts is None:
        raise ValueError("multi-variant campaigns need architectures and parameter_counts")

    payload = runtime.base_contract(args)
    payload["schema"] = schema
    payload["evidence_status"] = evidence_status
    payload["variants"] = list(variants)
    payload["seeds"] = list(seeds)
    payload["variant_configs"] = dict(variant_configs)
    payload["parameter_counts"] = dict(parameter_counts)
    payload["architecture"] = dict(architectures)
    if references is not None:
        payload["references"] = dict(references)
    for key, path in (extra_sources or {}).items():
        payload["source_sha256"][key] = runtime.digest(path)
    payload["source_sha256"][runner_source_key] = runtime.digest(runner_file)
    return payload


def inherited_names(parent: ModuleType, names: Iterable[str]) -> dict[str, Any]:
    """Re-export a parent runner's shared surface under a campaign runner."""
    return {name: getattr(parent, name) for name in names}


__all__ = ["build_derived", "campaign_contract", "inherited_names"]
