#!/usr/bin/env python3
"""Resolve an R2K3 campaign runner module from a command-line name.

Campaign queue and smoke entry points used to exist as one shim file per
runner, each of which imported a shared implementation and rebound its
module-global ``runner``.  This module replaces that pattern with an explicit
``--runner`` argument so a single queue and a single smoke script cover every
candidate.
"""

from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

RUNNER_PREFIX = "run_"
RUNNER_SUFFIX = "_imagenet100"


def canonical_runner_module(name: str) -> str:
    """Expand a short campaign alias to its full runner module name."""
    if name.startswith(RUNNER_PREFIX):
        return name
    return f"{RUNNER_PREFIX}{name}{RUNNER_SUFFIX}"


def resolve_runner(name: str) -> Any:
    """Import a campaign runner and verify it declares a variant surface."""
    module_name = canonical_runner_module(name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        message = f"unknown campaign runner: {name} (resolved to {module_name})"
        raise ValueError(message) from error
    if not hasattr(module, "VARIANT") and not hasattr(module, "VARIANTS"):
        message = f"{module_name} does not declare VARIANT or VARIANTS"
        raise ValueError(message)
    return module


def runner_variants(module: Any) -> tuple[str, ...]:
    """Return every variant a runner covers, single- or multi-variant."""
    variants = getattr(module, "VARIANTS", None)
    if variants:
        return tuple(variants)
    return (module.VARIANT,)


def add_runner_argument(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    """Register the shared ``--runner`` selector on a campaign parser."""
    parser.add_argument(
        "--runner",
        type=resolve_runner,
        required=default is None,
        default=resolve_runner(default) if default is not None else None,
        help="campaign runner module (full name or short alias)",
    )


def parse_runner(argv: Sequence[str] | None = None) -> Any:
    parser = argparse.ArgumentParser()
    add_runner_argument(parser)
    return parser.parse_args(argv).runner


__all__ = [
    "add_runner_argument",
    "canonical_runner_module",
    "parse_runner",
    "resolve_runner",
    "runner_variants",
]
