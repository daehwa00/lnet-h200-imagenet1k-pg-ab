"""Measured, shape-aware launch selection for the project's Triton kernels.

Callers provide tensors and workload dimensions; this module derives the
device/compiler scope and selects the geometry. Resolution order is

    explicit argument
      >  environment variable
      >  geometry tuned and stored for this device
      >  registered default

Eager and compiled execution race candidates for an unseen exact scope in the
context that will actually launch the kernel. Persisted real-step winners can
still pin an exact device, shape, dtype, compiler, and kernel revision. Normal
model calls require no launch arguments.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import triton

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from triton.runtime.autotuner import Autotuner


@dataclass(frozen=True, slots=True)
class LaunchGeometry:
    """One deterministic Triton launch configuration."""

    num_warps: int = 4
    num_stages: int = 1
    # Kernel ``tl.constexpr`` tile sizes as sorted pairs, e.g.
    # ``(("BLOCK_N", 32),)``.  Stored as a tuple rather than a mapping so the
    # geometry stays hashable and can key the Autotuner cache.
    block_items: tuple[tuple[str, int], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        num_warps: int = 4,
        num_stages: int = 1,
        blocks: Mapping[str, int] | None = None,
    ) -> LaunchGeometry:
        return cls(
            num_warps=num_warps,
            num_stages=num_stages,
            block_items=tuple(sorted((blocks or {}).items())),
        )

    @property
    def blocks(self) -> dict[str, int]:
        return dict(self.block_items)

    def with_overrides(
        self,
        *,
        num_warps: int | None = None,
        num_stages: int | None = None,
        blocks: Mapping[str, int] | None = None,
    ) -> LaunchGeometry:
        merged = self.blocks if blocks is None else {**self.blocks, **blocks}
        return replace(
            self,
            num_warps=self.num_warps if num_warps is None else num_warps,
            num_stages=self.num_stages if num_stages is None else num_stages,
            block_items=tuple(sorted(merged.items())),
        )

    def as_config(self) -> triton.Config:
        return triton.Config(
            self.blocks,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )


@dataclass(frozen=True, slots=True)
class LaunchScope:
    """Exact workload/compiler identity for one persisted launch winner.

    Triton's in-process autotuner already distinguishes tensor dtypes and the
    values named in its ``key`` argument.  The project-level store must retain
    the same distinction: a winner measured for one shape or compiler build is
    not evidence for another.
    """

    execution_regime: str
    compile_profile: str | None
    dtype: str
    shape_items: tuple[tuple[str, int], ...]
    torch_version: str
    triton_version: str
    cuda_version: str | None
    cuda_driver_version: int | None
    kernel_revision: str

    @classmethod
    def build(
        cls,
        *,
        execution_regime: str,
        compile_profile: str | None = None,
        dtype: torch.dtype | str,
        shape: Mapping[str, int],
        kernel_revision: str,
        torch_version: str | None = None,
        triton_version: str | None = None,
        cuda_version: str | None = None,
        cuda_driver_version: int | None = None,
    ) -> LaunchScope:
        """Build a canonical scope, with injectable versions for tests/tools."""
        return cls(
            execution_regime=str(execution_regime),
            compile_profile=compile_profile,
            dtype=str(dtype).removeprefix("torch."),
            shape_items=tuple(sorted((str(name), int(value)) for name, value in shape.items())),
            torch_version=torch.__version__ if torch_version is None else torch_version,
            triton_version=triton.__version__ if triton_version is None else triton_version,
            cuda_version=torch.version.cuda if cuda_version is None else cuda_version,
            cuda_driver_version=(
                _cuda_driver_version() if cuda_driver_version is None else cuda_driver_version
            ),
            kernel_revision=str(kernel_revision),
        )

    @property
    def key(self) -> str:
        """Return a deterministic JSON key suitable for the on-disk store."""
        return json.dumps(
            {
                "cuda_driver_version": self.cuda_driver_version,
                "cuda_version": self.cuda_version,
                "compile_profile": self.compile_profile,
                "dtype": self.dtype,
                "execution_regime": self.execution_regime,
                "kernel_revision": self.kernel_revision,
                "shape": dict(self.shape_items),
                "torch_version": self.torch_version,
                "triton_version": self.triton_version,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


class UnknownKernelError(KeyError):
    """Raised when a geometry is requested for an unregistered kernel."""


_DEFAULTS: dict[str, LaunchGeometry] = {}
_CANDIDATES: dict[str, tuple[LaunchGeometry, ...]] = {}
_ENV_PREFIX = "LNET_LAUNCH"
_CACHE_VARIABLE = "LNET_LAUNCH_CACHE"
_DISABLE_AUTOTUNE_VARIABLE = "LNET_DISABLE_LAUNCH_AUTOTUNE"
_STORE_SCHEMA_VERSION = 2
_LEGACY_SCOPE_KEY = "legacy-unscoped"
_SCOPE_CAPTURE: ContextVar[tuple[str, set[LaunchScope]] | None] = ContextVar(
    "lnet_launch_scope_capture",
    default=None,
)
_SCOPE_GEOMETRY_OVERRIDE: ContextVar[tuple[str, LaunchScope, LaunchGeometry] | None] = ContextVar(
    "lnet_launch_scope_geometry_override",
    default=None,
)


def _cuda_driver_version() -> int | None:
    getter = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except RuntimeError:
        return None
    return value if isinstance(value, int) else None


def make_launch_scope(
    kernel: object,
    reference: torch.Tensor,
    *,
    shape: Mapping[str, int],
    execution_regime: str | None = None,
) -> LaunchScope:
    """Describe the exact runtime contract used to tune ``kernel``."""
    revision = getattr(kernel, "cache_key", None)
    if not isinstance(revision, str) or not revision:
        message = "launch-scoped kernels must expose a stable cache_key revision"
        raise TypeError(message)
    regime = execution_regime or ("compiled" if torch.compiler.is_compiling() else "eager")
    compile_profile = os.environ.get("LNET_COMPILE_PROFILE") if regime == "compiled" else None
    return LaunchScope.build(
        execution_regime=regime,
        compile_profile=compile_profile,
        dtype=reference.dtype,
        shape=shape,
        kernel_revision=revision,
    )


@contextmanager
def capture_launch_scopes(name: str) -> Generator[set[LaunchScope]]:
    """Collect exact scopes exercised while a real-step candidate is timed."""
    scopes: set[LaunchScope] = set()
    token = _SCOPE_CAPTURE.set((name, scopes))
    try:
        yield scopes
    finally:
        _SCOPE_CAPTURE.reset(token)


@contextmanager
def override_launch_geometry(
    name: str,
    scope: LaunchScope,
    geometry: LaunchGeometry,
) -> Generator[None]:
    """Temporarily force one exact scope without changing sibling launches."""
    token = _SCOPE_GEOMETRY_OVERRIDE.set((name, scope, geometry))
    try:
        yield
    finally:
        _SCOPE_GEOMETRY_OVERRIDE.reset(token)


def _scope_geometry_override(
    name: str,
    scope: LaunchScope | None,
) -> LaunchGeometry | None:
    active = _SCOPE_GEOMETRY_OVERRIDE.get()
    if active is None or scope is None or active[0] != name or active[1] != scope:
        return None
    return active[2]


def _record_launch_scope(name: str, scope: LaunchScope | None) -> None:
    capture = _SCOPE_CAPTURE.get()
    if scope is not None and capture is not None and capture[0] == name:
        capture[1].add(scope)


def register_default(
    name: str,
    geometry: LaunchGeometry,
    *,
    candidates: tuple[LaunchGeometry, ...] = (),
) -> None:
    """Register a kernel's measured geometry and its autotune search space.

    ``geometry`` is what runs unless a stored per-device record or an explicit
    override says otherwise.  ``candidates`` is only consulted when autotuning
    is enabled, and always includes ``geometry`` itself.
    """
    _DEFAULTS[name] = geometry
    _CANDIDATES[name] = tuple(dict.fromkeys((geometry, *candidates)))


def registered_candidates(name: str) -> tuple[LaunchGeometry, ...]:
    return _CANDIDATES.get(name, (registered_default(name),))


def device_key() -> str:
    """Identify the GPU well enough that a tuned record should transfer."""
    if not torch.cuda.is_available():
        return "cpu"
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index).replace(" ", "_")
    major, minor = torch.cuda.get_device_capability(index)
    return f"{name}-sm{major}{minor}"


def _store_path() -> Path:
    root = os.environ.get(_CACHE_VARIABLE)
    base = Path(root) if root else Path.home() / ".cache" / "lnet"
    return base / "launch_geometry.json"


def _read_store() -> dict[str, object]:
    path = _store_path()
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    if loaded.get("schema_version") == _STORE_SCHEMA_VERSION:
        records = loaded.get("records")
        return records if isinstance(records, dict) else {}
    # Schema 1 was an unversioned {device: {kernel: geometry}} mapping.  Keep it
    # readable only for unscoped legacy callers; exact scopes never inherit it.
    return loaded


def _geometry_payload(geometry: LaunchGeometry) -> dict[str, object]:
    return {
        "num_warps": geometry.num_warps,
        "num_stages": geometry.num_stages,
        "blocks": geometry.blocks,
    }


def _geometry_from_payload(record: object) -> LaunchGeometry | None:
    if not isinstance(record, dict):
        return None
    blocks: object = record.get("blocks")
    if not isinstance(blocks, dict):
        return None
    try:
        return LaunchGeometry.build(
            num_warps=int(str(record["num_warps"])),
            num_stages=int(str(record["num_stages"])),
            blocks={str(key): int(str(value)) for key, value in blocks.items()},
        )
    except (KeyError, TypeError, ValueError):
        return None


def stored_geometry(
    name: str,
    *,
    scope: LaunchScope | None = None,
) -> LaunchGeometry | None:
    """Return an exact scoped winner, or a legacy winner when unscoped."""
    device_records = _read_store().get(device_key())
    if not isinstance(device_records, dict):
        return None
    kernel_records = device_records.get(name)
    if not isinstance(kernel_records, dict):
        return None
    # Direct geometry dictionaries are records written by the old schema.
    if "num_warps" in kernel_records:
        return _geometry_from_payload(kernel_records) if scope is None else None
    scope_key = _LEGACY_SCOPE_KEY if scope is None else scope.key
    return _geometry_from_payload(kernel_records.get(scope_key))


def store_geometry(
    name: str,
    geometry: LaunchGeometry,
    *,
    scope: LaunchScope | None = None,
) -> None:
    """Persist a winner under one exact scope using a locked atomic update."""
    path = _store_path()
    lock_path = path.with_suffix(".lock")
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            store = _read_store()
            device = device_key()
            device_records = store.setdefault(device, {})
            if not isinstance(device_records, dict):
                device_records = {}
                store[device] = device_records
            previous = device_records.get(name)
            if isinstance(previous, dict) and "num_warps" in previous:
                scoped_records: dict[str, object] = {
                    _LEGACY_SCOPE_KEY: previous,
                }
            elif isinstance(previous, dict):
                scoped_records = dict(previous)
            else:
                scoped_records = {}
            scope_key = _LEGACY_SCOPE_KEY if scope is None else scope.key
            scoped_records[scope_key] = _geometry_payload(geometry)
            device_records[name] = scoped_records
            document = {
                "schema_version": _STORE_SCHEMA_VERSION,
                "records": store,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(document, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
            temporary_path = None
    except OSError:
        # A read-only cache directory must not break training.
        return
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def registered_names() -> tuple[str, ...]:
    return tuple(sorted(_DEFAULTS))


def registered_default(name: str) -> LaunchGeometry:
    if name not in _DEFAULTS:
        message = f"no launch geometry registered for {name!r}"
        raise UnknownKernelError(message)
    return _DEFAULTS[name]


def _environment_overrides(name: str, geometry: LaunchGeometry) -> LaunchGeometry:
    """Read ``LNET_LAUNCH_<NAME>_{WARPS,STAGES,<BLOCK>}`` if present."""
    prefix = f"{_ENV_PREFIX}_{name.upper()}"
    warps = os.environ.get(f"{prefix}_WARPS")
    stages = os.environ.get(f"{prefix}_STAGES")
    blocks = {
        key: int(os.environ[f"{prefix}_{key.upper()}"])
        for key in geometry.blocks
        if f"{prefix}_{key.upper()}" in os.environ
    }
    return geometry.with_overrides(
        num_warps=None if warps is None else int(warps),
        num_stages=None if stages is None else int(stages),
        blocks=blocks or None,
    )


def _has_environment_override(name: str) -> bool:
    prefix = f"{_ENV_PREFIX}_{name.upper()}"
    names = {f"{prefix}_WARPS", f"{prefix}_STAGES"}
    names.update(f"{prefix}_{block.upper()}" for block in registered_default(name).blocks)
    return any(variable in os.environ for variable in names)


def resolve(
    name: str,
    *,
    scope: LaunchScope | None = None,
    geometry: LaunchGeometry | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    blocks: Mapping[str, int] | None = None,
) -> LaunchGeometry:
    """Resolve the geometry for one kernel.

    An explicit ``geometry`` replaces the registered default outright; the
    scalar arguments override individual fields on top of whatever remains.
    Environment variables sit between the default and the explicit arguments so
    a sweep can move a kernel without touching the caller.
    """
    _record_launch_scope(name, scope)
    scope_override = _scope_geometry_override(name, scope) if geometry is None else None
    base = (
        geometry or scope_override or stored_geometry(name, scope=scope) or registered_default(name)
    )
    if scope_override is None:
        base = _environment_overrides(name, base)
    return base.with_overrides(num_warps=num_warps, num_stages=num_stages, blocks=blocks)


_AUTOTUNERS: dict[
    tuple[
        str,
        str,
        tuple[LaunchGeometry, ...],
        tuple[str, ...],
        object | None,
        tuple[str, ...],
        tuple[str, ...],
    ],
    Autotuner,
] = {}


def autotuned(
    kernel: Callable[..., object],
    name: str,
    *,
    key: tuple[str, ...],
    scope: LaunchScope | None = None,
    geometry: LaunchGeometry | None = None,
    early_config_prune: Callable[..., object] | None = None,
    reset_to_zero: tuple[str, ...] = (),
    restore_value: tuple[str, ...] = (),
) -> Autotuner:
    """Wrap ``kernel`` in an automatically searched or pinned Autotuner.

    Some kernels are inlined by Inductor, which then substitutes its own launch
    parameters; an autotune config is the only form a geometry survives that in.
    An unseen device/shape/dtype key searches the registered candidates in its
    actual eager or compiled launch context and uses Triton's disk cache
    thereafter. Reproducibility controls remain single-config: an explicit
    argument, a stored winner, an environment override, or
    ``LNET_DISABLE_LAUNCH_AUTOTUNE=1`` all pin ``resolve(name)``.
    """
    _record_launch_scope(name, scope)
    scope_override = _scope_geometry_override(name, scope) if geometry is None else None
    # Unscoped persisted records are retained for compatibility with explicit
    # ``resolve`` callers, but must not disable Triton's shape/dtype autotuning.
    # Only an exact scope is safe evidence for pinning this wrapper.
    stored = stored_geometry(name, scope=scope) if scope is not None else None
    pinned = (
        geometry is not None
        or scope_override is not None
        or os.environ.get(_DISABLE_AUTOTUNE_VARIABLE) == "1"
        or stored is not None
        or _has_environment_override(name)
    )
    geometries = (
        (geometry or scope_override or stored or resolve(name, scope=scope),)
        if pinned
        else registered_candidates(name)
    )
    cache_key = (
        name,
        device_key(),
        geometries,
        key,
        early_config_prune,
        reset_to_zero,
        restore_value,
    )
    tuner = _AUTOTUNERS.get(cache_key)
    if tuner is None:
        tuner = triton.autotune(
            configs=[candidate.as_config() for candidate in geometries],
            key=list(key),
            cache_results=True,
            reset_to_zero=reset_to_zero or None,
            restore_value=restore_value or None,
            prune_configs_by=(
                {"early_config_prune": early_config_prune}
                if early_config_prune is not None
                else None
            ),
        )(kernel)
        _AUTOTUNERS[cache_key] = tuner
    return tuner
