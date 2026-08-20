"""Resolve and hash the local source closure for current R2K3 workflows."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


_LEGACY_TRAINING_ROOT = (
    "run_a2d_pgv2_h96_rank2_polewise_k3_q4_nopg_short_pole_termmax20_rawq_imagenet100"
)


def dependency_paths(repo: Path, roots: Sequence[str]) -> tuple[Path, ...]:
    scripts = repo / "scripts"
    modules = {path.stem: path for path in scripts.glob("*.py")}
    pending = list(roots)
    selected: set[Path] = set()
    while pending:
        name = pending.pop()
        try:
            path = modules[name]
        except KeyError as error:
            raise RuntimeError(f"missing local script dependency: {name}") from error
        if path in selected:
            continue
        selected.add(path)
        if name == "a2d_r2k3_runtime":
            pending.append(_LEGACY_TRAINING_ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.append(node.module.split(".", maxsplit=1)[0])
            pending.extend(name for name in imported if name in modules)

    paths = [*sorted((repo / "src/lnet").glob("*.py")), *sorted(selected)]
    paths.extend(path for name in ("pyproject.toml", "uv.lock") if (path := repo / name).is_file())
    return tuple(dict.fromkeys(paths))


def fingerprint(repo: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = ["dependency_paths", "fingerprint"]
