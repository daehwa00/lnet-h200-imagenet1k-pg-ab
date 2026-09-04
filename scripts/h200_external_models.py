"""Adapters for pinned external ImageNet-1K baseline implementations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "h200" / "baselines" / "sources.json"
DEFAULT_SOURCE_ROOT = Path("/app/scratch/input/lnet-h200-baseline-sources")

MODEL_SOURCES = {
    "parc_net_xs": "parc_net",
    "parc_net_s": "parc_net",
    "sret_tiny": "sret",
    "moganet_xt": "moganet",
    "uniconvnet_a": "uniconvnet",
    "efficientmod_xxs": "efficientmod",
    "emov2_1m": "emov2",
    "emov2_2m": "emov2",
    "tinynext_t": "tinynext",
    "tinynext_s": "tinynext",
    "tinynext_m": "tinynext",
    "tinyvim_s": "tinyvim",
    "efficientvim_m1": "efficientvim",
    "mambaout_femto": "mambaout",
}

_IMPORT_LOCK = threading.RLock()
_MODULE_CACHE: dict[tuple[Path, str], types.ModuleType] = {}
_DCNV3_PROBES: set[tuple[Path, str, str | None]] = set()
EXPECTED_MODEL_PARAMETERS = {
    "tinyvim_s": 5_684_084,
    "efficientvim_m1": 6_679_458,
    "mambaout_femto": 7_304_536,
}


class _LogitsAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return _extract_logits(self.model(*args, **kwargs))


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("logits", "out", "pred", "output"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        tensors = [value for value in output if isinstance(value, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        if len(tensors) > 1:
            msg = "external model returned ambiguous auxiliary logits"
            raise TypeError(msg)
    msg = f"external model returned unsupported output type: {type(output).__name__}"
    raise TypeError(msg)


def _sources() -> dict[str, dict[str, Any]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "lnet.h200.imagenet1k.external_sources.v1":
        msg = f"unsupported external-source manifest: {MANIFEST_PATH}"
        raise TypeError(msg)
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        msg = f"invalid external-source manifest: {MANIFEST_PATH}"
        raise TypeError(msg)
    return sources


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [shutil.which("git") or "git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_checkout(key: str, source_root: str | Path | None) -> tuple[Path, str]:
    if key not in MODEL_SOURCES:
        msg = f"unknown external model {key!r}; supported={sorted(MODEL_SOURCES)}"
        raise ValueError(msg)
    root_value = source_root or os.environ.get("H200_BASELINE_SOURCE_ROOT") or DEFAULT_SOURCE_ROOT
    root = Path(root_value).expanduser().resolve()
    source_name = MODEL_SOURCES[key]
    source = _sources()[source_name]
    if (
        source.get("license") == "NOASSERTION"
        and os.environ.get("H200_ALLOW_NOASSERTION_SOURCES") != "research-only"
    ):
        msg = f"{key!r} requires explicit research-only NOASSERTION opt-in"
        raise RuntimeError(msg)
    checkout = root / source_name
    if not (checkout / ".git").is_dir():
        msg = (
            f"missing external source checkout for {key!r}: {checkout}; run "
            f"scripts/bootstrap_h200_baseline_sources.py --source-root {root}"
        )
        raise RuntimeError(msg)
    expected_commit = str(source["commit"])
    if _git(checkout, "rev-parse", "--verify", "HEAD") != expected_commit:
        msg = f"external source commit mismatch for {key!r}: {checkout}"
        raise RuntimeError(msg)
    if _git(checkout, "config", "--get", "remote.origin.url") != source["repository"]:
        msg = f"external source remote mismatch for {key!r}: {checkout}"
        raise RuntimeError(msg)
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        msg = f"external source checkout is dirty for {key!r}: {checkout}"
        raise RuntimeError(msg)
    return checkout, source_name


@contextlib.contextmanager
def _compat_import_context(
    checkout: Path,
    prefixes: tuple[str, ...],
    *,
    stub_cv2: bool = False,
) -> Iterator[None]:
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for name in saved:
        del sys.modules[name]
    old_path = list(sys.path)
    old_dont_write_bytecode = sys.dont_write_bytecode
    registry_module: types.ModuleType | None = None
    original_register: Callable[..., object] | None = None
    try:
        sys.path.insert(0, str(checkout))
        sys.dont_write_bytecode = True
        if stub_cv2 and importlib.util.find_spec("cv2") is None:
            sys.modules["cv2"] = types.ModuleType("cv2")
        try:
            registry_module = importlib.import_module("timm.models.registry")
            original_register = registry_module.register_model
            registry_module.register_model = lambda function: function
        except (ImportError, AttributeError):
            registry_module = None
        yield
    finally:
        if registry_module is not None and original_register is not None:
            registry_module.register_model = original_register
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_dont_write_bytecode


def _external_module(
    checkout: Path,
    module_name: str,
    prefixes: tuple[str, ...],
    *,
    stub_cv2: bool = False,
) -> types.ModuleType:
    cache_key = (checkout, module_name)
    with _IMPORT_LOCK:
        cached = _MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            with (
                _compat_import_context(checkout, prefixes, stub_cv2=stub_cv2),
                contextlib.redirect_stdout(sys.stderr),
            ):
                module = importlib.import_module(module_name)
        except Exception as exc:
            msg = f"failed to import pinned external module {module_name!r} from {checkout}: {exc}"
            raise RuntimeError(msg) from exc
        _MODULE_CACHE[cache_key] = module
        return module


def _parc_options(key: str, num_classes: int) -> argparse.Namespace:
    options = argparse.Namespace()
    values: dict[str, object] = {
        "model.classification.n_classes": num_classes,
        "model.classification.classifier_dropout": 0.2,
        "model.layer.global_pool": "mean",
        "model.classification.edge.scale": "scale_xs" if key == "parc_net_xs" else "scale_s",
        "model.classification.edge.mode": "outer_frame_v1",
        "model.classification.edge.kernel": "gcc_ca",
        "model.classification.edge.fusion": "add" if key == "parc_net_xs" else "concat",
        "model.classification.edge.instance_kernel": "interpolation_bilinear",
        "model.classification.edge.mid_mix": False,
        "model.classification.edge.use_pe": True,
        "model.activation.name": "swish",
        "model.activation.inplace": False,
        "model.activation.neg_slope": 0.1,
        "model.normalization.name": "batch_norm_2d",
        "model.normalization.momentum": 0.1,
        "model.normalization.groups": 1,
        "model.layer.conv_init": "kaiming_normal",
        "model.layer.linear_init": "trunc_normal",
        "model.layer.linear_init_std_dev": 0.02,
    }
    for name, value in values.items():
        setattr(options, name, value)
    return options


def _probe_uniconv_dcnv3(checkout: Path) -> None:
    cache_key = (checkout, sys.executable, torch.version.cuda)
    if cache_key in _DCNV3_PROBES:
        return
    probe = r"""
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable')
from ops_dcnv3.modules import DCNv3
model = DCNv3(channels=32, kernel_size=3, stride=1, pad=1, group=4).cuda()
x = torch.randn(2, 8, 8, 32, device='cuda', requires_grad=True)
y = model(x)
if y.shape != x.shape:
    raise SystemExit(f'DCNv3 shape mismatch: {y.shape} != {x.shape}')
y.square().mean().backward()
torch.cuda.synchronize()
"""
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-B", "-c", probe, str(checkout)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        msg = "UniConvNet-A DCNv3 isolated CUDA probe timed out"
        raise RuntimeError(msg) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        msg = f"UniConvNet-A is disabled: DCNv3 isolated CUDA probe failed: {detail}"
        raise RuntimeError(msg)
    _DCNV3_PROBES.add(cache_key)


def _build_model(key: str, checkout: Path, num_classes: int) -> nn.Module:  # noqa: PLR0911
    if key.startswith("parc_net_"):
        module = _external_module(
            checkout,
            "cvnets.models.classification.edgeformer",
            ("cvnets", "utils"),
        )
        return module.edgeformer(_parc_options(key, num_classes))
    if key == "sret_tiny":
        module = _external_module(checkout, "SReT", ("SReT",))
        return module.SReT_T(pretrained=False, num_classes=num_classes)
    if key == "moganet_xt":
        module = _external_module(checkout, "models.moganet", ("models",))
        return module.moganet_xtiny(pretrained=False, num_classes=num_classes)
    if key == "uniconvnet_a":
        _probe_uniconv_dcnv3(checkout)
        module = _external_module(
            checkout,
            "models.UniConvNet",
            ("models", "ops_dcnv3"),
        )
        return module.UniConvNet_A(pretrained=False, num_classes=num_classes)
    if key == "efficientmod_xxs":
        module = _external_module(checkout, "models.EfficientMod", ("models",))
        model = module.efficientMod_xxs(pretrained=False)
        model.num_classes = num_classes
        model.head = nn.Linear(model.num_features, num_classes)
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)
        return model
    if key.startswith("emov2_"):
        module = _external_module(
            checkout,
            "model.lib_emo.emo2",
            ("model", "util", "cv2"),
            stub_cv2=True,
        )
        constructor = module.EMO2_1M_k5_hybrid if key == "emov2_1m" else module.EMO2_2M_k5_hybrid
        return constructor(pretrained=False, num_classes=num_classes)
    if key.startswith("tinynext_"):
        classification = checkout / "classification"
        module = _external_module(classification, "models.tinynext", ("models",))
        constructor = getattr(module, key)
        return constructor(pretrained=False, num_classes=num_classes, distillation=False)
    if key == "tinyvim_s":
        if importlib.util.find_spec("selective_scan_cuda") is None:
            msg = "TinyViM-S requires the pinned selective_scan_cuda compatibility wheel"
            raise RuntimeError(msg)
        helper_name = "timm.models.layers.helpers"
        saved_helper = sys.modules.get(helper_name)
        helper = types.ModuleType(helper_name)
        from timm.layers import to_2tuple

        helper.to_2tuple = to_2tuple  # type: ignore[attr-defined]
        sys.modules[helper_name] = helper
        try:
            module = _external_module(checkout, "model.tinyvim", ("model",))
        finally:
            if saved_helper is None:
                sys.modules.pop(helper_name, None)
            else:
                sys.modules[helper_name] = saved_helper
        return module.TinyViM_S(
            pretrained=False,
            num_classes=num_classes,
            distillation=False,
        )
    if key == "efficientvim_m1":
        classification = checkout / "classification"
        saved_fvcore = {name: sys.modules.get(name) for name in ("fvcore", "fvcore.nn")}
        fvcore = types.ModuleType("fvcore")
        fvcore_nn = types.ModuleType("fvcore.nn")
        fvcore_nn.flop_count = lambda *_args, **_kwargs: ({}, {})  # type: ignore[attr-defined]
        fvcore.nn = fvcore_nn  # type: ignore[attr-defined]
        sys.modules["fvcore"] = fvcore
        sys.modules["fvcore.nn"] = fvcore_nn
        try:
            module = _external_module(
                classification,
                "models.EfficientViM",
                ("models",),
            )
        finally:
            for name, saved in saved_fvcore.items():
                if saved is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = saved
        return module.EfficientViM_M1(
            pretrained=False,
            num_classes=num_classes,
            distillation=False,
        )
    if key == "mambaout_femto":
        module = _external_module(checkout, "models.mambaout", ("models",))
        return module.mambaout_femto(pretrained=False, num_classes=num_classes)
    msg = f"no external model builder for {key!r}"
    raise ValueError(msg)


def build_external_model(
    key: str,
    source_root: str | Path | None,
    num_classes: int,
) -> torch.nn.Module:
    """Build a classifier from an exact, clean official-source checkout."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        msg = f"num_classes must be a positive integer, got {num_classes!r}"
        raise ValueError(msg)
    checkout, _ = _resolve_checkout(key, source_root)
    model = _build_model(key, checkout, num_classes)
    if not isinstance(model, nn.Module):
        msg = f"external constructor for {key!r} did not return torch.nn.Module"
        raise TypeError(msg)
    expected_parameters = EXPECTED_MODEL_PARAMETERS.get(key)
    if expected_parameters is not None:
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        if actual_parameters != expected_parameters:
            msg = (
                f"parameter count changed for {key}: "
                f"{actual_parameters} != {expected_parameters}"
            )
            raise RuntimeError(msg)
    return _LogitsAdapter(model)


def external_source_provenance(
    key: str,
    source_root: str | Path | None,
) -> dict[str, object]:
    """Return verified immutable-source evidence for a campaign contract."""
    checkout, source_name = _resolve_checkout(key, source_root)
    source = _sources()[source_name]
    commit = _git(checkout, "rev-parse", "--verify", "HEAD")
    manifest_sha256 = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    return {
        "source": source_name,
        "repository": source["repository"],
        "expected_commit": source["commit"],
        "actual_commit": commit,
        "head_tree": _git(checkout, "rev-parse", "HEAD^{tree}"),
        "tree_clean": not bool(_git(checkout, "status", "--porcelain", "--untracked-files=all")),
        "license": source["license"],
        "redistribution_allowed": source["redistribution_allowed"],
        "manifest_sha256": manifest_sha256,
        "sources_manifest_sha256": manifest_sha256,
    }


external_model_provenance = external_source_provenance

__all__ = [
    "build_external_model",
    "external_model_provenance",
    "external_source_provenance",
]
