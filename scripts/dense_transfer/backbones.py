"""Four-scale, checkpoint-exact feature adapters; no pretrained downloads.

VA's terminal Raw-Q aggregation and classifier are classification readouts,
not spatial feature blocks. They are explicitly excluded from the dense model.
The spatial stem, scan/coarsening stages, and all same-resolution blocks remain.
TinyViM uses the existing classifier's four stage endpoints (no new BatchNorm).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


EXPECTED_PARAMETERS = {"va_k128": 5_083_176, "convnextv2_atto": 3_708_400, "tinyvim_s": 5_684_084}


def _checkpoint(path: Path, key: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
    # These are user-owned checkpoints from the already completed A runs.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("completed_epochs") != 100:
        raise ValueError("Expected a completed epoch-100 ImageNet checkpoint")
    if payload.get("parameters") != EXPECTED_PARAMETERS[key]:
        raise ValueError(f"Checkpoint parameter identity does not match {key}")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state")
    # Training-only torch.compile wrappers do not belong to the model identity.
    clean = {name.removeprefix("_orig_mod."): value for name, value in state.items()}
    if key in ("va_k128", "tinyvim_s"):
        if not all(name.startswith("model.") for name in clean):
            raise ValueError("Expected the original primary-logits wrapper")
        clean = {name[len("model."):].removeprefix("_orig_mod."): value for name, value in clean.items()}
    if len(clean) != len(state):
        raise ValueError("Checkpoint key normalization produced duplicate keys")
    return clean, {k: payload.get(k) for k in ("completed_epochs", "global_step", "parameters", "contract_sha256", "source_sha256")}


def _build_va_classifier() -> nn.Module:
    import a2d_r2k3_runtime as runtime
    import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity
    import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

    spec = allocation.StageAllocationSpec(
        excitation_modes=(128, 128, 128, 128),
        pole_modes=(160, 160, 160, 128),
        extra_blocks=(0, 0, 4, 0),
        family="dense_transfer_frozen_k128",
    )
    model = capacity._build_all_resolution(spec.as_insight_spec(), runtime.model_config(output_dim=1000))
    model = allocation._append_repeated_blocks(model, spec)
    # Matches the original A training parameter representation; not a new recipe.
    return model.prepare_for_compiled_training_()


class VASpatialBackbone(nn.Module):
    feature_channels = (256, 256, 256, 256)
    feature_strides = (4, 8, 16, 32)
    omitted_readouts = ("classifier", "terminal_raw_q")

    def __init__(self, classifier: nn.Module, *, checkpoint_blocks: bool = False) -> None:
        super().__init__()
        # Keep the original object/methods and discard only classification readouts.
        classifier.classifier = nn.Identity()
        classifier.terminal = nn.Identity()
        self.body = classifier
        self.checkpoint_blocks = checkpoint_blocks

    def _at(self, resolution: int, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return activation_checkpoint(
                lambda real, imag: self.body._apply_at(resolution, (real, imag)),
                *state, use_reentrant=False,
            )
        return self.body._apply_at(resolution, state)

    @staticmethod
    def _real(state: tuple[Tensor, Tensor]) -> Tensor:
        # Internal VA layout is BHWC. Concatenation retains phase information.
        return torch.cat(state, dim=-1).permute(0, 3, 1, 2).contiguous()

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected BCHW RGB images")
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError("VA dense inputs must be padded to a multiple of 32")
        state = self._at(56, self.body._initial_excitation(images))
        outputs = OrderedDict({"0": self._real(state)})
        for index, (stage, resolution) in enumerate(
            ((self.body.stage1, 28), (self.body.stage2, 14), (self.body.stage3, 7)), 1
        ):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                state = activation_checkpoint(
                    lambda real, imag, module=stage: module(real, imag)[0],
                    *state, use_reentrant=False,
                )
            else:
                state, _unused_global_descriptor = stage(*state)
            if state is None:
                raise RuntimeError("Spatial stage unexpectedly returned only a readout")
            state = self._at(resolution, state)
            outputs[str(index)] = self._real(state)
        return outputs


class ConvNeXtSpatialBackbone(nn.Module):
    feature_strides = (4, 8, 16, 32)

    def __init__(self, classifier: nn.Module, *, checkpoint_blocks: bool = False) -> None:
        super().__init__()
        self.stem = classifier.stem
        self.stages = classifier.stages
        self.feature_channels = tuple(item["num_chs"] for item in classifier.feature_info)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        value = self.stem(images)
        outputs = OrderedDict()
        for index, stage in enumerate(self.stages):
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                value = activation_checkpoint(stage, value, use_reentrant=False)
            else:
                value = stage(value)
            outputs[str(index)] = value
        return outputs


class TinyViMSpatialBackbone(nn.Module):
    feature_channels = (48, 64, 168, 224)
    feature_strides = (4, 8, 16, 32)

    def __init__(self, classifier: nn.Module, *, checkpoint_blocks: bool = False) -> None:
        super().__init__()
        self.patch_embed = classifier.patch_embed
        self.network = classifier.network
        # Preserve the pretrained final normalization instead of random output BNs.
        self.final_norm = classifier.norm
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        if self.checkpoint_blocks and self.training:
            raise ValueError("TinyViM activation checkpointing is disabled: recomputation changes BatchNorm statistics")
        value = self.patch_embed(images)
        outputs = OrderedDict()
        for index, block in enumerate(self.network):
            value = block(value)
            if index in (0, 2, 4, 6):
                outputs[str(index // 2)] = self.final_norm(value) if index == 6 else value
        if len(outputs) != 4:
            raise RuntimeError("TinyViM did not produce four stage endpoints")
        return outputs


def _load_native_scan(source_root: Path) -> dict[str, str]:
    existing = importlib.util.find_spec("selective_scan_cuda")
    if existing is not None and existing.origin is not None:
        library = Path(existing.origin)
        with library.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"filename": library.name, "sha256": digest}
    native = source_root.parent / "native"
    libraries = sorted(native.glob("selective_scan_cuda*.so"))
    if len(libraries) != 1:
        raise RuntimeError("TinyViM requires the existing selective_scan_cuda library in assets/native or the environment")
    spec = importlib.util.spec_from_file_location("selective_scan_cuda", libraries[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned selective-scan extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["selective_scan_cuda"] = module
    with libraries[0].open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"filename": libraries[0].name, "sha256": digest}


def source_identity(source_root: Path) -> dict[str, str]:
    checkout = source_root / "tinyvim"
    paths = sorted(checkout.rglob("*.py"))
    if not paths:
        raise FileNotFoundError(f"No TinyViM Python sources in {checkout}")
    digest = hashlib.sha256()
    for path in paths:
        if "__pycache__" not in path.parts:
            digest.update(path.relative_to(checkout).as_posix().encode() + b"\0" + path.read_bytes())
    return {"kind": "frozen_runtime_python_tree", "sha256": digest.hexdigest()}


def build_backbone(
    model_key: str, checkpoint: Path, source_root: Path | None = None,
    *, checkpoint_blocks: bool = False,
) -> nn.Module:
    if model_key not in EXPECTED_PARAMETERS:
        raise ValueError(f"Unsupported backbone: {model_key}")
    state, provenance = _checkpoint(Path(checkpoint), model_key)
    if model_key == "va_k128":
        model = _build_va_classifier()
        kernel = Path(__file__).resolve().parents[2] / "src/lnet/pac_triton_product_scan_coarse4.py"
        provenance["dense_kernel_sha256"] = hashlib.sha256(kernel.read_bytes()).hexdigest()
        provenance["dense_launch_autotune_disabled"] = os.environ.get("LNET_DISABLE_LAUNCH_AUTOTUNE") == "1"
        adapter = VASpatialBackbone
    elif model_key == "convnextv2_atto":
        import timm
        model = timm.create_model("convnextv2_atto", pretrained=False, num_classes=1000)
        adapter = ConvNeXtSpatialBackbone
    else:
        if source_root is None:
            raise ValueError("TinyViM needs --source-root pointing to the staged frozen sources")
        source_root = Path(source_root)
        provenance["external_source"] = source_identity(source_root)
        provenance["native_extension"] = _load_native_scan(source_root)
        import h200_external_models as external
        # Same classifier builder used by A, with a frozen tree rather than a .git checkout.
        model = external._build_model("tinyvim_s", source_root / "tinyvim", 1000)
        adapter = TinyViMSpatialBackbone
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS[model_key]:
        raise ValueError(f"Classifier parameter count changed for {model_key}: {count}")
    model.load_state_dict(state, strict=True)
    backbone = adapter(model, checkpoint_blocks=checkpoint_blocks)
    backbone.pretrained_provenance = provenance
    backbone.classifier_parameters = count
    return backbone
