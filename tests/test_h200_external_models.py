from __future__ import annotations

import hashlib

# ruff: noqa: SLF001, S607
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

bootstrap = importlib.import_module("bootstrap_h200_baseline_sources")
external = importlib.import_module("h200_external_models")

EXPECTED_KEYS = {
    "parc_net_xs",
    "parc_net_s",
    "sret_tiny",
    "moganet_xt",
    "uniconvnet_a",
    "efficientmod_xxs",
    "emov2_1m",
    "emov2_2m",
    "tinynext_t",
    "tinynext_s",
    "tinynext_m",
    "tinyvim_s",
    "efficientvim_m1",
    "mambaout_femto",
}


class _OutputModel(nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    def forward(self, value: torch.Tensor) -> object:
        del value
        return self.output


def test_source_manifest_pins_all_external_models_and_license_policy() -> None:
    manifest = json.loads(
        (ROOT / "h200/baselines/sources.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "lnet.h200.imagenet1k.external_sources.v1"
    sources = manifest["sources"]
    model_keys = {key for source in sources.values() for key in source["models"]}
    assert model_keys == EXPECTED_KEYS
    for source in sources.values():
        assert source["repository"].startswith("https://github.com/")
        assert len(source["commit"]) == 40
        int(source["commit"], 16)
        if source["license"] == "NOASSERTION":
            assert source["license_file"] is None
            assert source["redistribution_allowed"] is False
    assert sources["parc_net"]["license"] == "NOASSERTION"
    assert sources["emov2"]["license"] == "NOASSERTION"
    patch_path = ROOT / sources["uniconvnet"]["compatibility_patch"]
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == sources["uniconvnet"][
        "compatibility_patch_sha256"
    ]


def test_logits_adapter_normalizes_tensor_mapping_and_distillation_tuple() -> None:
    value = torch.tensor([[1.0, 3.0]])
    input_tensor = torch.zeros(1)
    assert torch.equal(external._LogitsAdapter(_OutputModel(value))(input_tensor), value)
    assert torch.equal(
        external._LogitsAdapter(_OutputModel({"out": value}))(input_tensor),
        value,
    )
    with pytest.raises(TypeError, match="ambiguous auxiliary"):
        external._LogitsAdapter(_OutputModel((value, value + 2)))(input_tensor)


def test_build_rejects_unknown_key_and_invalid_class_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown external model"):
        external.build_external_model("not_a_model", tmp_path, 1000)
    with pytest.raises(ValueError, match="positive integer"):
        external.build_external_model("sret_tiny", tmp_path, 0)


def test_missing_checkout_has_bootstrap_instruction(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"bootstrap_h200_baseline_sources\.py"):
        external.build_external_model("sret_tiny", tmp_path, 1000)


def test_noassertion_sources_require_explicit_research_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("H200_ALLOW_NOASSERTION_SOURCES", raising=False)
    with pytest.raises(RuntimeError, match="research-only"):
        bootstrap.bootstrap_sources(tmp_path, selected={"parc_net"})
    with pytest.raises(RuntimeError, match="research-only"):
        external.build_external_model("parc_net_xs", tmp_path, 1000)


def test_bootstrap_continues_after_one_source_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("H200_ALLOW_NOASSERTION_SOURCES", "research-only")
    attempted: list[str] = []

    def checkout(
        _root: Path,
        name: str,
        _source: dict[str, object],
    ) -> Path:
        attempted.append(name)
        if name == "parc_net":
            message = "simulated source outage"
            raise OSError(message)
        return tmp_path / name

    monkeypatch.setattr(bootstrap, "_checkout_source", checkout)
    result = bootstrap.bootstrap_sources(tmp_path, selected={"parc_net", "sret"})

    assert attempted == ["parc_net", "sret"]
    assert result["failures"] == {"parc_net": "OSError"}
    assert result["checkouts"] == {"sret": str(tmp_path / "sret")}


def test_checkout_verifier_rejects_dirty_tree(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/example/source.git"],
        cwd=checkout,
        check=True,
    )
    (checkout / "LICENSE").write_text("test", encoding="utf-8")
    subprocess.run(["git", "add", "LICENSE"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=checkout,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = {
        "repository": "https://github.com/example/source.git",
        "commit": commit,
        "license_file": "LICENSE",
    }
    bootstrap._verify_checkout(checkout, source)
    (checkout / "untracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        bootstrap._verify_checkout(checkout, source)


def test_uniconv_fails_before_import_when_dcnv3_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        external,
        "_resolve_checkout",
        lambda _key, _source_root: (tmp_path, "uniconvnet"),
    )

    def fail_probe(checkout: Path) -> None:
        del checkout
        message = "DCNv3 isolated CUDA probe failed"
        raise RuntimeError(message)

    monkeypatch.setattr(external, "_probe_uniconv_dcnv3", fail_probe)
    with pytest.raises(RuntimeError, match="DCNv3 isolated CUDA probe failed"):
        external.build_external_model("uniconvnet_a", tmp_path, 1000)
