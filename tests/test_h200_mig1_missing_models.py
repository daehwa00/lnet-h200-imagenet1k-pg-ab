from __future__ import annotations

import json
from pathlib import Path

from scripts import h200_external_models as external
from scripts import run_imagenet1k_mig1_missing_queue as queue
from scripts import run_imagenet1k_mig1_tinynext_t as tinynext

ROOT = Path(__file__).resolve().parents[1]


def test_mig1_queue_is_exact_and_seed_major() -> None:
    assert queue.MODELS == ("tinyvim_s", "efficientvim_m1", "mambaout_femto")
    assert queue.SEEDS == (501, 509, 521)
    assert queue.LEARNING_RATE == 3.0e-3
    assert external.EXPECTED_MODEL_PARAMETERS == {
        "tinyvim_s": 5_684_084,
        "efficientvim_m1": 6_679_458,
        "mambaout_femto": 7_304_536,
    }


def test_mig1_sources_are_official_and_commit_pinned() -> None:
    sources = json.loads((ROOT / "h200/baselines/sources.json").read_text())["sources"]
    assert sources["tinyvim"]["repository"] == "https://github.com/xwmaxwma/TinyViM.git"
    assert sources["tinyvim"]["commit"] == "cff64524117b1439becf89827a3428a288677b95"
    assert sources["efficientvim"]["commit"] == "304340cb9c339b61669250d058525c9cdadd5e93"
    assert sources["mambaout"]["commit"] == "9f2f2343eb0f99f2cf3ba6b92290b5a81be2bad1"


def test_mig1_entrypoint_is_isolated_and_memory_safe() -> None:
    source = (ROOT / "h200/run_baselines.sh").read_text()
    assert "refs/heads/control/imagenet1k-mig1-missing" in source
    assert "H200_OUTPUT_USER:-daehwa00" in source
    assert "H200_MIG1_MISSING_ONLY" in source
    assert "run_imagenet1k_mig1_missing_queue.py" in source
    assert "--batch-size 64" in source
    assert "H200_GPU_MEMORY_FRACTION=1.0" in source
    assert "7201849146fb3b517e1a89741c4042596652dea24f44a94e4a83e6246353f49e" in source


def test_tinynext_first_lane_is_single_clean_seed() -> None:
    assert tinynext.MODEL == "tinynext_t_mig1_clean"
    assert tinynext.SEED == 521
    source = (ROOT / "h200/run_baselines.sh").read_text()
    assert "refs/heads/control/imagenet1k-mig1-tinynext" in source
    assert "H200_MIG1_TINYNEXT_ONLY" in source
    assert "run_imagenet1k_mig1_tinynext_t.py" in source
    assert "--batch-size 256" in source
