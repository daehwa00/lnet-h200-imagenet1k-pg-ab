from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
import json
import math
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMBlock,
    AlphabetLMConfig,
    DynamicLowRankWrite,
    FixedComplexPoleMemory1D,
    IdentityComplexMemory1D,
    LowRankDecaySelector,
    QueryConditionedLowRankReadout,
    TensorProductPoleMemory1D,
)
from lnet.alphabet_lm_mamba import MambaLMConfig, build_parameter_matched_mamba
from lnet.pac_triton_recurrence_op import pac_triton_recurrence_opaque_op
from scripts.prepare_h200_alphabet_lm_data import _parquet_to_jsonl, _split_documents
from scripts.train_h200_alphabet_lm_10m import _copy_matching_legacy_initialization


def _small() -> AlphabetLMConfig:
    return AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=2,
        post_hidden=12,
        context_length=16,
    )


def test_alphabet_lm_default_parameter_count() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(AlphabetLMConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_794_496


def test_alphabet_lm_is_causal_and_has_finite_gradients() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(_small())
    tokens = torch.randint(64, (1, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = model(tokens[:, :-1])
        actual = model(changed[:, :-1])
    assert torch.allclose(expected[:, :10], actual[:, :10], atol=1.0e-6, rtol=0.0)
    logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_lifetime_palette_spans_two_to_8192_with_decay_dominant_modes() -> None:
    config = AlphabetLMConfig(pole_initialization="lifetime_palette")
    model = AlphabetLM(config)
    memory = FixedComplexPoleMemory1D(
        320,
        context_length=2_048,
        scan_fp32=True,
        initialization="lifetime_palette",
    )
    half_lives = math.log(2.0) / memory.damping().detach()
    expected = 2.0 ** torch.arange(1, 14, dtype=torch.float32)
    torch.testing.assert_close(torch.unique(half_lives).sort().values, expected)
    assert int((memory.frequency().detach() == 0).sum()) == 160
    assert memory.raw_damping.requires_grad
    assert memory.raw_frequency.requires_grad
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_794_496


def test_grouped_h8p128_memory_contract() -> None:
    config = AlphabetLMConfig(
        pole_initialization="lifetime_palette",
        memory_banks=8,
        bank_pole_modes=128,
    )
    model = AlphabetLM(config)
    assert config.total_pole_modes == 1_024
    assert sum(parameter.numel() for parameter in model.parameters()) == 31_373_824
    reader_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("blocks.0.reader.")
    )
    writer_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("blocks.0.writer.")
    )
    assert reader_parameters == 143_616
    assert writer_parameters == 65_536
    tokens = torch.randint(config.vocab_size, (1, 9))
    changed = tokens.clone()
    changed[:, 6:] = torch.randint(config.vocab_size, changed[:, 6:].shape)
    with torch.no_grad():
        expected = model(tokens)
        actual = model(changed)
    torch.testing.assert_close(expected[:, :6], actual[:, :6], atol=1.0e-6, rtol=0.0)


def test_dense_k3_starts_from_the_same_function_as_r2k3() -> None:
    torch.manual_seed(501)
    factorized = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="r2k3",
        )
    )
    torch.manual_seed(501)
    dense = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
        )
    )
    tokens = torch.randint(64, (2, 9))
    with torch.no_grad():
        expected = factorized(tokens)
        actual = dense(tokens)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)
    full = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3"))
    assert sum(parameter.numel() for parameter in full.parameters()) == 36_714_496


def test_dynamic_pole_routers_are_neutral_at_initialization() -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(AlphabetLMConfig())
    tokens = torch.randint(32_768, (1, 9))
    with torch.no_grad():
        expected = baseline(tokens)
    for routing, parameters in (
        ("dynamic_write", 35_117_056),
        ("dynamic_write_read", 35_239_936),
    ):
        torch.manual_seed(501)
        model = AlphabetLM(
            AlphabetLMConfig(
                pole_routing=cast(
                    "Literal['static', 'dynamic_write', 'dynamic_write_read']", routing
                )
            )
        )
        with torch.no_grad():
            actual = model(tokens)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        assert sum(parameter.numel() for parameter in model.parameters()) == parameters


def test_query_conditioned_low_rank_readout_is_small_nonzero_and_content_dependent() -> None:
    torch.manual_seed(501)
    readout = QueryConditionedLowRankReadout(
        8,
        12,
        rank=4,
        initial_scale=0.05,
    )
    state_real = torch.randn(2, 5, 12)
    state_imag = torch.randn(2, 5, 12)
    base_real = torch.randn(2, 5, 8)
    base_imag = torch.randn(2, 5, 8)
    query_real = torch.randn(2, 5, 8)
    query_imag = torch.randn(2, 5, 8)
    first = readout(
        query_real,
        query_imag,
        state_real,
        state_imag,
        base_real,
        base_imag,
    )
    second = readout(
        -query_real,
        query_imag,
        state_real,
        state_imag,
        base_real,
        base_imag,
    )
    delta = torch.cat((first[0] - base_real, first[1] - base_imag), dim=-1)
    base = torch.cat((base_real, base_imag), dim=-1)
    ratio = (delta.square().mean().sqrt() / base.square().mean().sqrt()).detach()
    assert 0.0 < float(ratio) < 0.2
    assert not torch.allclose(first[0], second[0])
    torch.testing.assert_close(readout.scale(), torch.tensor(0.05))


def test_query_read_r32_model_contract_and_gradients() -> None:
    full = AlphabetLM(AlphabetLMConfig(memory_readout="query_low_rank", query_read_rank=32))
    assert sum(parameter.numel() for parameter in full.parameters()) == 35_436_556
    torch.manual_seed(501)
    model = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            memory_readout="query_low_rank",
            query_read_rank=4,
        )
    )
    tokens = torch.randint(64, (2, 17))
    logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_dynamic_damping_zero_control_matches_fixed_memory_exactly() -> None:
    torch.manual_seed(501)
    memory = FixedComplexPoleMemory1D(12, context_length=16, scan_fp32=True)
    drive_real = torch.randn(2, 9, 12)
    drive_imag = torch.randn(2, 9, 12)
    control = torch.zeros_like(drive_real, requires_grad=True)
    fixed = memory(drive_real, drive_imag)
    dynamic = memory(drive_real, drive_imag, control)
    torch.testing.assert_close(dynamic[0], fixed[0], atol=3.0e-7, rtol=4.0e-6)
    torch.testing.assert_close(dynamic[1], fixed[1], atol=3.0e-7, rtol=4.0e-6)
    dynamic[0].square().mean().add(dynamic[1].square().mean()).backward()
    assert control.grad is not None
    assert bool(torch.isfinite(control.grad).all())
    assert memory.raw_damping.grad is not None


def test_delta_select_r16_dense_model_contract_and_gradients() -> None:
    full = AlphabetLM(
        AlphabetLMConfig(
            reader_type="dense_k3",
            pole_dynamics="delta_select",
            delta_select_rank=16,
            delta_select_initial_scale=0.3,
        )
    )
    assert sum(parameter.numel() for parameter in full.parameters()) == 36_877_324
    torch.manual_seed(501)
    config = AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=2,
        post_hidden=12,
        context_length=16,
        reader_type="dense_k3",
        pole_dynamics="delta_select",
        delta_select_rank=4,
        delta_select_initial_scale=0.3,
    )
    model = AlphabetLM(config)
    tokens = torch.randint(64, (2, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = model(tokens[:, :-1])
        actual = model(changed[:, :-1])
    torch.testing.assert_close(actual[:, :10], expected[:, :10], atol=1.0e-6, rtol=0.0)
    logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    selectors = [module for module in model.modules() if isinstance(module, LowRankDecaySelector)]
    assert len(selectors) == 2
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for selector in selectors
        for parameter in selector.parameters()
    )


def test_dense_delta_pair_copies_every_fixed_dense_parameter() -> None:
    torch.manual_seed(501)
    fixed = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3"))
    fixed_parameters = dict(fixed.named_parameters())
    torch.manual_seed(501)
    dynamic = AlphabetLM(
        AlphabetLMConfig(
            reader_type="dense_k3",
            pole_dynamics="delta_select",
            delta_select_rank=16,
            delta_select_initial_scale=0.3,
        )
    )
    _tensors, copied = _copy_matching_legacy_initialization(
        dynamic,
        vocab_size=32_768,
        seed=501,
        reader_type="dense_k3",
    )
    assert copied == 36_714_496
    for name, parameter in dynamic.named_parameters():
        source = fixed_parameters.get(name)
        if source is not None and source.shape == parameter.shape:
            torch.testing.assert_close(parameter, source, atol=0.0, rtol=0.0)


def test_decoder_readout_variants_copy_every_shape_compatible_legacy_parameter() -> None:
    torch.manual_seed(501)
    reference = AlphabetLM(AlphabetLMConfig())
    reference_parameters = dict(reference.named_parameters())
    for config, expected_copied in (
        (AlphabetLMConfig(post_hidden=512), 22_998_016),
        (
            AlphabetLMConfig(
                memory_readout="query_low_rank",
                query_read_rank=32,
                query_read_initial_scale=0.15,
            ),
            34_794_496,
        ),
    ):
        torch.manual_seed(501)
        model = AlphabetLM(config)
        if config.post_hidden == 512:
            assert sum(parameter.numel() for parameter in model.parameters()) == 38_726_656
        _tensors, copied = _copy_matching_legacy_initialization(
            model,
            vocab_size=32_768,
            seed=501,
        )
        assert copied == expected_copied
        for name, parameter in model.named_parameters():
            source = reference_parameters.get(name)
            if source is not None and source.shape == parameter.shape:
                torch.testing.assert_close(parameter, source, atol=0.0, rtol=0.0)


def test_tensor_product_memory_matches_complex_reference() -> None:
    torch.manual_seed(501)
    memory = TensorProductPoleMemory1D(
        5,
        4,
        half_lives=(4.0, 16.0, 64.0, 256.0),
        scan_fp32=True,
        initial_read_gain=0.6,
    )
    drive_real = torch.randn(2, 9, 5)
    drive_imag = torch.randn(2, 9, 5)
    actual_real, actual_imag = memory(drive_real, drive_imag)
    drive = torch.complex(drive_real, drive_imag)
    write = torch.complex(memory.write_real, memory.write_imag)
    read = torch.complex(memory.read_real, memory.read_imag)
    pole = torch.complex(-memory.damping(), memory.frequency())
    decay = torch.exp(pole)
    gamma = torch.expm1(pole) / pole
    state = torch.zeros(2, 5, 4, dtype=torch.complex64)
    expected = []
    for current in drive.unbind(dim=1):
        state = decay * state + gamma * write * current.unsqueeze(-1)
        expected.append((read * state).sum(dim=-1))
    reference = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual_real, reference.real, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(actual_imag, reference.imag, atol=2.0e-5, rtol=2.0e-5)


def test_tensorpole_m8_model_contract_causality_and_gradients() -> None:
    config = AlphabetLMConfig(reader_type="dense_k3", memory_layout="tensor_product")
    full = AlphabetLM(config)
    assert config.recurrent_state_modes == 2_048
    assert sum(parameter.numel() for parameter in full.parameters()) == 33_659_584
    memory = full.blocks[0].memory
    assert isinstance(memory, TensorProductPoleMemory1D)
    expected_half_lives = torch.tensor(config.tensor_half_lives)
    torch.testing.assert_close(memory.half_lives().detach(), expected_half_lives)
    row_energy = memory.write_real.square().add(memory.write_imag.square()).sum(dim=-1)
    torch.testing.assert_close(row_energy, torch.ones_like(row_energy))
    torch.testing.assert_close(memory.read_real, 0.6 * memory.write_real)
    torch.testing.assert_close(memory.read_imag, -0.6 * memory.write_imag)

    torch.manual_seed(501)
    small = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
            memory_layout="tensor_product",
            tensor_temporal_modes=4,
            tensor_half_lives=(4.0, 16.0, 64.0, 256.0),
        )
    )
    tokens = torch.randint(64, (2, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = small(tokens[:, :-1])
        actual = small(changed[:, :-1])
    torch.testing.assert_close(actual[:, :10], expected[:, :10], atol=1.0e-6, rtol=0.0)
    logits = small(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in small.parameters()
    )


def test_tensorpole_initial_memory_scale_matches_dense_fixed_at_full_context() -> None:
    torch.manual_seed(501)
    real = torch.randn(1, 2_048, 256)
    imag = torch.randn_like(real)
    ratios = []
    for layout in ("flat", "tensor_product"):
        torch.manual_seed(501)
        block = cast(
            "AlphabetLMBlock",
            AlphabetLM(AlphabetLMConfig(reader_type="dense_k3", memory_layout=layout)).blocks[0],
        )
        with torch.no_grad():
            drive = block.reader(real, imag)
            state = block.memory(*drive)
            memory = block.writer(*state)
        input_rms = real.square().add(imag.square()).mean().sqrt()
        memory_rms = memory[0].square().add(memory[1].square()).mean().sqrt()
        ratios.append(float(memory_rms / input_rms))
    assert 0.9 < ratios[1] / ratios[0] < 1.1


def test_dynamic_low_rank_write_changes_state_direction_with_content() -> None:
    torch.manual_seed(501)
    write = DynamicLowRankWrite(8, 12, rank=4, initial_scale=0.06)
    real = torch.randn(2, 7, 8)
    imag = torch.randn_like(real)
    base_real = torch.randn(2, 7, 12)
    base_imag = torch.randn_like(base_real)
    first = write(real, imag, base_real, base_imag)
    second = write(-real, imag, base_real, base_imag)
    assert not torch.allclose(first[0], second[0])
    first_delta = torch.cat((first[0] - base_real, first[1] - base_imag), dim=-1)
    second_delta = torch.cat((second[0] - base_real, second[1] - base_imag), dim=-1)
    assert float(first_delta.detach().square().mean().sqrt()) > 0.0
    cosine = torch.nn.functional.cosine_similarity(
        first_delta.flatten(),
        second_delta.flatten(),
        dim=0,
    )
    assert abs(float(cosine.detach())) < 0.999
    torch.testing.assert_close(write.scale(), torch.tensor(0.06))


def test_dense_dynamic_write_r4_contract_causality_and_gradients() -> None:
    config = AlphabetLMConfig(
        reader_type="dense_k3",
        write_map="dynamic_low_rank",
        dynamic_write_rank=4,
        dynamic_write_initial_scale=0.06,
    )
    full = AlphabetLM(config)
    assert sum(parameter.numel() for parameter in full.parameters()) == 36_797_452
    torch.manual_seed(501)
    small = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
            write_map="dynamic_low_rank",
            dynamic_write_rank=4,
            dynamic_write_initial_scale=0.06,
        )
    )
    tokens = torch.randint(64, (2, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = small(tokens[:, :-1])
        actual = small(changed[:, :-1])
    torch.testing.assert_close(actual[:, :10], expected[:, :10], atol=1.0e-6, rtol=0.0)
    logits = small(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    writes = [module for module in small.modules() if isinstance(module, DynamicLowRankWrite)]
    assert len(writes) == 2
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for write in writes
        for parameter in write.parameters()
    )


def test_dynamic_write_pair_copies_every_fixed_dense_parameter() -> None:
    torch.manual_seed(501)
    fixed = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3"))
    fixed_parameters = dict(fixed.named_parameters())
    torch.manual_seed(501)
    dynamic = AlphabetLM(
        AlphabetLMConfig(
            reader_type="dense_k3",
            write_map="dynamic_low_rank",
            dynamic_write_rank=4,
            dynamic_write_initial_scale=0.06,
        )
    )
    _tensors, copied = _copy_matching_legacy_initialization(
        dynamic,
        vocab_size=32_768,
        seed=501,
        reader_type="dense_k3",
    )
    assert copied == 36_714_496
    for name, parameter in dynamic.named_parameters():
        source = fixed_parameters.get(name)
        if source is not None and source.shape == parameter.shape:
            torch.testing.assert_close(parameter, source, atol=0.0, rtol=0.0)


def test_dynamic_write_initial_branch_is_ten_percent_of_static_drive() -> None:
    torch.manual_seed(501)
    block = cast(
        "AlphabetLMBlock",
        AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                write_map="dynamic_low_rank",
                dynamic_write_rank=4,
                dynamic_write_initial_scale=0.06,
            )
        ).blocks[0],
    )
    real = torch.randn(1, 2_048, 256)
    imag = torch.randn_like(real)
    dynamic_write = block.dynamic_write
    assert isinstance(dynamic_write, DynamicLowRankWrite)
    with torch.no_grad():
        base = block.reader(real, imag)
        routed = dynamic_write(real, imag, *base)
    branch = torch.cat((routed[0] - base[0], routed[1] - base[1]), dim=-1)
    static = torch.cat(base, dim=-1)
    ratio = branch.square().mean().sqrt() / static.square().mean().sqrt()
    assert 0.08 < float(ratio) < 0.12


def test_dense_local_only_removes_only_recurrent_transport() -> None:
    torch.manual_seed(501)
    fixed = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3"))
    fixed_parameters = dict(fixed.named_parameters())
    torch.manual_seed(501)
    local = AlphabetLM(
        AlphabetLMConfig(reader_type="dense_k3", memory_layout="local_only")
    )
    assert sum(parameter.numel() for parameter in local.parameters()) == 36_706_816
    identity_memories = [
        module for module in local.modules() if isinstance(module, IdentityComplexMemory1D)
    ]
    assert len(identity_memories) == 12
    assert not any(isinstance(module, FixedComplexPoleMemory1D) for module in local.modules())
    for name, parameter in local.named_parameters():
        torch.testing.assert_close(parameter, fixed_parameters[name], atol=0.0, rtol=0.0)

    block = cast("AlphabetLMBlock", local.blocks[0])
    real = torch.randn(2, 17, 256)
    imag = torch.randn_like(real)
    with torch.no_grad():
        drive = block.reader(real, imag)
        state = block.memory(*drive)
    torch.testing.assert_close(state[0], drive[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(state[1], drive[1], atol=0.0, rtol=0.0)


def test_dense_local_only_is_causal_and_has_finite_gradients() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
            memory_layout="local_only",
        )
    )
    tokens = torch.randint(64, (2, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = model(tokens[:, :-1])
        actual = model(changed[:, :-1])
    torch.testing.assert_close(actual[:, :10], expected[:, :10], atol=1.0e-6, rtol=0.0)
    logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_h200_mamba_runtime_dependency_contract_is_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "h200/alphabet_lm_preflight/requirements.txt").read_text(
        encoding="utf-8"
    )
    lock = (root / "h200/alphabet_lm_preflight/requirements.lock").read_text(encoding="utf-8")
    for requirement in (
        "einops==0.8.1",
        "ninja==1.13.0",
        "packaging==26.3",
        "pyarrow==23.0.1",
        "setuptools==84.0.0",
        "torch==2.9.1+cu130",
        "transformers==4.57.1",
        "triton==3.5.1",
        "wandb==0.22.3",
        "wheel==0.46.2",
    ):
        assert requirement in requirements
        assert requirement in lock
    for transitive in (
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
        "tokenizers==0.22.2",
    ):
        assert transitive in lock
    assert "transformers==5." not in requirements
    assert "transformers==5." not in lock


def test_parameter_matched_mamba_uses_official_lm_initialization() -> None:
    torch.manual_seed(501)
    model, parameters, relative_error = build_parameter_matched_mamba(34_794_496, MambaLMConfig())
    assert parameters == 35_425_280
    assert relative_error < 0.03
    state = model.model.state_dict()
    embedding = state["backbone.embedding.weight"]
    lm_head = state["lm_head.weight"]
    assert embedding.data_ptr() == lm_head.data_ptr()
    torch.testing.assert_close(embedding.std(), torch.tensor(0.02), atol=2.0e-4, rtol=0)
    out_projections = [
        parameter
        for name, parameter in model.model.named_parameters()
        if name.endswith("mixer.out_proj.weight")
    ]
    assert len(out_projections) == 11
    for projection in out_projections:
        std = projection.detach().std().item()
        assert 0.0052 < std < 0.0057


def test_opaque_recurrence_compiles_time_major_noncontiguous_input() -> None:
    batch, steps, modes = 2, 7, 5

    def inputs() -> tuple[torch.Tensor, torch.Tensor]:
        real = torch.randn(batch, modes, steps).transpose(1, 2).detach().requires_grad_()
        imag = torch.randn(batch, modes, steps).transpose(1, 2).detach().requires_grad_()
        assert real.stride() == (steps * modes, 1, steps)
        return real, imag

    def target(input_real: torch.Tensor, input_imag: torch.Tensor) -> torch.Tensor:
        shape = (1, 1, modes)
        decay_real = torch.full(shape, 0.8).expand_as(input_real)
        decay_imag = torch.full(shape, 0.1).expand_as(input_imag)
        states_real, states_imag = pac_triton_recurrence_opaque_op(
            decay_real, decay_imag, input_real, input_imag
        )
        assert states_real.is_contiguous()
        assert states_imag.is_contiguous()
        return states_real.square().mean() + states_imag.square().mean()

    expected_inputs = inputs()
    expected_loss = target(*expected_inputs)
    expected_grads = torch.autograd.grad(expected_loss, expected_inputs)
    actual_inputs = tuple(
        tensor.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        for tensor in expected_inputs
    )
    compiled_loss = torch.compile(target, fullgraph=True, dynamic=False)(*actual_inputs)
    actual_grads = torch.autograd.grad(compiled_loss, actual_inputs)
    torch.testing.assert_close(compiled_loss, expected_loss)
    torch.testing.assert_close(actual_grads, expected_grads)

    real, imag = inputs()
    shape = (1, 1, modes)
    opcheck = torch.library.opcheck(
        torch.ops.lnet.pac_real2d_recurrence_opaque.default,
        (
            torch.full(shape, 0.8).expand_as(real),
            torch.full(shape, 0.1).expand_as(imag),
            real,
            imag,
            False,
        ),
        raise_exception=True,
    )
    assert all(value == "SUCCESS" for value in opcheck.values())


def test_fineweb_document_conversion_splits_before_tokenization(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [f"doc-{index}" for index in range(100)],
                "text": [f"English document number {index}." for index in range(100)],
            }
        ),
        source,
    )
    documents = tmp_path / "documents.jsonl"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _parquet_to_jsonl(source, documents)
    _split_documents(
        documents,
        train,
        validation,
        validation_fraction=0.2,
        salt="fixed",
    )
    train_ids = {json.loads(line)["id"] for line in train.read_text().splitlines()}
    validation_ids = {json.loads(line)["id"] for line in validation.read_text().splitlines()}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {f"doc-{index}" for index in range(100)}
