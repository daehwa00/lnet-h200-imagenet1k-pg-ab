from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMBlock,
    AlphabetLMConfig,
    CausalCNNPoleMemory,
    ChunkedSemanticPoleMemory,
    DynamicLowRankWrite,
    FixedComplexPoleMemory1D,
    FixedPoleResidualSidecar,
    IdentityComplexMemory1D,
    LowRankDecaySelector,
    QueryConditionedLowRankReadout,
    SemanticEdgePoleMemory,
    SlowCausalCNNPoleMemory,
    TensorProductPoleMemory1D,
)
from lnet.alphabet_lm_mamba import MambaLMConfig, build_parameter_matched_mamba
from lnet.pac_triton_recurrence_op import pac_triton_recurrence_opaque_op
from scripts.evaluate_kau_alphabet_lm_context import _zero_memory
from scripts.prepare_h200_alphabet_lm_data import _parquet_to_jsonl, _split_documents
from scripts.train_h200_alphabet_lm_10m import (
    _copy_matching_legacy_initialization,
    _initialize_chunk_memory_from_trunk,
    _initialize_cnn_pole_from_trunk,
    _initialize_semantic_edge_from_trunk,
    _initialize_sidecar_from_trunk,
    _initialize_slow_cnn_pole_from_trunk,
    _initialize_slow_key_from_trunk,
    _initialize_slow_query_from_trunk,
    _initialize_slow_value_from_trunk,
)


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
    local = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3", memory_layout="local_only"))
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


def test_local_sidecar_preserves_the_complete_local_trunk() -> None:
    torch.manual_seed(501)
    local = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3", memory_layout="local_only"))
    local_parameters = dict(local.named_parameters())
    torch.manual_seed(501)
    sidecar = AlphabetLM(
        AlphabetLMConfig(
            reader_type="dense_k3",
            memory_layout="local_sidecar",
            sidecar_initial_scale=0.01,
        )
    )
    assert sum(parameter.numel() for parameter in sidecar.parameters()) == 40_652_800
    sidecar_parameters = dict(sidecar.named_parameters())
    for name, parameter in local_parameters.items():
        torch.testing.assert_close(sidecar_parameters[name], parameter, atol=0.0, rtol=0.0)
    sidecars = [
        module for module in sidecar.modules() if isinstance(module, FixedPoleResidualSidecar)
    ]
    assert len(sidecars) == 12
    assert all(torch.equal(module.beta, torch.full_like(module.beta, 0.01)) for module in sidecars)


def test_zero_beta_sidecar_is_exactly_local_only_and_receives_gradients() -> None:
    config = AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=2,
        post_hidden=12,
        context_length=16,
        reader_type="dense_k3",
        memory_layout="local_sidecar",
        sidecar_initial_scale=0.01,
    )
    torch.manual_seed(501)
    sidecar = AlphabetLM(config)
    torch.manual_seed(501)
    local = AlphabetLM(
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
    for module in sidecar.modules():
        if isinstance(module, FixedPoleResidualSidecar):
            module.beta.data.zero_()
    tokens = torch.randint(64, (2, 17))
    with torch.no_grad():
        torch.testing.assert_close(sidecar(tokens), local(tokens), atol=0.0, rtol=0.0)

    for module in sidecar.modules():
        if isinstance(module, FixedPoleResidualSidecar):
            module.beta.data.fill_(0.01)
    logits = sidecar(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    sidecar_parameters = [
        parameter
        for module in sidecar.modules()
        if isinstance(module, FixedPoleResidualSidecar)
        for parameter in module.parameters()
    ]
    assert sidecar_parameters
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in sidecar_parameters
    )


def test_sidecar_memory_zero_preserves_the_local_trunk() -> None:
    def config(
        memory_layout: Literal["local_only", "local_sidecar"],
    ) -> AlphabetLMConfig:
        return AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
            memory_layout=memory_layout,
        )

    torch.manual_seed(501)
    local = AlphabetLM(config("local_only"))
    torch.manual_seed(501)
    sidecar = AlphabetLM(config("local_sidecar"))
    tokens = torch.randint(64, (2, 17))
    handles = _zero_memory(sidecar)
    try:
        with torch.no_grad():
            torch.testing.assert_close(sidecar(tokens), local(tokens), atol=0.0, rtol=0.0)
    finally:
        for handle in handles:
            handle.remove()


def test_normalized_scalar_sidecar_enforces_per_token_branch_rms() -> None:
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
            memory_layout="local_sidecar",
            sidecar_initial_scale=0.01,
            sidecar_normalize_memory=True,
            sidecar_channelwise_scale=False,
        )
    )
    sidecar = next(
        module for module in model.modules() if isinstance(module, FixedPoleResidualSidecar)
    )
    assert sidecar.beta.shape == ()
    real = torch.randn(2, 16, 8)
    imag = torch.randn_like(real)
    output_real, output_imag = sidecar(real, imag)
    trunk_rms = torch.sqrt(real.square().add(imag.square()).mean(dim=-1))
    branch_rms = torch.sqrt(
        (output_real - real).square().add((output_imag - imag).square()).mean(dim=-1)
    )
    torch.testing.assert_close(
        branch_rms / trunk_rms,
        torch.full_like(trunk_rms, 0.01),
        atol=2.0e-6,
        rtol=2.0e-4,
    )


def test_frozen_sidecar_owns_only_new_memory_parameters(tmp_path: Path) -> None:
    def config(
        memory_layout: Literal["local_only", "local_sidecar"],
    ) -> AlphabetLMConfig:
        return AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
            memory_layout=memory_layout,
            sidecar_normalize_memory=memory_layout == "local_sidecar",
            sidecar_channelwise_scale=memory_layout != "local_sidecar",
        )

    torch.manual_seed(501)
    local = AlphabetLM(config("local_only"))
    checkpoint = tmp_path / "local-only.pt"
    torch.save({"model": local.state_dict()}, checkpoint)
    torch.manual_seed(9)
    sidecar = AlphabetLM(config("local_sidecar"))
    contract = _initialize_sidecar_from_trunk(sidecar, checkpoint, freeze_trunk=True)
    assert contract["enabled"] is True
    assert contract["frozen"] is True
    local_state = local.state_dict()
    for name, value in sidecar.state_dict().items():
        if ".sidecar." not in name:
            torch.testing.assert_close(value, local_state[name], atol=0.0, rtol=0.0)
    trainable = [name for name, parameter in sidecar.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".sidecar." in name for name in trainable)


def test_no_recurrence_sidecar_changes_only_temporal_carry() -> None:
    def config(*, use_recurrence: bool) -> AlphabetLMConfig:
        return AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=2,
            post_hidden=12,
            context_length=16,
            reader_type="dense_k3",
            memory_layout="local_sidecar",
            sidecar_normalize_memory=True,
            sidecar_channelwise_scale=False,
            sidecar_use_recurrence=use_recurrence,
        )

    torch.manual_seed(501)
    recurrent = AlphabetLM(config(use_recurrence=True))
    torch.manual_seed(501)
    local = AlphabetLM(config(use_recurrence=False))
    assert sum(parameter.numel() for parameter in recurrent.parameters()) == sum(
        parameter.numel() for parameter in local.parameters()
    )
    for name, parameter in recurrent.named_parameters():
        torch.testing.assert_close(
            parameter,
            dict(local.named_parameters())[name],
            atol=0.0,
            rtol=0.0,
        )
    tokens = torch.randint(64, (2, 17))
    logits = local(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    for block in local.blocks:
        sidecar = cast("FixedPoleResidualSidecar", block.sidecar)
        assert sidecar.reader.weight_real.grad is not None
        assert sidecar.writer.weight_real.grad is not None
        assert sidecar.beta.grad is not None
        assert sidecar.memory.raw_damping.grad is None
        assert sidecar.memory.raw_frequency.grad is None


def test_chunk_memory_is_delayed_causal_and_preserves_the_first_chunk() -> None:
    def config(*, chunk_memory: bool) -> AlphabetLMConfig:
        return AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=4,
            post_hidden=12,
            context_length=64,
            reader_type="dense_k3",
            memory_layout="local_only",
            chunk_memory=chunk_memory,
            chunk_size=4,
            chunk_summary_width=8,
            chunk_pole_modes=8,
            chunk_upper_blocks=2,
            chunk_beta_initial=0.01,
            chunk_minimum_half_life=1.0,
            chunk_maximum_half_life=8.0,
        )

    torch.manual_seed(501)
    local = AlphabetLM(config(chunk_memory=False)).eval()
    torch.manual_seed(501)
    chunked = AlphabetLM(config(chunk_memory=True)).eval()
    local_parameters = dict(local.named_parameters())
    for name, parameter in chunked.named_parameters():
        if not name.startswith("chunk_memory."):
            torch.testing.assert_close(parameter, local_parameters[name], atol=0.0, rtol=0.0)
    tokens = torch.randint(64, (2, 17))
    with torch.no_grad():
        local_logits = local(tokens)
        chunked_logits = chunked(tokens)
    torch.testing.assert_close(chunked_logits[:, :4], local_logits[:, :4], atol=0.0, rtol=0.0)

    memory = cast("ChunkedSemanticPoleMemory", chunked.chunk_memory)
    memory.beta.data.zero_()
    with torch.no_grad():
        torch.testing.assert_close(chunked(tokens), local_logits, atol=0.0, rtol=0.0)


def test_chunk_memory_handles_partial_chunks_without_leakage() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=4,
            post_hidden=12,
            context_length=64,
            reader_type="dense_k3",
            memory_layout="local_only",
            chunk_memory=True,
            chunk_size=4,
            chunk_summary_width=8,
            chunk_pole_modes=8,
            chunk_upper_blocks=2,
            chunk_minimum_half_life=1.0,
            chunk_maximum_half_life=8.0,
        )
    ).eval()
    for steps in (1, 4, 5, 9, 16):
        tokens = torch.randint(64, (2, steps))
        with torch.no_grad():
            assert model(tokens).shape == (2, steps, 64)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = model(tokens)
        actual = model(changed)
    torch.testing.assert_close(actual[:, :8], expected[:, :8], atol=1.0e-6, rtol=0.0)


def test_chunk_memory_checkpoint_freezes_lower_trunk_and_trains_upper_blocks(
    tmp_path: Path,
) -> None:
    base = AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=4,
        post_hidden=12,
        context_length=64,
        reader_type="dense_k3",
        memory_layout="local_only",
    )
    torch.manual_seed(501)
    local = AlphabetLM(base)
    checkpoint = tmp_path / "local.pt"
    torch.save({"model": local.state_dict()}, checkpoint)
    torch.manual_seed(9)
    chunked = AlphabetLM(
        replace(
            base,
            chunk_memory=True,
            chunk_size=4,
            chunk_summary_width=8,
            chunk_pole_modes=8,
            chunk_upper_blocks=2,
            chunk_minimum_half_life=1.0,
            chunk_maximum_half_life=8.0,
        )
    )
    contract = _initialize_chunk_memory_from_trunk(
        chunked,
        checkpoint,
        train_upper_blocks=2,
    )
    assert contract["enabled"] is True
    trainable = [name for name, parameter in chunked.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith(("chunk_memory.", "blocks.2.", "blocks.3.")) for name in trainable)
    frozen_state = local.state_dict()
    for name, value in chunked.state_dict().items():
        if not name.startswith("chunk_memory."):
            torch.testing.assert_close(value, frozen_state[name], atol=0.0, rtol=0.0)


def test_semantic_level_detail_is_energy_preserving() -> None:
    previous = torch.randn(3, 5, 16, dtype=torch.complex64)
    current = torch.randn_like(previous)
    level = math.sqrt(0.5) * (previous + current)
    detail = math.sqrt(0.5) * (current - previous)
    source_energy = previous.abs().square() + current.abs().square()
    edge_energy = level.abs().square() + detail.abs().square()
    torch.testing.assert_close(edge_energy, source_energy, atol=2.0e-6, rtol=2.0e-6)


def test_semantic_edge_memory_is_delayed_and_semi_orthogonal() -> None:
    def config(*, use_recurrence: bool) -> AlphabetLMConfig:
        return AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=4,
            post_hidden=12,
            context_length=64,
            reader_type="dense_k3",
            memory_layout="local_only",
            semantic_edge_memory=True,
            semantic_edge_stride=4,
            semantic_edge_pole_modes=8,
            semantic_edge_upper_blocks=2,
            semantic_edge_beta_initial=0.01,
            semantic_edge_use_recurrence=use_recurrence,
            semantic_edge_minimum_half_life=1.0,
            semantic_edge_maximum_half_life=8.0,
        )

    torch.manual_seed(501)
    recurrent = AlphabetLM(config(use_recurrence=True)).eval()
    torch.manual_seed(501)
    control = AlphabetLM(config(use_recurrence=False)).eval()
    for name, parameter in recurrent.named_parameters():
        torch.testing.assert_close(
            parameter,
            dict(control.named_parameters())[name],
            atol=0.0,
            rtol=0.0,
        )
    edge = cast("SemanticEdgePoleMemory", recurrent.semantic_edge_memory)
    gram = (
        edge.excitation.weight_real @ edge.excitation.weight_real.T
        + edge.excitation.weight_imag @ edge.excitation.weight_imag.T
    )
    torch.testing.assert_close(gram, torch.eye(8), atol=2.0e-6, rtol=0.0)

    torch.manual_seed(501)
    local = AlphabetLM(replace(config(use_recurrence=True), semantic_edge_memory=False)).eval()
    tokens = torch.randint(64, (2, 17))
    with torch.no_grad():
        recurrent_logits = recurrent(tokens)
        local_logits = local(tokens)
    torch.testing.assert_close(recurrent_logits[:, :4], local_logits[:, :4], atol=0.0, rtol=0.0)


def test_semantic_edge_memory_is_causal_for_partial_anchor_sequences() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(
        AlphabetLMConfig(
            vocab_size=64,
            modes=8,
            pole_modes=12,
            layers=4,
            post_hidden=12,
            context_length=64,
            reader_type="dense_k3",
            memory_layout="local_only",
            semantic_edge_memory=True,
            semantic_edge_stride=4,
            semantic_edge_pole_modes=8,
            semantic_edge_upper_blocks=2,
            semantic_edge_minimum_half_life=1.0,
            semantic_edge_maximum_half_life=8.0,
        )
    ).eval()
    for steps in (1, 4, 5, 9, 16):
        tokens = torch.randint(64, (2, steps))
        with torch.no_grad():
            assert model(tokens).shape == (2, steps, 64)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = model(tokens)
        actual = model(changed)
    torch.testing.assert_close(actual[:, :8], expected[:, :8], atol=1.0e-6, rtol=0.0)


def test_semantic_edge_checkpoint_freezes_every_local_parameter(tmp_path: Path) -> None:
    base = AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=4,
        post_hidden=12,
        context_length=64,
        reader_type="dense_k3",
        memory_layout="local_only",
    )
    torch.manual_seed(501)
    local = AlphabetLM(base)
    checkpoint = tmp_path / "local.pt"
    torch.save({"model": local.state_dict()}, checkpoint)
    torch.manual_seed(9)
    edge_model = AlphabetLM(
        replace(
            base,
            semantic_edge_memory=True,
            semantic_edge_stride=4,
            semantic_edge_pole_modes=8,
            semantic_edge_upper_blocks=2,
            semantic_edge_minimum_half_life=1.0,
            semantic_edge_maximum_half_life=8.0,
        )
    )
    contract = _initialize_semantic_edge_from_trunk(edge_model, checkpoint)
    assert contract["enabled"] is True
    trainable = [
        name for name, parameter in edge_model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("semantic_edge_memory.") for name in trainable)
    local_state = local.state_dict()
    for name, value in edge_model.state_dict().items():
        if not name.startswith("semantic_edge_memory."):
            torch.testing.assert_close(value, local_state[name], atol=0.0, rtol=0.0)


def _cnn_pole_config(*, use_recurrence: bool = True) -> AlphabetLMConfig:
    return AlphabetLMConfig(
        vocab_size=64,
        modes=8,
        pole_modes=12,
        layers=4,
        post_hidden=12,
        context_length=64,
        reader_type="dense_k3",
        memory_layout="local_only",
        cnn_pole_memory=True,
        cnn_pole_interval=2,
        cnn_pole_modes=8,
        cnn_pole_evidence_width=16,
        cnn_pole_kernel_size=4,
        cnn_pole_beta_initial=0.01,
        cnn_pole_use_recurrence=use_recurrence,
        cnn_pole_minimum_half_life=1.0,
        cnn_pole_maximum_half_life=8.0,
    )


def test_cnn_pole_memory_is_repeated_causal_and_non_expansive() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(_cnn_pole_config()).eval()
    memories = cast("torch.nn.ModuleList", model.cnn_pole_memories)
    assert len(memories) == 2
    for module in memories:
        assert isinstance(module, CausalCNNPoleMemory)
        gram = module.analysis.weight @ module.analysis.weight.T
        torch.testing.assert_close(gram, torch.eye(16), atol=2.0e-6, rtol=0.0)
        assert not module.analysis.weight.requires_grad

    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = model(tokens)
        actual = model(changed)
    torch.testing.assert_close(actual[:, :8], expected[:, :8], atol=2.0e-6, rtol=0.0)


def test_cnn_pole_zero_beta_is_exact_local_and_recurrent_control_is_paired() -> None:
    torch.manual_seed(501)
    recurrent = AlphabetLM(_cnn_pole_config(use_recurrence=True))
    torch.manual_seed(501)
    control = AlphabetLM(_cnn_pole_config(use_recurrence=False))
    recurrent_state = recurrent.state_dict()
    control_state = control.state_dict()
    assert recurrent_state.keys() == control_state.keys()
    for name, value in recurrent_state.items():
        torch.testing.assert_close(value, control_state[name], atol=0.0, rtol=0.0)

    torch.manual_seed(501)
    local = AlphabetLM(replace(_cnn_pole_config(), cnn_pole_memory=False)).eval()
    local_state = local.state_dict()
    recurrent.load_state_dict(local_state, strict=False)
    for module in cast("torch.nn.ModuleList", recurrent.cnn_pole_memories):
        memory = cast("CausalCNNPoleMemory", module)
        memory.beta.data.zero_()
    recurrent.eval()
    tokens = torch.randint(64, (2, 17))
    with torch.no_grad():
        torch.testing.assert_close(recurrent(tokens), local(tokens), atol=0.0, rtol=0.0)

    for module in cast("torch.nn.ModuleList", recurrent.cnn_pole_memories):
        cast("CausalCNNPoleMemory", module).beta.data.fill_(0.01)
    handles = _zero_memory(recurrent)
    assert len(handles) == 2
    with torch.no_grad():
        torch.testing.assert_close(recurrent(tokens), local(tokens), atol=0.0, rtol=0.0)
    for handle in handles:
        handle.remove()


def test_cnn_pole_checkpoint_freezes_the_complete_local_trunk(tmp_path: Path) -> None:
    config = _cnn_pole_config()
    torch.manual_seed(501)
    local = AlphabetLM(replace(config, cnn_pole_memory=False))
    checkpoint = tmp_path / "local.pt"
    torch.save({"model": local.state_dict()}, checkpoint)
    torch.manual_seed(9)
    model = AlphabetLM(config)
    contract = _initialize_cnn_pole_from_trunk(model, checkpoint)
    assert contract["enabled"] is True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("cnn_pole_memories.") for name in trainable)
    assert not any(name.endswith(".analysis.weight") for name in trainable)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert trainable_parameters == 1_218
    for name, value in model.state_dict().items():
        if not name.startswith("cnn_pole_memories."):
            torch.testing.assert_close(value, local.state_dict()[name], atol=0.0, rtol=0.0)


def _slow_cnn_pole_config(*, use_recurrence: bool = True) -> AlphabetLMConfig:
    return replace(
        _cnn_pole_config(use_recurrence=False),
        slow_cnn_pole_memory=True,
        slow_cnn_pole_stride=4,
        slow_cnn_pole_modes=8,
        slow_cnn_pole_evidence_width=16,
        slow_cnn_pole_kernel_size=4,
        slow_cnn_pole_upper_blocks=2,
        slow_cnn_pole_beta_initial=0.01,
        slow_cnn_pole_use_recurrence=use_recurrence,
        slow_cnn_pole_minimum_half_life=1.0,
        slow_cnn_pole_maximum_half_life=8.0,
    )


def test_slow_cnn_pole_is_delayed_causal_and_paired() -> None:
    torch.manual_seed(501)
    recurrent = AlphabetLM(_slow_cnn_pole_config(use_recurrence=True)).eval()
    torch.manual_seed(501)
    control = AlphabetLM(_slow_cnn_pole_config(use_recurrence=False)).eval()
    for name, value in recurrent.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[name], atol=0.0, rtol=0.0)

    baseline = AlphabetLM(
        replace(_slow_cnn_pole_config(), slow_cnn_pole_memory=False)
    ).eval()
    recurrent.load_state_dict(baseline.state_dict(), strict=False)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = recurrent(tokens)
        actual = recurrent(changed)
        baseline_logits = baseline(tokens)
    torch.testing.assert_close(actual[:, :8], expected[:, :8], atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(expected[:, :4], baseline_logits[:, :4], atol=0.0, rtol=0.0)


def test_slow_cnn_pole_checkpoint_freezes_fast_sidecars_and_trunk(tmp_path: Path) -> None:
    config = _slow_cnn_pole_config()
    torch.manual_seed(501)
    fast_model = AlphabetLM(replace(config, slow_cnn_pole_memory=False))
    checkpoint = tmp_path / "fast.pt"
    torch.save({"model": fast_model.state_dict()}, checkpoint)
    torch.manual_seed(9)
    model = AlphabetLM(config)
    contract = _initialize_slow_cnn_pole_from_trunk(model, checkpoint)
    assert contract["enabled"] is True
    slow = cast("SlowCausalCNNPoleMemory", model.slow_cnn_pole_memory)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("slow_cnn_pole_memory.") for name in trainable)
    assert not slow.analysis.weight.requires_grad
    for name, value in model.state_dict().items():
        if not name.startswith("slow_cnn_pole_memory."):
            torch.testing.assert_close(value, fast_model.state_dict()[name], atol=0.0, rtol=0.0)


def _addressed_slow_config(*, mode: Literal["anchor", "token"]) -> AlphabetLMConfig:
    return replace(
        _slow_cnn_pole_config(use_recurrence=True),
        slow_cnn_pole_query=mode,
        slow_cnn_pole_query_rho=0.5,
    )


def test_addressed_slow_query_is_identity_centered_and_causal() -> None:
    torch.manual_seed(501)
    anchor_query = AlphabetLM(_addressed_slow_config(mode="anchor")).eval()
    torch.manual_seed(501)
    token_query = AlphabetLM(_addressed_slow_config(mode="token")).eval()
    source = AlphabetLM(
        replace(_addressed_slow_config(mode="anchor"), slow_cnn_pole_query="none")
    ).eval()
    anchor_query.load_state_dict(source.state_dict(), strict=False)
    token_query.load_state_dict(source.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", token_query.slow_cnn_pole_memory)
    gate = slow.query_gate(torch.randn(2, 7, 16))
    torch.testing.assert_close(gate.mean(dim=-1), torch.ones(2, 7), atol=0.0, rtol=0.0)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        source_logits = source(tokens)
        anchor_logits = anchor_query(tokens)
        token_logits = token_query(tokens)
        changed_logits = token_query(changed)
    torch.testing.assert_close(anchor_logits, source_logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(token_logits, source_logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(changed_logits[:, :8], token_logits[:, :8], atol=2e-6, rtol=0.0)


def test_addressed_slow_checkpoint_trains_only_query(tmp_path: Path) -> None:
    config = _addressed_slow_config(mode="token")
    torch.manual_seed(501)
    source = AlphabetLM(replace(config, slow_cnn_pole_query="none"))
    checkpoint = tmp_path / "v1.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    model = AlphabetLM(config)
    contract = _initialize_slow_query_from_trunk(model, checkpoint)
    assert contract["enabled"] is True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert set(trainable) == {
        "slow_cnn_pole_memory.query_norm.weight",
        "slow_cnn_pole_memory.query.weight",
    }
    slow = cast("SlowCausalCNNPoleMemory", model.slow_cnn_pole_memory)
    assert slow.query is not None
    torch.testing.assert_close(slow.query.weight, torch.zeros_like(slow.query.weight))


def _qk_slow_config() -> AlphabetLMConfig:
    return replace(
        _addressed_slow_config(mode="token"),
        slow_cnn_pole_key=True,
        slow_cnn_pole_key_rho=0.5,
    )


def test_qk_slow_memory_is_identity_initialized_mean_one_and_causal() -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token")).eval()
    torch.manual_seed(501)
    qk = AlphabetLM(_qk_slow_config()).eval()
    qk.load_state_dict(token_q.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", qk.slow_cnn_pole_memory)
    key_gate = slow.key_gate(torch.randn(2, 7, 16))
    torch.testing.assert_close(key_gate.mean(dim=-1), torch.ones(2, 7), atol=0.0, rtol=0.0)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = token_q(tokens)
        actual = qk(tokens)
        changed_logits = qk(changed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(changed_logits[:, :8], actual[:, :8], atol=2e-6, rtol=0.0)


def test_qk_checkpoint_trains_only_mandatory_key(tmp_path: Path) -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token"))
    checkpoint = tmp_path / "token-q.pt"
    torch.save({"model": token_q.state_dict()}, checkpoint)
    qk = AlphabetLM(_qk_slow_config())
    contract = _initialize_slow_key_from_trunk(qk, checkpoint)
    assert contract["enabled"] is True
    trainable = [name for name, parameter in qk.named_parameters() if parameter.requires_grad]
    assert set(trainable) == {
        "slow_cnn_pole_memory.key_norm.weight",
        "slow_cnn_pole_memory.key.weight",
    }
    slow = cast("SlowCausalCNNPoleMemory", qk.slow_cnn_pole_memory)
    assert slow.key is not None
    torch.testing.assert_close(slow.key.weight, torch.zeros_like(slow.key.weight))


def _vector_slow_config() -> AlphabetLMConfig:
    return replace(
        _addressed_slow_config(mode="token"),
        slow_cnn_pole_value_width=4,
    )


def test_vector_pole_memory_preserves_scalar_coordinate_exactly() -> None:
    torch.manual_seed(501)
    memory = FixedComplexPoleMemory1D(8, context_length=32, scan_fp32=False)
    drive_real = torch.randn(2, 9, 8)
    drive_imag = torch.randn_like(drive_real)
    scalar = memory(drive_real, drive_imag)
    value = torch.randn(2, 9, 1, 4)
    value[..., 0] = 1.0
    vector = memory(drive_real.unsqueeze(-1) * value, drive_imag.unsqueeze(-1) * value)
    torch.testing.assert_close(vector[0][..., 0], scalar[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(vector[1][..., 0], scalar[1], atol=0.0, rtol=0.0)


def test_vector_slow_memory_is_baseline_preserving_and_causal() -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token")).eval()
    torch.manual_seed(501)
    vector = AlphabetLM(_vector_slow_config()).eval()
    vector.load_state_dict(token_q.state_dict(), strict=False)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = token_q(tokens)
        actual = vector(tokens)
        changed_logits = vector(changed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(changed_logits[:, :8], actual[:, :8], atol=2e-6, rtol=0.0)


def test_vector_checkpoint_trains_only_value_and_extra_synthesis(tmp_path: Path) -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token"))
    checkpoint = tmp_path / "token-q.pt"
    torch.save({"model": token_q.state_dict()}, checkpoint)
    vector = AlphabetLM(_vector_slow_config())
    contract = _initialize_slow_value_from_trunk(vector, checkpoint)
    assert contract["enabled"] is True
    trainable = [name for name, parameter in vector.named_parameters() if parameter.requires_grad]
    assert set(trainable) == {
        "slow_cnn_pole_memory.value_norm.weight",
        "slow_cnn_pole_memory.value.weight",
        "slow_cnn_pole_memory.extra_synthesis.weight",
    }
    slow = cast("SlowCausalCNNPoleMemory", vector.slow_cnn_pole_memory)
    assert slow.extra_synthesis is not None
    torch.testing.assert_close(
        slow.extra_synthesis.weight,
        torch.zeros_like(slow.extra_synthesis.weight),
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
