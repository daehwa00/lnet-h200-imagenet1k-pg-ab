from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lnet.alphabet_lm import (
    AlphabetLM,
    AlphabetLMBlock,
    AlphabetLMConfig,
    CausalCNNPoleMemory,
    ChunkedSemanticPoleMemory,
    ComplexDepthwiseCausalPredictor,
    DynamicLowRankWrite,
    FactorizedTokenRateVectorPoleBlock,
    FixedComplexPoleMemory1D,
    FixedPoleResidualSidecar,
    IdentityComplexMemory1D,
    LaplaceMambaBlock,
    LaplaceMambaLM,
    LaplaceMambaLMConfig,
    LowRankDecaySelector,
    PoleSpecificCausalVectorReader,
    QueryConditionedLowRankReadout,
    SemanticEdgePoleMemory,
    SlowCausalCNNPoleMemory,
    TensorProductPoleMemory1D,
    TokenRateVectorPoleBlock,
)
from lnet.alphabet_lm_mamba import MambaLMConfig, build_parameter_matched_mamba
from lnet.pac_triton_recurrence_op import pac_triton_recurrence_opaque_op
from scripts.evaluate_kau_alphabet_lm_context import (
    _factorized_extra_coordinate_override,
    _factorized_pca_override,
    _zero_memory,
)
from scripts.prepare_h200_alphabet_lm_data import _parquet_to_jsonl, _split_documents
from scripts.train_h200_alphabet_lm_10m import (
    _copy_matching_legacy_initialization,
    _initialize_chunk_memory_from_trunk,
    _initialize_cnn_pole_from_trunk,
    _initialize_repeated_factorized_expansion,
    _initialize_repeated_mamba_outer,
    _initialize_repeated_retained_factor_state,
    _initialize_semantic_edge_from_trunk,
    _initialize_sidecar_from_trunk,
    _initialize_slow_cnn_pole_from_trunk,
    _initialize_slow_complex_vector_from_trunk,
    _initialize_slow_dynamic_transport_from_trunk,
    _initialize_slow_full_complex_vector_from_trunk,
    _initialize_slow_independent_value_from_trunk,
    _initialize_slow_innovation_from_trunk,
    _initialize_slow_key_from_trunk,
    _initialize_slow_matrix_from_trunk,
    _initialize_slow_query_from_trunk,
    _initialize_slow_semantic_clock_from_trunk,
    _initialize_slow_value_from_trunk,
    _initialize_slow_vector_pole_from_trunk,
    _initialize_slow_write_scheduler_from_trunk,
    _validate_repeated_factorized_source,
    _validate_repeated_mamba_outer_source,
    _validate_repeated_retained_source,
    _validate_slow_complex_vector_source,
    _validate_slow_dynamic_transport_source,
    _validate_slow_full_complex_vector_source,
    _validate_slow_innovation_source,
    _validate_slow_semantic_clock_source,
    _validate_slow_vector_pole_source,
    _validate_slow_write_scheduler_source,
    _validate_vector_initialization_selection,
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


def _vector_pole_slow_config() -> AlphabetLMConfig:
    return replace(
        _addressed_slow_config(mode="token"),
        slow_cnn_pole_vector_width=4,
    )


def test_vector_pole_r4_preserves_token_q_and_has_live_query_gradient() -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token")).eval()
    torch.manual_seed(501)
    vector_pole = AlphabetLM(_vector_pole_slow_config()).eval()
    vector_pole.load_state_dict(token_q.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", vector_pole.slow_cnn_pole_memory)
    assert slow.vector_excitation is not None
    assert slow.vector_query is not None
    assert slow.vector_excitation.weight.square().sum() > 0
    torch.testing.assert_close(
        slow.vector_query.weight,
        torch.zeros_like(slow.vector_query.weight),
    )
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = token_q(tokens)
        actual = vector_pole(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    vector_pole(tokens).square().mean().backward()
    assert slow.vector_query.weight.grad is not None
    assert slow.vector_query.weight.grad.abs().sum() > 0
    with torch.no_grad():
        torch.nn.init.normal_(slow.vector_query.weight, std=0.02)
        active_logits = vector_pole(tokens)
        changed_logits = vector_pole(changed)
    torch.testing.assert_close(
        changed_logits[:, :8], active_logits[:, :8], atol=2e-6, rtol=0.0
    )


def test_vector_pole_checkpoint_trains_only_extra_coordinates(tmp_path: Path) -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token"))
    checkpoint = tmp_path / "token-q.pt"
    torch.save({"model": token_q.state_dict()}, checkpoint)
    vector_pole = AlphabetLM(_vector_pole_slow_config())
    contract = _initialize_slow_vector_pole_from_trunk(vector_pole, checkpoint)
    assert contract["enabled"] is True
    trainable = [
        name for name, parameter in vector_pole.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.vector_excitation_norm.weight",
        "slow_cnn_pole_memory.vector_excitation.weight",
        "slow_cnn_pole_memory.vector_query_norm.weight",
        "slow_cnn_pole_memory.vector_query.weight",
    }


def test_vector_pole_configuration_requires_an_active_recurrent_slow_bank() -> None:
    with pytest.raises(ValueError, match="slow pole addressing requires"):
        replace(_small(), slow_cnn_pole_vector_width=4)
    with pytest.raises(ValueError, match="vector pole memory requires"):
        replace(
            _addressed_slow_config(mode="token"),
            slow_cnn_pole_vector_width=4,
            slow_cnn_pole_use_recurrence=False,
        )


def test_vector_pole_source_digest_is_enforced_inside_the_trainer() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"token_q_10m_sha256": "expected"}}
    _validate_slow_vector_pole_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="source checkpoint digest changed"):
        _validate_slow_vector_pole_source(
            initialization,
            {"source": {"token_q_10m_sha256": "different"}},
            enabled=True,
        )


def _complex_vector_pole_config() -> AlphabetLMConfig:
    return replace(
        _vector_pole_slow_config(),
        slow_cnn_pole_complex_vector_excitation=True,
    )


def test_complex_vector_excitation_preserves_shared_phase_and_has_live_gradient(
    tmp_path: Path,
) -> None:
    torch.manual_seed(501)
    shared_phase = AlphabetLM(_vector_pole_slow_config()).eval()
    shared_slow = cast("SlowCausalCNNPoleMemory", shared_phase.slow_cnn_pole_memory)
    assert shared_slow.vector_query is not None
    torch.nn.init.normal_(shared_slow.vector_query.weight, std=0.02)
    checkpoint = tmp_path / "shared-phase.pt"
    torch.save({"model": shared_phase.state_dict()}, checkpoint)
    torch.manual_seed(501)
    complex_vector = AlphabetLM(_complex_vector_pole_config()).eval()
    contract = _initialize_slow_complex_vector_from_trunk(complex_vector, checkpoint)
    assert contract["enabled"] is True
    complex_slow = cast("SlowCausalCNNPoleMemory", complex_vector.slow_cnn_pole_memory)
    assert complex_slow.vector_excitation_imag is not None
    torch.testing.assert_close(
        complex_slow.vector_excitation_imag.weight,
        torch.zeros_like(complex_slow.vector_excitation_imag.weight),
    )
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = shared_phase(tokens)
        actual = complex_vector(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    complex_vector(tokens).square().mean().backward()
    assert complex_slow.vector_excitation_imag.weight.grad is not None
    assert complex_slow.vector_excitation_imag.weight.grad.abs().sum() > 0
    trainable = [
        name for name, parameter in complex_vector.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.vector_excitation_imag_norm.weight",
        "slow_cnn_pole_memory.vector_excitation_imag.weight",
    }


def test_complex_vector_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"vector_pole_r4_4m_sha256": "expected"}}
    _validate_slow_complex_vector_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="complex vector source checkpoint digest changed"):
        _validate_slow_complex_vector_source(
            initialization,
            {"source": {"vector_pole_r4_4m_sha256": "different"}},
            enabled=True,
        )


def test_complex_vector_provenance_does_not_require_token_q_again() -> None:
    disabled_token_q: dict[str, object] = {"enabled": False}
    complex_initialization: dict[str, object] = {"checkpoint_sha256": "shared-phase"}
    runtime = {"source": {"vector_pole_r4_4m_sha256": "shared-phase"}}
    _validate_slow_vector_pole_source(
        disabled_token_q,
        runtime,
        enabled=False,
    )
    _validate_slow_complex_vector_source(
        complex_initialization,
        runtime,
        enabled=True,
    )


def test_full_complex_r16_preserves_token_q_with_live_query_gradient(
    tmp_path: Path,
) -> None:
    torch.manual_seed(501)
    token_q = AlphabetLM(_addressed_slow_config(mode="token")).eval()
    checkpoint = tmp_path / "token-q.pt"
    torch.save({"model": token_q.state_dict()}, checkpoint)
    config = replace(
        _addressed_slow_config(mode="token"),
        slow_cnn_pole_vector_width=16,
        slow_cnn_pole_complex_vector_excitation=True,
    )
    torch.manual_seed(501)
    candidate = AlphabetLM(config).eval()
    contract = _initialize_slow_full_complex_vector_from_trunk(candidate, checkpoint)
    assert contract["enabled"] is True
    slow = cast("SlowCausalCNNPoleMemory", candidate.slow_cnn_pole_memory)
    assert slow.vector_excitation is not None
    assert slow.vector_excitation_imag is not None
    assert slow.vector_query is not None
    assert slow.vector_excitation.weight.square().sum() > 0
    assert slow.vector_excitation_imag.weight.square().sum() > 0
    torch.testing.assert_close(slow.vector_query.weight, torch.zeros_like(slow.vector_query.weight))
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = token_q(tokens)
        actual = candidate(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    candidate(tokens).square().mean().backward()
    assert slow.vector_query.weight.grad is not None
    assert slow.vector_query.weight.grad.abs().sum() > 0
    trainable = [
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.vector_excitation_norm.weight",
        "slow_cnn_pole_memory.vector_excitation.weight",
        "slow_cnn_pole_memory.vector_query_norm.weight",
        "slow_cnn_pole_memory.vector_query.weight",
        "slow_cnn_pole_memory.vector_excitation_imag_norm.weight",
        "slow_cnn_pole_memory.vector_excitation_imag.weight",
    }


def test_full_complex_r16_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"token_q_10m_sha256": "expected"}}
    _validate_slow_full_complex_vector_source(
        initialization, runtime, enabled=True
    )
    with pytest.raises(RuntimeError, match="full complex vector source checkpoint"):
        _validate_slow_full_complex_vector_source(
            initialization,
            {"source": {"token_q_10m_sha256": "different"}},
            enabled=True,
        )


def test_vector_initialization_source_must_be_unambiguous(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _validate_vector_initialization_selection(checkpoint, None, None)
    with pytest.raises(RuntimeError, match="initialization source is ambiguous"):
        _validate_vector_initialization_selection(checkpoint, checkpoint, None)


def _complex_query_r16_config(*, coordinate_read: bool = False) -> AlphabetLMConfig:
    return replace(
        _addressed_slow_config(mode="token"),
        slow_cnn_pole_vector_width=16,
        slow_cnn_pole_complex_vector_excitation=True,
        slow_cnn_pole_complex_vector_query=True,
        slow_cnn_pole_coordinate_read=coordinate_read,
    )


def test_complex_query_zero_imag_preserves_real_query_and_remains_causal() -> None:
    torch.manual_seed(501)
    real_query = AlphabetLM(
        replace(
            _addressed_slow_config(mode="token"),
            slow_cnn_pole_vector_width=16,
            slow_cnn_pole_complex_vector_excitation=True,
        )
    ).eval()
    torch.manual_seed(501)
    complex_query = AlphabetLM(_complex_query_r16_config()).eval()
    complex_query.load_state_dict(real_query.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", complex_query.slow_cnn_pole_memory)
    assert slow.vector_query_imag is not None
    torch.nn.init.zeros_(slow.vector_query_imag.weight)
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = real_query(tokens)
        actual = complex_query(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.nn.init.normal_(slow.vector_query_imag.weight, std=0.02)
    with torch.no_grad():
        active = complex_query(tokens)
        changed_logits = complex_query(changed)
    torch.testing.assert_close(changed_logits[:, :8], active[:, :8], atol=2e-6, rtol=0.0)


def test_coordinate_read_returns_model_width_and_requires_complex_query() -> None:
    model = AlphabetLM(_complex_query_r16_config(coordinate_read=True))
    tokens = torch.randint(64, (1, 16))
    assert model(tokens).shape == (1, 16, 64)
    with pytest.raises(ValueError, match="coordinate read requires complex vector query"):
        replace(
            _addressed_slow_config(mode="token"),
            slow_cnn_pole_vector_width=16,
            slow_cnn_pole_coordinate_read=True,
        )


def _dynamic_transport_config() -> AlphabetLMConfig:
    return replace(
        _complex_query_r16_config(coordinate_read=True),
        slow_cnn_pole_dynamic_transport=True,
        slow_cnn_pole_transport_rank=4,
    )


def test_vector_dynamic_damping_zero_control_preserves_fixed_recurrence() -> None:
    memory = FixedComplexPoleMemory1D(8, context_length=32, scan_fp32=False)
    real = torch.randn(2, 7, 8, 4)
    imag = torch.randn_like(real)
    fixed = memory(real, imag)
    dynamic = memory(real, imag, damping_control=torch.zeros(2, 7, 8))
    torch.testing.assert_close(dynamic, fixed, atol=0.0, rtol=0.0)


def test_dynamic_transport_checkpoint_is_exact_and_selector_is_live(tmp_path: Path) -> None:
    torch.manual_seed(501)
    fixed = AlphabetLM(_complex_query_r16_config(coordinate_read=True)).eval()
    checkpoint = tmp_path / "coordinate.pt"
    torch.save({"model": fixed.state_dict()}, checkpoint)
    torch.manual_seed(501)
    dynamic = AlphabetLM(_dynamic_transport_config()).eval()
    contract = _initialize_slow_dynamic_transport_from_trunk(dynamic, checkpoint)
    assert contract["enabled"] is True
    slow = cast("SlowCausalCNNPoleMemory", dynamic.slow_cnn_pole_memory)
    assert slow.transport_selector is not None
    torch.testing.assert_close(
        slow.transport_selector.output.weight,
        torch.zeros_like(slow.transport_selector.output.weight),
    )
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = fixed(tokens)
        actual = dynamic(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    dynamic(tokens).square().mean().backward()
    assert slow.transport_selector.output.weight.grad is not None
    assert slow.transport_selector.output.weight.grad.abs().sum() > 0
    trainable = [name for name, parameter in dynamic.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("slow_cnn_pole_memory.transport_selector.") for name in trainable)


def test_dynamic_transport_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"coordinate_30m_sha256": "expected"}}
    _validate_slow_dynamic_transport_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="dynamic transport source checkpoint"):
        _validate_slow_dynamic_transport_source(
            initialization,
            {"source": {"coordinate_30m_sha256": "different"}},
            enabled=True,
        )


def test_pole_specific_vector_reader_is_causal_and_vector_valued() -> None:
    torch.manual_seed(501)
    reader = PoleSpecificCausalVectorReader(8, 4, 3, kernel_size=4)
    real = torch.randn(2, 9, 8)
    imag = torch.randn_like(real)
    changed_real = real.clone()
    changed_imag = imag.clone()
    changed_real[:, 5:] = torch.randn_like(changed_real[:, 5:])
    changed_imag[:, 5:] = torch.randn_like(changed_imag[:, 5:])
    output = reader(real, imag)
    changed = reader(changed_real, changed_imag)
    assert output[0].shape == (2, 9, 4, 3)
    assert output[1].shape == (2, 9, 4, 3)
    torch.testing.assert_close(changed[0][:, :5], output[0][:, :5], atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed[1][:, :5], output[1][:, :5], atol=0.0, rtol=0.0)


def test_pole_specific_reader_requires_token_rate_late_fusion() -> None:
    config = replace(
        _complex_query_r16_config(coordinate_read=True),
        slow_cnn_pole_stride=1,
        slow_cnn_pole_minimum_half_life=16.0,
        slow_cnn_pole_maximum_half_life=4_096.0,
        slow_cnn_pole_specific_reader=True,
    )
    model = AlphabetLM(config)
    assert model(torch.randint(64, (1, 16))).shape == (1, 16, 64)
    with pytest.raises(ValueError, match="pole-specific reader requires"):
        replace(config, slow_cnn_pole_stride=16)


def _scheduled_pole_reader_config() -> AlphabetLMConfig:
    return replace(
        _complex_query_r16_config(coordinate_read=True),
        slow_cnn_pole_stride=1,
        slow_cnn_pole_minimum_half_life=16.0,
        slow_cnn_pole_maximum_half_life=4_096.0,
        slow_cnn_pole_specific_reader=True,
        slow_cnn_pole_write_scheduler=True,
    )


def test_write_scheduler_is_identity_initialized_pole_wise_and_live() -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_scheduled_pole_reader_config(), slow_cnn_pole_write_scheduler=False)
    ).eval()
    torch.manual_seed(501)
    scheduled = AlphabetLM(_scheduled_pole_reader_config()).eval()
    scheduled.load_state_dict(baseline.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", scheduled.slow_cnn_pole_memory)
    assert slow.write_scheduler is not None
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        packed = torch.randn(2, 9, 16)
        gate = slow.write_gate(packed)
        expected = baseline(tokens)
        actual = scheduled(tokens)
    assert gate.shape == (2, 9, slow.memory.modes)
    torch.testing.assert_close(gate, torch.ones_like(gate), atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    scheduled(tokens).square().mean().backward()
    assert slow.write_scheduler.weight.grad is not None
    assert slow.write_scheduler.weight.grad.abs().sum() > 0


def test_write_scheduler_does_not_perturb_matching_seeded_initialization() -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_scheduled_pole_reader_config(), slow_cnn_pole_write_scheduler=False)
    )
    torch.manual_seed(501)
    scheduled = AlphabetLM(_scheduled_pole_reader_config())
    baseline_state = baseline.state_dict()
    for name, value in scheduled.state_dict().items():
        if "write_scheduler" not in name:
            torch.testing.assert_close(value, baseline_state[name], atol=0.0, rtol=0.0)


def test_write_scheduler_checkpoint_freezes_dense_k3_and_trains_only_gate(
    tmp_path: Path,
) -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_scheduled_pole_reader_config(), slow_cnn_pole_write_scheduler=False)
    ).eval()
    checkpoint = tmp_path / "dense-k3.pt"
    torch.save({"model": baseline.state_dict()}, checkpoint)
    torch.manual_seed(501)
    scheduled = AlphabetLM(_scheduled_pole_reader_config()).eval()
    contract = _initialize_slow_write_scheduler_from_trunk(scheduled, checkpoint)
    assert contract["enabled"] is True
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        torch.testing.assert_close(
            scheduled(tokens), baseline(tokens), atol=0.0, rtol=0.0
        )
    trainable = [
        name for name, parameter in scheduled.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.write_scheduler_norm.weight",
        "slow_cnn_pole_memory.write_scheduler.weight",
        "slow_cnn_pole_memory.write_scheduler.bias",
    }


def test_write_scheduler_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"pole_reader_30m_sha256": "expected"}}
    _validate_slow_write_scheduler_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="write-scheduler source checkpoint"):
        _validate_slow_write_scheduler_source(
            initialization,
            {"source": {"pole_reader_30m_sha256": "different"}},
            enabled=True,
        )


def test_write_scheduler_requires_pole_specific_reader() -> None:
    with pytest.raises(ValueError, match="write scheduler requires"):
        replace(
            _complex_query_r16_config(coordinate_read=True),
            slow_cnn_pole_write_scheduler=True,
        )


def _innovation_pole_reader_config() -> AlphabetLMConfig:
    return replace(
        _scheduled_pole_reader_config(),
        slow_cnn_pole_write_scheduler=False,
        slow_cnn_pole_innovation=True,
        slow_cnn_pole_innovation_kernel=3,
    )


def test_complex_depthwise_predictor_is_strictly_causal_lag_one_identity() -> None:
    predictor = ComplexDepthwiseCausalPredictor(4, 3, kernel_size=3)
    real = torch.randn(2, 9, 4, 3)
    imag = torch.randn_like(real)
    predicted_real, predicted_imag = predictor(real, imag)
    torch.testing.assert_close(predicted_real[:, :1], torch.zeros_like(real[:, :1]))
    torch.testing.assert_close(predicted_imag[:, :1], torch.zeros_like(imag[:, :1]))
    torch.testing.assert_close(predicted_real[:, 1:], real[:, :-1])
    torch.testing.assert_close(predicted_imag[:, 1:], imag[:, :-1])
    changed_real = real.clone()
    changed_imag = imag.clone()
    changed_real[:, 5:] = torch.randn_like(changed_real[:, 5:])
    changed_imag[:, 5:] = torch.randn_like(changed_imag[:, 5:])
    changed = predictor(changed_real, changed_imag)
    torch.testing.assert_close(changed[0][:, :6], predicted_real[:, :6])
    torch.testing.assert_close(changed[1][:, :6], predicted_imag[:, :6])


def test_innovation_filter_is_exact_baseline_at_zero_strength_and_live() -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_innovation_pole_reader_config(), slow_cnn_pole_innovation=False)
    ).eval()
    torch.manual_seed(501)
    innovation = AlphabetLM(_innovation_pole_reader_config()).eval()
    innovation.load_state_dict(baseline.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", innovation.slow_cnn_pole_memory)
    assert slow.innovation_filter is not None
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = baseline(tokens)
        actual = innovation(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    innovation(tokens).square().mean().backward()
    assert slow.innovation_filter.raw_strength.grad is not None
    assert slow.innovation_filter.raw_strength.grad.abs().sum() > 0


def test_innovation_checkpoint_freezes_dense_k3_and_trains_only_filter(
    tmp_path: Path,
) -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_innovation_pole_reader_config(), slow_cnn_pole_innovation=False)
    ).eval()
    checkpoint = tmp_path / "dense-k3.pt"
    torch.save({"model": baseline.state_dict()}, checkpoint)
    torch.manual_seed(501)
    innovation = AlphabetLM(_innovation_pole_reader_config()).eval()
    contract = _initialize_slow_innovation_from_trunk(innovation, checkpoint)
    assert contract["enabled"] is True
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        torch.testing.assert_close(
            innovation(tokens), baseline(tokens), atol=0.0, rtol=0.0
        )
    trainable = [
        name for name, parameter in innovation.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.innovation_filter.raw_strength",
        "slow_cnn_pole_memory.innovation_filter.predictor.weight_real",
        "slow_cnn_pole_memory.innovation_filter.predictor.weight_imag",
    }


def test_innovation_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"pole_reader_30m_sha256": "expected"}}
    _validate_slow_innovation_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="innovation source checkpoint"):
        _validate_slow_innovation_source(
            initialization,
            {"source": {"pole_reader_30m_sha256": "different"}},
            enabled=True,
        )


def test_innovation_requires_pole_specific_reader_and_excludes_scheduler() -> None:
    with pytest.raises(ValueError, match="innovation filter requires"):
        replace(
            _complex_query_r16_config(coordinate_read=True),
            slow_cnn_pole_innovation=True,
        )
    with pytest.raises(ValueError, match="must be isolated"):
        replace(
            _scheduled_pole_reader_config(),
            slow_cnn_pole_innovation=True,
        )


def _semantic_clock_pole_reader_config() -> AlphabetLMConfig:
    return replace(
        _scheduled_pole_reader_config(),
        slow_cnn_pole_write_scheduler=False,
        slow_cnn_pole_semantic_clock=True,
    )


def test_semantic_clock_unit_step_matches_fixed_and_zero_step_holds_state() -> None:
    torch.manual_seed(501)
    memory = FixedComplexPoleMemory1D(4, context_length=16, scan_fp32=False)
    real = torch.randn(2, 7, 4)
    imag = torch.randn_like(real)
    fixed = memory(real, imag)
    unit_clock = memory(real, imag, clock_step=torch.ones(2, 7))
    torch.testing.assert_close(unit_clock, fixed, atol=0.0, rtol=0.0)
    clock = torch.zeros(2, 7)
    clock[:, 0] = 1.0
    held_real, held_imag = memory(real, imag, clock_step=clock)
    torch.testing.assert_close(
        held_real[:, 1:], held_real[:, :1].expand_as(held_real[:, 1:])
    )
    torch.testing.assert_close(
        held_imag[:, 1:], held_imag[:, :1].expand_as(held_imag[:, 1:])
    )


def test_semantic_clock_is_identity_initialized_shared_and_live() -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_semantic_clock_pole_reader_config(), slow_cnn_pole_semantic_clock=False)
    ).eval()
    torch.manual_seed(501)
    clocked = AlphabetLM(_semantic_clock_pole_reader_config()).eval()
    clocked.load_state_dict(baseline.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", clocked.slow_cnn_pole_memory)
    assert slow.semantic_clock is not None
    packed = torch.randn(2, 9, 16)
    step = slow.semantic_clock(packed)
    assert step.shape == (2, 9)
    torch.testing.assert_close(step, torch.ones_like(step), atol=0.0, rtol=0.0)
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = baseline(tokens)
        actual = clocked(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    clocked(tokens).square().mean().backward()
    assert slow.semantic_clock.hold.weight.grad is not None
    assert slow.semantic_clock.hold.weight.grad.abs().sum() > 0


def test_semantic_clock_checkpoint_freezes_dense_k3_and_trains_only_clock(
    tmp_path: Path,
) -> None:
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(_semantic_clock_pole_reader_config(), slow_cnn_pole_semantic_clock=False)
    ).eval()
    checkpoint = tmp_path / "dense-k3.pt"
    torch.save({"model": baseline.state_dict()}, checkpoint)
    torch.manual_seed(501)
    clocked = AlphabetLM(_semantic_clock_pole_reader_config()).eval()
    contract = _initialize_slow_semantic_clock_from_trunk(clocked, checkpoint)
    assert contract["enabled"] is True
    trainable = [
        name for name, parameter in clocked.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.semantic_clock.norm.weight",
        "slow_cnn_pole_memory.semantic_clock.hold.weight",
        "slow_cnn_pole_memory.semantic_clock.hold.bias",
    }


def test_semantic_clock_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"pole_reader_30m_sha256": "expected"}}
    _validate_slow_semantic_clock_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="semantic-clock source checkpoint"):
        _validate_slow_semantic_clock_source(
            initialization,
            {"source": {"pole_reader_30m_sha256": "different"}},
            enabled=True,
        )


def test_semantic_clock_requires_reader_and_excludes_other_transport_controls() -> None:
    with pytest.raises(ValueError, match="semantic clock requires"):
        replace(
            _complex_query_r16_config(coordinate_read=True),
            slow_cnn_pole_semantic_clock=True,
        )
    with pytest.raises(ValueError, match="without other transport controls"):
        replace(
            _scheduled_pole_reader_config(),
            slow_cnn_pole_semantic_clock=True,
        )


def _repeated_vector_pole_config() -> AlphabetLMConfig:
    return replace(
        _small(),
        layers=2,
        reader_type="dense_k3",
        memory_layout="local_only",
        repeated_vector_pole_memory=True,
        repeated_vector_pole_interval=1,
        repeated_vector_pole_modes=4,
        repeated_vector_pole_width=2,
        repeated_vector_pole_reader_kernel=3,
        repeated_vector_pole_beta_initial=0.01,
        repeated_vector_pole_minimum_half_life=4.0,
        repeated_vector_pole_maximum_half_life=16.0,
    )


def test_repeated_vector_poles_are_depth_local_token_rate_and_causal() -> None:
    torch.manual_seed(501)
    model = AlphabetLM(_repeated_vector_pole_config()).eval()
    banks = model.repeated_vector_pole_memories
    assert banks is not None
    assert len(banks) == 2
    assert all(isinstance(bank, TokenRateVectorPoleBlock) for bank in banks)
    first = cast("TokenRateVectorPoleBlock", banks[0])
    second = cast("TokenRateVectorPoleBlock", banks[1])
    assert (
        first.pole_memory.raw_damping.data_ptr()
        != second.pole_memory.raw_damping.data_ptr()
    )
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        logits = model(tokens)
        changed_logits = model(changed)
    assert logits.shape == (1, 16, 64)
    torch.testing.assert_close(changed_logits[:, :8], logits[:, :8], atol=2e-6, rtol=0.0)


def test_repeated_vector_poles_preserve_matching_local_initialization() -> None:
    repeated_config = _repeated_vector_pole_config()
    torch.manual_seed(501)
    baseline = AlphabetLM(
        replace(repeated_config, repeated_vector_pole_memory=False)
    )
    torch.manual_seed(501)
    repeated = AlphabetLM(repeated_config)
    baseline_state = baseline.state_dict()
    for name, value in repeated.state_dict().items():
        if not name.startswith("repeated_vector_pole_memories."):
            torch.testing.assert_close(value, baseline_state[name], atol=0.0, rtol=0.0)


def test_repeated_vector_poles_have_live_reader_memory_query_and_synthesis() -> None:
    model = AlphabetLM(_repeated_vector_pole_config())
    model(torch.randint(64, (1, 16))).square().mean().backward()
    memories = model.repeated_vector_pole_memories
    assert memories is not None
    bank = cast("TokenRateVectorPoleBlock", memories[0])
    parameters = (
        bank.reader.weight_real,
        bank.pole_memory.raw_damping,
        bank.vector_query_imag.weight,
        bank.synthesis.weight,
        bank.beta,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )


def test_repeated_vector_poles_exclude_single_bank_and_invalid_schedules() -> None:
    config = _repeated_vector_pole_config()
    with pytest.raises(ValueError, match="invalid"):
        replace(config, slow_cnn_pole_memory=True)
    with pytest.raises(ValueError, match="invalid repeated"):
        replace(config, repeated_vector_pole_interval=3)
    with pytest.raises(ValueError, match="invalid repeated"):
        replace(config, repeated_vector_pole_width=1)


def _factorized_repeated_vector_pole_config(
    *, write_rank: int = 4, query_rank: int = 4
) -> AlphabetLMConfig:
    return replace(
        _repeated_vector_pole_config(),
        repeated_vector_pole_width=16,
        repeated_vector_pole_factorized=True,
        repeated_vector_pole_write_rank=write_rank,
        repeated_vector_pole_query_rank=query_rank,
        repeated_vector_pole_synthesis_rank=4,
    )


@pytest.mark.parametrize(("write_rank", "query_rank"), [(4, 4), (8, 4), (4, 8), (8, 8)])
def test_factorized_expansion_preserves_p32r4_function_exactly(
    tmp_path: Path,
    write_rank: int,
    query_rank: int,
) -> None:
    dense_config = replace(_repeated_vector_pole_config(), repeated_vector_pole_width=4)
    torch.manual_seed(501)
    dense = AlphabetLM(dense_config).eval()
    checkpoint = tmp_path / "dense-repeated.pt"
    torch.save({"model": dense.state_dict()}, checkpoint)
    torch.manual_seed(501)
    factorized = AlphabetLM(
        _factorized_repeated_vector_pole_config(
            write_rank=write_rank,
            query_rank=query_rank,
        )
    ).eval()
    contract = _initialize_repeated_factorized_expansion(factorized, checkpoint)
    assert contract["enabled"] is True
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = dense(tokens)
        actual = factorized(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_factorized_expansion_trains_only_new_state_interface(tmp_path: Path) -> None:
    dense = AlphabetLM(
        replace(_repeated_vector_pole_config(), repeated_vector_pole_width=4)
    )
    checkpoint = tmp_path / "dense-repeated.pt"
    torch.save({"model": dense.state_dict()}, checkpoint)
    factorized = AlphabetLM(_factorized_repeated_vector_pole_config())
    _initialize_repeated_factorized_expansion(factorized, checkpoint)
    trainable = [
        name for name, parameter in factorized.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(
        name.startswith("repeated_vector_pole_memories.")
        and any(
            part in name
            for part in (
                "content_basis",
                "content_delta",
                "query_basis",
                "extra_projection_basis",
                "extra_synthesis",
            )
        )
        for name in trainable
    )
    factorized(torch.randint(64, (1, 16))).square().mean().backward()
    banks = factorized.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    assert bank.extra_synthesis.weight.grad is not None
    assert bank.extra_synthesis.weight.grad.abs().sum() > 0


def test_factorized_r4_interface_preserves_source_and_has_live_synthesis(
    tmp_path: Path,
) -> None:
    dense_config = replace(_repeated_vector_pole_config(), repeated_vector_pole_width=4)
    torch.manual_seed(501)
    dense = AlphabetLM(dense_config).eval()
    checkpoint = tmp_path / "dense-repeated.pt"
    torch.save({"model": dense.state_dict()}, checkpoint)
    factorized_config = replace(
        dense_config,
        repeated_vector_pole_factorized=True,
        repeated_vector_pole_write_rank=4,
        repeated_vector_pole_query_rank=4,
        repeated_vector_pole_synthesis_rank=4,
    )
    torch.manual_seed(501)
    factorized = AlphabetLM(factorized_config).eval()
    _initialize_repeated_factorized_expansion(factorized, checkpoint)
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = dense(tokens)
        actual = factorized(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    banks = factorized.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    assert torch.count_nonzero(bank.extra_projection_basis) > 0
    factorized.train()(tokens).square().mean().backward()
    assert bank.extra_synthesis.weight.grad is not None
    assert bank.extra_synthesis.weight.grad.abs().sum() > 0


def test_factorized_write_is_instantaneously_low_rank_with_wide_state() -> None:
    model = AlphabetLM(_factorized_repeated_vector_pole_config()).eval()
    banks = model.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    real = torch.randn(1, 9, 8)
    imag = torch.randn_like(real)
    packed = torch.cat((real, imag), dim=-1)
    coefficient = bank.reader(real, imag)
    basis = bank.content_basis(packed)
    excitation = bank.complex_factor_product(
        coefficient[0], coefficient[1], basis[0], basis[1]
    )
    singular = torch.linalg.svdvals(torch.complex(excitation[0], excitation[1]))
    assert int((singular > 1.0e-4).sum(dim=-1).max()) <= bank.write_rank
    assert excitation[0].shape[-1] == 16


def test_factorized_state_functional_overrides_act_on_memory_coordinates() -> None:
    model = AlphabetLM(_factorized_repeated_vector_pole_config()).eval()
    banks = model.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    excitation_real = torch.randn(1, 8, bank.pole_modes, bank.vector_width)
    excitation_imag = torch.randn_like(excitation_real)
    expected_real, expected_imag = bank.pole_memory(
        excitation_real,
        excitation_imag,
    )

    handles = _factorized_extra_coordinate_override(model)
    truncated_real, truncated_imag = bank.pole_memory(
        excitation_real,
        excitation_imag,
    )
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(
        truncated_real[..., : bank.baseline_width],
        expected_real[..., : bank.baseline_width],
    )
    torch.testing.assert_close(
        truncated_imag[..., : bank.baseline_width],
        expected_imag[..., : bank.baseline_width],
    )
    assert torch.count_nonzero(truncated_real[..., bank.baseline_width :]) == 0
    assert torch.count_nonzero(truncated_imag[..., bank.baseline_width :]) == 0

    bases = [torch.eye(bank.vector_width, dtype=torch.complex64) for _ in banks]
    handles = _factorized_pca_override(model, bases, rank=4)
    projected_real, projected_imag = bank.pole_memory(
        excitation_real,
        excitation_imag,
    )
    for handle in handles:
        handle.remove()
    assert torch.count_nonzero(projected_real[..., :-4]) == 0
    assert torch.count_nonzero(projected_imag[..., :-4]) == 0
    torch.testing.assert_close(projected_real[..., -4:], expected_real[..., -4:])
    torch.testing.assert_close(projected_imag[..., -4:], expected_imag[..., -4:])


def test_retained_factor_state_fixed_read_matches_collapsed_factorization() -> None:
    collapsed_config = _factorized_repeated_vector_pole_config()
    torch.manual_seed(501)
    collapsed = AlphabetLM(collapsed_config).eval()
    retained = AlphabetLM(
        replace(
            collapsed_config,
            repeated_vector_pole_retain_factor_state=True,
        )
    ).eval()
    retained.load_state_dict(collapsed.state_dict())
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = collapsed(tokens)
        actual = retained(tokens)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.parametrize(
    "write_law",
    ["row_specific", "shared_outer", "pole_outer"],
)
def test_retained_factor_write_laws_preserve_the_p32r4_source(
    tmp_path: Path,
    write_law: str,
) -> None:
    dense_config = replace(_repeated_vector_pole_config(), repeated_vector_pole_width=4)
    torch.manual_seed(501)
    dense = AlphabetLM(dense_config).eval()
    checkpoint = tmp_path / f"dense-{write_law}.pt"
    torch.save({"model": dense.state_dict()}, checkpoint)
    target_config = replace(
        _factorized_repeated_vector_pole_config(),
        repeated_vector_pole_retain_factor_state=True,
        repeated_vector_pole_learned_factor_read=True,
        repeated_vector_pole_factor_write_law=cast("Any", write_law),
    )
    torch.manual_seed(501)
    target = AlphabetLM(target_config).eval()
    _initialize_repeated_factorized_expansion(target, checkpoint)
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = dense(tokens)
        actual = target(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("write_law", ["shared_outer", "pole_outer"])
def test_outer_product_write_laws_are_rank_one_per_pole(write_law: str) -> None:
    config = replace(
        _factorized_repeated_vector_pole_config(),
        repeated_vector_pole_retain_factor_state=True,
        repeated_vector_pole_learned_factor_read=True,
        repeated_vector_pole_factor_write_law=cast("Any", write_law),
    )
    model = AlphabetLM(config).eval()
    banks = model.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    real = torch.randn(1, 9, config.modes)
    imag = torch.randn_like(real)
    packed = torch.cat((real, imag), dim=-1)
    coefficient = bank.reader(real, imag)
    basis = bank.content_basis(packed)
    drive = bank.factor_state_drive(
        packed,
        coefficient[0],
        coefficient[1],
        basis[0],
        basis[1],
    )
    extra = torch.complex(
        drive[0][..., bank.baseline_width :],
        drive[1][..., bank.baseline_width :],
    )
    singular = torch.linalg.svdvals(extra)
    assert int((singular > 1.0e-4).sum(dim=-1).max()) <= 1


def test_from_scratch_variants_share_every_common_initial_tensor() -> None:
    base = replace(
        _factorized_repeated_vector_pole_config(),
        repeated_vector_pole_retain_factor_state=True,
    )
    configs = (
        base,
        replace(base, repeated_vector_pole_learned_factor_read=True),
        replace(
            base,
            repeated_vector_pole_learned_factor_read=True,
            repeated_vector_pole_factor_write_law="shared_outer",
        ),
        replace(
            base,
            repeated_vector_pole_learned_factor_read=True,
            repeated_vector_pole_factor_write_law="pole_outer",
        ),
    )
    states: list[dict[str, torch.Tensor]] = []
    for config in configs:
        torch.manual_seed(501)
        states.append(AlphabetLM(config).state_dict())
    common = set.intersection(*(set(state) for state in states))
    assert common
    for name in common:
        torch.testing.assert_close(
            states[0][name],
            states[1][name],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            states[0][name],
            states[2][name],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            states[0][name],
            states[3][name],
            atol=0.0,
            rtol=0.0,
        )


def test_learned_factor_read_is_identity_initialized_and_trainable() -> None:
    fixed_config = replace(
        _factorized_repeated_vector_pole_config(),
        repeated_vector_pole_retain_factor_state=True,
    )
    torch.manual_seed(501)
    fixed = AlphabetLM(fixed_config).eval()
    learned = AlphabetLM(
        replace(
            fixed_config,
            repeated_vector_pole_learned_factor_read=True,
        )
    ).eval()
    incompatible = learned.load_state_dict(fixed.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all("factor_read" in name for name in incompatible.missing_keys)
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = fixed(tokens)
        actual = learned(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    learned.train()(tokens).square().mean().backward()
    banks = learned.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    assert bank.factor_read_real is not None
    assert bank.factor_read_real.weight.grad is not None
    assert bank.factor_read_real.weight.grad.abs().sum() > 0


def test_retained_factor_checkpoint_contract_freezes_the_source_trunk(
    tmp_path: Path,
) -> None:
    source_config = _factorized_repeated_vector_pole_config()
    source = AlphabetLM(source_config)
    checkpoint = tmp_path / "factorized.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    retained = AlphabetLM(
        replace(
            source_config,
            repeated_vector_pole_retain_factor_state=True,
            repeated_vector_pole_learned_factor_read=True,
        )
    )
    contract = _initialize_repeated_retained_factor_state(retained, checkpoint)
    assert contract["enabled"] is True
    trainable = [
        name for name, parameter in retained.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("repeated_vector_pole_memories.") for name in trainable)
    assert any("factor_read" in name for name in trainable)
    digest = cast("str", contract["checkpoint_sha256"])
    _validate_repeated_retained_source(
        contract,
        {"source": {"factorized_p32r32_js4_4m_sha256": digest}},
        enabled=True,
    )


@pytest.mark.parametrize(
    ("direct", "gate"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mamba_outer_scaffold_is_source_preserving_and_live(
    tmp_path: Path,
    direct: bool,
    gate: bool,
) -> None:
    source_config = replace(
        _factorized_repeated_vector_pole_config(),
        repeated_vector_pole_retain_factor_state=True,
        repeated_vector_pole_learned_factor_read=True,
    )
    torch.manual_seed(501)
    source = AlphabetLM(source_config).eval()
    checkpoint = tmp_path / f"source-{direct}-{gate}.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    target = AlphabetLM(
        replace(
            source_config,
            repeated_vector_pole_mamba_outer=True,
            repeated_vector_pole_outer_direct=direct,
            repeated_vector_pole_outer_gate=gate,
        )
    ).eval()
    contract = _initialize_repeated_mamba_outer(target, checkpoint)
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = source(tokens)
        actual = target(tokens)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    target.train()(tokens).square().mean().backward()
    banks = target.repeated_vector_pole_memories
    assert banks is not None
    bank = cast("FactorizedTokenRateVectorPoleBlock", banks[0])
    assert bank.outer_output is not None
    assert bank.outer_output.weight.grad is not None
    assert bank.outer_output.weight.grad.abs().sum() > 0
    digest = cast("str", contract["checkpoint_sha256"])
    _validate_repeated_mamba_outer_source(
        contract,
        {"source": {"retained_factor_learned_30m_sha256": digest}},
        enabled=True,
    )


def test_factorized_expansion_source_digest_is_enforced() -> None:
    initialization: dict[str, object] = {"checkpoint_sha256": "expected"}
    runtime = {"source": {"repeated_p32r4_30m_sha256": "expected"}}
    _validate_repeated_factorized_source(initialization, runtime, enabled=True)
    with pytest.raises(RuntimeError, match="factorized expansion source checkpoint"):
        _validate_repeated_factorized_source(
            initialization,
            {"source": {"repeated_p32r4_30m_sha256": "different"}},
            enabled=True,
        )


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


def _matrix_slow_config() -> AlphabetLMConfig:
    return replace(
        _vector_slow_config(),
        slow_cnn_pole_matrix_key_width=4,
    )


def test_matrix_slow_memory_preserves_vector_d4_and_is_causal() -> None:
    torch.manual_seed(501)
    vector = AlphabetLM(_vector_slow_config()).eval()
    torch.manual_seed(501)
    matrix = AlphabetLM(_matrix_slow_config()).eval()
    matrix.load_state_dict(vector.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", matrix.slow_cnn_pole_memory)
    assert slow.matrix_key is not None
    assert slow.matrix_query is not None
    assert slow.matrix_key.weight.square().sum() > 0
    torch.testing.assert_close(
        slow.matrix_query.weight,
        torch.zeros_like(slow.matrix_query.weight),
    )
    tokens = torch.randint(64, (1, 16))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(64, changed[:, 8:].shape)
    with torch.no_grad():
        expected = vector(tokens)
        actual = matrix(tokens)
        changed_logits = matrix(changed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(changed_logits[:, :8], actual[:, :8], atol=2e-6, rtol=0.0)


def test_matrix_checkpoint_trains_only_extra_query_and_key_axes(tmp_path: Path) -> None:
    torch.manual_seed(501)
    vector = AlphabetLM(_vector_slow_config())
    checkpoint = tmp_path / "vector.pt"
    torch.save({"model": vector.state_dict()}, checkpoint)
    matrix = AlphabetLM(_matrix_slow_config())
    contract = _initialize_slow_matrix_from_trunk(matrix, checkpoint)
    assert contract["enabled"] is True
    trainable = [name for name, parameter in matrix.named_parameters() if parameter.requires_grad]
    assert set(trainable) == {
        "slow_cnn_pole_memory.matrix_key_norm.weight",
        "slow_cnn_pole_memory.matrix_key.weight",
        "slow_cnn_pole_memory.matrix_query_norm.weight",
        "slow_cnn_pole_memory.matrix_query.weight",
    }


def _nonseparable_slow_config() -> AlphabetLMConfig:
    return replace(
        _matrix_slow_config(),
        slow_cnn_pole_independent_matrix_value=True,
    )


def test_nonseparable_value_preserves_matrix_k4v4_and_breaks_value_sharing() -> None:
    torch.manual_seed(501)
    matrix = AlphabetLM(_matrix_slow_config()).eval()
    slow_matrix = cast("SlowCausalCNNPoleMemory", matrix.slow_cnn_pole_memory)
    assert slow_matrix.matrix_query is not None
    torch.nn.init.normal_(slow_matrix.matrix_query.weight, std=0.02)
    torch.manual_seed(501)
    nonseparable = AlphabetLM(_nonseparable_slow_config()).eval()
    nonseparable.load_state_dict(matrix.state_dict(), strict=False)
    slow = cast("SlowCausalCNNPoleMemory", nonseparable.slow_cnn_pole_memory)
    assert slow.matrix_value is not None
    torch.testing.assert_close(
        slow.matrix_value.weight,
        torch.zeros_like(slow.matrix_value.weight),
    )
    tokens = torch.randint(64, (1, 16))
    with torch.no_grad():
        expected = matrix(tokens)
        actual = nonseparable(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.nn.init.normal_(slow.matrix_value.weight, std=0.02)
    packed = torch.randn(2, 3, 16)
    value_axes = slow.matrix_value_axes(packed, slow.anchor_value(packed))
    assert not torch.allclose(value_axes[..., 0, :], value_axes[..., 1, :])


def test_nonseparable_checkpoint_trains_only_independent_values(tmp_path: Path) -> None:
    torch.manual_seed(501)
    matrix = AlphabetLM(_matrix_slow_config())
    checkpoint = tmp_path / "matrix.pt"
    torch.save({"model": matrix.state_dict()}, checkpoint)
    nonseparable = AlphabetLM(_nonseparable_slow_config())
    contract = _initialize_slow_independent_value_from_trunk(nonseparable, checkpoint)
    assert contract["enabled"] is True
    trainable = [
        name for name, parameter in nonseparable.named_parameters() if parameter.requires_grad
    ]
    assert set(trainable) == {
        "slow_cnn_pole_memory.matrix_value_norm.weight",
        "slow_cnn_pole_memory.matrix_value.weight",
    }


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


def test_parameter_matched_mamba2_uses_matrix_state_geometry() -> None:
    model, parameters, relative_error = build_parameter_matched_mamba(
        48_587_020,
        MambaLMConfig(
            architecture="Mamba2",
            state_size=128,
        ),
    )
    assert parameters == 47_739_744
    assert relative_error < 0.03
    assert model.config.layers == 18
    assert model.config.state_size == 128
    assert model.config.architecture == "Mamba2"


def _small_laplace_mamba() -> LaplaceMambaLMConfig:
    return LaplaceMambaLMConfig(
        vocab_size=64,
        model_width=16,
        layers=2,
        pole_modes=2,
        state_size=2,
        head_width=8,
        conv_width=3,
        context_length=16,
        minimum_half_life=4.0,
        maximum_half_life=16.0,
    )


def test_laplace_mamba_is_causal_and_has_live_integrated_paths() -> None:
    torch.manual_seed(501)
    model = LaplaceMambaLM(_small_laplace_mamba())
    tokens = torch.randint(64, (2, 17))
    changed = tokens.clone()
    changed[:, 10:] = torch.randint(64, changed[:, 10:].shape)
    with torch.no_grad():
        expected = model(tokens[:, :-1])
        actual = model(changed[:, :-1])
    torch.testing.assert_close(actual[:, :10], expected[:, :10], atol=1.0e-6, rtol=0.0)
    logits = model(tokens[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        tokens[:, 1:].flatten(),
    )
    loss.backward()
    block = cast("LaplaceMambaBlock", model.blocks[0])
    for parameter in (
        block.input_projection.weight,
        block.conv.weight,
        block.memory.raw_damping,
        block.memory.raw_frequency,
        block.direct_scale,
        block.output_norm_weight,
        block.output_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_laplace_mamba_matches_mamba_depth_and_parameter_scale() -> None:
    model = LaplaceMambaLM(LaplaceMambaLMConfig())
    assert len(model.blocks) == 19
    assert sum(parameter.numel() for parameter in model.parameters()) == 46_838_464
    block = cast("LaplaceMambaBlock", model.blocks[0])
    assert sum(parameter.numel() for parameter in block.parameters()) == 1_582_144
    assert model.config.inner_real_width == 1_024


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
