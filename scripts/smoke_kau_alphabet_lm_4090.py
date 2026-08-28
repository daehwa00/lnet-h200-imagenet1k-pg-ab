#!/usr/bin/env python3
"""Compiled full-context RTX 4090 gate for pole-init variants and Mamba."""

from __future__ import annotations

import argparse
import json
import math
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.alphabet_lm import AlphabetLM, AlphabetLMConfig
from lnet.alphabet_lm_mamba import MambaLMConfig, build_parameter_matched_mamba


def _loss(model: nn.Module, tokens: Tensor) -> Tensor:
    logits = model(tokens[:, :-1])
    return functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())


def _step(model: nn.Module, tokens: Tensor) -> dict[str, float]:
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.1, fused=True)
    compiled = cast("nn.Module", torch.compile(model, fullgraph=False, dynamic=False))
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = _loss(compiled, tokens).float()
    initial_loss = float(loss.detach())
    if not 0.5 * math.log(32_768) <= initial_loss <= 2.0 * math.log(32_768):
        raise RuntimeError(f"invalid RTX 4090 initial loss: {initial_loss}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "initial_loss": initial_loss,
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated()),
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
    }
    del compiled, optimizer, model
    torch.compiler.reset()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "palette",
            "grouped",
            "dense",
            "routing",
            "qread",
            "delta_select",
            "tensorpole",
            "dynamic_write_r4",
            "local_only",
            "local_sidecar",
            "normalized_sidecar",
            "normalized_local_sidecar",
            "chunked_semantic_p128",
        ),
        default="all",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("KAU ALPHABET-LM smoke requires the RTX 4090")
    torch.manual_seed(501)
    tokens = torch.randint(32_768, (2, 2_049), device="cuda")
    results = {}
    alphabet_variants = (
        ("alphabet-legacy", "legacy"),
        ("alphabet-palette", "lifetime_palette"),
    )
    if args.only == "palette":
        alphabet_variants = alphabet_variants[1:]
    elif args.only in {
        "grouped",
        "dense",
        "routing",
        "qread",
        "delta_select",
        "tensorpole",
        "dynamic_write_r4",
        "local_only",
        "local_sidecar",
        "normalized_sidecar",
        "normalized_local_sidecar",
        "chunked_semantic_p128",
    }:
        alphabet_variants = ()
    for label, initialization in alphabet_variants:
        torch.manual_seed(501)
        results[label] = _step(
            AlphabetLM(
                AlphabetLMConfig(
                    pole_initialization=cast(
                        "Literal['legacy', 'lifetime_palette']", initialization
                    )
                )
            ),
            tokens,
        )
    if args.only == "grouped":
        torch.manual_seed(501)
        grouped = AlphabetLM(
            AlphabetLMConfig(
                pole_initialization="lifetime_palette",
                memory_banks=8,
                bank_pole_modes=128,
            )
        )
        if sum(parameter.numel() for parameter in grouped.parameters()) != 31_373_824:
            raise RuntimeError("grouped H8P128 parameter contract changed")
        results["alphabet-grouped-h8p128"] = _step(grouped, tokens)
    if args.only == "dense":
        torch.manual_seed(501)
        dense = AlphabetLM(AlphabetLMConfig(reader_type="dense_k3"))
        if sum(parameter.numel() for parameter in dense.parameters()) != 36_714_496:
            raise RuntimeError("dense K3 P320 parameter contract changed")
        results["alphabet-dense-k3-p320"] = _step(dense, tokens)
    if args.only == "routing":
        torch.manual_seed(501)
        routed = AlphabetLM(AlphabetLMConfig(pole_routing="dynamic_write_read"))
        if sum(parameter.numel() for parameter in routed.parameters()) != 35_239_936:
            raise RuntimeError("dynamic write/read parameter contract changed")
        results["alphabet-dynamic-write-read"] = _step(routed, tokens)
    if args.only == "qread":
        torch.manual_seed(501)
        qread = AlphabetLM(
            AlphabetLMConfig(
                memory_readout="query_low_rank",
                query_read_rank=32,
                query_read_initial_scale=0.15,
            )
        )
        if sum(parameter.numel() for parameter in qread.parameters()) != 35_436_556:
            raise RuntimeError("query-read R32 parameter contract changed")
        results["alphabet-qread-r32"] = _step(qread, tokens)
    if args.only == "delta_select":
        torch.manual_seed(501)
        delta_select = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                pole_dynamics="delta_select",
                delta_select_rank=16,
                delta_select_initial_scale=0.3,
            )
        )
        if sum(parameter.numel() for parameter in delta_select.parameters()) != 36_877_324:
            raise RuntimeError("DenseK3 DeltaSelect-R16 parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-delta-select-r16"] = _step(
            delta_select,
            full_microbatch,
        )
    if args.only == "tensorpole":
        torch.manual_seed(501)
        tensorpole = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                memory_layout="tensor_product",
                tensor_temporal_modes=8,
            )
        )
        if sum(parameter.numel() for parameter in tensorpole.parameters()) != 33_659_584:
            raise RuntimeError("DenseK3 TensorPole-M8 parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-tensorpole-m8"] = _step(
            tensorpole,
            full_microbatch,
        )
    if args.only == "dynamic_write_r4":
        torch.manual_seed(501)
        dynamic_write = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                write_map="dynamic_low_rank",
                dynamic_write_rank=4,
                dynamic_write_initial_scale=0.06,
            )
        )
        if sum(parameter.numel() for parameter in dynamic_write.parameters()) != 36_797_452:
            raise RuntimeError("DenseK3 DynamicWrite-R4 parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-dynamic-write-r4"] = _step(
            dynamic_write,
            full_microbatch,
        )
    if args.only == "local_only":
        torch.manual_seed(501)
        local_only = AlphabetLM(
            AlphabetLMConfig(reader_type="dense_k3", memory_layout="local_only")
        )
        if sum(parameter.numel() for parameter in local_only.parameters()) != 36_706_816:
            raise RuntimeError("DenseK3 LocalOnly parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-local-only"] = _step(local_only, full_microbatch)
    if args.only == "local_sidecar":
        torch.manual_seed(501)
        local_sidecar = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                memory_layout="local_sidecar",
                sidecar_initial_scale=0.01,
            )
        )
        if sum(parameter.numel() for parameter in local_sidecar.parameters()) != 40_652_800:
            raise RuntimeError("DenseK3 LocalSidecar parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-local-sidecar"] = _step(local_sidecar, full_microbatch)
    if args.only == "normalized_sidecar":
        torch.manual_seed(501)
        normalized_sidecar = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                memory_layout="local_sidecar",
                sidecar_initial_scale=0.01,
                sidecar_normalize_memory=True,
                sidecar_channelwise_scale=False,
            )
        )
        if sum(parameter.numel() for parameter in normalized_sidecar.parameters()) != 40_649_740:
            raise RuntimeError("DenseK3 NormalizedScalarSidecar parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-normalized-scalar-sidecar"] = _step(
            normalized_sidecar,
            full_microbatch,
        )
    if args.only == "normalized_local_sidecar":
        torch.manual_seed(501)
        normalized_local_sidecar = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                memory_layout="local_sidecar",
                sidecar_initial_scale=0.01,
                sidecar_normalize_memory=True,
                sidecar_channelwise_scale=False,
                sidecar_use_recurrence=False,
            )
        )
        if (
            sum(parameter.numel() for parameter in normalized_local_sidecar.parameters())
            != 40_649_740
        ):
            raise RuntimeError("DenseK3 NormalizedLocalSidecar parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-dense-k3-normalized-local-sidecar"] = _step(
            normalized_local_sidecar,
            full_microbatch,
        )
    if args.only == "chunked_semantic_p128":
        torch.manual_seed(501)
        chunked = AlphabetLM(
            AlphabetLMConfig(
                reader_type="dense_k3",
                memory_layout="local_only",
                chunk_memory=True,
                chunk_size=32,
                chunk_summary_width=128,
                chunk_pole_modes=128,
                chunk_upper_blocks=4,
                chunk_beta_initial=0.01,
                chunk_minimum_half_life=1.0,
                chunk_maximum_half_life=128.0,
            )
        )
        if sum(parameter.numel() for parameter in chunked.parameters()) != 36_920_580:
            raise RuntimeError("Chunked Semantic P128 parameter contract changed")
        full_microbatch = torch.randint(32_768, (8, 2_049), device="cuda")
        results["alphabet-chunked-semantic-p128"] = _step(chunked, full_microbatch)
    if args.only == "all":
        torch.manual_seed(501)
        mamba, parameters, relative_error = build_parameter_matched_mamba(
            34_794_496, MambaLMConfig()
        )
        if parameters != 35_425_280 or relative_error >= 0.03:
            raise RuntimeError("RTX 4090 Mamba parameter match changed")
        results["mamba"] = _step(mamba, tokens)
    print("KAU_ALPHABET_LM_SMOKE=" + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
