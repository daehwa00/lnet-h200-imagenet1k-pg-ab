"""Use pinned official Vim math with the installed causal-conv1d ABI bridge."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

from h200_external_models import _compat_import_context


def build_vision_mamba(checkout: Path, num_classes: int):
    with _compat_import_context(checkout/'vim',('models_mamba','rope','mamba_ssm')):
        sys.path.insert(0,str(checkout/'mamba-1p1p1'))
        # Upstream imports unused text-generation classes during package init.
        # Transformers 5 merged these result-container names; no classifier math changes.
        import transformers.generation as generation
        for name in ('GreedySearchDecoderOnlyOutput','SampleDecoderOnlyOutput'):
            if not hasattr(generation,name):
                setattr(generation,name,generation.GenerateDecoderOnlyOutput)
        scan=importlib.import_module('mamba_ssm.ops.selective_scan_interface')
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function
        scan.causal_conv1d_cuda=SimpleNamespace(
            causal_conv1d_fwd=causal_conv1d_fwd_function,
            causal_conv1d_bwd=causal_conv1d_bwd_function,
        )
        module=importlib.import_module('models_mamba')
        return module.vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
            pretrained=False,num_classes=num_classes,drop_path_rate=0.0)
