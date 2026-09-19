"""The five frozen model definitions, freshly initialized for 200 classes."""
import importlib.util
import sys
import types
from pathlib import Path
import torch

MODELS = ('va_k96','va_k128','convnextv2_atto','tinyvim_s','parc_net_s')


def build(key, source_root, num_classes=200):
    if key not in MODELS or num_classes != 200:
        raise ValueError('CUB supports the fixed five-model panel and 200 classes only')
    if key in ('va_k96','va_k128'):
        import a2d_r2k3_runtime as runtime
        if key == 'va_k96':
            import run_lnet_k96_p128_d2262_imagenet1k as spec
            model = spec.family._build(spec.VARIANT,runtime.model_config(output_dim=num_classes))
        else:
            import run_lnet_k128_p160_160_160_128_d2262_h200_imagenet1k as spec
            model = spec.anchor._build(spec.VARIANT,runtime.model_config(output_dim=num_classes))
        model = spec.PrimaryLogitsAdapter(model.prepare_for_compiled_training_())
        expected = 2842824 if key == 'va_k96' else 4672776
        if sum(p.numel() for p in model.parameters()) != expected:
            raise ValueError('VA architecture parameter identity changed')
        return model
    if key == 'convnextv2_atto':
        import timm
        return timm.create_model(key,pretrained=False,num_classes=num_classes)
    import h200_external_models as external
    if key == 'parc_net_s':
        # Fixed 224x224 classification needs no custom rectangular adapter.
        return external._build_model(key,Path(source_root)/'parc_net',num_classes)
    if importlib.util.find_spec('selective_scan_cuda') is None:
        raise RuntimeError('Pinned selective_scan_cuda extension missing; refusing a fallback model')
    helper_name = 'timm.models.layers.helpers'
    original = sys.modules.get(helper_name)
    helper = types.ModuleType(helper_name)
    from timm.layers import to_2tuple
    helper.to_2tuple = to_2tuple
    sys.modules[helper_name] = helper
    try:
        module = external._external_module(Path(source_root)/'tinyvim','model.tinyvim',('model',))
    finally:
        if original is None: sys.modules.pop(helper_name,None)
        else: sys.modules[helper_name] = original
    return module.TinyViM_S(pretrained=False,num_classes=num_classes,distillation=False)
