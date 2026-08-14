# lnet

`lnet` is a research prototype for a trainable Laplace-domain neural layer whose
parameters are continuous-time poles and residues. The implementation uses the
exact zero-order-hold discretization of

\[
q_m'(\xi)=p_m q_m(\xi) + x(\xi), \qquad
y(\xi)=D x(\xi) + \sum_{m=1}^M R_m q_m(\xi) + b.
\]

The current artifact focuses on stable real poles for a verifiable first
prototype. It includes:

- a differentiable pole-residue Laplace layer,
- a synthetic teacher-student regression task,
- a deterministic training loop,
- tests for forward correctness, gradient flow, and learnability.

## Final ALPHABET

The canonical ALPHABET is the radial-log modal autocorrelation model:

```python
from lnet import Alphabet

model = Alphabet(
    config,
    output_dim=num_classes,
    dwconv_kernel_size=5,
    dwconv_dilation=4,
)
```

`K=5, D=4` is the optimized production geometry. Other positive kernel sizes
and dilations preserve sequence length and use the exact PyTorch fallback;
only geometries selected for production need a dedicated fused kernel.

For irregular sampling, `time_delta` has shape `[B,T]` or `[B,T,1]` and is
expressed in the same physical unit as lags 1, 2, and 4. Pole transitions use
exact interval-aware ZOH updates; both depthwise maps and modal lag statistics
use physical-time interpolation. `observation_mask` may additionally have
shape `[B,T,C]` for channel-asynchronous observations, while `valid_mask`
remains `[B,T]` or `[B,T,1]` and excludes padding. The metadata-free unit grid
still dispatches to the measured fused CUDA/Triton kernels.

The fixed-shape exact-split training runtime is intentionally metadata-free.
Irregular datasets must route metadata on every eager model call (including
partial batches and validation) until a dedicated metadata-aware capture
runtime is implemented.

Its backbone lives in `lnet.alphabet_backbone`; the final class contains
its single affine readout directly. Historical descriptor/head candidates have
been removed from the runtime package, while their recorded results remain
under `artifacts/`. No aliases for the retired model class names are installed.
Internal `pac_*` kernel modules retain their historical research filenames, but
they are implementation details rather than model API.

## Run training

```bash
PYTHONPATH=src python -m lnet.cli --epochs 250 --device cpu
```

## Run benchmarks

```bash
PYTHONPATH=src python -m lnet.benchmark_cli
```

## Verification

```bash
python -m pytest
uvx ruff check .
uvx basedpyright
python -m compileall src
```
