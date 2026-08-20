"""Complex associative-scan models and optimized CUDA kernels.

Import concrete APIs from their defining modules. Keeping package import
side-effect free avoids loading Torch and Triton when only metadata is needed.
"""

from __future__ import annotations
