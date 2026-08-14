from __future__ import annotations

import re
from typing import Literal

import torch

PACDeviceChoice = str
_CUDA_DEVICE_PATTERN = re.compile(r"cuda:\d+")


def resolve_device(choice: PACDeviceChoice) -> str:
    """Resolve a PAC runtime device without importing the broader hybrid stack."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice in {"cpu", "cuda"} or _CUDA_DEVICE_PATTERN.fullmatch(choice):
        return choice
    message = f"Unsupported device: {choice}"
    raise ValueError(message)
