"""Runtime compatibility for the Python 3.10 B200 experiment environment."""

from __future__ import annotations

import typing
from typing import NoReturn

if not hasattr(typing, "assert_never"):

    def _assert_never(value: object) -> NoReturn:
        message = f"Expected code to be unreachable, got: {value!r}"
        raise AssertionError(message)

    typing.assert_never = _assert_never  # type: ignore[attr-defined]
