"""Remove document metadata, XMP, and trailer identifiers from a PDF."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_INFO_LITERAL = re.compile(
    rb"/(Title|Author|Subject|Keywords|Creator|Producer|CreationDate|ModDate)\s*\([^()\r\n]*\)"
)
_INFO_REFERENCE = re.compile(
    rb"/Metadata\s+\d+\s+\d+\s+R"
)
_TRAILER_ID = re.compile(rb"/ID\s*\[[^\]]*\]", re.DOTALL)


def _blank_match(match: re.Match[bytes]) -> bytes:
    """Keep the original byte length so existing PDF offsets remain valid."""
    value = match.group(0)
    open_paren = value.find(b"(")
    if open_paren < 0:
        return b" " * len(value)
    prefix = value[: open_paren + 1]
    return prefix + b")" + b" " * (len(value) - len(prefix) - 1)


def _scrub_metadata_bytes(data: bytes) -> bytes:
    data = _INFO_LITERAL.sub(_blank_match, data)
    data = _INFO_REFERENCE.sub(lambda match: b" " * len(match.group(0)), data)
    data = _TRAILER_ID.sub(
        lambda match: match.group(0)[: match.group(0).find(b"[") + 1]
        + b" " * (len(match.group(0)) - match.group(0).find(b"[") - 2)
        + b"]",
        data,
    )

    # XMP is usually a plain stream. Blank its payload and the type marker
    # without changing object offsets; references to the object are removed
    # above, so parsers do not see a metadata stream afterward.
    marker = b"/Type/Metadata"
    cursor = 0
    while True:
        start = data.find(marker, cursor)
        if start < 0:
            break
        data = data[:start] + b" " * len(marker) + data[start + len(marker) :]
        stream = data.find(b"stream", start + len(marker))
        end = data.find(b"endstream", stream + len(b"stream")) if stream >= 0 else -1
        if stream >= 0 and end >= 0:
            payload_start = stream + len(b"stream")
            data = data[:payload_start] + b" " * (end - payload_start) + data[end:]
            cursor = end + len(b"endstream")
        else:
            cursor = start + len(marker)
    return data


def scrub(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise RuntimeError("Ghostscript (gs) is required for PDF metadata scrubbing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdf-metadata-scrub-") as directory:
        rendered = Path(directory) / "rendered.pdf"
        subprocess.run(
            [
                ghostscript,
                "-q",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.7",
                f"-sOutputFile={rendered}",
                "-f",
                str(input_path),
                "-c",
                "[ /Title () /Author () /Subject () /Creator () /Producer () /Keywords () "
                "/CreationDate () /ModDate () /DOCINFO pdfmark",
            ],
            check=True,
        )
        scrubbed = _scrub_metadata_bytes(rendered.read_bytes())
        final = Path(directory) / "scrubbed.pdf"
        final.write_bytes(scrubbed)
        final.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    scrub(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
