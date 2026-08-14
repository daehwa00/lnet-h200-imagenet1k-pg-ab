#!/usr/bin/env python3
"""Render every submission PDF page and fail on retired/private text.

The JSON report is a mechanical precursor to, not a replacement for, the
required page-by-page visual review.  Each contact sheet contains at most 12
pages so a reviewer can inspect text flow, floats, clipping, and blank pages at
a useful scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

BANNED_PATTERNS = {
    "retired_or_private_model": re.compile(
        r"H-compact|tuned-EFP|bidirectional|forward then backward|"
        r"InceptionTime|Mini[ _-]?Rocket|PA2WP|EFP16|"
        r"DiagnosticModel|DiagnosticArchitecture|energy-complete|information-complete",
        re.IGNORECASE,
    ),
    "private_filesystem_path": re.compile(r"/(?:home|Users)/(?:local_gpu|secondary_host)/"),
}


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    # Commands are assembled exclusively from the fixed PDF utilities below and
    # repository-controlled paths supplied by the release workflow.
    return subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    """Return a repository-relative path when possible to avoid host leakage."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return resolved.name


def _page_count(pdf: Path) -> int:
    output = _run("pdfinfo", str(pdf)).stdout
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", maxsplit=1)[1])
    message = f"pdfinfo did not report a page count for {pdf}"
    raise RuntimeError(message)


def _page_text(pdf: Path, page: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt") as handle:
        _run(
            "pdftotext",
            "-f",
            str(page),
            "-l",
            str(page),
            str(pdf),
            handle.name,
        )
        return Path(handle.name).read_text(encoding="utf-8", errors="replace")


def _audit_pdf(label: str, pdf: Path, output_root: Path) -> dict[str, Any]:
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    page_count = _page_count(pdf)
    target = output_root / label
    target.mkdir(parents=True, exist_ok=False)
    prefix = target / "page"
    _run("pdftoppm", "-png", "-r", "120", str(pdf), str(prefix))
    pages = sorted(target.glob("page-*.png"))
    if len(pages) != page_count:
        message = f"{label}: rendered {len(pages)}/{page_count} pages"
        raise RuntimeError(message)

    hits: list[dict[str, Any]] = []
    for page in range(1, page_count + 1):
        text = _page_text(pdf, page)
        for name, pattern in BANNED_PATTERNS.items():
            matches = sorted({match.group(0) for match in pattern.finditer(text)})
            if matches:
                hits.append({"page": page, "pattern": name, "matches": matches})

    contact_sheets: list[str] = []
    for offset in range(0, len(pages), 12):
        chunk = pages[offset : offset + 12]
        sheet = target / f"contact-{offset + 1:03d}-{offset + len(chunk):03d}.png"
        _run(
            "montage",
            *(str(path) for path in chunk),
            "-thumbnail",
            "360x",
            "-tile",
            "3x4",
            "-geometry",
            "+8+8",
            "-colorspace",
            "sRGB",
            "-depth",
            "8",
            str(sheet),
        )
        contact_sheets.append(str(sheet))
    return {
        "pdf": _report_path(pdf),
        "sha256": _sha256(pdf),
        "pages": page_count,
        "rendered_pages": [_report_path(path) for path in pages],
        "contact_sheets": [_report_path(Path(path)) for path in contact_sheets],
        "banned_text_hits": hits,
        "mechanical_status": "PASS" if not hits else "FAIL",
        "visual_review": "PENDING",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    for executable in ("pdfinfo", "pdftoppm", "pdftotext", "montage"):
        if shutil.which(executable) is None:
            message = f"required executable is unavailable: {executable}"
            raise RuntimeError(message)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "alphabet.pdf_page_audit.v1",
        "main": _audit_pdf("main", args.main.resolve(), args.output_dir.resolve()),
        "supplement": _audit_pdf(
            "supplement", args.supplement.resolve(), args.output_dir.resolve()
        ),
    }
    report["status"] = (
        "PASS"
        if all(report[key]["mechanical_status"] == "PASS" for key in ("main", "supplement"))
        else "FAIL"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        message = f"PDF text audit failed; see {args.report}"
        raise RuntimeError(message)
    print(  # noqa: T201
        f"PASS: rendered {report['main']['pages']} main and "
        f"{report['supplement']['pages']} supplement pages to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
