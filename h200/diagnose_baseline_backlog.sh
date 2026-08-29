#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT

cd "${PROJECT_ROOT}"
if [[ ! "${H200_EXPECTED_DIAGNOSTIC_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_DIAGNOSTIC_COMMIT must be the exact diagnostic commit" >&2
  exit 2
fi
ACTUAL_COMMIT="$(git rev-parse --verify HEAD)"
readonly ACTUAL_COMMIT
if [[ "${ACTUAL_COMMIT}" != "${H200_EXPECTED_DIAGNOSTIC_COMMIT}" ]]; then
  echo "ERROR: diagnostic commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: diagnostic checkout is not clean" >&2
  exit 2
fi

exec python3 scripts/diagnose_h200_baseline_backlog.py
