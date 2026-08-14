#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
platform=${2:?platform must be b200 or pro6000}

case "$platform" in
  b200)
    external_expected=424
    ucr_expected=630
    shard_glob='shard-[0-6]'
    ;;
  pro6000)
    external_expected=56
    ucr_expected=90
    shard_glob='shard-7'
    ;;
  *)
    echo "platform must be b200 or pro6000" >&2
    exit 2
    ;;
esac

external_root="$project_root/.omx/results/pac-revised-external-final-20260712"
ucr_root="$project_root/.omx/results/pac-revised-ucr-official-test-20260712"
status_file="$project_root/.omx/logs/pac-revised-final-20260712/supervisor-$platform.status"

while true; do
  bash "$project_root/scripts/launch_pac_revised_final_queues.sh" "$project_root" "$platform"
  external_done=$(find "$external_root/jobs" -path '*/results/external_comparisons.csv' \
    -type f -exec grep -l ',done,' {} + 2>/dev/null | wc -l)
  ucr_done=$(PYTHONPATH="$project_root/src" python - "$ucr_root/shards/$shard_glob" <<'PY'
import csv
import glob
import sys

keys = set()
for pattern in sys.argv[1:]:
    for path in glob.glob(pattern + "/results/low_data_recommended_real.csv"):
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") == "done":
                    keys.add(row.get("job_key", ""))
print(len(keys))
PY
)
  printf '%s external=%s/%s ucr=%s/%s\n' \
    "$(date -Is)" "$external_done" "$external_expected" "$ucr_done" "$ucr_expected" \
    >"$status_file"
  if (( external_done >= external_expected && ucr_done >= ucr_expected )); then
    touch "${status_file}.COMPLETE"
    exit 0
  fi
  sleep 60
done
