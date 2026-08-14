#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PAC_LOGSIG_V51_Q1_SOURCE_SNAPSHOT:?set PAC_LOGSIG_V51_Q1_SOURCE_SNAPSHOT}"
: "${PAC_LOGSIG_V51_Q1_EXPECTED_HASH:?set PAC_LOGSIG_V51_Q1_EXPECTED_HASH}"

export PAC_LOGSIG_Q1_SOURCE_SNAPSHOT=$PAC_LOGSIG_V51_Q1_SOURCE_SNAPSHOT
export PAC_LOGSIG_Q1_EXPECTED_HASH=$PAC_LOGSIG_V51_Q1_EXPECTED_HASH
export PAC_LOGSIG_Q1_ROOT=.omx/results/pac-logsig-v51-q1-final-20260721
export PAC_LOGSIG_Q1_SMOKE_ROOT=.omx/results/pac-logsig-v51-q1-smoke-20260721
export PAC_LOGSIG_Q1_CLI_MODULE=lnet.pac_logsig_v51_q1_cli
export PAC_LOGSIG_Q1_CAMPAIGN_MODULE=lnet.pac_logsig_v51_q1_campaign
export PAC_LOGSIG_Q1_PREFLIGHT_MODULE=lnet.pac_logsig_v51_preflight
export PAC_LOGSIG_Q1_SLUG=pac-logsig-v51-q1
export PAC_LOGSIG_Q1_LABEL=V5.1
export PAC_LOGSIG_Q1_SMOKE_KEY=ucr:ECGFiveDays:logsig_v51_direct_stem:w1-t2:split7:seed7
export PAC_LOGSIG_Q1_LOG=.omx/logs/pac-logsig-v51-q1-final-20260721-supervisor.log
export PAC_LOGSIG_Q1_LOCK=.omx/locks/pac-logsig-v51-q1-final-20260721.lock

exec bash "$script_dir/supervise_pac_logsig_q1_generic.sh" "$project_root"
