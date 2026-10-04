#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}" \
BS=1 \
LR="${LR:-1e-5}" \
DRIFT_AWARE=true \
USE_MUON=true \
BASELINE=subzero_muon_da \
"${SCRIPT_DIR}/subzero.sh" \
    --use_muon true \
    --drift_aware true \
    --da_beta_min "${DA_BETA_MIN:-0.50}" \
    --da_beta_max "${DA_BETA_MAX:-0.99}" \
    --gg_ns_eta_g "${GG_NS_ETA_G:-0.50}" \
    --gg_ns_pre_steps "${GG_NS_PRE_STEPS:-3}" \
    "$@"
