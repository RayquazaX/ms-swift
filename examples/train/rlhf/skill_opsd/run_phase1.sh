#!/bin/bash
# Phase-1 driver: run skill_extract.py on a collected trajectories JSONL.
#
# Usage:
#   bash run_phase1.sh [trajectories.jsonl] [additional CLI args passed through]
# Env overrides:
#   EXTRACTOR=Qwen/Qwen3-4B          (override extractor model)
set -euo pipefail

# shellcheck disable=SC1090
source ~/.bashrc
conda activate opsd

SKILL_OPSD_DIR="/home/qzheng19/ms-swift/examples/train/rlhf/skill_opsd"
cd "$SKILL_OPSD_DIR"

TRAJ="${1:-trajectories.jsonl}"
shift || true

SKILL_LIB="$(python -c 'import yaml; print(yaml.safe_load(open("config.yaml"))["paths"]["skill_library"])')"

python skill_extract.py \
    --trajectories "$TRAJ" \
    --skill_library "$SKILL_LIB" \
    --extractor_model "${EXTRACTOR:-Qwen/Qwen3-4B}" \
    --config "$SKILL_OPSD_DIR/config.yaml" \
    "$@"
