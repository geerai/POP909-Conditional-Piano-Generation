#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

BASELINE_DIR="${BASELINE_DIR:?Set BASELINE_DIR to the baseline generation run}"
CANDIDATE_DIR="${CANDIDATE_DIR:?Set CANDIDATE_DIR to the candidate generation run}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for comparison results}"

python -m pop909_cp.compare_generation_grammar \
  --baseline-dir "$BASELINE_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output-dir "$OUTPUT_DIR"
