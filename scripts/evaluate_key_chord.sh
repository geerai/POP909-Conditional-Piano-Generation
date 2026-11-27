#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to a generation run directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}-evaluation}"

python -m pop909_cp.evaluate_key_chord_generation \
  --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
