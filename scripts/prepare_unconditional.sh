#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/data/unconditional-piano-pop-909}"
require_dir "$POP909_ORIGINAL"

python -m pop909_cp.prepare_pop909_piano \
  --source-root "$POP909_ORIGINAL" \
  --output-dir "$OUTPUT_DIR" \
  --vocab "$PROJECT_ROOT/vocab/remi-plus-vocab-v4.json" \
  --index-csv "$PROJECT_ROOT/metadata/pop909-index.csv" \
  --overwrite

python -m pop909_cp.validate_pop909_piano \
  --data-dir "$OUTPUT_DIR" \
  --expected-main-midis 909 \
  --require-zero-failures
