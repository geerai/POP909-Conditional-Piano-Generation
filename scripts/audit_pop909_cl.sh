#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-audit}"
require_dir "$POP909_ORIGINAL"
require_dir "$POP909_CL_ROOT"

python -m pop909_cp.audit_pop909_cl \
  --pop909-original "$POP909_ORIGINAL" \
  --pop909-cl "$POP909_CL_ROOT" \
  --output-dir "$OUTPUT_DIR"
