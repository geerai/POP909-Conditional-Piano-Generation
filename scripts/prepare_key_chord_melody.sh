#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

AUDIT_DIR="${AUDIT_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-to-piano-v1}"
require_dir "$POP909_ORIGINAL"
require_dir "$POP909_CL_PROCESSED"
require_file "$AUDIT_DIR/accepted-songs.json"

python -m pop909_cp.key_chord_melody_to_piano \
  --pop909-original "$POP909_ORIGINAL" \
  --pop909-cl "$POP909_CL_PROCESSED" \
  --accepted-songs "$AUDIT_DIR/accepted-songs.json" \
  --output-dir "$OUTPUT_DIR" \
  --base-vocab "$PROJECT_ROOT/vocab/remi-plus-vocab-v4.json" \
  --index-csv "$PROJECT_ROOT/metadata/pop909-index.csv"

python -m pop909_cp.validate_key_chord_melody_to_piano --data-dir "$OUTPUT_DIR"
