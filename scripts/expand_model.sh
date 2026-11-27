#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SOURCE_MODEL="${SOURCE_MODEL:-$QWEN_MIDI_ROOT/runs/pop909-melody-to-piano-v1/final-model}"
DEPTH_MODEL="${DEPTH_MODEL:-$QWEN_MIDI_ROOT/runs/pop909-melody-to-piano-150m-initial}"
DATA_DIR="${DATA_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-to-piano-v1}"
OUTPUT_MODEL="${OUTPUT_MODEL:-$QWEN_MIDI_ROOT/runs/pop909-key-chord-melody-to-piano-150m-initial}"
NEW_VOCAB="$DATA_DIR/remi-plus-vocab-key-chord-v1.json"

require_file "$SOURCE_MODEL/config.json"
require_file "$NEW_VOCAB"

python -m pop909_cp.expand_qwen3_depth \
  --source-model "$SOURCE_MODEL" \
  --output-model "$DEPTH_MODEL" \
  --new-layers 48 \
  --max-verification-difference 1e-5 \
  --overwrite

python -m pop909_cp.expand_qwen3_vocab \
  --source-model "$DEPTH_MODEL" \
  --new-vocab "$NEW_VOCAB" \
  --output-model "$OUTPUT_MODEL"
