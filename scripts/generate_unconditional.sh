#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-$QWEN_MIDI_ROOT/runs/unconditional-piano-pop-909/final-model}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/runs/unconditional-piano-generation10}"

python -m pop909_cp.generate_unconditional_piano \
  --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --sample-count 10 --meter 2/4 --tempo 100 --top-k 5 \
  --temperature 1.0 --seed 20260924
