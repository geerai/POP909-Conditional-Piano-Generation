#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-$QWEN_MIDI_ROOT/runs/pop909-key-chord-melody-to-piano-150m-v1/final-model}"
DATA_DIR="${DATA_DIR:-$QWEN_MIDI_ROOT/data/pop909-key-chord-melody-to-piano-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_MIDI_ROOT/runs/pop909-key-chord-melody-to-piano-150m-v2-long}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ -z "$RESUME_FROM_CHECKPOINT" ]]; then
  shopt -s nullglob
  current_checkpoints=("$OUTPUT_DIR"/checkpoints/checkpoint-*)
  previous_checkpoints=("$QWEN_MIDI_ROOT"/runs/pop909-key-chord-melody-to-piano-150m-v1/checkpoints/checkpoint-*)
  if (( ${#current_checkpoints[@]} > 0 )); then
    RESUME_FROM_CHECKPOINT="$(printf '%s\n' "${current_checkpoints[@]}" | sort -V | tail -n 1)"
  elif (( ${#previous_checkpoints[@]} > 0 )); then
    RESUME_FROM_CHECKPOINT="$(printf '%s\n' "${previous_checkpoints[@]}" | sort -V | tail -n 1)"
  else
    echo "ERROR: no checkpoint is available to resume" >&2
    exit 1
  fi
fi

torchrun --standalone --nproc_per_node=2 \
  -m pop909_cp.train_key_chord_melody_to_piano \
  --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
  --mode train --max-steps 2000 --learning-rate 1e-5 --warmup-ratio 0.05 \
  --per-device-batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 4 --max-grad-norm 1.0 \
  --eval-steps 100 --save-steps 100 --logging-steps 10 \
  --save-total-limit 5 --resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"
