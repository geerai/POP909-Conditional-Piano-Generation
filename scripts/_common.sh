#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QWEN_MIDI_ROOT="${QWEN_MIDI_ROOT:?Set QWEN_MIDI_ROOT to the data/model workspace}"
POP909_ORIGINAL="${POP909_ORIGINAL:-$QWEN_MIDI_ROOT/datasets/POP909-original}"
POP909_CL_ROOT="${POP909_CL_ROOT:-$QWEN_MIDI_ROOT/datasets/POP909-CL-Dataset}"
POP909_CL_PROCESSED="${POP909_CL_PROCESSED:-$POP909_CL_ROOT/POP909_processed}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: required directory not found: $1" >&2
    exit 1
  fi
}
