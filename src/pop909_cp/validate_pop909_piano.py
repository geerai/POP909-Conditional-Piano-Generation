#!/usr/bin/env python3
"""Strict read-only validation for POP909 PIANO-only REMI+ data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


MAX_TOKENS = 2048
FORBIDDEN_EXACT = {
    "<TASK_PIANO_LM>",
    "<TASK_MELODY_TO_ARRANGEMENT>",
    "<TASK_ARRANGEMENT_LM>",
    "<TASK_MELODY_TO_ROLE>",
    "<CONDITION_START>",
    "<CONDITION_END>",
    "<ROLE_MELODY>",
    "<ROLE_GUITAR>",
    "<ROLE_BASS>",
    "<ROLE_DRUMS>",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_vocab(path: Path) -> list[str]:
    rows = sorted(json.loads(path.read_text(encoding="utf-8"))["tokens"], key=lambda row: int(row["id"]))
    tokens = [str(row["token"]) for row in rows]
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("vocabulary IDs are not contiguous")
    if len(tokens) != 1727 or len(tokens) != len(set(tokens)):
        raise ValueError("expected 1,727 unique vocabulary tokens")
    return tokens


def validate_grammar(tokens: list[str], true_start: bool, true_end: bool) -> dict[str, int]:
    index = 0
    if true_start:
        if not tokens or tokens[0] != "<SONG_START>":
            raise ValueError("true-start sample lacks SONG_START")
        index += 1
    elif tokens and tokens[0] == "<SONG_START>":
        raise ValueError("non-start sample contains SONG_START")

    if index >= len(tokens) or not re.fullmatch(r"<METER_\d+_\d+>", tokens[index]):
        raise ValueError("expected METER after optional SONG_START")
    index += 1
    if index >= len(tokens) or not re.fullmatch(r"<TEMPO_\d{3}>", tokens[index]):
        raise ValueError("expected TEMPO")
    index += 1
    expected_header = ["<ROLE_PIANO>", "<PIANO_000>", "<ARRANGEMENT_START>"]
    if tokens[index:index + 3] != expected_header:
        raise ValueError(f"invalid header: {tokens[index:index + 3]}")
    index += 3

    bar_count = 0
    note_count = 0
    while index < len(tokens) and tokens[index] == "<BAR>":
        bar_count += 1
        index += 1
        if index >= len(tokens) or tokens[index] != "<ROLE_PIANO>":
            raise ValueError(f"bar {bar_count}: expected ROLE_PIANO")
        index += 1
        while index < len(tokens) and tokens[index] != "<ROLE_END>":
            if index + 2 >= len(tokens):
                raise ValueError(f"bar {bar_count}: truncated note")
            if not re.fullmatch(r"<POSITION_\d{3}>", tokens[index]):
                raise ValueError(f"bar {bar_count}: expected POSITION at {index}")
            if not re.fullmatch(r"<PITCH_\d{3}>", tokens[index + 1]):
                raise ValueError(f"bar {bar_count}: expected PITCH at {index + 1}")
            if not re.fullmatch(r"<DURATION_\d{3}>", tokens[index + 2]):
                raise ValueError(f"bar {bar_count}: expected DURATION at {index + 2}")
            note_count += 1
            index += 3
        if index >= len(tokens) or tokens[index] != "<ROLE_END>":
            raise ValueError(f"bar {bar_count}: missing ROLE_END")
        index += 1
        if index >= len(tokens) or tokens[index] != "<BAR_END>":
            raise ValueError(f"bar {bar_count}: missing BAR_END")
        index += 1
    if not bar_count or not note_count:
        raise ValueError("sample must contain at least one bar and one piano note")

    if true_end:
        if tokens[index:] != ["<ARRANGEMENT_END>", "<SONG_END>"]:
            raise ValueError(f"true-end sample has invalid suffix: {tokens[index:]}")
        index = len(tokens)
    elif index != len(tokens):
        raise ValueError(f"non-end sample has illegal suffix: {tokens[index:]}")
    return {"bars": bar_count, "notes": note_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-main-midis", type=int, default=909)
    parser.add_argument("--require-zero-failures", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    vocab = load_vocab(data_dir / "remi-plus-vocab-v4.json")
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary["main_midi_count"]) != args.expected_main_midis:
        raise SystemExit(f"expected {args.expected_main_midis} main MIDIs")
    if summary.get("task_token_emitted") is not False:
        raise SystemExit("summary does not prove that Task tokens are absent")
    if summary.get("track_policy") != "use exactly one explicitly named PIANO track; ignore MELODY and BRIDGE":
        raise SystemExit("unexpected track policy")

    with (data_dir / "conversion-audit.csv").open(encoding="utf-8-sig", newline="") as handle:
        audits = list(csv.DictReader(handle))
    if len(audits) != args.expected_main_midis:
        raise SystemExit(f"audit has {len(audits)} rows, expected {args.expected_main_midis}")
    failures = [row for row in audits if row["status"] != "PASS"]
    if args.require_zero_failures and failures:
        raise SystemExit(f"conversion has {len(failures)} failures; inspect conversion-audit.csv")
    passed_song_ids = {row["song_id"] for row in audits if row["status"] == "PASS"}
    for row in audits:
        if row["status"] != "PASS":
            continue
        if row.get("meter_source") != "POP909 index.xlsx num_beats_per_measure / 4":
            raise SystemExit(f"{row['song_id']}: meter does not come from official POP909 index")
        if row.get("bar_alignment_source") != "POP909 beat_midi.txt column 2":
            raise SystemExit(f"{row['song_id']}: bars are not aligned to official downbeats")
        error = row.get("median_primary_downbeat_interval_error_grid", "")
        if error == "" or float(error) > 2:
            raise SystemExit(
                f"{row['song_id']}: downbeat/bar interval mismatch is {error!r} grids"
            )
    cleaned_midis = list((data_dir / "cleaned-midi").rglob("*.mid"))
    if len(cleaned_midis) != len(passed_song_ids):
        raise SystemExit(
            f"cleaned MIDI count {len(cleaned_midis)} != passed songs {len(passed_song_ids)}"
        )

    split_songs: dict[str, set[str]] = {}
    record_counts: dict[str, int] = {}
    total_bars = 0
    total_notes = 0
    maximum_length = 0
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        songs: set[str] = set()
        records = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = [int(value) for value in row["input_ids"]]
                if not 1 <= int(row["loss_start_token_index"]) < len(ids) <= MAX_TOKENS:
                    raise ValueError(f"{path}:{line_number}: invalid length/loss start")
                if min(ids) < 0 or max(ids) >= len(vocab):
                    raise ValueError(f"{path}:{line_number}: token ID outside vocabulary")
                tokens = [vocab[value] for value in ids]
                forbidden = sorted(set(tokens) & FORBIDDEN_EXACT)
                forbidden += sorted(
                    token for token in set(tokens)
                    if token.startswith(("<TARGET_ROLE_", "<DRUM_", "<GUITAR_", "<BASS_", "<MELODY_", "<FLUTE_"))
                )
                if forbidden:
                    raise ValueError(f"{path}:{line_number}: forbidden tokens {forbidden}")
                parsed = validate_grammar(
                    tokens,
                    bool(row["contains_true_song_start"]),
                    bool(row["contains_true_song_end"]),
                )
                if row.get("roles") != ["PIANO"] or row.get("task") != "UNCONDITIONAL_PIANO_ACCOMPANIMENT":
                    raise ValueError(f"{path}:{line_number}: wrong task/role metadata")
                songs.add(str(row["song_id"]))
                records += 1
                total_bars += parsed["bars"]
                total_notes += parsed["notes"]
                maximum_length = max(maximum_length, len(ids))
        if not records:
            raise SystemExit(f"{path} is empty")
        split_songs[split] = songs
        record_counts[split] = records

    intersections = {
        "train_validation": sorted(split_songs["train"] & split_songs["validation"]),
        "train_test": sorted(split_songs["train"] & split_songs["test"]),
        "validation_test": sorted(split_songs["validation"] & split_songs["test"]),
    }
    if any(intersections.values()):
        raise SystemExit(f"song leakage detected: {intersections}")

    with (data_dir / "roundtrip-audit.csv").open(encoding="utf-8-sig", newline="") as handle:
        roundtrip = list(csv.DictReader(handle))
    if not roundtrip or any(row["status"] != "PASS" for row in roundtrip):
        raise SystemExit("roundtrip audit is absent or contains failures")

    report = {
        "status": "PASS",
        "vocabulary_size": len(vocab),
        "main_midi_count": summary["main_midi_count"],
        "passed_song_count": len(passed_song_ids),
        "failed_song_count": len(failures),
        "cleaned_midi_count": len(cleaned_midis),
        "record_counts": record_counts,
        "song_counts": {split: len(songs) for split, songs in split_songs.items()},
        "maximum_sequence_length": maximum_length,
        "total_parsed_bars_across_overlapping_windows": total_bars,
        "total_parsed_notes_across_overlapping_windows": total_notes,
        "song_group_leakage": {key: len(value) for key, value in intersections.items()},
        "task_token_present": False,
        "only_piano_role_present": True,
        "roundtrip_example_count": len(roundtrip),
        "test_opened_during_training": False,
    }
    write_json(data_dir / "validation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
