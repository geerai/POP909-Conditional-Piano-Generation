#!/usr/bin/env python3
"""Strictly validate Key+Chord+Melody -> Piano JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


TASK = "<TASK_KEY_CHORD_MELODY_TO_PIANO>"
MAX_TOKENS = 2048
MAX_BARS = 32


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_vocab(path: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    payload = read_json(path)
    rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in rows]
    tokens = [str(row["token"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("vocabulary IDs are not contiguous")
    if len(tokens) != len(set(tokens)):
        raise ValueError("vocabulary contains duplicate tokens")
    if int(payload.get("base_vocab_size", -1)) != 1727:
        raise ValueError("extended vocabulary does not declare base_vocab_size=1727")
    if not payload.get("base_vocab_ids_preserved"):
        raise ValueError("extended vocabulary does not preserve old token IDs")
    families = {str(row["token"]): str(row.get("family", "")) for row in rows}
    required = {
        TASK, "<ROLE_CHORD>", "<KEY_MODE_MAJOR>", "<KEY_MODE_MINOR>",
        "<NO_CHORD>", "<CHORD_OTHER>", "<CONDITION_START>",
        "<CONDITION_END>", "<ARRANGEMENT_START>", "<ROLE_MELODY>",
        "<ROLE_PIANO>", "<ROLE_END>", "<BAR_END>",
    }
    missing = sorted(required - set(tokens))
    if missing:
        raise ValueError(f"extended vocabulary lacks required tokens: {missing}")
    return tokens, families, payload


def expect(body: Sequence[str], index: int, token: str) -> int:
    if index >= len(body) or body[index] != token:
        found = body[index] if index < len(body) else "<EOF>"
        raise ValueError(f"expected {token} at body index {index}, found {found}")
    return index + 1


def parse_position(token: str, bar_size: int) -> int:
    match = re.fullmatch(r"<POSITION_(\d{3})>", token)
    if not match:
        raise ValueError(f"invalid position token: {token}")
    value = int(match.group(1))
    if not 0 <= value < bar_size:
        raise ValueError(f"position {value} outside bar size {bar_size}")
    return value


def parse_note_role(
    body: Sequence[str], index: int, role: str, bar_size: int
) -> tuple[int, int]:
    index = expect(body, index, f"<ROLE_{role}>")
    count = 0
    previous_position = -1
    while index < len(body) and body[index] != "<ROLE_END>":
        if index + 2 >= len(body):
            raise ValueError(f"truncated {role} note triple")
        position = parse_position(body[index], bar_size)
        pitch = re.fullmatch(r"<PITCH_(\d{3})>", body[index + 1])
        duration = re.fullmatch(r"<DURATION_(\d{3})>", body[index + 2])
        if not pitch or not 0 <= int(pitch.group(1)) <= 127:
            raise ValueError(f"invalid {role} pitch token: {body[index + 1]}")
        if not duration or not 1 <= int(duration.group(1)) <= 768:
            raise ValueError(f"invalid {role} duration token: {body[index + 2]}")
        if position < previous_position:
            raise ValueError(f"{role} positions are not sorted")
        previous_position = position
        count += 1
        index += 3
    index = expect(body, index, "<ROLE_END>")
    return index, count


def parse_chord_role(
    body: Sequence[str], index: int, bar_size: int
) -> tuple[int, int, int, int]:
    index = expect(body, index, "<ROLE_CHORD>")
    event_count = no_chord_count = other_count = 0
    previous_position = -1
    while index < len(body) and body[index] != "<ROLE_END>":
        position = parse_position(body[index], bar_size)
        if position <= previous_position:
            raise ValueError("chord positions must be strictly increasing")
        if event_count == 0 and position != 0:
            raise ValueError("each bar's chord state must begin at POSITION_000")
        previous_position = position
        index += 1
        if index >= len(body):
            raise ValueError("truncated chord event")
        if body[index] == "<NO_CHORD>":
            no_chord_count += 1
            index += 1
        elif body[index] == "<CHORD_OTHER>":
            other_count += 1
            index += 1
        else:
            triple = body[index:index + 3]
            if len(triple) != 3:
                raise ValueError("truncated factorized chord")
            if not re.fullmatch(r"<CHORD_ROOT_[A-Z_]+>", triple[0]):
                raise ValueError(f"invalid chord root: {triple[0]}")
            if not re.fullmatch(r"<CHORD_QUALITY_[A-Za-z0-9_]+>", triple[1]):
                raise ValueError(f"invalid chord quality: {triple[1]}")
            if not re.fullmatch(r"<CHORD_BASS_[A-Z_]+>", triple[2]):
                raise ValueError(f"invalid chord bass: {triple[2]}")
            index += 3
        event_count += 1
    if not event_count:
        raise ValueError("empty chord role")
    index = expect(body, index, "<ROLE_END>")
    return index, event_count, no_chord_count, other_count


def parse_condition(
    body: Sequence[str], bar_size: int
) -> tuple[list[int], dict[str, int], Counter[str]]:
    index = 0
    bars: list[int] = []
    stats = {"chord_events": 0, "no_chord_events": 0, "other_chord_events": 0, "melody_notes": 0}
    keys: Counter[str] = Counter()
    while index < len(body):
        match = re.fullmatch(r"<BAR_(\d{3})>", body[index])
        if not match:
            raise ValueError(f"expected condition BAR token, found {body[index]}")
        bar_number = int(match.group(1))
        if bar_number != len(bars) or bar_number >= MAX_BARS:
            raise ValueError(f"condition bar sequence is not relative/contiguous: {bar_number}")
        bars.append(bar_number)
        index += 1
        if index + 1 >= len(body):
            raise ValueError("truncated Key header")
        root, mode = body[index:index + 2]
        if not re.fullmatch(r"<KEY_ROOT_[A-Z_]+>", root):
            raise ValueError(f"invalid Key root: {root}")
        if mode not in {"<KEY_MODE_MAJOR>", "<KEY_MODE_MINOR>"}:
            raise ValueError(f"invalid Key mode: {mode}")
        keys[f"{root} {mode}"] += 1
        index += 2
        index, chord_count, no_count, other_count = parse_chord_role(body, index, bar_size)
        stats["chord_events"] += chord_count
        stats["no_chord_events"] += no_count
        stats["other_chord_events"] += other_count
        index, melody_count = parse_note_role(body, index, "MELODY", bar_size)
        stats["melody_notes"] += melody_count
        index = expect(body, index, "<BAR_END>")
    if not bars:
        raise ValueError("condition has no bars")
    return bars, stats, keys


def parse_target(body: Sequence[str], bar_size: int) -> tuple[list[int], int]:
    index = 0
    bars: list[int] = []
    piano_notes = 0
    while index < len(body):
        if body[index] == "<ARRANGEMENT_END>":
            index += 1
            index = expect(body, index, "<SONG_END>")
            if index != len(body):
                raise ValueError("tokens found after SONG_END")
            break
        match = re.fullmatch(r"<BAR_(\d{3})>", body[index])
        if not match:
            raise ValueError(f"expected target BAR token, found {body[index]}")
        bar_number = int(match.group(1))
        if bar_number != len(bars) or bar_number >= MAX_BARS:
            raise ValueError(f"target bar sequence is not relative/contiguous: {bar_number}")
        bars.append(bar_number)
        index += 1
        index, count = parse_note_role(body, index, "PIANO", bar_size)
        piano_notes += count
        index = expect(body, index, "<BAR_END>")
    if not bars:
        raise ValueError("target has no bars")
    return bars, piano_notes


def validate_tokens(tokens: list[str], loss_start: int) -> dict[str, Any]:
    if tokens.count("<CONDITION_START>") != 1 or tokens.count("<CONDITION_END>") != 1:
        raise ValueError("condition boundaries must occur exactly once")
    if tokens.count("<ARRANGEMENT_START>") != 1:
        raise ValueError("ARRANGEMENT_START must occur exactly once")
    cs = tokens.index("<CONDITION_START>")
    ce = tokens.index("<CONDITION_END>")
    target = tokens.index("<ARRANGEMENT_START>")
    if not cs < ce < target or target != ce + 1:
        raise ValueError("bad condition/target boundary order")
    if loss_start != target + 1:
        raise ValueError("loss_start_token_index must point after ARRANGEMENT_START")
    header = tokens[:cs]
    if header and header[0] == "<SONG_START>":
        header = header[1:]
    if len(header) != 3 or header[0] != TASK:
        raise ValueError(f"invalid task/header: {header}")
    meter = re.fullmatch(r"<METER_(\d+)_(\d+)>", header[1])
    if not meter:
        raise ValueError(f"invalid meter token: {header[1]}")
    numerator, denominator = int(meter.group(1)), int(meter.group(2))
    raw_bar_size = numerator * 4 * 48
    if denominator <= 0 or raw_bar_size % denominator:
        raise ValueError("meter cannot be represented on 48-grid")
    bar_size = raw_bar_size // denominator
    if not re.fullmatch(r"<TEMPO_\d{3}>", header[2]):
        raise ValueError(f"invalid tempo token: {header[2]}")
    condition_bars, stats, keys = parse_condition(tokens[cs + 1:ce], bar_size)
    target_bars, piano_notes = parse_target(tokens[target + 1:], bar_size)
    if condition_bars != target_bars:
        raise ValueError("condition and target BAR IDs differ")
    if stats["melody_notes"] <= 0 or piano_notes <= 0:
        raise ValueError("sample has no Melody notes or no Piano notes")
    stats["piano_notes"] = piano_notes
    return {
        "bar_count": len(condition_bars),
        "meter": f"{numerator}/{denominator}",
        "stats": stats,
        "keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    vocab_path = data_dir / "remi-plus-vocab-key-chord-v1.json"
    vocab, _families, vocab_payload = load_vocab(vocab_path)

    record_counts: dict[str, int] = {}
    song_counts: dict[str, int] = {}
    songs_by_split: dict[str, set[str]] = {}
    length_values: list[int] = []
    bar_values: list[int] = []
    aggregate = Counter()
    meter_counts = Counter()
    key_counts = Counter()
    sample_ids: set[str] = set()

    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = 0
        songs: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id", ""))
                location = f"{path}:{line_number} sample_id={sample_id}"
                if not sample_id or sample_id in sample_ids:
                    raise ValueError(f"{location}: missing or duplicate sample_id")
                sample_ids.add(sample_id)
                if row.get("task") != "KEY_CHORD_MELODY_TO_PIANO":
                    raise ValueError(f"{location}: wrong task")
                if row.get("split") != split:
                    raise ValueError(f"{location}: row split does not match file")
                ids = [int(value) for value in row["input_ids"]]
                tokens = [str(value) for value in row.get("tokens", [])]
                if not 1 <= len(ids) <= MAX_TOKENS:
                    raise ValueError(f"{location}: invalid sequence length {len(ids)}")
                if min(ids) < 0 or max(ids) >= len(vocab):
                    raise ValueError(f"{location}: token ID outside vocabulary")
                decoded = [vocab[value] for value in ids]
                if tokens != decoded:
                    raise ValueError(f"{location}: tokens and input_ids disagree")
                if int(row.get("token_count", -1)) != len(ids):
                    raise ValueError(f"{location}: token_count field is incorrect")
                try:
                    parsed = validate_tokens(tokens, int(row["loss_start_token_index"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                if int(row.get("source_bar_count", -1)) != parsed["bar_count"]:
                    raise ValueError(f"{location}: source_bar_count field is incorrect")
                rows += 1
                songs.add(str(row["song_id"]))
                length_values.append(len(ids))
                bar_values.append(parsed["bar_count"])
                aggregate.update(parsed["stats"])
                meter_counts[parsed["meter"]] += 1
                key_counts.update(parsed["keys"])
        if not rows:
            raise ValueError(f"empty split: {path}")
        record_counts[split] = rows
        song_counts[split] = len(songs)
        songs_by_split[split] = songs

    leakage = (
        (songs_by_split["train"] & songs_by_split["validation"])
        | (songs_by_split["train"] & songs_by_split["test"])
        | (songs_by_split["validation"] & songs_by_split["test"])
    )
    if leakage:
        raise ValueError(f"song-level split leakage: {sorted(leakage)[:10]}")

    def numeric_summary(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "mean": round(sum(ordered) / len(ordered), 3),
            "max": ordered[-1],
        }

    report = {
        "status": "PASS",
        "task": "KEY_CHORD_MELODY_TO_PIANO",
        "vocabulary": {
            "path": str(vocab_path),
            "base_size": int(vocab_payload["base_vocab_size"]),
            "extended_size": len(vocab),
            "old_ids_preserved": True,
        },
        "records": record_counts,
        "songs": song_counts,
        "unique_sample_ids": len(sample_ids),
        "sequence_lengths": numeric_summary(length_values),
        "bars_per_sample": numeric_summary(bar_values),
        "event_counts": dict(sorted(aggregate.items())),
        "meter_counts": dict(sorted(meter_counts.items())),
        "key_bar_counts": dict(sorted(key_counts.items())),
        "condition_bar_order": "Key -> Chord -> Melody",
        "target_role": "PIANO",
        "loss_mask": "loss_start_token_index is the first BAR token after ARRANGEMENT_START",
        "condition_target_bar_ids_equal": True,
        "song_split_leakage": False,
        "maximum_tokens": MAX_TOKENS,
        "test_opened_for_validation_only": True,
    }
    write_json(data_dir / "validation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
