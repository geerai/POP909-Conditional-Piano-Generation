#!/usr/bin/env python3
"""Audit paired POP909 and POP909-CL MIDI files before tokenization.

The script is read-only with respect to both datasets. It pairs songs 001-909,
checks the independent MELODY/PIANO tracks in original POP909, checks the
curated chord/key/meter metadata in POP909-CL, and writes machine-readable
reports describing accepted, excluded, and warning-only songs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import miditoolkit
except ImportError as exc:  # pragma: no cover - provides a useful server error
    raise SystemExit(
        "miditoolkit is required. Activate the qwen-midi environment and run "
        "`pip install miditoolkit`."
    ) from exc


EXPECTED_SONG_IDS = tuple(f"{number:03d}" for number in range(1, 910))

# These exclusions come from missing metadata found in POP909-CL and the two
# alignment caveats documented by the POP909-CL maintainers.
KNOWN_EXCLUSIONS: dict[str, list[str]] = {
    "063": ["known_missing_key_signature"],
    "367": ["known_missing_key_signature", "known_missing_chord_track"],
    "518": ["known_downbeat_misalignment"],
    "620": ["known_possible_hand_misalignment"],
    "658": ["known_missing_chord_track"],
}

ALLOWED_OPERATION_TYPES = {
    "shift_start_beat",
    "add_key_change",
    "change_time_signature",
}


def canonical_track_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", (value or "").upper())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def note_range(notes: Iterable[Any]) -> dict[str, int | None]:
    rows = list(notes)
    if not rows:
        return {"note_count": 0, "first_tick": None, "last_tick": None}
    return {
        "note_count": len(rows),
        "first_tick": min(int(note.start) for note in rows),
        "last_tick": max(int(note.end) for note in rows),
    }


def ranges_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left["note_count"] or not right["note_count"]:
        return False
    return max(left["first_tick"], right["first_tick"]) < min(
        left["last_tick"], right["last_tick"]
    )


def ranges_overlap_in_quarters(
    left: dict[str, Any], left_ppq: int, right: dict[str, Any], right_ppq: int
) -> bool:
    if not left["note_count"] or not right["note_count"] or left_ppq <= 0 or right_ppq <= 0:
        return False
    left_start = int(left["first_tick"]) / left_ppq
    left_end = int(left["last_tick"]) / left_ppq
    right_start = int(right["first_tick"]) / right_ppq
    right_end = int(right["last_tick"]) / right_ppq
    return max(left_start, right_start) < min(left_end, right_end)


def find_original_tracks(midi: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_name: dict[str, Any] = {}
    inventory: list[dict[str, Any]] = []
    for index, instrument in enumerate(midi.instruments):
        canonical = canonical_track_name(instrument.name)
        inventory.append(
            {
                "index": index,
                "name": instrument.name,
                "canonical_name": canonical,
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                **note_range(instrument.notes),
            }
        )
        if canonical and canonical not in by_name:
            by_name[canonical] = instrument
    return by_name, inventory


def normalize_operations(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise SystemExit("midi_operations.json must contain a JSON object")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for raw_song_id, rows in payload.items():
        try:
            song_id = f"{int(raw_song_id):03d}"
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid song ID in midi_operations.json: {raw_song_id!r}") from exc
        if song_id not in EXPECTED_SONG_IDS:
            raise SystemExit(f"Out-of-range song ID in midi_operations.json: {raw_song_id!r}")
        if not isinstance(rows, list):
            raise SystemExit(f"Operations for {raw_song_id!r} must be a list")
        normalized[song_id] = rows
    return normalized


def discover_cl_midis(processed_root: Path) -> dict[str, Path]:
    """Index CL files while tolerating the upstream `043 .mid` filename."""
    result: dict[str, Path] = {}
    for path in sorted(processed_root.glob("*.mid")):
        match = re.fullmatch(r"\s*(\d{3})\s*", path.stem)
        if not match:
            continue
        song_id = match.group(1)
        if song_id in result:
            raise SystemExit(
                f"Duplicate POP909-CL MIDI for song {song_id}: "
                f"{result[song_id]} and {path}"
            )
        result[song_id] = path
    return result


def operation_summary(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    unknown: list[str] = []
    for row in rows:
        name = str(row.get("operation", "<missing>")) if isinstance(row, dict) else "<invalid>"
        names.append(name)
        if name not in ALLOWED_OPERATION_TYPES:
            unknown.append(name)
    return names, unknown


def inspect_song(
    song_id: str,
    original_path: Path,
    cl_path: Path,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = list(KNOWN_EXCLUSIONS.get(song_id, []))
    warnings: list[str] = []
    result: dict[str, Any] = {
        "song_id": song_id,
        "original_path": str(original_path),
        "cl_path": str(cl_path),
        "excluded": False,
        "exclusion_reasons": reasons,
        "warnings": warnings,
    }
    if cl_path.name != f"{song_id}.mid":
        warnings.append("noncanonical_cl_filename")

    operation_names, unknown_operations = operation_summary(operations)
    result["operations"] = operation_names
    if unknown_operations:
        reasons.append("unknown_midi_operation")
        result["unknown_operations"] = unknown_operations

    if not original_path.is_file():
        reasons.append("missing_original_midi")
    if not cl_path.is_file():
        reasons.append("missing_cl_midi")
    if reasons and (not original_path.is_file() or not cl_path.is_file()):
        result["excluded"] = True
        result["exclusion_reasons"] = sorted(set(reasons))
        return result

    try:
        original = miditoolkit.MidiFile(original_path)
    except Exception as exc:  # MIDI parser errors vary by backend/version
        reasons.append("unreadable_original_midi")
        result["original_error"] = f"{type(exc).__name__}: {exc}"
        original = None
    try:
        cl_midi = miditoolkit.MidiFile(cl_path)
    except Exception as exc:
        reasons.append("unreadable_cl_midi")
        result["cl_error"] = f"{type(exc).__name__}: {exc}"
        cl_midi = None
    if original is None or cl_midi is None:
        result["excluded"] = True
        result["exclusion_reasons"] = sorted(set(reasons))
        return result

    original_tracks, original_inventory = find_original_tracks(original)
    result["original"] = {
        "ppq": int(original.ticks_per_beat),
        "tracks": original_inventory,
        "tempo_change_count": len(original.tempo_changes),
        "time_signature_change_count": len(original.time_signature_changes),
        "key_signature_change_count": len(original.key_signature_changes),
    }

    melody = original_tracks.get("MELODY")
    piano = original_tracks.get("PIANO")
    if melody is None:
        reasons.append("missing_original_melody_track")
        melody_stats = note_range([])
    else:
        melody_stats = note_range(melody.notes)
        if not melody_stats["note_count"]:
            reasons.append("empty_original_melody_track")
    if piano is None:
        reasons.append("missing_original_piano_track")
        piano_stats = note_range([])
    else:
        piano_stats = note_range(piano.notes)
        if not piano_stats["note_count"]:
            reasons.append("empty_original_piano_track")
    result["original"]["melody"] = melody_stats
    result["original"]["piano"] = piano_stats

    cl_inventory = []
    for index, instrument in enumerate(cl_midi.instruments):
        cl_inventory.append(
            {
                "index": index,
                "name": instrument.name,
                "canonical_name": canonical_track_name(instrument.name),
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                **note_range(instrument.notes),
            }
        )

    score = cl_midi.instruments[0] if cl_midi.instruments else None
    # In this release the human chord annotation is the final/second track.
    # Some files name it "chords" and others "MIDI 01", so its position is
    # more reliable than its text name.
    chord = cl_midi.instruments[-1] if len(cl_midi.instruments) >= 2 else None
    score_stats = note_range(score.notes if score is not None else [])
    chord_stats = note_range(chord.notes if chord is not None else [])

    result["cl"] = {
        "ppq": int(cl_midi.ticks_per_beat),
        "tracks": cl_inventory,
        "score": score_stats,
        "chord": chord_stats,
        "tempo_change_count": len(cl_midi.tempo_changes),
        "time_signatures": [
            {
                "time": int(event.time),
                "numerator": int(event.numerator),
                "denominator": int(event.denominator),
            }
            for event in sorted(cl_midi.time_signature_changes, key=lambda event: int(event.time))
        ],
        "key_signatures": [
            {"time": int(event.time), "key_name": event.key_name}
            for event in sorted(cl_midi.key_signature_changes, key=lambda event: int(event.time))
        ],
        "tempo_changes": [
            {"time": int(event.time), "tempo": float(event.tempo)}
            for event in sorted(cl_midi.tempo_changes, key=lambda event: int(event.time))
        ],
    }

    if score is None or not score_stats["note_count"]:
        reasons.append("missing_or_empty_cl_score_track")
    if chord is None:
        reasons.append("missing_cl_chord_track")
    elif not chord_stats["note_count"]:
        reasons.append("empty_cl_chord_track")
    if not cl_midi.key_signature_changes:
        reasons.append("missing_cl_key_signature")
    if not cl_midi.time_signature_changes:
        reasons.append("missing_cl_time_signature")
    if not cl_midi.tempo_changes:
        reasons.append("missing_cl_tempo")
    if int(original.ticks_per_beat) <= 0 or int(cl_midi.ticks_per_beat) <= 0:
        reasons.append("invalid_ppq")
    elif int(original.ticks_per_beat) != int(cl_midi.ticks_per_beat):
        # Tick values cannot be compared directly, but both sources can be
        # converted through quarter-note units during dataset construction.
        warnings.append("ppq_mismatch_requires_rescaling")

    if melody_stats["note_count"] and piano_stats["note_count"]:
        if not ranges_overlap(melody_stats, piano_stats):
            reasons.append("no_melody_piano_time_overlap")
        ppq = int(original.ticks_per_beat)
        first_gap_beats = abs(
            int(melody_stats["first_tick"]) - int(piano_stats["first_tick"])
        ) / max(1, ppq)
        result["original"]["melody_piano_first_gap_beats"] = round(first_gap_beats, 4)
        if first_gap_beats > 128:
            warnings.append("large_melody_piano_start_gap")

    if chord_stats["note_count"] and piano_stats["note_count"]:
        # These ranges come from two source MIDIs. A total lack of overlap is
        # fatal; a large first-event gap is only a warning because intros and
        # delayed melody/chord entrances are musically normal.
        if not ranges_overlap_in_quarters(
            chord_stats,
            int(cl_midi.ticks_per_beat),
            piano_stats,
            int(original.ticks_per_beat),
        ):
            reasons.append("no_chord_piano_time_overlap_before_alignment")
        first_gap_beats = abs(
            int(chord_stats["first_tick"]) / max(1, int(cl_midi.ticks_per_beat))
            - int(piano_stats["first_tick"]) / max(1, int(original.ticks_per_beat))
        )
        result["cl"]["chord_original_piano_first_gap_beats"] = round(first_gap_beats, 4)
        if first_gap_beats > 16:
            warnings.append("large_chord_piano_start_gap_before_alignment")

    result["excluded"] = bool(reasons)
    result["exclusion_reasons"] = sorted(set(reasons))
    result["warnings"] = sorted(set(warnings))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit paired original POP909 and POP909-CL data."
    )
    parser.add_argument("--pop909-original", type=Path, required=True)
    parser.add_argument("--pop909-cl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    original_root = args.pop909_original.expanduser().resolve()
    cl_root = args.pop909_cl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    processed_root = cl_root / "POP909_processed"
    operations_path = cl_root / "midi_operations.json"

    if not original_root.is_dir():
        raise SystemExit(f"Original POP909 directory not found: {original_root}")
    if not processed_root.is_dir():
        raise SystemExit(f"POP909_processed directory not found: {processed_root}")

    operations = normalize_operations(load_json(operations_path))
    cl_midis = discover_cl_midis(processed_root)
    rows: list[dict[str, Any]] = []
    for index, song_id in enumerate(EXPECTED_SONG_IDS, 1):
        row = inspect_song(
            song_id=song_id,
            original_path=original_root / song_id / f"{song_id}.mid",
            cl_path=cl_midis.get(song_id, processed_root / f"{song_id}.mid"),
            operations=operations.get(song_id, []),
        )
        rows.append(row)
        if index % 100 == 0 or index == len(EXPECTED_SONG_IDS):
            print(f"Audited {index}/{len(EXPECTED_SONG_IDS)} songs")

    accepted = [row["song_id"] for row in rows if not row["excluded"]]
    excluded = {
        row["song_id"]: row["exclusion_reasons"]
        for row in rows
        if row["excluded"]
    }
    warnings = {
        row["song_id"]: row["warnings"]
        for row in rows
        if row["warnings"]
    }
    reason_counts = Counter(
        reason for row in rows for reason in row["exclusion_reasons"]
    )
    warning_counts = Counter(warning for row in rows for warning in row["warnings"])
    operation_counts = Counter(name for row in rows for name in row["operations"])

    report = {
        "schema_version": 1,
        "inputs": {
            "pop909_original": str(original_root),
            "pop909_cl": str(cl_root),
            "pop909_cl_processed": str(processed_root),
            "midi_operations": str(operations_path),
        },
        "summary": {
            "expected_songs": len(EXPECTED_SONG_IDS),
            "audited_songs": len(rows),
            "accepted_songs": len(accepted),
            "excluded_songs": len(excluded),
            "songs_with_warnings": len(warnings),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
            "operation_counts": dict(sorted(operation_counts.items())),
        },
        "songs": rows,
    }

    write_json(output_dir / "dataset-audit.json", report)
    write_json(output_dir / "accepted-songs.json", accepted)
    write_json(output_dir / "excluded-songs.json", excluded)
    write_json(output_dir / "warning-songs.json", warnings)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Reports written to: {output_dir}")


if __name__ == "__main__":
    main()
