#!/usr/bin/env python3
"""Evaluate POP909-CL key+chord+melody conditioned piano generations.

The evaluator uses the text token dumps so that results exactly follow the
model representation.  For malformed generations, the first complete
occurrence of each BAR id is retained; duplicated, truncated, or out-of-order
bars are reported as validity errors and excluded from paired comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


BAR_RE = re.compile(r"<BAR_(\d+)>")
TOKEN_VALUE_RE = re.compile(r"<[^>]+_(\d+)>")
ROOT_TO_PC = {
    "C": 0, "C_SHARP": 1, "D": 2, "D_SHARP": 3, "E": 4, "F": 5,
    "F_SHARP": 6, "G": 7, "G_SHARP": 8, "A": 9, "A_SHARP": 10, "B": 11,
}
QUALITY_INTERVALS = {
    "M": (0, 4, 7),
    "m": (0, 3, 7),
    "D7": (0, 4, 7, 10),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "half_o7": (0, 3, 6, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}
DURATION_BINS = ("<=12", "13-24", "25-48", "49-96", ">96")


def token_number(token: str) -> int:
    match = TOKEN_VALUE_RE.fullmatch(token)
    if not match:
        raise ValueError(f"token has no integer suffix: {token}")
    return int(match.group(1))


def split_bar_occurrences(tokens: list[str]) -> list[dict]:
    starts = [i for i, token in enumerate(tokens) if BAR_RE.fullmatch(token)]
    occurrences = []
    for occurrence_index, start in enumerate(starts):
        end = starts[occurrence_index + 1] if occurrence_index + 1 < len(starts) else len(tokens)
        segment = tokens[start:end]
        bar_id = int(BAR_RE.fullmatch(segment[0]).group(1))
        complete = len(segment) >= 3 and segment[-2:] == ["<ROLE_END>", "<BAR_END>"]
        occurrences.append({
            "bar_id": bar_id,
            "occurrence_index": occurrence_index,
            "tokens": segment,
            "complete": complete,
        })
    return occurrences


def parse_notes(segment: list[str]) -> tuple[list[dict], int]:
    notes = []
    invalid_positions = 0
    for index, token in enumerate(segment):
        if not token.startswith("<POSITION_"):
            continue
        if index + 2 >= len(segment):
            invalid_positions += 1
            continue
        pitch_token, duration_token = segment[index + 1:index + 3]
        if not pitch_token.startswith("<PITCH_") or not duration_token.startswith("<DURATION_"):
            invalid_positions += 1
            continue
        notes.append({
            "position": token_number(token),
            "pitch": token_number(pitch_token),
            "duration": token_number(duration_token),
        })
    return notes, invalid_positions


def parse_piano(path: Path) -> dict:
    tokens = path.read_text(encoding="utf-8").split()
    occurrences = split_bar_occurrences(tokens)
    first_complete = {}
    duplicate_ids = []
    invalid_positions = 0
    complete_occurrences = 0
    for occurrence in occurrences:
        notes, invalid = parse_notes(occurrence["tokens"])
        invalid_positions += invalid
        occurrence["notes"] = notes
        if not occurrence["complete"]:
            continue
        complete_occurrences += 1
        bar_id = occurrence["bar_id"]
        if bar_id in first_complete:
            duplicate_ids.append(bar_id)
        else:
            first_complete[bar_id] = notes
    ids_in_order = [o["bar_id"] for o in occurrences]
    transitions = sum(b != a + 1 for a, b in zip(ids_in_order, ids_in_order[1:]))
    return {
        "tokens": tokens,
        "bars": first_complete,
        "occurrences": occurrences,
        "bar_ids_in_order": ids_in_order,
        "complete_occurrences": complete_occurrences,
        "duplicate_bar_ids": duplicate_ids,
        "non_contiguous_transitions": transitions,
        "truncated_bar_occurrences": sum(not o["complete"] for o in occurrences),
        "invalid_note_positions": invalid_positions,
        "ends_with_complete_bar": tokens[-2:] == ["<ROLE_END>", "<BAR_END>"],
    }


def parse_chords(path: Path) -> tuple[dict[int, list[dict]], str]:
    tokens = path.read_text(encoding="utf-8").split()
    meter = next((t[7:-1] for t in tokens if t.startswith("<METER_") and t.endswith(">")), "unknown")
    bars = {}
    for occurrence in split_bar_occurrences(tokens):
        segment = occurrence["tokens"]
        chords = []
        for i, token in enumerate(segment):
            if not token.startswith("<POSITION_") or i + 3 >= len(segment):
                continue
            root_token, quality_token, bass_token = segment[i + 1:i + 4]
            if not (root_token.startswith("<CHORD_ROOT_") and quality_token.startswith("<CHORD_QUALITY_")
                    and bass_token.startswith("<CHORD_BASS_")):
                continue
            root = root_token[len("<CHORD_ROOT_"):-1]
            quality = quality_token[len("<CHORD_QUALITY_"):-1]
            chords.append({"position": token_number(token), "root": root, "quality": quality})
        bars[occurrence["bar_id"]] = sorted(chords, key=lambda c: c["position"])
    return bars, meter


def active_chord(chords: list[dict], position: int) -> dict | None:
    active = None
    for chord in chords:
        if chord["position"] <= position:
            active = chord
        else:
            break
    return active


def active_chord_at(chord_bars: dict[int, list[dict]], bar_id: int, position: int) -> dict | None:
    """Return the latest chord event, allowing harmony to carry across bars."""
    active = active_chord(chord_bars.get(bar_id, []), position)
    if active is not None:
        return active
    for prior_bar in sorted((b for b in chord_bars if b < bar_id), reverse=True):
        if chord_bars[prior_bar]:
            return chord_bars[prior_bar][-1]
    return None


def chord_tones(chord: dict) -> set[int] | None:
    root = ROOT_TO_PC.get(chord["root"])
    intervals = QUALITY_INTERVALS.get(chord["quality"])
    if root is None or intervals is None:
        return None
    return {(root + interval) % 12 for interval in intervals}


def harmony_metrics(bars: dict[int, list[dict]], chord_bars: dict[int, list[dict]], bar_ids: list[int]) -> dict:
    all_hits = all_total = first_hits = first_total = 0
    per_bar = []
    for bar_id in bar_ids:
        hits = total = beat_hits = beat_total = 0
        for note in bars[bar_id]:
            chord = active_chord_at(chord_bars, bar_id, note["position"])
            tones = chord_tones(chord) if chord else None
            if tones is None:
                continue
            hit = note["pitch"] % 12 in tones
            total += 1
            hits += int(hit)
            if note["position"] == 0:
                beat_total += 1
                beat_hits += int(hit)
        all_hits += hits
        all_total += total
        first_hits += beat_hits
        first_total += beat_total
        per_bar.append({
            "bar_id": bar_id,
            "hits": hits,
            "evaluable_notes": total,
            "overall_ctr": hits / total if total else None,
            "first_beat_hits": beat_hits,
            "first_beat_evaluable_notes": beat_total,
            "first_beat_ctr": beat_hits / beat_total if beat_total else None,
        })
    return {
        "overall_ctr": all_hits / all_total if all_total else None,
        "overall_hits": all_hits,
        "overall_evaluable_notes": all_total,
        "first_beat_ctr": first_hits / first_total if first_total else None,
        "first_beat_hits": first_hits,
        "first_beat_evaluable_notes": first_total,
        "per_bar": per_bar,
    }


def duration_bin(duration: int) -> str:
    if duration <= 12:
        return "<=12"
    if duration <= 24:
        return "13-24"
    if duration <= 48:
        return "25-48"
    if duration <= 96:
        return "49-96"
    return ">96"


def rhythm_metrics(bars: dict[int, list[dict]], bar_ids: list[int]) -> dict:
    note_counts = [len(bars[b]) for b in bar_ids]
    iois = []
    durations = []
    position_hist = Counter()
    patterns = []
    for bar_id in bar_ids:
        notes = bars[bar_id]
        onsets = sorted({note["position"] for note in notes})
        iois.extend(b - a for a, b in zip(onsets, onsets[1:]))
        durations.extend(note["duration"] for note in notes)
        position_hist.update(note["position"] for note in notes)
        patterns.append(tuple(onsets))
    pattern_counts = Counter(patterns)
    repeated_bars = sum(count for count in pattern_counts.values() if count > 1)
    return {
        "notes_per_bar_mean": statistics.mean(note_counts) if note_counts else None,
        "notes_per_bar_median": statistics.median(note_counts) if note_counts else None,
        "unique_onset_ioi_mean": statistics.mean(iois) if iois else None,
        "unique_onset_ioi_median": statistics.median(iois) if iois else None,
        "duration_histogram": dict(Counter(duration_bin(d) for d in durations)),
        "onset_position_histogram": {str(k): v for k, v in sorted(position_hist.items())},
        "bar_rhythm_pattern_repetition_rate": repeated_bars / len(patterns) if patterns else None,
        "note_counts_per_bar": dict(zip(map(str, bar_ids), note_counts)),
    }


def mean_bar_onset_jaccard(a_bars: dict, b_bars: dict, bar_ids: list[int]) -> float | None:
    values = []
    for bar_id in bar_ids:
        a = {n["position"] for n in a_bars[bar_id]}
        b = {n["position"] for n in b_bars[bar_id]}
        union = a | b
        values.append(len(a & b) / len(union) if union else 1.0)
    return statistics.mean(values) if values else None


def bar_feature(notes: list[dict]) -> list[float]:
    pitch = [0.0] * 12
    rhythm = [0.0] * 16
    for note in notes:
        pitch[note["pitch"] % 12] += 1.0
        rhythm[min(15, note["position"] // 12)] += 1.0
    total = max(1, len(notes))
    return [v / total for v in pitch] + [v / total for v in rhythm] + [len(notes) / 20.0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def self_similarity(bars: dict[int, list[dict]], bar_ids: list[int]) -> list[list[float]]:
    features = [bar_feature(bars[b]) for b in bar_ids]
    return [[cosine(a, b) for b in features] for a in features]


def structure_metrics(matrix: list[list[float]]) -> dict:
    n = len(matrix)
    off_diagonal = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    lag = []
    for distance in range(1, n):
        values = [matrix[i][i + distance] for i in range(n - distance)]
        lag.append({"lag_bars": distance, "mean_similarity": statistics.mean(values)})
    return {
        "repetition_rate_at_0_8": (
            sum(value >= 0.8 for value in off_diagonal) / len(off_diagonal) if off_diagonal else None
        ),
        "mean_off_diagonal_similarity": statistics.mean(off_diagonal) if off_diagonal else None,
        "lag_similarity": lag,
    }


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return numerator / (da * db) if da and db else None


def matrix_correlation(a: list[list[float]], b: list[list[float]]) -> float | None:
    n = len(a)
    av = [a[i][j] for i in range(n) for j in range(i + 1, n)]
    bv = [b[i][j] for i in range(n) for j in range(i + 1, n)]
    return pearson(av, bv)


def write_matrix(path: Path, bar_ids: list[int], matrix: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bar_id", *bar_ids])
        for bar_id, row in zip(bar_ids, matrix):
            writer.writerow([bar_id, *[f"{value:.8f}" for value in row]])


def safe_delta(a: float | None, b: float | None) -> float | None:
    return b - a if a is not None and b is not None else None


def mean_present(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def evaluate_sample(sample_dir: Path, output_dir: Path) -> tuple[dict, dict]:
    report = json.loads((sample_dir / "generation-report.json").read_text(encoding="utf-8"))
    chord_bars, meter = parse_chords(sample_dir / "prompt-tokens.txt")
    reference = parse_piano(sample_dir / "reference-tokens.txt")
    generated = parse_piano(sample_dir / "generated-tokens.txt")
    common_ids = sorted(set(chord_bars) & set(reference["bars"]) & set(generated["bars"]))

    ref_harmony = harmony_metrics(reference["bars"], chord_bars, common_ids)
    gen_harmony = harmony_metrics(generated["bars"], chord_bars, common_ids)
    ref_rhythm = rhythm_metrics(reference["bars"], common_ids)
    gen_rhythm = rhythm_metrics(generated["bars"], common_ids)
    ref_ssm = self_similarity(reference["bars"], common_ids)
    gen_ssm = self_similarity(generated["bars"], common_ids)
    ref_structure = structure_metrics(ref_ssm)
    gen_structure = structure_metrics(gen_ssm)

    sample_output = output_dir / sample_dir.name
    sample_output.mkdir(parents=True, exist_ok=True)
    write_matrix(sample_output / "reference-ssm.csv", common_ids, ref_ssm)
    write_matrix(sample_output / "generated-ssm.csv", common_ids, gen_ssm)
    with (sample_output / "per-bar-harmony.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "bar_id", "reference_overall_ctr", "generated_overall_ctr",
            "reference_first_beat_ctr", "generated_first_beat_ctr",
            "reference_evaluable_notes", "generated_evaluable_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ref_bar, gen_bar in zip(ref_harmony["per_bar"], gen_harmony["per_bar"]):
            writer.writerow({
                "bar_id": ref_bar["bar_id"],
                "reference_overall_ctr": ref_bar["overall_ctr"],
                "generated_overall_ctr": gen_bar["overall_ctr"],
                "reference_first_beat_ctr": ref_bar["first_beat_ctr"],
                "generated_first_beat_ctr": gen_bar["first_beat_ctr"],
                "reference_evaluable_notes": ref_bar["evaluable_notes"],
                "generated_evaluable_notes": gen_bar["evaluable_notes"],
            })

    expected = report["expected_bars"]
    summary = {
        "sample": sample_dir.name,
        "sample_id": report["sample_id"],
        "meter": meter,
        "expected_bars": expected,
        "generated_unique_complete_bars": len(generated["bars"]),
        "paired_common_bars": len(common_ids),
        "unique_complete_bar_ratio": len(generated["bars"]) / expected,
        "paired_bar_ratio": len(common_ids) / expected,
        "generated_complete_occurrences": generated["complete_occurrences"],
        "duplicate_bar_occurrences": len(generated["duplicate_bar_ids"]),
        "non_contiguous_bar_transitions": generated["non_contiguous_transitions"],
        "truncated_bar_occurrences": generated["truncated_bar_occurrences"],
        "invalid_note_positions": generated["invalid_note_positions"],
        "ends_with_complete_bar": generated["ends_with_complete_bar"],
        "grammar_valid": report["grammar_valid"],
        "stopping_reason": report.get("stopping_reason"),
        "generated_tokens": report["generated_tokens"],
        "reference_overall_ctr": ref_harmony["overall_ctr"],
        "generated_overall_ctr": gen_harmony["overall_ctr"],
        "overall_ctr_delta": safe_delta(ref_harmony["overall_ctr"], gen_harmony["overall_ctr"]),
        "reference_first_beat_ctr": ref_harmony["first_beat_ctr"],
        "generated_first_beat_ctr": gen_harmony["first_beat_ctr"],
        "first_beat_ctr_delta": safe_delta(ref_harmony["first_beat_ctr"], gen_harmony["first_beat_ctr"]),
        "reference_notes_per_bar": ref_rhythm["notes_per_bar_mean"],
        "generated_notes_per_bar": gen_rhythm["notes_per_bar_mean"],
        "reference_mean_ioi": ref_rhythm["unique_onset_ioi_mean"],
        "generated_mean_ioi": gen_rhythm["unique_onset_ioi_mean"],
        "mean_bar_onset_jaccard": mean_bar_onset_jaccard(reference["bars"], generated["bars"], common_ids),
        "reference_repetition_rate": ref_structure["repetition_rate_at_0_8"],
        "generated_repetition_rate": gen_structure["repetition_rate_at_0_8"],
        "ssm_correlation": matrix_correlation(ref_ssm, gen_ssm),
    }
    detail = {
        "summary": summary,
        "alignment_policy": "first complete occurrence of each BAR id; paired by shared BAR id",
        "paired_bar_ids": common_ids,
        "generation_report": report,
        "validity": {
            "bar_ids_in_emission_order": generated["bar_ids_in_order"],
            "duplicate_bar_ids": generated["duplicate_bar_ids"],
        },
        "harmony": {"reference": ref_harmony, "generated": gen_harmony},
        "rhythm": {"reference": ref_rhythm, "generated": gen_rhythm},
        "structure": {"reference": ref_structure, "generated": gen_structure},
    }
    (sample_output / "evaluation.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    return summary, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    summaries, details = [], []
    for sample_dir in sample_dirs:
        summary, detail = evaluate_sample(sample_dir, args.output_dir)
        summaries.append(summary)
        details.append(detail)

    with (args.output_dir / "evaluation-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    macro_fields = [
        "unique_complete_bar_ratio", "paired_bar_ratio", "reference_overall_ctr",
        "generated_overall_ctr", "overall_ctr_delta", "reference_first_beat_ctr",
        "generated_first_beat_ctr", "first_beat_ctr_delta", "reference_notes_per_bar",
        "generated_notes_per_bar", "reference_mean_ioi", "generated_mean_ioi",
        "mean_bar_onset_jaccard", "reference_repetition_rate",
        "generated_repetition_rate", "ssm_correlation",
    ]
    ref_hits = sum(d["harmony"]["reference"]["overall_hits"] for d in details)
    ref_notes = sum(d["harmony"]["reference"]["overall_evaluable_notes"] for d in details)
    gen_hits = sum(d["harmony"]["generated"]["overall_hits"] for d in details)
    gen_notes = sum(d["harmony"]["generated"]["overall_evaluable_notes"] for d in details)
    ref_first_hits = sum(d["harmony"]["reference"]["first_beat_hits"] for d in details)
    ref_first_notes = sum(d["harmony"]["reference"]["first_beat_evaluable_notes"] for d in details)
    gen_first_hits = sum(d["harmony"]["generated"]["first_beat_hits"] for d in details)
    gen_first_notes = sum(d["harmony"]["generated"]["first_beat_evaluable_notes"] for d in details)
    aggregate = {
        "samples": len(summaries),
        "paired_common_bars_total": sum(row["paired_common_bars"] for row in summaries),
        "grammar_valid_samples": sum(bool(row["grammar_valid"]) for row in summaries),
        "complete_bar_end_samples": sum(bool(row["ends_with_complete_bar"]) for row in summaries),
        "context_limit_samples": sum(row["stopping_reason"] == "context_limit" for row in summaries),
        "macro_means": {field: mean_present([row[field] for row in summaries]) for field in macro_fields},
        "micro_harmony": {
            "reference_overall_ctr": ref_hits / ref_notes if ref_notes else None,
            "generated_overall_ctr": gen_hits / gen_notes if gen_notes else None,
            "reference_overall_hits": ref_hits,
            "reference_overall_evaluable_notes": ref_notes,
            "generated_overall_hits": gen_hits,
            "generated_overall_evaluable_notes": gen_notes,
            "reference_first_beat_ctr": ref_first_hits / ref_first_notes if ref_first_notes else None,
            "generated_first_beat_ctr": gen_first_hits / gen_first_notes if gen_first_notes else None,
            "reference_first_beat_hits": ref_first_hits,
            "reference_first_beat_evaluable_notes": ref_first_notes,
            "generated_first_beat_hits": gen_first_hits,
            "generated_first_beat_evaluable_notes": gen_first_notes,
        },
        "methodology": {
            "harmony": "note-onset chord-tone ratio; first-beat means POSITION_000",
            "rhythm": "unique-onset IOI within bars; exact onset-position Jaccard across paired bars",
            "duration_bins_grid_units": list(DURATION_BINS),
            "structure": "bar features = normalized pitch-class histogram + 16-bin onset histogram + density",
            "repetition": "fraction of off-diagonal SSM values >= 0.8",
            "alignment": "first complete unique emitted BAR id paired to the same condition/reference BAR id",
        },
    }
    (args.output_dir / "dataset-summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (args.output_dir / "evaluation-details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")

    with (args.output_dir / "aggregate-duration-histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["duration_bin_grid_units", "reference_count", "generated_count"])
        for duration_label in DURATION_BINS:
            writer.writerow([
                duration_label,
                sum(d["rhythm"]["reference"]["duration_histogram"].get(duration_label, 0) for d in details),
                sum(d["rhythm"]["generated"]["duration_histogram"].get(duration_label, 0) for d in details),
            ])

    lag_rows = {}
    for detail in details:
        for role in ("reference", "generated"):
            for item in detail["structure"][role]["lag_similarity"]:
                lag_rows.setdefault(item["lag_bars"], {"reference": [], "generated": []})[role].append(
                    item["mean_similarity"]
                )
    with (args.output_dir / "aggregate-lag-similarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag_bars", "reference_mean_similarity", "generated_mean_similarity", "samples"])
        for lag_value in sorted(lag_rows):
            ref_values = lag_rows[lag_value]["reference"]
            gen_values = lag_rows[lag_value]["generated"]
            writer.writerow([
                lag_value,
                statistics.mean(ref_values),
                statistics.mean(gen_values),
                min(len(ref_values), len(gen_values)),
            ])
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
