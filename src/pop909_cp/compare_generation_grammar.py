#!/usr/bin/env python3
"""Compare grammar quality for two paired conditional-generation runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ERROR_CATEGORIES = (
    "invalid_note_triple",
    "orphan_or_truncated_position",
    "invalid_duration_token",
    "position_outside_bar",
    "bar_sequence",
    "positions_not_sorted",
    "unexpected_target_token",
    "other",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def classify_error(message: str) -> str:
    text = message.lower()
    if "invalid note triple" in text:
        return "invalid_note_triple"
    if "orphan/truncated position" in text:
        return "orphan_or_truncated_position"
    if "duration token" in text:
        return "invalid_duration_token"
    if "outside bar size" in text:
        return "position_outside_bar"
    if "bar sequence" in text or "bar id" in text or "bar ids" in text:
        return "bar_sequence"
    if "positions are not sorted" in text:
        return "positions_not_sorted"
    if "expected target" in text or "unexpected" in text:
        return "unexpected_target_token"
    return "other"


def load_run(directory: Path) -> dict[str, dict[str, Any]]:
    summary_path = directory / "generation-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = read_json(summary_path)
    samples = summary.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No samples in {summary_path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in samples:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample_id in {summary_path}: {sample_id}")
        by_id[sample_id] = row
    return by_id


def sample_metrics(row: dict[str, Any]) -> dict[str, Any]:
    expected = int(row["expected_bars"])
    generated = int(row["generated_bars_detected"])
    errors = [str(value) for value in row.get("grammar_errors", [])]
    categories = Counter(classify_error(error) for error in errors)
    reference_errors = [str(value) for value in row.get("reference_parse_errors", [])]
    return {
        "grammar_valid": bool(row.get("grammar_valid")),
        "grammar_error_count": len(errors),
        "error_categories": {name: categories.get(name, 0) for name in ERROR_CATEGORIES},
        "expected_bars": expected,
        "generated_bars": generated,
        "bar_completion_ratio": generated / expected if expected else None,
        "full_bar_count_reached": generated >= expected,
        "stopping_reason": row.get("stopping_reason", "unknown"),
        "generated_tokens": int(row.get("generated_tokens", 0)),
        "generated_notes": int(row.get("generated_note_count", 0)),
        "reference_error_count": len(reference_errors),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [sample_metrics(row) for row in rows]
    error_totals = Counter()
    for item in metrics:
        error_totals.update(item["error_categories"])
    completion = [item["bar_completion_ratio"] for item in metrics if item["bar_completion_ratio"] is not None]
    error_counts = [item["grammar_error_count"] for item in metrics]
    return {
        "samples": len(metrics),
        "grammar_valid_count": sum(item["grammar_valid"] for item in metrics),
        "grammar_valid_rate": sum(item["grammar_valid"] for item in metrics) / len(metrics),
        "total_grammar_errors": sum(error_counts),
        "mean_grammar_errors_per_sample": statistics.mean(error_counts),
        "median_grammar_errors_per_sample": statistics.median(error_counts),
        "error_category_totals": {name: error_totals.get(name, 0) for name in ERROR_CATEGORIES},
        "mean_bar_completion_ratio": statistics.mean(completion),
        "median_bar_completion_ratio": statistics.median(completion),
        "full_bar_count_reached": sum(item["full_bar_count_reached"] for item in metrics),
        "complete_non_context_limit": sum(
            item["full_bar_count_reached"] and item["stopping_reason"] != "context_limit"
            for item in metrics
        ),
        "stopping_reasons": dict(Counter(item["stopping_reason"] for item in metrics)),
        "samples_with_reference_errors": sum(item["reference_error_count"] > 0 for item in metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_run(args.baseline_dir.expanduser().resolve())
    candidate = load_run(args.candidate_dir.expanduser().resolve())
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Runs are not paired on identical sample IDs. "
            f"Missing from candidate={missing}; extra in candidate={extra}"
        )

    paired_rows = []
    for sample_id in sorted(baseline_ids):
        before = sample_metrics(baseline[sample_id])
        after = sample_metrics(candidate[sample_id])
        if before["expected_bars"] != after["expected_bars"]:
            raise ValueError(f"Expected-bar mismatch for {sample_id}")
        row = {
            "sample_id": sample_id,
            "baseline_valid": before["grammar_valid"],
            "candidate_valid": after["grammar_valid"],
            "baseline_errors": before["grammar_error_count"],
            "candidate_errors": after["grammar_error_count"],
            "error_delta_candidate_minus_baseline": (
                after["grammar_error_count"] - before["grammar_error_count"]
            ),
            "baseline_completion_ratio": before["bar_completion_ratio"],
            "candidate_completion_ratio": after["bar_completion_ratio"],
            "completion_delta_candidate_minus_baseline": (
                after["bar_completion_ratio"] - before["bar_completion_ratio"]
            ),
            "baseline_stop": before["stopping_reason"],
            "candidate_stop": after["stopping_reason"],
        }
        for category in ERROR_CATEGORIES:
            row[f"baseline_{category}"] = before["error_categories"][category]
            row[f"candidate_{category}"] = after["error_categories"][category]
        paired_rows.append(row)

    baseline_summary = aggregate([baseline[key] for key in sorted(baseline)])
    candidate_summary = aggregate([candidate[key] for key in sorted(candidate)])
    comparison = {
        "status": "PASS",
        "baseline_dir": str(args.baseline_dir.expanduser().resolve()),
        "candidate_dir": str(args.candidate_dir.expanduser().resolve()),
        "pairing": "identical sample_id required",
        "sample_count": len(paired_rows),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "paired_change": {
            "became_valid": sum(not row["baseline_valid"] and row["candidate_valid"] for row in paired_rows),
            "became_invalid": sum(row["baseline_valid"] and not row["candidate_valid"] for row in paired_rows),
            "fewer_errors": sum(row["error_delta_candidate_minus_baseline"] < 0 for row in paired_rows),
            "same_errors": sum(row["error_delta_candidate_minus_baseline"] == 0 for row in paired_rows),
            "more_errors": sum(row["error_delta_candidate_minus_baseline"] > 0 for row in paired_rows),
            "higher_completion": sum(row["completion_delta_candidate_minus_baseline"] > 0 for row in paired_rows),
            "same_completion": sum(row["completion_delta_candidate_minus_baseline"] == 0 for row in paired_rows),
            "lower_completion": sum(row["completion_delta_candidate_minus_baseline"] < 0 for row in paired_rows),
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "grammar-comparison.json", comparison)
    with (output_dir / "grammar-comparison-per-sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
