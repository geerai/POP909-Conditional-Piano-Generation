from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryArtifactTests(unittest.TestCase):
    def test_base_vocab_is_contiguous_and_fixed(self) -> None:
        payload = json.loads(
            (ROOT / "vocab" / "remi-plus-vocab-v4.json").read_text(encoding="utf-8")
        )
        rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
        self.assertEqual(len(rows), 1727)
        self.assertEqual([int(row["id"]) for row in rows], list(range(1727)))

    def test_experiment_configs_are_valid_json(self) -> None:
        configs = sorted((ROOT / "configs").glob("*.json"))
        self.assertGreaterEqual(len(configs), 5)
        for path in configs:
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_extended_vocab_preserves_base_rows(self) -> None:
        base = json.loads(
            (ROOT / "vocab" / "remi-plus-vocab-v4.json").read_text(encoding="utf-8")
        )
        extended = json.loads(
            (ROOT / "vocab" / "remi-plus-vocab-key-chord-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(extended["base_vocab_size"], 1727)
        self.assertEqual(extended["vocab_size"], 1782)
        self.assertTrue(extended["base_vocab_ids_preserved"])
        self.assertEqual(extended["tokens"][:1727], base["tokens"])

    def test_grammar_comparison_is_paired(self) -> None:
        payload = json.loads(
            (ROOT / "results" / "grammar-comparison" / "grammar-comparison.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["sample_count"], 10)
        self.assertEqual(payload["pairing"], "identical sample_id required")


if __name__ == "__main__":
    unittest.main()
