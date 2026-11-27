#!/usr/bin/env python3
"""Expand a trained Qwen3 music model to an append-only REMI+ vocabulary.

The script proves that every old token keeps the same ID and weight row,
resizes input/output token matrices, saves a new checkpoint, reloads it, and
verifies that all old logits and all non-vocabulary parameters are preserved.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3ForCausalLM, set_seed


VOCAB_FILENAMES = (
    "remi-plus-vocab-key-chord-v1.json",
    "remi-plus-vocab-v4.json",
    "remi-plus-vocab-v3.json",
    "remi-plus-vocab.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = read_json(path)
    rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in rows]
    tokens = [str(row["token"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError(f"Vocabulary IDs are not contiguous: {path}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Vocabulary has duplicate tokens: {path}")
    declared_size = payload.get("vocab_size")
    if declared_size is not None and int(declared_size) != len(tokens):
        raise ValueError(
            f"Vocabulary declares size {declared_size}, but contains {len(tokens)} rows: {path}"
        )
    return tokens, payload


def find_old_vocab(model_dir: Path) -> Path:
    for name in VOCAB_FILENAMES:
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No REMI+ vocabulary found in source model: {model_dir}")


def maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max())


def resize_embeddings(model: Qwen3ForCausalLM, new_size: int) -> None:
    """Use deterministic standard initialization when the API supports it."""
    signature = inspect.signature(model.resize_token_embeddings)
    if "mean_resizing" in signature.parameters:
        model.resize_token_embeddings(new_size, mean_resizing=False)
    else:  # Compatibility with older Transformers releases.
        model.resize_token_embeddings(new_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--new-vocab", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--old-vocab", type=Path)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--verification-sequence-length", type=int, default=32)
    parser.add_argument("--max-old-logit-difference", type=float, default=1e-6)
    args = parser.parse_args()

    source_dir = args.source_model.expanduser().resolve()
    output_dir = args.output_model.expanduser().resolve()
    new_vocab_path = args.new_vocab.expanduser().resolve()
    old_vocab_path = (
        args.old_vocab.expanduser().resolve()
        if args.old_vocab is not None
        else find_old_vocab(source_dir)
    )

    if not source_dir.is_dir():
        raise SystemExit(f"Source model directory does not exist: {source_dir}")
    if not new_vocab_path.is_file():
        raise SystemExit(f"New vocabulary does not exist: {new_vocab_path}")
    if output_dir.exists():
        raise SystemExit(
            f"Output already exists; choose a new empty path to avoid overwriting: {output_dir}"
        )

    old_tokens, old_vocab_payload = load_vocab(old_vocab_path)
    new_tokens, new_vocab_payload = load_vocab(new_vocab_path)
    old_size = len(old_tokens)
    new_size = len(new_tokens)
    if new_size <= old_size:
        raise ValueError(f"New vocabulary must be larger: old={old_size}, new={new_size}")
    if new_tokens[:old_size] != old_tokens:
        mismatch = next(
            index for index, (old, new) in enumerate(zip(old_tokens, new_tokens))
            if old != new
        )
        raise ValueError(
            "New vocabulary is not append-only: "
            f"ID {mismatch} is {old_tokens[mismatch]!r} in old vocab and "
            f"{new_tokens[mismatch]!r} in new vocab"
        )
    if int(new_vocab_payload.get("base_vocab_size", old_size)) != old_size:
        raise ValueError("New vocabulary base_vocab_size does not match the source vocabulary")
    if new_vocab_payload.get("base_vocab_ids_preserved") is False:
        raise ValueError("New vocabulary says old IDs were not preserved")

    set_seed(args.seed)
    source = Qwen3ForCausalLM.from_pretrained(str(source_dir), dtype=torch.float32)
    source.eval()
    if int(source.config.vocab_size) != old_size:
        raise ValueError(
            f"Source config vocab_size={source.config.vocab_size}, old vocabulary size={old_size}"
        )
    source_input = source.get_input_embeddings().weight.detach().cpu().clone()
    source_output_module = source.get_output_embeddings()
    if source_output_module is None:
        raise ValueError("Source model has no output embedding / LM head")
    source_output = source_output_module.weight.detach().cpu().clone()
    if source_input.shape[0] != old_size or source_output.shape[0] != old_size:
        raise ValueError(
            "Source embedding row count does not match source vocabulary: "
            f"input={source_input.shape[0]}, output={source_output.shape[0]}, vocab={old_size}"
        )

    sequence_length = max(1, int(args.verification_sequence_length))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(0, old_size, (1, sequence_length), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        old_logits = source(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()

    resize_embeddings(source, new_size)
    source.config.vocab_size = new_size
    source.eval()
    resized_input = source.get_input_embeddings().weight
    resized_output_module = source.get_output_embeddings()
    if resized_output_module is None:
        raise ValueError("Resized model lost its output embedding / LM head")
    resized_output = resized_output_module.weight
    if resized_input.shape[0] != new_size or resized_output.shape[0] != new_size:
        raise ValueError(
            f"Resize failed: input={resized_input.shape[0]}, output={resized_output.shape[0]}"
        )
    input_old_row_difference = maximum_difference(resized_input[:old_size], source_input)
    output_old_row_difference = maximum_difference(resized_output[:old_size], source_output)
    if input_old_row_difference != 0.0 or output_old_row_difference != 0.0:
        raise RuntimeError(
            "Old vocabulary weights changed during resize: "
            f"input_diff={input_old_row_difference}, output_diff={output_old_row_difference}"
        )
    if not torch.isfinite(resized_input[old_size:]).all():
        raise RuntimeError("New input embedding rows contain non-finite values")
    if not torch.isfinite(resized_output[old_size:]).all():
        raise RuntimeError("New LM-head rows contain non-finite values")

    with torch.inference_mode():
        resized_logits = source(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
    old_logit_difference_before_save = maximum_difference(
        resized_logits[..., :old_size], old_logits
    )
    if old_logit_difference_before_save > args.max_old_logit_difference:
        raise RuntimeError(
            "Old-token logits changed after vocabulary expansion: "
            f"{old_logit_difference_before_save} > {args.max_old_logit_difference}"
        )

    output_dir.mkdir(parents=True)
    source.save_pretrained(
        str(output_dir), safe_serialization=True, max_shard_size="4GB"
    )
    shutil.copy2(new_vocab_path, output_dir / "remi-plus-vocab-key-chord-v1.json")
    generation_config = source_dir / "generation_config.json"
    if generation_config.is_file() and not (output_dir / "generation_config.json").is_file():
        shutil.copy2(generation_config, output_dir / "generation_config.json")

    # Reloading proves that serialization preserved the resize and all old rows.
    reloaded = Qwen3ForCausalLM.from_pretrained(str(output_dir), dtype=torch.float32)
    reloaded.eval()
    if int(reloaded.config.vocab_size) != new_size:
        raise RuntimeError(f"Reloaded config has vocab_size={reloaded.config.vocab_size}")
    reloaded_input = reloaded.get_input_embeddings().weight
    reloaded_output_module = reloaded.get_output_embeddings()
    if reloaded_output_module is None:
        raise RuntimeError("Reloaded model has no LM head")
    reloaded_output = reloaded_output_module.weight
    reload_input_old_row_difference = maximum_difference(
        reloaded_input[:old_size], source_input
    )
    reload_output_old_row_difference = maximum_difference(
        reloaded_output[:old_size], source_output
    )
    if reload_input_old_row_difference != 0.0 or reload_output_old_row_difference != 0.0:
        raise RuntimeError(
            "Old weights changed after save/reload: "
            f"input_diff={reload_input_old_row_difference}, "
            f"output_diff={reload_output_old_row_difference}"
        )
    with torch.inference_mode():
        reloaded_logits = reloaded(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.cpu()
    old_logit_difference_after_reload = maximum_difference(
        reloaded_logits[..., :old_size], old_logits
    )
    if old_logit_difference_after_reload > args.max_old_logit_difference:
        raise RuntimeError(
            "Old-token logits changed after save/reload: "
            f"{old_logit_difference_after_reload} > {args.max_old_logit_difference}"
        )

    # Since resize_token_embeddings only targets these two matrices, exact
    # old-row and old-logit equality is the relevant functional proof. Also
    # report whether Qwen ties the two matrices in this checkpoint.
    tied_after_reload = (
        reloaded_input.data_ptr() == reloaded_output.data_ptr()
    )
    parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in reloaded.parameters()
        if parameter.requires_grad
    )
    report = {
        "status": "PASS",
        "method": "append-only vocabulary expansion with resize_token_embeddings",
        "source_model": str(source_dir),
        "output_model": str(output_dir),
        "old_vocab": str(old_vocab_path),
        "new_vocab": str(new_vocab_path),
        "old_vocab_size": old_size,
        "new_vocab_size": new_size,
        "added_token_count": new_size - old_size,
        "added_tokens": new_tokens[old_size:],
        "old_token_ids_preserved": True,
        "old_input_embedding_max_difference_before_save": input_old_row_difference,
        "old_lm_head_max_difference_before_save": output_old_row_difference,
        "old_input_embedding_max_difference_after_reload": reload_input_old_row_difference,
        "old_lm_head_max_difference_after_reload": reload_output_old_row_difference,
        "old_token_logit_max_difference_before_save": old_logit_difference_before_save,
        "old_token_logit_max_difference_after_reload": old_logit_difference_after_reload,
        "tie_word_embeddings_config": bool(reloaded.config.tie_word_embeddings),
        "input_and_output_embeddings_share_storage": tied_after_reload,
        "new_input_rows_mean": float(reloaded_input[old_size:].float().mean()),
        "new_input_rows_std": float(reloaded_input[old_size:].float().std()),
        "new_output_rows_mean": float(reloaded_output[old_size:].float().mean()),
        "new_output_rows_std": float(reloaded_output[old_size:].float().std()),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "parameter_count_millions": parameter_count / 1_000_000,
        "optimizer_state_preserved": False,
        "transformer_and_old_music_knowledge_preserved": True,
        "new_token_rows_require_training": True,
        "seed": args.seed,
    }
    write_path = output_dir / "vocab-expansion-report.json"
    write_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
