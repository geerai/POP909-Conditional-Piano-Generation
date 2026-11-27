#!/usr/bin/env python3
"""Expand a trained Qwen3 music model by inserting identity decoder layers.

All trained tensors are copied unchanged. Original decoder layers retain their
order, while newly inserted layers have zero attention-output and MLP-output
projections, making each new residual block an identity function at migration.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import Qwen3ForCausalLM, set_seed


def original_positions(old_count: int, new_count: int) -> list[int]:
    if old_count < 2 or new_count <= old_count:
        raise ValueError("new layer count must exceed an old layer count >= 2")
    result = [round(index * (new_count - 1) / (old_count - 1)) for index in range(old_count)]
    if len(set(result)) != old_count:
        raise ValueError("layer mapping collided")
    return result


def zero_identity_outputs(layer: torch.nn.Module) -> None:
    # Qwen3 decoder block: residual + o_proj(attention), then residual +
    # down_proj(MLP). Zeroing these two projections makes the block identity.
    torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
    if getattr(layer.self_attn.o_proj, "bias", None) is not None:
        torch.nn.init.zeros_(layer.self_attn.o_proj.bias)
    torch.nn.init.zeros_(layer.mlp.down_proj.weight)
    if getattr(layer.mlp.down_proj, "bias", None) is not None:
        torch.nn.init.zeros_(layer.mlp.down_proj.bias)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--new-layers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-verification-difference", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_model.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {args.output_model}")
        shutil.rmtree(args.output_model)
    args.output_model.mkdir(parents=True)
    set_seed(args.seed)

    source = Qwen3ForCausalLM.from_pretrained(
        str(args.source_model), dtype=torch.float32
    )
    source.eval()
    old_count = len(source.model.layers)
    positions = original_positions(old_count, args.new_layers)

    config = source.config
    old_layer_types = list(getattr(config, "layer_types", []) or [])
    if old_layer_types and len(old_layer_types) != old_count:
        raise ValueError(
            f"source config has {len(old_layer_types)} layer_types for {old_count} layers"
        )
    config.num_hidden_layers = args.new_layers
    if old_layer_types:
        # Recent Transformers versions serialize one attention implementation
        # label per Qwen3 layer. Expanding only num_hidden_layers leaves this
        # list at the old length and Qwen3Attention raises IndexError. Assign
        # every new position the type of its nearest preserved source layer.
        config.layer_types = [
            old_layer_types[min(
                range(old_count),
                key=lambda old_index: abs(positions[old_index] - new_index),
            )]
            for new_index in range(args.new_layers)
        ]
    expanded = Qwen3ForCausalLM(config).float().cpu()
    expanded.eval()

    expanded.model.embed_tokens.load_state_dict(source.model.embed_tokens.state_dict())
    expanded.model.norm.load_state_dict(source.model.norm.state_dict())
    # Copy any same-shaped, non-layer parameters/buffers not covered above.
    source_state = source.state_dict()
    expanded_state = expanded.state_dict()
    for name, value in source_state.items():
        if name.startswith("model.layers."):
            continue
        if name in expanded_state and expanded_state[name].shape == value.shape:
            expanded_state[name].copy_(value)

    old_at_new = dict(zip(positions, range(old_count)))
    for new_index, layer in enumerate(expanded.model.layers):
        if new_index in old_at_new:
            layer.load_state_dict(source.model.layers[old_at_new[new_index]].state_dict())
        else:
            zero_identity_outputs(layer)

    # A deterministic forward equivalence check proves the migration preserved
    # the trained function before any further optimization.
    vocab_size = int(source.config.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(0, vocab_size, (1, 32), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        old_logits = source(input_ids=input_ids, attention_mask=attention_mask).logits
        new_logits = expanded(input_ids=input_ids, attention_mask=attention_mask).logits
    maximum_difference = float((old_logits - new_logits).abs().max())
    if maximum_difference > args.max_verification_difference:
        raise RuntimeError(
            f"equivalence failed: max logit difference {maximum_difference} > "
            f"{args.max_verification_difference}"
        )

    parameter_count = sum(parameter.numel() for parameter in expanded.parameters())
    trainable_count = sum(parameter.numel() for parameter in expanded.parameters() if parameter.requires_grad)
    expanded.save_pretrained(
        str(args.output_model), safe_serialization=True, max_shard_size="4GB"
    )
    for name in (
        "remi-plus-vocab-v4.json", "remi-plus-vocab.json", "generation_config.json"
    ):
        source_file = args.source_model / name
        if source_file.is_file():
            shutil.copy2(source_file, args.output_model / name)

    report = {
        "status": "PASS",
        "method": "function-preserving depth expansion with identity residual blocks",
        "source_model": str(args.source_model),
        "output_model": str(args.output_model),
        "old_layer_count": old_count,
        "new_layer_count": args.new_layers,
        "old_layer_to_new_layer": {
            str(old_index): new_index for old_index, new_index in enumerate(positions)
        },
        "inserted_identity_layer_count": args.new_layers - old_count,
        "source_layer_types": old_layer_types,
        "expanded_layer_types": list(getattr(config, "layer_types", []) or []),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "parameter_count_millions": parameter_count / 1_000_000,
        "maximum_logit_difference": maximum_difference,
        "optimizer_state_preserved": False,
        "trained_model_function_preserved_at_initialization": True,
    }
    (args.output_model / "expansion-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
