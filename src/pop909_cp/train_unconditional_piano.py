#!/usr/bin/env python3
"""Continue the 38.6M ARIA checkpoint on POP909 piano accompaniment only."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_vocab(path: Path) -> list[str]:
    rows = sorted(read_json(path)["tokens"], key=lambda row: int(row["id"]))
    tokens = [str(row["token"]) for row in rows]
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"invalid vocabulary IDs: {path}")
    if len(tokens) != 1727 or len(tokens) != len(set(tokens)):
        raise ValueError(f"expected fixed 1,727-token vocabulary: {path}")
    return tokens


def find_model_vocab(root: Path) -> Path:
    for name in (
        "remi-plus-vocab-v4.json",
        "remi-plus-vocab-v3.json",
        "remi-plus-vocab.json",
    ):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no REMI+ vocabulary stored with model: {root}")


class JsonlMusicDataset(Dataset):
    def __init__(self, path: Path, vocab_size: int) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = [int(value) for value in row["input_ids"]]
                loss_start = int(row["loss_start_token_index"])
                if not 1 <= loss_start < len(ids) <= 2048:
                    raise ValueError(f"{path}:{line_number}: invalid sequence")
                if min(ids) < 0 or max(ids) >= vocab_size:
                    raise ValueError(f"{path}:{line_number}: token ID out of range")
                if row.get("task") != "UNCONDITIONAL_PIANO_ACCOMPANIMENT":
                    raise ValueError(f"{path}:{line_number}: unexpected task")
                if row.get("roles") != ["PIANO"]:
                    raise ValueError(f"{path}:{line_number}: unexpected roles")
                self.rows.append({"input_ids": ids, "loss_start": loss_start})
        if not self.rows:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class MusicCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(row["input_ids"]) for row in features)
        batch = len(features)
        input_ids = torch.full((batch, longest), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, longest), dtype=torch.long)
        labels = torch.full((batch, longest), -100, dtype=torch.long)
        for index, row in enumerate(features):
            values = torch.tensor(row["input_ids"], dtype=torch.long)
            length = len(values)
            loss_start = int(row["loss_start"])
            input_ids[index, :length] = values
            attention_mask[index, :length] = 1
            labels[index, loss_start:length] = values[loss_start:length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "train"), default="validate")
    parser.add_argument("--epochs", type=float, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_vocab_path = args.data_dir / "remi-plus-vocab-v4.json"
    tokens = load_vocab(data_vocab_path)
    model_tokens = load_vocab(find_model_vocab(args.model_path))
    if tokens != model_tokens:
        raise ValueError("model and POP909 data do not use identical Token IDs")

    validation_report = read_json(args.data_dir / "validation-report.json")
    if validation_report.get("status") != "PASS":
        raise ValueError("data validation did not pass")
    if validation_report.get("task_token_present") is not False:
        raise ValueError("Task token unexpectedly present in data")
    if validation_report.get("only_piano_role_present") is not True:
        raise ValueError("data is not strict Piano-only")

    train = JsonlMusicDataset(args.data_dir / "train.jsonl", len(tokens))
    validation = JsonlMusicDataset(args.data_dir / "validation.jsonl", len(tokens))
    # Test is deliberately not opened during training.
    model = Qwen3ForCausalLM.from_pretrained(str(args.model_path))
    model.config.use_cache = False
    if model.get_input_embeddings().num_embeddings != len(tokens):
        raise ValueError("model embedding size does not equal 1,727")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != trainable_count:
        raise ValueError("this experiment requires full-model training")

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    preflight = {
        "mode": args.mode,
        "objective": "unconditional PIANO accompaniment next-token modeling",
        "initial_model": str(args.model_path),
        "task_token_in_data": False,
        "roles_in_data": ["PIANO"],
        "melody_and_bridge_in_data": False,
        "loss": "ordinary teacher-forced causal cross entropy after ARRANGEMENT_START",
        "train_windows": len(train),
        "validation_windows": len(validation),
        "test_opened_during_training": False,
        "vocabulary_size": len(tokens),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "fully_trainable": parameter_count == trainable_count,
        "optimization": {
            "learning_rate": args.learning_rate,
            "schedule": "5% warmup then constant",
            "warmup_ratio": args.warmup_ratio,
            "epochs": args.epochs,
            "max_grad_norm": args.max_grad_norm,
            "weight_decay": 0.01,
            "effective_batch_size_all_gpus": (
                args.per_device_batch_size
                * args.gradient_accumulation_steps
                * world_size
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(args.output_dir / "preflight-report.json", preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "validate":
        return

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="constant_with_warmup",
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.01,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=MusicCollator(tokens.index("<PAD>")),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )
    trainer.train()
    final_model = args.output_dir / "final-model"
    trainer.save_model(str(final_model))
    metrics = trainer.evaluate()
    metrics["validation_perplexity"] = math.exp(min(20, float(metrics["eval_loss"])))
    if trainer.is_world_process_zero():
        shutil.copy2(data_vocab_path, final_model / "remi-plus-vocab-v4.json")
        shutil.copy2(data_vocab_path, final_model / "remi-plus-vocab.json")
        write_json(args.output_dir / "final-validation-metrics.json", metrics)
        write_json(args.output_dir / "trainer-log-history.json", trainer.state.log_history)
        write_json(args.output_dir / "training-result.json", {
            "final_model": str(final_model),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "validation": metrics,
        })
        print(json.dumps({
            "final_model": str(final_model),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "validation": metrics,
        }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
