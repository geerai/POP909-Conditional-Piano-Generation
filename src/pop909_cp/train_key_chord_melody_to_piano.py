#!/usr/bin/env python3
"""Fine-tune Qwen3 on POP909 Key+Chord+Melody -> Piano samples."""

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
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


TASK_NAME = "KEY_CHORD_MELODY_TO_PIANO"
TASK_TOKEN = "<TASK_KEY_CHORD_MELODY_TO_PIANO>"
VOCAB_FILENAME = "remi-plus-vocab-key-chord-v1.json"
MAX_LENGTH = 2048


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_vocab(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = read_json(path)
    rows = sorted(payload["tokens"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in rows]
    tokens = [str(row["token"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError(f"Vocabulary IDs are not contiguous: {path}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Vocabulary contains duplicate tokens: {path}")
    if int(payload.get("vocab_size", len(tokens))) != len(tokens):
        raise ValueError(f"Vocabulary size field is incorrect: {path}")
    if int(payload.get("base_vocab_size", -1)) != 1727:
        raise ValueError(f"Expected an extended 1,727-token base vocabulary: {path}")
    if not payload.get("base_vocab_ids_preserved"):
        raise ValueError(f"Vocabulary does not certify preserved base IDs: {path}")
    if TASK_TOKEN not in tokens:
        raise ValueError(f"Vocabulary lacks {TASK_TOKEN}: {path}")
    return tokens, payload


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
                location = f"{path}:{line_number}"
                if not 1 <= loss_start < len(ids) <= MAX_LENGTH:
                    raise ValueError(
                        f"{location}: expected 1 <= loss_start < length <= {MAX_LENGTH}"
                    )
                if min(ids) < 0 or max(ids) >= vocab_size:
                    raise ValueError(f"{location}: token ID outside vocabulary")
                if row.get("task") != TASK_NAME:
                    raise ValueError(f"{location}: unexpected task {row.get('task')!r}")
                tokens = row.get("tokens")
                if not isinstance(tokens, list) or len(tokens) != len(ids):
                    raise ValueError(f"{location}: missing or mismatched token strings")
                if tokens[loss_start - 1] != "<ARRANGEMENT_START>":
                    raise ValueError(
                        f"{location}: loss_start does not immediately follow ARRANGEMENT_START"
                    )
                self.rows.append({"input_ids": ids, "loss_start": loss_start})
        if not self.rows:
            raise ValueError(f"Dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class TargetOnlyCollator:
    """Pad sequences and mask every label before the generated Piano body."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = int(pad_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(row["input_ids"]) for row in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, longest), self.pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, longest), dtype=torch.long)
        labels = torch.full((batch_size, longest), -100, dtype=torch.long)
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
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Exact optimizer-update budget. A positive value overrides --epochs "
            "and training always runs all steps; validation never stops training."
        ),
    )
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="A checkpoint directory, or 'latest' to let Trainer select the latest checkpoint.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Use only if memory is ample; checkpointing is enabled by default.",
    )
    return parser.parse_args()


def validate_hyperparameters(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup ratio must be in [0, 1)")
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max steps must be -1 or a positive integer")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        raise ValueError("eval steps and save steps must be positive")
    if args.max_steps > 0 and args.save_steps % args.eval_steps != 0:
        raise ValueError("save steps must be a multiple of eval steps")


def main() -> None:
    args = parse_args()
    validate_hyperparameters(args)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    model_path = args.model_path.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_vocab_path = data_dir / VOCAB_FILENAME
    model_vocab_path = model_path / VOCAB_FILENAME
    data_tokens, data_vocab_payload = load_vocab(data_vocab_path)
    model_tokens, _model_vocab_payload = load_vocab(model_vocab_path)
    if data_tokens != model_tokens:
        raise ValueError("Model and dataset vocabularies do not have identical token IDs")
    vocab_size = len(data_tokens)

    validation_report = read_json(data_dir / "validation-report.json")
    if validation_report.get("status") != "PASS":
        raise ValueError("Dataset validation-report.json does not have status PASS")
    if validation_report.get("task") != TASK_NAME:
        raise ValueError("Dataset validation report proves the wrong task")
    if validation_report.get("target_role") != "PIANO":
        raise ValueError("Dataset validation report does not prove a Piano target")
    if validation_report.get("song_split_leakage") is not False:
        raise ValueError("Dataset validation report does not rule out split leakage")
    if int(validation_report["vocabulary"]["extended_size"]) != vocab_size:
        raise ValueError("Dataset validation report has a different vocabulary size")

    # Deliberately never open test.jsonl during training or preflight.
    train = JsonlMusicDataset(data_dir / "train.jsonl", vocab_size)
    validation = JsonlMusicDataset(data_dir / "validation.jsonl", vocab_size)

    model = Qwen3ForCausalLM.from_pretrained(str(model_path), dtype=torch.float32)
    model.config.use_cache = False
    if int(model.config.vocab_size) != vocab_size:
        raise ValueError(
            f"Model config vocab_size={model.config.vocab_size}, dataset vocab_size={vocab_size}"
        )
    input_rows = int(model.get_input_embeddings().num_embeddings)
    output_module = model.get_output_embeddings()
    if output_module is None:
        raise ValueError("Model has no output embedding / LM head")
    output_rows = int(output_module.weight.shape[0])
    if input_rows != vocab_size or output_rows != vocab_size:
        raise ValueError(
            f"Embedding rows do not match vocabulary: input={input_rows}, output={output_rows}"
        )
    if int(model.config.max_position_embeddings) < MAX_LENGTH:
        raise ValueError("Model context length is smaller than 2,048")
    if int(model.config.num_hidden_layers) != 48:
        raise ValueError(
            f"Expected the expanded 48-layer model, got {model.config.num_hidden_layers} layers"
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count != trainable_count:
        raise ValueError("This experiment requires full-model training")
    if not 100_000_000 <= parameter_count <= 300_000_000:
        raise ValueError(f"Model parameter count is outside 100M-300M: {parameter_count}")

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", "0"))
    eval_batch_size = args.eval_batch_size or args.per_device_batch_size
    gradient_checkpointing = not args.disable_gradient_checkpointing
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    precision = "bf16" if bf16 else "fp16" if fp16 else "fp32"
    preflight = {
        "status": "PASS",
        "mode": args.mode,
        "objective": "Key+Chord+Melody-conditioned Piano next-token modeling",
        "initial_model": str(model_path),
        "data_dir": str(data_dir),
        "task": TASK_NAME,
        "task_token": TASK_TOKEN,
        "condition_order": "Key -> Chord -> Melody",
        "target_role": "PIANO",
        "loss": "teacher-forced causal cross entropy on target tokens after ARRANGEMENT_START only",
        "train_windows": len(train),
        "validation_windows": len(validation),
        "test_opened_during_training": False,
        "vocabulary_size": vocab_size,
        "base_vocabulary_size": int(data_vocab_payload["base_vocab_size"]),
        "old_vocabulary_ids_preserved": True,
        "model_layers": int(model.config.num_hidden_layers),
        "model_hidden_size": int(model.config.hidden_size),
        "model_context_length": int(model.config.max_position_embeddings),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "fully_trainable": parameter_count == trainable_count,
        "precision": precision,
        "gradient_checkpointing": gradient_checkpointing,
        "optimization": {
            "learning_rate": args.learning_rate,
            "schedule": "constant_with_warmup",
            "warmup_ratio": args.warmup_ratio,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "stop_policy": (
                "run_all_max_steps_no_early_stopping"
                if args.max_steps > 0
                else "run_all_epochs_no_early_stopping"
            ),
            "max_grad_norm": args.max_grad_norm,
            "weight_decay": args.weight_decay,
            "per_device_train_batch_size": args.per_device_batch_size,
            "per_device_eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "world_size": world_size,
            "effective_batch_size_all_gpus": (
                args.per_device_batch_size
                * args.gradient_accumulation_steps
                * world_size
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(output_dir / "preflight-report.json", preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "validate":
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Training mode requires CUDA")
    checkpoints_dir = output_dir / "checkpoints"
    existing_checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"))
    if existing_checkpoints and args.resume_from_checkpoint is None:
        raise RuntimeError(
            f"Existing checkpoints found in {checkpoints_dir}; explicitly pass "
            "--resume-from-checkpoint latest or choose a new output directory"
        )
    resume: str | bool | None
    if args.resume_from_checkpoint == "latest":
        resume = True
    elif args.resume_from_checkpoint:
        resume = str(Path(args.resume_from_checkpoint).expanduser().resolve())
    else:
        resume = None

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="constant_with_warmup",
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.eval_steps if args.max_steps > 0 else None,
        save_steps=args.save_steps if args.max_steps > 0 else None,
        save_total_limit=args.save_total_limit,
        # The requested experiment must finish every update and preserve the
        # last-step weights. Validation is observational, not a stop/selection
        # signal. Intermediate checkpoints remain available for comparison.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        tf32=(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        ),
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=TargetOnlyCollator(data_tokens.index("<PAD>")),
    )
    train_result = trainer.train(resume_from_checkpoint=resume)
    final_model = output_dir / "final-model"
    if trainer.is_world_process_zero() and final_model.exists():
        raise RuntimeError(f"Refusing to overwrite existing final model: {final_model}")
    trainer.save_model(str(final_model))
    metrics = trainer.evaluate()
    metrics["validation_perplexity"] = math.exp(
        min(20.0, float(metrics["eval_loss"]))
    )
    if trainer.is_world_process_zero():
        shutil.copy2(data_vocab_path, final_model / VOCAB_FILENAME)
        write_json(output_dir / "final-validation-metrics.json", metrics)
        write_json(output_dir / "trainer-log-history.json", trainer.state.log_history)
        result = {
            "status": "PASS",
            "final_model": str(final_model),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "train_metrics": train_result.metrics,
            "validation": metrics,
            "test_opened_during_training": False,
        }
        write_json(output_dir / "training-result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
