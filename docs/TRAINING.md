# Training methodology

## Objective

All experiments use full-parameter teacher-forced causal language modeling.
There is no weighted loss and no LoRA/QLoRA in the reported POP909 runs. For
conditional tasks, only Piano target tokens contribute to loss.

## Model lineage

```text
38.6M piano-domain model
  -> POP909 Piano unconditional
  -> POP909 Melody to Piano
  -> depth expansion: 12 to 48 layers
  -> vocabulary expansion: 1,727 to 1,782 tokens
  -> Key + Chord + Melody to Piano, 5 epochs
  -> resumed long run to global optimizer step 2,000
```

The depth expander distributes the 12 trained layers through a 48-layer model
and initializes new layers with zero output projections so the residual path is
initially identity-like. It refuses to save when the output-logit equivalence
error exceeds the configured threshold.

The vocabulary expander copies all old input/output rows exactly, initializes
only new rows, and verifies old-token logits before and after reload.

## Optimization

The 150M key/chord runs use AdamW, constant learning rate after warmup,
gradient clipping at 1.0, gradient checkpointing, and mixed precision when the
GPU supports it.

The five-epoch run uses learning rate `1e-5`, 10% warmup, per-device batch 2,
gradient accumulation 8, and two GPUs (effective batch 32).

The long run uses learning rate `1e-5`, 5% warmup, per-device batch 4,
gradient accumulation 4, two GPUs, evaluation/checkpointing every 100 optimizer
steps, and an exact total budget of 2,000 optimizer updates. Validation remains
observational and cannot stop a positive `max_steps` run.

## Interpretation

The long run reduced training loss but increased final validation loss. This is
evidence that additional optimization steps alone did not improve
generalization. Intermediate checkpoints should be evaluated rather than
assuming the last checkpoint is best.

The held-out test split is not opened by the training programs.

