# Reproducibility status

## Versioned here

- Source code for cleaning, tokenization, validation, training, generation, and
  evaluation.
- Base 1,727-token vocabulary.
- POP909 metadata export and POP909-CL audit records.
- Exact experiment and sampling configurations.
- Compact evaluation outputs and selected generated examples.

## Not versioned in ordinary Git

- Source datasets and derived JSONL/MIDI corpora.
- Model weights and optimizer/scheduler checkpoints.
- Virtual environments and caches.
- Full raw training logs.

## Exact checkpoint reproduction still requires

- The trained model checkpoint or a published model-weight artifact.
- The generated 1,782-token vocabulary from the prepared dataset.
- Cleaned run metadata: model `config.json`, expansion reports, trainer log
  history, and final validation metrics.
- The same PyTorch/Transformers/CUDA environment used on the training server.

Large artifacts should be published through a model host or release storage
with SHA-256 checksums. They should not be committed as ordinary Git blobs.

