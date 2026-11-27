# Model card

## Intended task

Generate a symbolic Piano accompaniment from an aligned sequence of Key,
Chord, and Melody events derived from POP909/POP909-CL.

## Architecture

The current model is a fully trainable 48-layer Qwen3-style decoder-only
Transformer with hidden size 512, intermediate size 1,536, eight attention
heads, four KV heads, a 2,048-token context, and a 1,782-token REMI+
vocabulary. It has 151,963,136 parameters.

## Initialization

The model inherits a trained 38.6M Melody-to-Piano model. Depth expansion adds
identity-like residual blocks and verifies output equivalence. Vocabulary
expansion copies every old input/output row and initializes only the 55 added
Key/Chord rows.

## Training objective

Full-parameter teacher-forced causal cross-entropy on Piano target tokens after
`ARRANGEMENT_START`. Condition labels are masked. No LoRA, weighted loss, or
generation-time grammar constraint is used in the reported final experiment.

## Limitations

The model frequently emits non-contiguous bar IDs or malformed target token
patterns. On the 100-sample diagnostic set it underperforms reference Piano on
chord-tone ratio, overproduces short notes, and is more repetitive. The current
checkpoint should be treated as a research artifact rather than a production
accompaniment system.

## Weights

Weights are not yet published in this Git repository. A release should include
model weights, vocabulary, configuration, expansion reports, and SHA-256
checksums together.

