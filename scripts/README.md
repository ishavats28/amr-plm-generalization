# Scientific scripts

These are selected implementations of the published methods. They preserve recorded model configurations and analysis rules but require original data, selected checkpoints, external databases, or GPU resources for full execution.

- `data_preparation/`: construct and independently verify the curated homology-disjoint split.
- `training/frozen/`: ESM-2 models M0–M3 and ProtBERT models M4–M7, with configuration JSONs and required model/cache modules.
- `training/lora/`: the three adapted ESM-2 architectures and training protocol.
- `evaluation/`: held-out inference, original 24-block Group-OOD analysis, ESKAPE application, and required validation-policy helpers. Synthetic self-tests run without model weights.
- `annotation/`: preparation and integration of reviewed Swiss-Prot and InterProScan evidence for the fixed candidate cohort.

The retained modules use relative paths or command-line study roots. Full evaluation checks the original source-lock manifests; a provenance-locked rerun requires the original code and manifests as well as weights and data. The Colab-only candidate-export helper and redundant preflight copies are excluded from the public release. See [reproducibility scope](../REPRODUCIBILITY.md).

