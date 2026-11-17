# Reproducibility scope

The released TSVs and Word supplement contain the completed, audited results. The scripts document original analytical methods; most full workflows require inputs and checkpoints that are not distributed here.

## Provided scientific code

| Directory | Role | Required external inputs |
| --- | --- | --- |
| `scripts/data_preparation/` | Dataset construction, homology-component partitioning, and split verification | Original AMRFinderPlus/CARD-derived inputs, clustering evidence, solver environment, and source manifests |
| `scripts/training/frozen/` | Eight ESM-2/ProtBERT frozen-encoder classifiers, extraction, cache handling, and model configurations | Pinned encoder weights, sequences, partitions, GPUs, and large embedding caches |
| `scripts/training/lora/` | M0, M1, and M3 LoRA trainers and selection protocol | Frozen heads, the same dataset, pinned ESM-2 weights, and GPU resources |
| `scripts/evaluation/` | Selected-checkpoint Test/Group-OOD inference, original 24-block analysis, and six-proteome application | Selected weights, source manifests, held-out sequences, and proteome inputs |
| `scripts/annotation/` | Swiss-Prot and InterProScan preparation, merge, and final evidence integration | The fixed candidate export, reviewed Swiss-Prot release 2026_04, InterProScan 6.0.3.0 / InterPro 111.0 results |

The eight frozen configurations and LoRA protocol preserve model settings. Nine LoRA runs, six selected checkpoints, 12 held-out evaluations, archived paired component-bootstrap intervals, 24 Group-OOD blocks, six ESKAPE proteomes, and the 552-candidate evidence are released in [Supplementary Tables S1–S5](data/supplementary/README.md). The standalone code used to calculate the archived bootstrap intervals was not present in the inspected source set, so those intervals are released as verified data rather than claimed as directly rerunnable here.

## Validated lightweight commands

From the repository root, after installing `requirements.txt`:

```bash
python scripts/evaluation/locked_holdout.py --self-test
python scripts/evaluation/eskape_inference.py --self-test
python scripts/evaluation/group_ood_blocks.py --help
python scripts/annotation/prepare_interproscan.py --help
```

The self-tests do not load checkpoints or run inference. Scripts that operate on research data accept an explicit study root or use relative paths; inspect each script's help and provide external inputs before running. Cache staging defaults to a local `.cache/` directory, which is excluded from the release.

Some evaluation code checks hashes against the original locked source manifest. Repository path-only portability edits mean a complete provenance-locked rerun needs the original source versions and manifests as well as weights and data. No claim of end-to-end execution from this release is made.

## Figures and interpretation

The three [final PNG/PDF figure pairs](figures/README.md) are provided as approved publication outputs. Their generation scripts are not part of this release. The Figure 2 CSV links displayed B01–B24 labels to verified Group-OOD block and component identifiers; other plotted values are available in the supplementary tables.

Group-OOD and ESKAPE analyses are exploratory. Core AMRFinderPlus agreement has small, unequal denominators and is not experimentally measured sensitivity. Swiss-Prot alignments and InterPro signatures provide computational sequence and domain evidence rather than proof of AMR function.

