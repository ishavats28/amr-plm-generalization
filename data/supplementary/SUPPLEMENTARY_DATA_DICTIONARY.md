# HypoSeq supplementary data dictionary

This dictionary covers the authoritative machine-readable files for Supplementary Tables S1-S5. Column-level definitions are in `SUPPLEMENTARY_DATA_DICTIONARY.csv`.

## Conventions and relationships

- UTF-8 tab-separated files use one header row; the dictionary CSV is comma-separated.
- Empty text denotes unavailable or inapplicable evidence. It is not zero and must not be interpreted as a negative experimental result. Numeric zero is an observed count, score, or binary value.
- `protein_key` is the parent key for the 552-candidate cohort. Swiss-Prot alignments are one-to-many by `protein_key`. InterPro matches are one-to-many by `protein_key`; locations link to matches by `match_id`.
- `original_match_json` and `original_location_json` preserve complete source objects, including member-library-specific fields. Individual InterPro member-library versions are preserved and do not all equal the umbrella InterPro 111.0 release.
- Boolean values may appear as `True`/`False` or as source-preserved `0`/`1`; the column definition identifies their meaning.
- Model scores and flags are descriptive screening outputs at threshold 0.5. They are not phenotype measurements, resistance prevalence estimates, or proof of protein novelty.
- A Swiss-Prot similarity alignment or exact reviewed-sequence match is sequence evidence, not experimental phenotype validation. Mapped InterPro entries and member signatures are computational annotations.
- The published S4 summary reports direct whole-proteome model-positive totals separately from core-annotation agreement.

## File roles

- S1 reports all nine LoRA development runs and the three held-out selections.
- S2 reports 12 locked Test/Group-OOD evaluations plus 18 archived component-bootstrap rows.
- S3 reports 24 verified positive blocks, 144 block/model recovery rows, and six model summaries. Negative-class block metrics are unavailable because negatives were not assigned to these blocks.
- S4 reports six reference panels, direct E2 whole-proteome positive totals, distinct E3 core-positive recovery, and core/plus/other annotation counts. `Table_S4_Annotation_concordance.tsv` supplies the long-form 36 panel/model concordance records.
- S5 reports 552 audited candidates, 552 qualified interpretations, 4,312 Swiss-Prot alignments, 7,296 InterPro matches and 8,948 InterPro locations.
