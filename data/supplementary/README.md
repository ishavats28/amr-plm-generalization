# Supplementary data

These files accompany Supplementary Tables S1–S5. The S4 proteome summary is copied from the approved direct-count source; the S5 InterPro match table is losslessly gzip-compressed. Column definitions are in [SUPPLEMENTARY_DATA_DICTIONARY.csv](SUPPLEMENTARY_DATA_DICTIONARY.csv), with relationships and interpretation limits in [SUPPLEMENTARY_DATA_DICTIONARY.md](SUPPLEMENTARY_DATA_DICTIONARY.md).

| Table | Machine-readable files | Records |
| --- | --- | ---: |
| S1 | `Table_S1_LoRA_runs.tsv` | 9 |
| S2 | `Table_S2_Heldout_metrics.tsv`; `Table_S2_Component_bootstrap.tsv` | 12; 18 |
| S3 | `Table_S3_Block_definitions.tsv`; `Table_S3_Block_recovery.tsv`; `Table_S3_Block_summaries.tsv` | 24; 144; 6 |
| S4 | `Table_S4_Proteome_summary.tsv`; `Table_S4_Annotation_concordance.tsv` | 6; 36 |
| S5 | `Table_S5_Candidates_552.tsv`; `Table_S5_Qualified_interpretations.tsv`; `Table_S5_SwissProt_alignments.tsv`; `Table_S5_InterPro_matches.tsv.gz`; `Table_S5_InterPro_locations.tsv` | 552; 552; 4,312; 7,296; 8,948 |

Join S5 tables by `protein_key`; InterPro locations additionally join matches by `match_id`. Empty annotation fields are not negative experimental findings. Predictions, similarity, and domain signatures do not establish resistance phenotype or protein novelty.

## Source attribution and reuse

The software [MIT license](../../LICENSE) does not govern third-party data. Swiss-Prot-derived accession, title, and annotation fields use [UniProtKB/Swiss-Prot, CC BY 4.0](https://www.expasy.org/resources/uniprotkb-swiss-prot); credit the UniProt Consortium. [InterPro's license page](https://interpro-documentation.readthedocs.io/en/latest/license.html) identifies InterPro, Pfam, PRINTS, and SFLD downloads as CC0, while other member resources can have separate terms. The release contains derived InterProScan match and location tables, not the InterProScan installation or member applications. [Phobius and SignalP require separate application licenses to rerun](https://interproscan6.readthedocs.io/stable/licensed_applications/).

Protein identifiers and AMRFinderPlus flags derive from [NCBI RefSeq and AMRFinderPlus](https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/AMRFinder/). [NCBI's molecular-data policy](https://www.ncbi.nlm.nih.gov/home/about/policies/) places no NCBI restriction on distribution but notes possible third-party rights. The original classification work used [CARD](https://card.mcmaster.ca/about) inputs; no CARD database download or reference sequences are redistributed in this package. CARD's terms distinguish its CC BY 4.0 ontology from other site materials. Cite and obtain each upstream resource under its own terms when rerunning the study.

