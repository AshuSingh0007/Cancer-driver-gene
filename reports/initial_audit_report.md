# Initial project audit

**Audit date:** 2026-10-28  
**Scope:** Read-only review of existing project files, existing outputs, and the local Python environment. No training or dataset modifications were performed.

## Findings

### Project structure

- Root scripts: `build_ppi_network.py`, `preprocess_ncg.py`, `preprocess_string.py`, `train_baseline.py`, `validate_ppi_network.py`, and `plot_project_results.py`.
- Source annotations: two NCG TSV files and COSMIC Cancer Gene Census v104 tables for GRCh37 and GRCh38, each with a README and extracted TSV plus compressed copy.
- STRING: a 497,998,517-byte `9606.protein.links.v12.5.txt` and a 199,702,624-byte file named `9606.protein.aliases.v12.0.txt` inside a directory named `9606.protein.aliases.v12.5.txt`.
- TCGA: `data/tcga/expression` contains 1,228 STAR augmented gene-count TSVs (5,210,744,298 bytes total), 1,228 `.parcel` download metadata files and 32 `annotations.txt` files. `tcga_tsv_files.txt` is a 179,288-byte path list.
- Existing outputs include NCG merged data, a filtered gene-level STRING edge list and graph summaries, a logistic-regression experiment with metrics/predictions/model, and six PNG figures. No notebooks, dependency lock/requirements file, project tests, GNN/Transformer implementation, or quantum-computing implementation were found at the project root.
- `.venv` is present; its individual installed files are represented by a summary in `project_inventory.json` to avoid duplicating environment internals in the inventory.

### Dataset schemas and current status

**TCGA-BRCA expression.** All 1,228 matching expression TSVs found are non-empty and larger than 100 KB. The inspected files have the GENCODE v36 header and columns `gene_id`, `gene_name`, `gene_type`, `unstranded`, `stranded_first`, `stranded_second`, `tpm_unstranded`, `fpkm_unstranded`, and `fpkm_uq_unstranded`. They contain four STAR summary records (`N_unmapped`, `N_multimapping`, `N_noFeature`, `N_ambiguous`) before gene records. Gene IDs are versioned Ensembl IDs and gene symbols are present. The integer count fields and normalized TPM/FPKM fields must not be conflated. The directory layout and filenames alone do not establish unique patient/sample count or sample type. The 32 GDC `annotations.txt` files inspected are exception/observation annotations, not a complete sample phenotype table. File-to-case/sample mapping and tumor/normal status remain unverified. The historical cart/manifest counts cannot be reconciled from the TSV count alone; the manifest mentioned in the request is outside the project and was not read.

**STRING.** The links file header is `protein1 protein2 combined_score`; examples use `9606.ENSP...` protein identifiers. It contains 10,826,054 interaction rows plus header. Its filename indicates STRING v12.5. The aliases file is actually named/content-versioned v12.0 despite its parent directory name saying v12.5; therefore the current mapping combines mismatched releases. Existing code selects `Ensembl_HGNC_entrez_id`, filters score >=700, maps to the NCG candidate list, removes self-links, canonicalizes undirected pairs, and keeps the maximum score per gene pair. Saved graph summary: 3,347 NCG IDs; 3,326 mapped STRING proteins; 24,956 unique edges among 2,799 observed nodes; 27 connected components (largest 2,743); no saved self-loops; mean degree about 17.83, maximum 402, density about 0.00637. This is an NCG-restricted subgraph, not a genome-wide graph. Link scores are STRING combined confidence scores, not experimentally measured physical binding strengths. Duplicate protein-pair and evidence-channel inspection was not completed.

**NCG.** `NCG_cancerdrivers_annotation_supporting_evidence.tsv` has 8,886 rows, 14 columns and 3,347 unique Entrez IDs; many rows repeat a gene across papers, tumor types, or methods (5,539 repeated Entrez occurrences after the first). Columns include gene identifiers, publication/method/cancer context, evidence annotations, and `NCG_oncogene`/`NCG_tsg`. The systems-properties table has 3,347 rows, 19 columns, one row per Entrez ID, with topology, expression/essentiality, and constraint fields. The saved merged table has 3,347 rows and 33 columns, all marked `both`. Its labels preserve missing values: oncogene 256 positive, 335 zero, 2,756 missing; TSG 254 positive, 337 zero, 2,756 missing. A value of zero is an explicit source annotation, not proof of biological absence. Existing NCG preprocessing aggregates any positive per gene, retains zero only when no positive exists, combines evidence strings, and outer-merges systems properties.

**COSMIC CGC.** Both v104 genome-build exports have 768 unique symbols and 21 columns. Fields include `GENE_SYMBOL`, COSMIC gene ID, coordinates, somatic/germline status, tumor context, `ROLE_IN_CANCER`, and `TIER`. The role field distinguishes oncogene, TSG, fusion, and combinations; collapsing these into a single positive label without a task-specific rule would lose meaning. Each export shares 656 gene symbols with the NCG annotation symbols. The GRCh37 and GRCh38 tables should be treated as coordinate-build alternatives, not independent evidence sets. Their README describes fields and links the COSMIC Census source; usage conditions must be checked before redistribution.

### Existing pipeline and outputs

1. `preprocess_ncg.py` aggregates annotation evidence by Entrez ID and merges the systems-level table. It writes `outputs/NCG_preprocessed_merged.csv` and prints merge counts.
2. `preprocess_string.py` only inspects sample rows/columns; it does not preprocess or build a graph.
3. `build_ppi_network.py` reads the above merged NCG table and STRING aliases/links, filters at 700, maps to Entrez IDs and emits an undirected NCG-induced edge list and summary. Its current aliases/links release mismatch is a major reproducibility issue. It also accumulates filtered chunks before concatenation, which can consume significant memory.
4. `validate_ppi_network.py` validates the saved edge list and writes degree and component-size tables, but does not validate identifier correctness or the source evidence.
5. `train_baseline.py` trains a class-weighted Logistic Regression and most-frequent dummy for NCG oncogene labels among labeled NCG rows, using 16 NCG systems-level features and a single stratified 80/20 random gene holdout (seed 42). The target remains 2,756 genes unlabeled. Existing saved test metrics: logistic accuracy 0.647, positive F1 0.618, average precision 0.594, ROC-AUC 0.675; dummy accuracy 0.563 and ROC-AUC 0.5. This is a preliminary source-defined cohort result, not an independent validation. Some features are PPI-derived and could encode ascertainment. `plot_project_results.py` redraws six existing baseline/network figures from saved artifacts; it overwrites those named PNGs when run.

The experiment output is timestamped 2026-10-14 and includes a joblib model and holdout predictions. No model checkpoints for deep learning or saved TCGA-derived matrices were found.

### Environment and resources

- Windows 11, Python 3.13.5, 12 logical CPUs.
- `.venv` uses `C:\Python313`, does not include system packages.
- Observed packages include NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, NetworkX 3.7, SciPy 1.18.1, Matplotlib 3.11.2, seaborn 0.13.2, and psutil 7.2.2.
- PyTorch is not installed; no GPU availability could be assessed through PyTorch. RAM: 7.68 GiB total, about 0.83 GiB available at audit time. Free disk space on the project volume: about 207 GB.
- No `requirements.txt`, `pyproject.toml`, or lockfile was found. Deep graph learning would require careful environment planning and a small-memory design.

## Key methodological issues

1. Expression files are at sample level; the current model is gene level. A sample-to-gene representation must be defined. A reasonable initial representation is per-gene summary features from a carefully selected homogeneous sample cohort (e.g. primary tumor only), such as log-CPM mean/dispersion and detection fraction. Cohort selection and patient/sample mapping are not yet available, so expression should not yet be joined to labels or graph nodes.
2. Gene-level classification with a handful of hundred labeled NCG genes does not support treating all unannotated genes as negatives. A positive-unlabeled or ranking design, or a clearly bounded classification against explicitly NCG-labeled zeros, needs a predeclared interpretation and source-aware evaluation.
3. NCG labels and COSMIC CGC roles differ in evidence and granularity. Use one label source for training and reserve the other for carefully described reference comparison, or define a conflict policy before any integration. Avoid using the same catalogue for both features and target.
4. Random gene splits may leak correlated gene/network information. Compare a repeatable stratified split with cross-validation and an explicitly designed graph-aware/source-aware validation. Avoid choosing thresholds on the final test set.
5. STRING aliases/link releases need reconciliation. The current graph is restricted to NCG genes and cannot directly support ranking the broader set of candidate genes.
6. Existing outputs contain no expression-to-PPI mapping or data-quality report. No accuracy claims beyond the existing single split are supported.

## Proposed HMGAT-ET design (provisional)

The proposed name is project-specific and is not asserted to be an established or novel published architecture. After TCGA sample linkage and identifier harmonization, form one gene-level expression vector with train-fitted transformations only. An expression MLP encoder maps it to a fixed-width embedding. A weighted, undirected STRING graph encoder (initially a compact GAT) consumes this representation plus only non-target-derived graph features. Project graph embeddings to the same width, then apply learned sigmoid gated fusion and a small residual prediction head. Compare expression-only, graph-only, GAT-only, and concatenation versus gate on identical splits. Add a second GraphSAGE branch or cross-attention only if the dimensions and sample size support a meaningful hypothesis and an ablation justifies it. The most defensible initial target is ranking/labeled-positive versus explicitly labeled-zero genes within the NCG cohort, with unlabeled genes kept unlabeled; the interpretation and validation design must be settled in Phase 3. A broader candidate ranking can follow only if the training/evaluation evidence supports it.

No Transformer or cross-attention is currently justified: each gene would initially have a compact statistical feature vector rather than a naturally ordered token sequence. No quantum-computing component exists or should be added. If the formal project title implies quantum methodology, flag the title/method mismatch to the supervisor; do not rename it without approval.

## Staged implementation plan

1. **Audit (complete):** inventory and initial audit saved under `reports/`.
2. **Data validation:** establish TCGA file/case/sample metadata linkage and sample types; quantify file completeness, schema consistency, duplicate sample/gene records, missingness; resolve STRING release mismatch and measure mapping/overlap; quantify NCG/COSMIC conflicts; write `data_quality_report.md`.
3. **Task design:** define gene-level prediction unit, exact label cohort and unknown handling, candidate universe, and graph/source-aware evaluation before model selection.
4. **Architecture specification:** freeze expression feature dimensions, graph node coverage, encoder widths, fusion and output semantics; document the HMGAT-ET design and ablations.
5. **Pipeline implementation:** add reproducible, non-destructive preprocessing and mapping outputs, configurations, logs, and focused tests.
6. **Model work:** first validate a tiny CPU-capable smoke configuration, then train controlled baselines and the primary hybrid model only if RAM/dependencies permit.
7. **Evaluation and interpretation:** compare consistent splits, report variability and limitations, produce rankings only for eligible unlabeled candidates, and generate figures from measured outputs.
8. **Final documentation:** record actual and failed/not-run experiments, commands, resources, and limitations in a reproducible research report.

## Audit limits

The full 5.2 GB of expression matrices was not parsed row-by-row. Therefore exact unique sample/case counts, per-file gene row counts, numeric/missing-value summaries, tumor/normal composition, duplicate sample detection, expression/graph overlap, and expression/annotation overlap are **not yet established**. Those belong to the dedicated data-validation phase. No expensive training, downloads, or modifications to existing data/model artifacts were performed.
