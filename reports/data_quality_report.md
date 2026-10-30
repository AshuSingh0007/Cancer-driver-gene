# Phase 2 data quality and validation report

**Run date:** 2026-10-28  
**Validation code:** `src/validate_data.py`, `src/export_mapping_outputs.py`, `src/integrate_gdc_metadata.py`, `src/build_gene_expression_features.py`, `src/export_model_input_components.py`  
**Latest machine-readable validation:** `data/processed/validation_20261028_152218/`  
**Phase 2 historical snapshot:** Original expression, STRING links, and annotation sources were preserved. GDC case/sample metadata and STRING v12.5 human aliases were downloaded to a new reference folder after approval. Subsequent approved follow-up acquired the NCBI Gene reference, installed CPU PyTorch in the project virtual environment, and trained/evaluated models; see the superseding Phase 6 sections below.

## Executive status

All 1,228 local matrices parse successfully, share schema/gene universe, and match the local manifest size and MD5. The GDC Files API metadata join now maps all 1,228 local files to 1,223 unique samples and 1,093 unique cases. Five samples have multiple expression files. The local cohort has 1,103 primary tumors, 113 solid-tissue normals, and 7 metastases. Three manifest files are absent locally.

The full STRING v12.5 links file validates structurally, and a human-filtered STRING v12.5 aliases file is now acquired and header-validated. NCG-to-STRING mapping covers 3,325/3,347 genes unambiguously. The release-matched NCG graph has 2,799 connected nodes, 24,956 edges, 27 components, and a 2,743-node largest component. NCG labels preserve positive, explicit-zero, unknown, and conflict states. COSMIC CGC remains a separate reference catalogue.

**Phase 2 readiness snapshot (superseded):** At the time of the original Phase 2 run, integrated training was blocked on an Ensembl-to-Entrez mapping. The NCBI stable-ID crosswalk and model run were completed afterward; see the final follow-up sections below and `phase6_model_evaluation.md`.

The detailed sections below summarize the initial local-only audit. The **Phase 2 follow-up** at the end supersedes their earlier unresolved sample-metadata and STRING-alias statements.

## 1. TCGA-BRCA expression files

### Inventory, completeness, and identity

| Measure | Verified result |
|---|---:|
| Local augmented STAR TSV files | 1,228 |
| Total file bytes | 5,210,744,298 |
| Successfully parsed and validated | 1,228 / 1,228 |
| Empty/corrupt/schema-invalid files | 0 |
| GDC manifest entries | 1,231 |
| Local files found in manifest | 1,228 / 1,228 |
| Manifest entries missing locally | 3 |
| MD5 matches / mismatches among local files | 1,228 / 0 |
| Manifest size matches / mismatches | 1,228 / 0 |
| Duplicate file-content groups (MD5) | 0 |

For every local file, the manifest file ID matches the enclosing directory UUID; filenames match manifest filenames. These identify GDC files. They do **not** establish sample or patient counts. The three absent filenames are listed in `tcga_validation_summary.json` and `tcga_file_validation.csv`.

### Schema and row integrity

All files have the same GENCODE v36 header:

`gene_id`, `gene_name`, `gene_type`, `unstranded`, `stranded_first`, `stranded_second`, `tpm_unstranded`, `fpkm_unstranded`, `fpkm_uq_unstranded`.

Across files there are 74,490,480 gene rows (60,660 per file) plus exactly four STAR summary rows per file: `N_unmapped`, `N_multimapping`, `N_noFeature`, and `N_ambiguous`. Every file has the same 60,660 gene IDs. There are no duplicate gene rows, malformed records, missing gene symbols, missing numeric cells, negative/non-finite values, nonnumeric expression cells, or fractional raw-count values. The gene IDs are versioned Ensembl IDs; 44 legitimate GENCODE PAR-Y entries use the `_PAR_Y` suffix. That suffix was included in the final Ensembl validity rule rather than treated as a malformed identifier.

The raw count fields and normalized TPM/FPKM fields are separate. No normalization or expression aggregation was applied in Phase 2.

### Sample/case mapping and cohort eligibility

The local GDC manifest contains file ID, filename, MD5, size, and state only. The 32 local `annotations.txt` files contain GDC case/aliquot observations and exceptions but do not link the expression file UUIDs to a barcode, case, or sample type. Thus:

- Unique GDC **file IDs:** 1,228.
- Unique biological samples: **unavailable**.
- Unique patients/cases: **unavailable**.
- Primary tumor, normal, recurrent, metastatic, and other sample counts: **unavailable**.
- Duplicate biological samples or aliquots: **unresolved**; no file-content duplicates were found, which is not a biological duplicate check.

The file manifest is provided in `tcga_sample_case_manifest.csv`, with sample/case/type values explicitly null. No samples are included/excluded by filename inference. The user-provided rule in `reports/prediction_task_specification.md` requires eligible primary tumor sample metadata before calculating per-gene cohort expression summaries.

## 2. STRING network


The links file is STRING v12.5, header `protein1 protein2 combined_score`, and has 10,826,054 data rows. There are 19,649 distinct protein IDs, scores range from 150 to 999, and all scores lie within the expected 0â€“1000 range. Score bins: below 400: 8,673,606; 400â€“699: 1,586,720; 700â€“899: 315,040; 900+: 250,688. The source pairs are monotonically ordered, with zero adjacent duplicate pairs and zero self-links.

The available alias file contains 3,889,207 rows and is STRING v12.0 (the directory name saying v12.5 does not change the file's actual name/version). No compatible v12.5 alias file is present. Of 19,649 linked proteins, 19,185 have an unambiguous Entrez alias. At the row endpoint level, 98.93% of interaction endpoints map; 97.64% of distinct linked proteins map. There is one protein with multiple distinct Entrez aliases. Within NCG, 3,325 of 3,347 distinct Entrez genes map to at least one unambiguous protein (99.34%).

Rebuilding the NCG-restricted graph with score >=700 and this provisional alias map yields 49,914 mapped high-score link rows before gene self-link removal, no mapped gene self-links, 24,958 repeated canonical gene-edge occurrences, and 24,956 unique gene edges. The saved graph has 2,799 nodes, 27 components, largest component 2,743, density 0.006373, mean degree 17.832, and maximum degree 402. It is an NCG-induced subgraph, not a genome-wide graph. STRING combined score is an association/confidence score, not a measured binding strength. The graph copy at `ppi_edges_provisional.csv` is explicitly provisional; the original `outputs/NCG_STRING_PPI_edges.csv` remains untouched.

**Resolution:** the official STRING v12.5 download offers the human-filtered `9606.protein.aliases.v12.5.txt.gz` (18.9 MB). After approval, acquire it, record source/version/checksum, and rerun validation/mapping. The [STRING 12.5 download page](https://string-db.org/cgi/download?species=Homo+sapiens) documents the human-prefixed file. No network download was initiated during Phase 2.

## 3. NCG labels and COSMIC reference comparison

NCG annotation has 8,886 evidence rows, 3,347 unique Entrez IDs and 3,347 unique symbols. The systems-properties table has 3,347 rows with one row per Entrez ID. There are no gene-symbol-to-multiple-Entrez conflicts in the NCG file and no repeated-row 0/1 conflicts for either NCG label.

| NCG target | Positive | Explicit zero | Unknown | Conflicting repeated records |
|---|---:|---:|---:|---:|
| Oncogene | 256 | 335 | 2,756 | 0 |
| TSG | 254 | 337 | 2,756 | 0 |

The output `ncg_gene_label_table.csv` preserves the label status per gene. For this task, positive means NCG explicitly marks the target 1; an explicit 0 is only a source-defined comparison label, not proof of biological non-driver status; conflict means both 0 and 1 across repeated records and is excluded; unknown remains unlabeled and is never made negative. The two targets remain separate, and all label-derived fields are forbidden as input predictors.

Both COSMIC CGC v104 GRCh37 and GRCh38 exports have 768 unique gene symbols and 21 columns. They are alternative genome-build tables, not two independent validation cohorts. GRCh38 shares 656 symbols with NCG; 112 COSMIC symbols are not in NCG, and 2,691 NCG symbols are not in COSMIC. COSMIC roles comprise TSG 193, oncogene+fusion 143, fusion 140, oncogene 111, TSG+fusion 61, oncogene+TSG 39, oncogene+TSG+fusion 32, fusion+oncogene ordering variant 3, and missing role 46. The reference table includes multi-role genes and cannot be reduced to an all-purpose binary target without losing meaning.

There are 63 oncogene-role and 61 TSG-role discordances between the NCG status and the COSMIC role field under the descriptive comparison; these are **not proven biological conflicts**. A missing COSMIC role or a role that does not list a target role is not equivalent to a curated negative. See `ncg_cosmic_gene_comparison.csv` and `annotation_conflicts.csv`.

The local COSMIC README describes the fields but omits licensing conditions. COSMIC's official [download page](https://cancer.sanger.ac.uk/cosmic/download/cosmic) states that non-commercial users must register and commercial users need a commercial license. The official [non-commercial terms](https://cancer.sanger.ac.uk/cancergenome/assets/COSMIC_non-commercial_license_May2021.pdf) limit non-commercial use and restrict sharing/distribution, with a narrow publication-enablement allowance requiring acknowledgement and the release version. Do not distribute the source tables or derived COSMIC-containing tables outside the applicable terms; verify the license accepted for the supplied v104 files before publication or sharing.

## 4. Identifier mappings and overlap

Expression has 60,660 Ensembl gene IDs and gene symbols in GENCODE v36. NCG labels use Entrez IDs and symbols. STRING uses Ensembl protein IDs and its aliases. The local source set has no authoritative Ensembl v36 gene-to-Entrez mapping table. Exact symbol matching gives 3,343 unique expression-symbol/NCG-symbol candidates, but 1,343 symbols are repeated across multiple expression gene IDs; this is a provisional bridge, not authoritative stable-ID mapping. The candidate crosswalk excludes ambiguous symbol-to-gene cases from its `unique_symbol_candidate_only` status.

The detailed files are `expression_symbol_ncg_candidate_crosswalk.csv` and `string_protein_entrez_mapping_v12_0_aliases.csv`; interpretation is in `reports/gene_identifier_mapping_report.md`. Exact expression-to-PPI node overlap by stable identifier remains unavailable pending an authoritative Ensembl-to-Entrez bridge and release-matched STRING aliases.

## 5. Prediction task and data leakage controls

The prediction unit is gene-level. The initial target is NCG oncogene label, in the bounded 3,347-gene NCG cohort; it is not a genome-wide driver/non-driver task. Explicit zeros are source-defined comparisons; unknown and conflicting genes are excluded from supervised fitting but retained with status. COSMIC is kept separate as a reference comparison and never used as input or merged into the target. No label-derived graph statistic is allowed as a feature.

After cohort metadata is available, the planned expression summary is per gene across a predeclared primary solid-tumor cohort: mean and standard deviation of `log2(TPM+1)` plus the fraction of eligible samples with TPM > 0. This is gene-level cohort context, not patient-level expression classification. Train-fold transformations and feature selection must be fitted inside each training fold. A held-out test set is scored once after model selection. The random gene split is supplemented with connected-component/group sensitivity analysis; any use of unlabeled graph structure is declared transductive. Full rules and limitations are in `reports/prediction_task_specification.md`.

## 6. Model-readiness and missing outputs

Created: file validation table, GDC file identity/checksum manifest, summary JSONs, NCG label table, NCG/COSMIC comparison and conflict table, detailed provisional STRING protein mapping, provisional symbol crosswalk, copied provisional PPI edge list, and this report. Raw data and previous experiment outputs were preserved.

Not created because inputs do not support them defensibly: final primary-tumor sample cohort, patient/case/sample count report, gene-level expression feature table, authoritative expression-to-Entrez/STRING node mapping, or joined model-ready feature matrix. No train/test split or training run was performed. Memory was kept bounded during the scan; the input feature matrix was not materialized.

The environment check from Phase 1 measured 7.68 GiB total RAM and approximately 0.83 GiB available before this phase. The expression validator streamed one file at a time and kept only per-file audit summaries and the shared 60,660-ID set (observed Python working set about 85 MB during the scan). No 60,660 Ã— 1,228 matrix was built; even a single float32 matrix of that shape would require roughly 284 MiB before pandas/working-copy overhead.

## Reproducible commands

From PowerShell in `D:\Quantum`:

```powershell
.\.venv\Scripts\python.exe src\validate_data.py
.\.venv\Scripts\python.exe src\export_mapping_outputs.py
.\.venv\Scripts\python.exe src\finalize_phase2_outputs.py
.\.venv\Scripts\python.exe src\check_data_integrity.py
```

`validate_data.py` expects the local GDC manifest at `C:\Users\Hp\Downloads\gdc_manifest.2026-10-27.234917.txt`. It reads the raw TCGA, STRING, NCG, and COSMIC files. It writes only to the dated validation output directory. The finalizer corrects the GENCODE `_PAR_Y` suffix accounting after confirming the shared 60,660-gene universe.

## Phase 2 follow-up: metadata acquisition and model inputs

### GDC file-to-sample/case join

The GDC Files API was queried for all 1,231 file UUIDs in the local manifest. It returned all 1,231 records; none were missing. Of the 1,228 local expression files, all 1,228 mapped uniquely to one case, one sample, and one sample type. The other three records are manifest files absent from the workspace. Counts below are based on GDC IDs and API metadata, not filename inference.

| Measure | Verified result |
|---|---:|
| Validated local expression files | 1,228 |
| Unique local biological samples | 1,223 |
| Unique local cases | 1,093 |
| Primary Tumor samples | 1,103 |
| Solid Tissue Normal samples | 113 |
| Metastatic samples | 7 |
| Samples represented by multiple local files | 5 |
| Files with missing or ambiguous GDC mapping | 0 |

There are 1,108 local files assigned to 1,103 primary-tumor samples, which belong to 1,092 cases. Eleven cases have multiple primary-tumor samples. These data are recorded in `data/processed/validation_20261028_152218/tcga_sample_case_manifest.csv`; API records are preserved in `data/reference/metadata_20261028_151433/gdc_file_sample_case_metadata.json`.

### Feature aggregation and memory

Before feature construction, available physical memory was checked at approximately 0.72 GiB (7.68 GiB total). The feature builder streamed one expression file at a time and did not allocate a patient-by-gene matrix. Its declared rule is: average TPM across files for the same sample; transform with `log2(TPM+1)`; for cases with multiple primary-tumor samples, take the gene-wise median across samples; then calculate mean, population SD, and fraction with TPM > 0 across 1,092 cases. The result has 60,660 versioned Ensembl gene rows at `data/processed/expression_features_20261028_153824/gene_expression_features.csv`; exact cohort and aggregation details are in its companion JSON.

### STRING v12.5 aliases and graph

Human STRING v12.5 aliases were acquired after approval and validated against the expected header. The compressed file is preserved with its extracted table under `data/reference/metadata_20261028_151433/`. Version-matched mapping gives 19,197 proteins with an Entrez alias, 19,196 with one unambiguous Entrez ID, and one ambiguous protein mapping. There are 19,649 linked proteins; 19,185 linked proteins have an unambiguous Entrez mapping. Of the 3,347 NCG Entrez genes, 3,325 map to at least one unambiguous STRING protein (99.34%).

At the fixed score threshold 700, the NCG-induced graph has 2,799 connected nodes and 24,956 unique undirected gene edges; it has zero self-loops, 27 components, and a largest component of 2,743 nodes. The version-matched graph is `data/processed/validation_20261028_152218/ppi_edges_string_v12_5.csv`. Detailed mapping is in `string_protein_entrez_mapping_v12_5_aliases.csv` in the same directory.

### Identifier crosswalk and modeling readiness

The versioned expression feature table is keyed by Ensembl IDs; NCG labels and the graph use Entrez IDs. The planned Ensembl 102 BioMart query returned the current Ensembl archive help HTML rather than tabular mappings. That response is retained as an invalid-response artifact and **is not used as data**. No authoritative Ensembl v36-to-Entrez crosswalk has been produced. Exact symbol candidates are still available in the crosswalk report, but are not joined into a target-bearing matrix.

Separated prediction inputs are in `data/processed/model_inputs_20261028_154017/`: `graph_nodes.csv` (3,325 NCG genes with unambiguous v12.5 mapping, including isolated nodes), `graph_edges.csv` (24,956 edges), `gene_labels.csv` (positive/explicit-zero/unknown states), and a copy of the 60,660-row Ensembl-keyed expression features. At that historical point no split manifest or integrated matrix had been created. This statement was superseded by the NCBI crosswalk, integrated inputs, and completed model run documented in the final follow-up sections below. Unknown genes remain unlabeled; COSMIC remains reference-only.

### Reproducibility and integrity

Scripts executed include `src/acquire_missing_metadata.py`, `src/integrate_gdc_metadata.py`, `src/validate_data.py`, `src/export_mapping_outputs.py`, `src/build_gene_expression_features.py`, and `src/export_model_input_components.py`. The acquisition and validation JSON summaries, cohort manifest, mapping table, graph, feature table, labels, and machine-readable integrity checks are retained under the paths above. Raw expression, links, and annotation files were not changed. No dependencies were installed and no model training was started.
# Phase 6 model implementation and evaluation

**Run:** `outputs/experiments/integrated_20261028_161519/`  
**Status:** Completed CPU training and evaluation for nine neural configurations plus a regularized logistic reference; no training was started on COSMIC labels. The current run uses the residual hybrid prediction head and the matched Transformer encoder in both HMGAT fusion ablations. Earlier runs under `integrated_20261028_160239` and `integrated_20261028_160856` are retained as exploratory/interim artifacts and are not the final architecture run.

## Task and cohort frozen for this experiment

- Unit: one gene per node, not one patient. This predicts the NCG pan-cancer `NCG_oncogene` source annotation using BRCA-derived gene-level expression summaries and STRING graph context; it is not BRCA-specific causal driver discovery.
- Candidate universe: 3,304 NCG genes with unambiguous Ensembl-to-Entrez mapping, expression representation, and STRING graph-node mapping. Of these, 254 are positive, 333 explicitly annotated zero, and 2,717 unknown. Unknowns are masked, never negatives. No conflicting NCG records were present in the selected target; genes failing mapping/intersection are excluded and listed in the crosswalk summary.
- Expression: three cohort-level features (mean and population SD of case-level `log2(TPM+1)`, and nonzero fraction) summarized over 1,092 primary-tumor cases. These are not patient-specific predictors.
- Graph: STRING v12.5 human, score threshold >=700, edge weight `combined_score / 1000`, NCG candidate induced graph. It contains 3,304 nodes and 24,606 undirected edges (49,212 directed message edges). Topology-only features are log degree, log weighted strength, and weighted local clustering.
- Inputs exclude target labels and COSMIC. Scaling is fit on training genes within each split. Training is transductive: unlabeled test-node graph and non-target features are visible; test labels are not used in fitting or model selection.
- Split: seed 42, stratified 80/20 held-out gene split; 3-fold stratified CV on development labels for architecture/epoch/threshold selection; held-out test measured once. Component-held-out evaluation is infeasible under the predeclared minimum class-support rule: nodes outside the largest component contain 7 positives and 8 explicit zeros total.

## Measured comparison

Sorted by development CV average precision; test metrics are shown for reference only and did not select the model. The best-validation model is GCN based on CV AP.

| Model | CV AP mean Â± SD | Held-out AP | Held-out ROC-AUC | Held-out MCC |
|---|---:|---:|---:|---:|
| GCN | 0.5693 Â± 0.0517 | 0.6030 | 0.6178 | 0.1377 |
| GraphSAGE | 0.5687 Â± 0.0505 | 0.5657 | 0.6248 | 0.2419 |
| Expression + GAT | 0.5527 Â± 0.0241 | 0.5275 | 0.6073 | 0.1789 |
| GAT | 0.5320 Â± 0.0095 | 0.4943 | 0.5941 | 0.2492 |
| Expression + GraphSAGE | 0.5266 Â± 0.0472 | 0.4907 | 0.4966 | 0.0000 |
| Expression Transformer | 0.5244 Â± 0.0157 | 0.5284 | 0.5830 | 0.0563 |
| Expression residual MLP | 0.5221 Â± 0.0324 | 0.5980 | 0.6637 | 0.1102 |
| HMGAT-ET gated fusion | 0.5039 Â± 0.0173 | 0.4105 | 0.4393 | 0.0000 |
| Logistic expression + topology | 0.4912 Â± 0.0074 | 0.5986 | 0.6494 | 0.1532 |
| HMGAT-ET concatenation ablation | 0.4888 Â± 0.0334 | 0.4359 | 0.4592 | 0.0180 |

GCN's held-out set has 118 genes (51 positive, 67 explicit zero); at the threshold selected on development OOF predictions (0.585), precision is 0.552, recall 0.314, balanced accuracy 0.560, and macro F1 0.546. Ranking metrics at top 10/25/50 are precision 0.80/0.64/0.54. The small labeled cohort and modest CV performance mean these are exploratory source-label prediction results, not a clinically validated model. The hybrid did not outperform graph-only or simple reference approaches in this run.

## Reproducibility and outputs

- `configs/experiment.json`: seed, split/training settings, inputs, runtime environment.
- `metrics/model_comparison.csv` and `.json`: full comparison; each model has fold metrics under `metrics/*_cv_folds.csv`.
- `predictions/split_manifest.csv`, per-model development OOF, one-time held-out predictions, all-candidate scores, and top unknown candidate PPI neighborhoods.
- `checkpoints/<model>/`: fold-best and final checkpoints for each neural model.
- `logs/<model>/`: epoch-by-epoch training/validation logs.
- `plots/`: training curves, comparison, ROC and precision-recall visuals.
- `metrics/component_holdout_feasibility.json`: component split feasibility.
- `metrics/experiment_integrity_checks.json`: 19/19 automated artifact and data integrity checks passed.
- Peak observed process RSS was 0.503 GiB; CPU run elapsed 117.97 seconds. `torch 2.14.1+cpu` is recorded in the config. A sparse tensor invariant warning was emitted by PyTorch, but smoke tests and all integrity checks passed.

## Phase 3 readiness

The reproducible initial gene-level training/evaluation loop is operational. It is ready for Phase 3 research iteration, but results should remain preliminary until label provenance/definition is reviewed, external gene-group or time-split validation is designed, and component-level generalization limitations are addressed. Do not claim independent validation from COSMIC: keep it as a separate descriptive reference. The latest model input and label files are documented in `reports/prediction_task_specification.md` and their source summaries. No expensive tuning or further training was run beyond the stated comparison.

## Superseding resolution: NCBI exact Ensembl crosswalk and integrated model

An official NCBI Gene `Homo_sapiens.gene_info.gz` source was acquired into a new dated reference directory after approval. `src/build_ncbi_identifier_crosswalk.py` joins exact Ensembl stable IDs from NCBI `dbXrefs`; it strips the expression version suffix and recognized `_PAR_Y` suffix only. It does not choose mappings by symbol. The machine-readable crosswalk and summary are in `data/processed/identifier_mapping_20261028_155713/`.

NCBI lists 193,952 human records and 38,267 Ensembl stable IDs with Entrez cross-references. Of 60,660 expression IDs, 36,327 map uniquely, 251 are ambiguous, and 24,082 are unmapped. In the NCG-expression-graph intersection, 3,304 candidate genes remain (254 positive, 333 explicit zero, 2,717 unknown); 43 NCG genes were excluded for missing/ambiguous mapping or graph/expression mismatch. NCBI and expression symbols differ for 36 of the 3,304 eligible genes; these are flagged for review and do not alter stable-ID joins. Symbol discordance among all expression IDs is expected to include changed official symbols and aliases; no symbol fallback was used.

The integrated inputs at `data/processed/integrated_model_inputs_20261028_155809/` are aligned and mask unknown labels. The final experiment is `outputs/experiments/integrated_20261028_161519/`; the CV-selected graph convolutional model reached CV AP 0.5693 Â± 0.0517 and held-out AP 0.6030 (118 genes). The full interpretation, per-model comparison, evaluation limits, and output map are in `reports/phase6_model_evaluation.md`. Automated checks passed 19/19. See also the [NCBI Gene FTP documentation](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/data-processing/gene-processing/geneftp/) for the official Gene FTP source format.


