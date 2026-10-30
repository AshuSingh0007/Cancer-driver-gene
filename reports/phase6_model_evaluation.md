# Phase 6 model implementation and evaluation

**Final run:** `outputs/experiments/integrated_20261028_161519/` (completed 2026-10-28). This is the final current-code run with residual hybrid prediction heads and matched Transformer encoders for gated and concatenation HMGAT ablations. Earlier timestamped runs are preserved as exploratory/interim results and are not used below.

## Prediction task

This is gene-level prediction of the pan-cancer NCG `NCG_oncogene` source annotation in a bounded cohort, using gene-level summaries of TCGA-BRCA primary tumor expression and STRING protein association context. It is not patient-level classification, BRCA-specific causal driver discovery, or a genome-wide confirmed driver-versus-nondriver task. The candidate universe is 3,304 genes that align by stable Ensembl-to-Entrez cross-reference, expression, and STRING node. Labels are 254 positive, 333 explicitly annotated zero, and 2,717 unknown. Only positive/explicit-zero enter supervised loss/evaluation; unknown is masked, not negative. COSMIC CGC is separate and never used in training targets or predictors.

NCBI exact crosswalk source is `data/reference/ncbi_gene_info_20261028/Homo_sapiens.gene_info.gz`; code uses NCBI `dbXrefs` Ensembl stable IDs, strips the GENCODE version and recognized `_PAR_Y` suffix only, and does not resolve by symbol. Of 60,660 expression IDs, 36,327 map uniquely, 251 are ambiguous, and 24,082 have no unique mapping. There are 36 expression-vs-NCBI symbol discordances among the 3,304 eligible genes; stable-ID mapping is retained and all such cases are flagged. Full table and counts: `data/processed/identifier_mapping_20261028_155713/`.

## Feature and graph construction

Expression features are mean and population SD of case-level `log2(TPM+1)` and fraction of cases with TPM > 0, from 1,092 primary-tumor cases. These are cohort summaries, not patient-specific values. Expression and graph topology have 3 features each. Graph topology features are log degree, log weighted strength, and weighted clustering, from the STRING v12.5 candidate-induced graph at combined score >=700; edge weights are combined score / 1000. The eligible graph has 3,304 nodes, 24,606 canonical undirected edges (49,212 directed message edges). No label-derived features are included. Feature scalers are fit on training genes within each CV/final fit.

The split is gene-level and transductive: seed 42 stratified 80/20 holdout; 3-fold stratified CV on the development partition for model and epoch selection and threshold selection; the heldout test is evaluated once. Graph and unlabeled node features are visible across partitions, but only training labels contribute to optimization. This does not test generalization to new disconnected biological modules. The largest component has 2,723 nodes and 572 known labels (247 positive, 325 explicit zero). All other components combined have 7 positives and 8 explicit zeros, so a class-balanced component-held-out split was not feasible under the predefined support rule.

## Measured model comparison

Model selection uses mean development 3-fold average precision (AP); the test set is for final reference and did not choose the model.

| Model | CV AP mean ± SD | Test AP | Test ROC-AUC | Test MCC |
|---|---:|---:|---:|---:|
| Graph GCN | 0.5693 ± 0.0517 | 0.6030 | 0.6178 | 0.1377 |
| GraphSAGE | 0.5687 ± 0.0505 | 0.5657 | 0.6248 | 0.2419 |
| Expression + GAT | 0.5527 ± 0.0241 | 0.5275 | 0.6073 | 0.1789 |
| Graph GAT | 0.5320 ± 0.0095 | 0.4943 | 0.5941 | 0.2492 |
| Expression + GraphSAGE | 0.5266 ± 0.0472 | 0.4907 | 0.4966 | 0.0000 |
| Expression Transformer | 0.5244 ± 0.0157 | 0.5284 | 0.5830 | 0.0563 |
| Expression residual MLP | 0.5221 ± 0.0324 | 0.5980 | 0.6637 | 0.1102 |
| HMGAT-ET gated fusion | 0.5039 ± 0.0173 | 0.4105 | 0.4393 | 0.0000 |
| Logistic expression + topology | 0.4912 ± 0.0074 | 0.5986 | 0.6494 | 0.1532 |
| HMGAT-ET concatenation ablation | 0.4888 ± 0.0334 | 0.4359 | 0.4592 | 0.0180 |

GCN was selected by CV AP. On the single heldout set (118 genes; 51 positive, 67 explicit zeros), using development-OOF MCC threshold 0.585, its balanced accuracy was 0.560, macro F1 0.546, precision 0.552, recall 0.314, and MCC 0.138. At top 10/25/50, precision was 0.80/0.64/0.54. The AP is close to the heldout positive prevalence (51/118 = 0.432), and performance is modest. HMGAT-ET did not improve over graph-only models or logistic reference in this run. These results are exploratory and do not establish biological or clinical validity.

## Outputs and reproducibility

- Inputs: `data/processed/integrated_model_inputs_20261028_155809/` contains the aligned feature/label table, NumPy arrays, indexed graph edges, and input summary.
- Full comparison: `metrics/model_comparison.csv` and `.json`; per-fold CV metrics are `metrics/*_cv_folds.csv`.
- Split: `predictions/split_manifest.csv`; OOF, heldout and all-candidate score files are per model. Unknown candidate ranks are provided as hypotheses, preserving unknown label status.
- Neural checkpoints: `checkpoints/<model>/fold_*_best.pt` and `final.pt`; logs and plots under `logs/` and `plots/`.
- Runtime/config: `configs/experiment.json`; CPU torch 2.14.1; runtime 117.97 s; observed peak process RSS 0.503 GiB.
- Automated checks: `src/check_experiment_integrity.py` ran successfully with 19/19 checks; results at `metrics/experiment_integrity_checks.json`. A neural forward/backward smoke check passed for all nine neural configurations. PyTorch emitted a sparse invariant warning during graph tensor creation; training completed, finite-output/gradient smoke checks passed, and graph integrity checks passed.
- Full metrics, confusion counts, precision/recall at k, test predictions, and candidate neighborhoods are in the experiment output directory. Raw source data remain unchanged.

## Readiness

The data-to-training path is operational and Phase 3 research iteration can proceed. Before making biological claims, the label definition and provenance need review; additional external validation and leakage/generalization analysis should be designed. COSMIC is a separate descriptive catalogue and has licensing/redistribution restrictions documented in `reports/data_quality_report.md`; it is not independent validation here. No further training or expensive tuning was run in this phase.

## Reproduction commands and non-destructive output behavior

The raw expression, STRING, NCG, and COSMIC inputs are read-only to the processing pipeline. Dated builders create new directories and refuse a colliding output directory (`exist_ok=False`); experiment runs use a fresh timestamped directory. Do not edit the original source files. With the project `.venv` active, the validated final stages can be reproduced with:

```powershell
.\.venv\Scripts\python.exe src\build_gene_expression_features.py
.\.venv\Scripts\python.exe src\build_ncbi_identifier_crosswalk.py
.\.venv\Scripts\python.exe src\build_integrated_model_inputs.py
.\.venv\Scripts\python.exe src\check_integrated_alignment.py
.\.venv\Scripts\python.exe src\train_integrated_models.py --smoke-only
.\.venv\Scripts\python.exe src\train_integrated_models.py
.\.venv\Scripts\python.exe src\check_experiment_integrity.py <outputs/experiments/integrated_RUN_ID>
```

Data acquisition is not part of this command sequence; its already-approved reference inputs are preserved under `data/reference/`. The complete Phase 2 file validation and existing summaries remain under `data/processed/validation_20261028_152218/`. Each training execution uses a new experiment directory and performs no hyperparameter sweep beyond the declared CV/early-stopping procedure.
