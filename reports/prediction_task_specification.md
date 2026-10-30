# Prediction-task specification

**Status:** Task definition, cohort aggregation, stable-ID mapping, integrated input construction, and model evaluation are implemented. See the final update at the end and `phase6_model_evaluation.md`.

**Phase 3 decision:** The primary task is pan-cancer NCG `NCG_oncogene` annotation prediction among the NCG cohort, with BRCA-derived expression context. It must not be described as BRCA-specific driver discovery. The breast-evidence subset is a sensitivity cohort only. See [Phase 3 task decision](phase3_task_decision.md) and [Phase 4 HMGAT-ET design](phase4_hmgat_et_design.md).

## Prediction unit and candidate universe

- **Prediction unit:** gene, with one row per stable Entrez Gene ID after identifier reconciliation. This is not patient-level tumor classification.
- **Initial bounded candidate universe:** 3,347 unique Entrez IDs represented in the NCG annotation/systems-properties cohort. It is a curated cohort, not a genome-wide set of confirmed drivers and non-drivers. A gene must also have valid expression representation and identifier mapping before it can enter a multimodal model.
- **Graph representation:** STRING v12.5 links with the matching human v12.5 aliases; confidence scores are retained as undirected edge weights. The NCG-induced score-700 graph has 2,799 connected nodes and 24,956 edges. Full release-matched mapping and component statistics are in `data/processed/validation_20261028_152218/string_validation_summary.json`.

## Initial target and label policy

The initial supervised target is **NCG_oncogene**, a source-defined gene-level annotation. This narrow target avoids collapsing oncogene, tumor-suppressor, and fusion roles into a generic driver class.

- Positive: at least one gene-level NCG record explicitly labels the target 1, and no record labels it 0.
- Explicit comparison zero: one or more NCG records explicitly label it 0, and none label it 1. This is an NCG source-defined comparison class; it does not prove the gene is biologically not an oncogene.
- Conflict: repeated NCG records contain both 0 and 1 for this target. Exclude from supervised fitting and report separately rather than silently applying positive-wins aggregation.
- Unknown: all target records are missing. Keep unlabeled; never recode as 0.
- Keep `NCG_tsg` separate. Do not use either NCG target or any derived annotation field as a predictor.
- COSMIC CGC v104 GRCh38 is reserved for an annotation comparison after training labels are frozen. It is not a feature and is not merged into the training target. Genes with both COSMIC oncogene and TSG roles remain multi-role reference records; role-specific concordance is reported separately.

## Expression-to-gene feature definition

GDC metadata now maps all local files to case/sample/sample type. Retain only `Primary Tumor` samples (1,103 samples across 1,092 cases). Exclude the 113 solid-tissue normal samples and 7 metastases. Five biological samples have multiple local files; average their raw TPM values across files before log transformation. Eleven cases have multiple primary-tumor samples; take the gene-wise median of sample-level `log2(TPM+1)` values within a case. Then compute mean, population standard deviation, and fraction of cases with TPM > 0 across the 1,092 cases. This produces one expression feature row per versioned Ensembl gene ID; implementation and exact counts are documented in `data/processed/expression_features_20261028_153824/expression_feature_summary.json`.

The implemented expression table contains mean, population SD, and nonzero fraction of case-level `log2(TPM+1)` summaries. These are gene-level cohort features, not patient-specific predictions. The source raw counts remain unchanged. Any imputation/scaling used by a predictive model must be fit on its training genes within each split; cohort/sample inclusion is fixed before model evaluation.

The historical blocker statement is superseded by the Phase 2 follow-up: GDC metadata now resolves local file-to-case/sample/type mapping, and the primary-tumor features have been computed using the aggregation rule above. These features remain Ensembl-keyed until an authoritative Entrez crosswalk is available.

## Identifier alignment and graph features

Expression identifiers are versioned Ensembl gene IDs. NCG uses Entrez IDs and symbols; COSMIC uses HGNC-style symbols and Ensembl versions embedded only in the synonyms field. Use a release-matched Ensembl-to-Entrez mapping as the canonical join, with symbol mapping as a separately audited fallback only. STRING protein identifiers must be mapped via same-release aliases; ambiguous protein-to-gene mappings are excluded from the primary mapping and counted. Do not equate gene symbols alone without ambiguity handling.

The node universe for an initial NCG-cohort experiment is eligible NCG IDs intersected with mapped STRING nodes. Genes lacking graph coverage may remain for expression-only analyses but cannot be passed to the GNN without an explicitly defined isolated-node policy. The whole STRING graph can provide unlabeled topology; target labels, label counts, and label-derived graph statistics are never input features. A held-out gene's unlabeled PPI neighborhood may be visible only under an explicitly transductive evaluation, which must be reported as such.

## Evaluation design

Use a predeclared gene-level holdout stratified by the resolved positive/explicit-zero labels, keeping conflicts and unknowns out of supervised scoring. Use repeated/nested cross-validation on the training partition for model selection and threshold choice; evaluate the final test partition once. Report PR-AUC/average precision, ROC-AUC, MCC, balanced accuracy, class-wise precision/recall/F1, and uncertainty across training folds. Fit feature transformations and any feature selection only inside training folds.

Because connected or functionally related genes are not independent, a random gene split is only a baseline estimate. Add a sensitivity analysis that holds out connected components or gene groups where feasible, and state whether graph topology is transductive. COSMIC overlap/concordance is a descriptive cross-catalogue comparison, not independent validation when the catalogues share genes or evidence.

## Readiness decision

The integrated model was subsequently built with exact NCBI Ensembl stable-ID cross-references and evaluated. See the follow-up below for the selected GCN, full model comparison, and limitations.

## Final implemented input and evaluation status

Exact Ensembl stable-ID mapping uses NCBI Gene dbXrefs; no symbol-only join. The final eligible universe is 3,304 genes, with 254 positive, 333 explicit-zero, and 2,717 unknown labels. Unknowns remain masked. Candidate intersection, feature construction, graph, split, model comparison, and validation limitations are detailed in `phase6_model_evaluation.md`. The CV-selected GCN reached 3-fold development AP 0.5693 ± 0.0517 and held-out AP 0.6030 on 118 genes. The evaluation is gene-level and transductive; component-held-out evaluation was infeasible due to insufficient labeled support outside the largest connected component. Results are preliminary, not independent or clinical validation.
