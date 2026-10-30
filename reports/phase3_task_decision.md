# Phase 3 prediction-task decision

**Decision status:** Task scope and label semantics are fixed; cohort feature construction and version-matched graph validation are complete. An authoritative Ensembl-to-Entrez mapping is still unresolved, so the integrated hybrid matrix, split manifest, and training are not ready.

## Primary task

**Gene-level prediction of the pan-cancer NCG `NCG_oncogene` annotation** among the 3,347 genes in the NCG annotation/systems cohort. The prediction unit is one Entrez gene. This is a source-label prediction task, not a biological ground-truth classification and not patient-level breast-cancer classification.

NCG label counts are 256 positive, 335 explicitly labeled zero, and 2,756 unknown. The zero class means only that NCG explicitly gives zero for this field; it is not a claim that those genes cannot be drivers. Unknown genes remain unlabeled. Repeated NCG records produced zero 0/1 conflicts for this target. `NCG_tsg` remains a separate task and will not be blended into oncogene labels.

### Scope limitation

The expression modality is TCGA-BRCA, while the primary target is a pan-cancer NCG annotation. The model can therefore answer: **how well do BRCA-derived gene-expression summaries, PPI structure, and non-target gene features predict NCG's pan-cancer oncogene annotation in the curated NCG cohort?** It cannot establish BRCA-specific driver status or discover a novel driver from the present labels. Reports and candidate lists must use this narrower language.

The NCG evidence table contains 572 breast-associated evidence rows across 298 unique genes. If restricted to those rows, the NCG oncogene field gives 52 positive, 99 explicit zero, and 147 unknown genes, with no within-subset conflicts. This 151-labeled-gene cohort is retained as a **breast-evidence sensitivity analysis**, not as an independent test set. It is smaller and still source-defined; the label field is not guaranteed to mean breast-specific causal driver status. Its data and COSMIC comparison are in `data/processed/phase3_task_20261028_145404/`.

## Candidate universe and eligibility

The annotation candidate universe is the 3,347 NCG Entrez IDs, not all human genes. STRING v12.5 aliases map 3,325 NCG genes unambiguously. A symbol-only expression crosswalk remains a candidate bridge and is not authoritative; an Ensembl v102 BioMart query returned HTML rather than tabular data. The cohort feature table is computed for 60,660 versioned Ensembl IDs across 1,092 primary-tumor cases. Freeze the integrated model universe only after stable-ID/Entrez mapping is resolved and reviewed.

For the multimodal analysis, a gene is eligible only if it has a stable-ID expression mapping, a gene-level expression vector computed from the predeclared TCGA cohort, and a valid STRING protein mapping. Expression-only comparisons may retain genes without graph edges. For mapped proteins with no edge at the fixed confidence threshold, include the node as isolated with a self-message path; do not silently drop it. Genes without any mapped protein cannot be represented in the graph branch and are excluded from the hybrid comparison, with counts reported.

## Feature and label separation

The implemented expression vector for each Ensembl gene is `[mean(log2(TPM+1)), population SD(log2(TPM+1)), fraction(TPM > 0)]` across 1,092 primary-tumor cases. Duplicate files for five samples are averaged in TPM space; for eleven cases with multiple primary samples, sample-level log TPM is median-combined per gene before across-case statistics. These summarize a BRCA cohort and are not per-patient predictive features. The sample/case manifest records all local files and sample types.

The STRING graph uses undirected confidence-weighted edges. Edge scores describe association confidence. Graph-derived predictors may include topology computed without cancer labels; do not use target labels, annotation membership, target counts, or label-derived centrality. COSMIC CGC v104 is reference-only and must not enter predictors, target construction, tuning, or threshold selection.

The crosswalk from Ensembl IDs to NCG currently uses exact symbol matches and is not authoritative. Replace it with a versioned gene mapping before any expression/label join. Mapping ambiguity must be tracked, never resolved by choosing the first row.

## Evaluation protocol

1. Freeze the eligible gene universe and labels before splitting. Keep unknown and conflicting labels out of supervised scoring; retain their status for later ranking.
2. Create a stratified gene-level 80/20 holdout from the eligible positive/explicit-zero genes once, with a fixed seed. Do not place a gene in both partitions. Save the split manifest and input hashes.
3. Use only the development partition for repeated/nested stratified cross-validation and all model/threshold choices. Fit imputation, scaling, feature selection, and calibration within training folds.
4. Score the final holdout once. Report PR-AUC/average precision, ROC-AUC, MCC, balanced accuracy, precision, recall, F1, specificity, and uncertainty. Do not use accuracy alone. Do not call scores calibrated probabilities unless a held-out calibration assessment supports that claim.
5. The default graph protocol is **transductive**: unlabeled graph structure and expression inputs for train/test genes may be visible, but test target labels are masked and never used for feature generation or tuning. State this explicitly.
6. Add a graph-component sensitivity evaluation. Assign complete connected components to folds where feasible; report when giant-component structure or class support makes the result infeasible. The current graph has 27 components and one component of 2,743 nodes, so component-held-out estimates may be highly imbalanced.
7. COSMIC overlap is descriptive catalogue concordance only. NCG and COSMIC share genes and may share underlying evidence; do not call this independent biological validation.

The labeled primary cohort is small for a high-capacity neural network. HMGAT-ET must be regularized and compared against simple models on identical splits. If the hybrid does not improve robust metrics or ranking utility, report that result and simplify rather than tune on the holdout.

## Phase 3 exit criteria

The prediction unit, target, candidate cohort, label handling, modality meaning, leakage controls, and evaluation protocol are specified. This phase 3 decision predates the NCBI exact-ID crosswalk; the mapping gate was subsequently cleared, and the integrated model was evaluated. See `prediction_task_specification.md` and `phase6_model_evaluation.md` for the current status.

