# Phase 5: gene-universe expansion, supervised retraining, and pseudo-label decision

**Final expanded dataset:** `outputs/experiments/phase5_expansion_20261028_181232/`  
**Supervised model/evaluation outputs:** `outputs/experiments/phase5_expansion_20261028_181232/training_replay_20261028_182511/`  
**Status:** The 5,000-gene target was exceeded using only existing local files. Three supervised model families were trained on the expanded network; out-of-fold metrics and a final development-only refit were verified. Pseudo-label retraining was stopped because the development data did not support reliable confidence thresholds. The old, previously exposed heldout set was not re-evaluated.

## Dataset audit and expansion

The former integrated universe had **3,304 genes**: 254 NCG positives, 333 NCG explicit-zero comparison labels, and 2,717 unknowns. The expanded feature/network universe contains **18,998 unique Entrez genes**, adding **15,694**. Of these, 18,998 have the existing TCGA-BRCA expression summaries and an unambiguous STRING v12.5 alias mapping; **17,076** have at least one retained score-700-or-higher edge, and 1,922 are isolated at that edge cutoff. The gene universe is determined by expression availability and exact identifier/network mapping, not by labels.

| Expanded universe measure | Count |
|---|---:|
| Unique genes | 18,998 |
| Genes with expression features in this universe | 18,998 |
| Genes with unambiguous STRING alias mapping | 18,998 |
| Genes with at least one retained PPI edge | 17,076 |
| Verified NCG positive labels | 254 |
| Verified explicit NCG zero labels | 333 |
| Unknown/unlabeled | 18,411 |
| Pseudo-labeled | 0 |

The source expression table contains 60,660 versioned GENCODE v36 Ensembl IDs. NCBI Gene `gene_info` exact stable-ID cross-references provided 36,327 unique Ensembl-to-Entrez mappings, 251 ambiguous Ensembl rows, and 24,082 unmapped rows. To avoid arbitrarily combining separate expression records, a further 62 rows representing 31 Entrez IDs with multiple Ensembl records were excluded. This leaves 36,265 strict one-to-one mapped expression IDs. Of those, 17,267 Entrez IDs had no unambiguous STRING mapping; the remaining 18,998 form the expanded universe. No symbol-only join was used.

The STRING v12.5 links file has 10,826,054 rows. The streamed score-700-or-higher pass saw 565,728 rows; 555,100 had both endpoints mapped through unambiguous v12.5 aliases. After the candidate-universe restriction, six mapped gene self-links were discarded and protein interactions collapsing to a repeated canonical Entrez pair were reduced by retaining the maximum score. The result is **273,053 unique undirected edges** (**546,106 directed message edges**), with mean degree 28.75, maximum degree 862, and 1,922 isolated nodes. Raw expression and STRING files were preserved.

### Expression cohort provenance

The reused GDC join maps 1,228 local expression files to **1,223 unique samples and 1,093 cases/patients**. Across all sample types there are 1,103 primary-tumour samples, 113 solid-tissue normals, and 7 metastases. Expression features were built from the **1,103 primary-tumour samples across 1,092 primary-tumour cases**, using the documented duplicate-file/sample/case aggregation. Five biological samples had multiple files and 11 cases had multiple primary-tumour samples. All 1,228 local matrices validated; three manifest files are absent locally. The new dataset uses these existing gene-level mean, population SD, and TPM-positive-fraction summaries; it does not create patient-level rows.

### Labels and sources

The frozen NCG `NCG_oncogene` table remains the only training target. The strict expanded intersection retains the same 254 positive and 333 explicit-zero labels as the former integrated dataset. **No new verified labels were obtained**, and no new source or external dataset was downloaded. Of the 3,347 NCG source genes, 43 remain outside the strict expression/network universe. Every other added gene is unknown unless the frozen NCG table explicitly provides a valid target value. Unknowns were never converted to negatives.

An explicit NCG zero is a source-defined comparison label, not proof that the gene is biologically non-driver. Conflicting or missing target states remain masked/excluded under the existing label policy. COSMIC CGC v104 GRCh38 remains outside candidate construction, model inputs, training targets, pseudo-labeling, threshold selection, and hyperparameter selection. It was consulted only after predictions were frozen for a limited descriptive reference overlap. No COSMIC records were redistributed.

## Split and leakage controls

The Phase 2 development assignments and three development folds were mapped into the expanded gene index by Entrez ID. The same **469 development genes** (203 positive, 266 explicit zero) and **118 previously exposed heldout genes** (51 positive, 67 explicit zero) remain the scored cohorts. There were no additional verified NCG labels to add to training. All 18,411 unknowns remain unscored as labels. Feature scaling was fitted separately on fold training genes. The loss was evaluated only at known development training indices; fold validation labels were used for early stopping/metrics, and heldout labels were not indexed during selection.

The graph is transductive: PPI edges and unlabeled node features, including those of heldout nodes, are visible. Labels are not. The heldout set has already been evaluated in earlier project phases, so Phase 5 did not repeat its metrics or call it an untouched test set. COSMIC is a cross-catalogue reference, not an independent performance claim. The available COSMIC comparison table is reconciled to the NCG reference cohort and does not cover all 18,411 unknown genes.

## Supervised model comparison

The same old development genes and folds were used for the expanded models. Logistic regression used the three expression summaries plus log degree and log weighted STRING strength, with train-fitted standardization and balanced class weights. The expanded GCN used the two topology features, 32 hidden units, two layers, dropout 0.30, AdamW learning rate 0.008, weight decay 0.001, and unweighted loss. The expanded sparse graph Transformer used the two topology features, two four-head attention blocks, hidden dimension 32, dropout 0.25, AdamW learning rate 0.003, weight decay 0.005, and training-fold positive weighting. Neural models were evaluated over three seeds and three folds, with a 25-epoch maximum and patience 5. These are small CPU-sized models on the full 18,998-node graph.

| Model | Mean fold AP ± SD | Pooled OOF AP | Pooled OOF ROC-AUC | Fits |
|---|---:|---:|---:|---:|
| Expanded sparse graph Transformer | **0.5169 ± 0.0317** | 0.4912 | 0.5336 | 9 |
| Expanded GCN | 0.4777 ± 0.0399 | **0.4999** | **0.5905** | 9 |
| Logistic expression + topology | 0.4638 ± 0.0032 | 0.4472 | 0.5059 | 3 |

The sparse Transformer was selected among the three Phase 5 models by the predeclared **mean fold AP** rule. Pooled OOF AP instead favors the GCN, and the Transformer’s pooled ROC-AUC is also lower. This ranking disagreement means the apparent winner is not robust. At the same time, the Phase 4 sparse Transformer on the old 3,304-gene universe scored 0.5898 ± 0.0518 mean-fold AP and 0.5723 pooled OOF AP; Phase 3 selected GCN scored 0.5692 ± 0.0399 and 0.5509 pooled OOF AP. Thus the expanded-universe runs **did not improve development performance** under comparable old validation genes/folds. Expansion changed the graph context and topology-feature distribution, so these are comparative diagnostics, not a controlled claim that node count alone caused the drop.

At the default 0.5 threshold, the fold/seed mean ± SD secondary measures were:

| Model | Accuracy | Precision | Recall | F1 | Balanced accuracy | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Expanded sparse graph Transformer | 0.5344 ± 0.0627 | 0.4566 ± 0.2020 | 0.4078 ± 0.3493 | 0.3585 ± 0.2094 | 0.5184 ± 0.0294 | 0.0453 ± 0.0822 |
| Expanded GCN | 0.5764 ± 0.0211 | 0.1345 ± 0.2694 | 0.0719 ± 0.1444 | 0.0923 ± 0.1834 | 0.5164 ± 0.0335 | 0.0398 ± 0.0831 |
| Logistic expression + topology | 0.5138 ± 0.0435 | 0.4423 ± 0.0434 | 0.4289 ± 0.0551 | 0.4324 ± 0.0178 | 0.5039 ± 0.0334 | 0.0090 ± 0.0685 |

On the selected Transformer’s pooled development OOF scores at 0.5, accuracy was **53.7%**, AP 0.4912, ROC-AUC 0.5336, precision 0.4563, recall 0.3596, F1 0.4022, balanced accuracy 0.5163, and MCC 0.0340. The confusion matrix was TN 179, FP 87, FN 130, TP 73 among 266 explicit zeros and 203 positives. The aspirational **80% accuracy target was not reached**; this is a development OOF result, not a heldout-test score.

For completeness, MCC thresholds were selected from the development OOF scores only. The selected Transformer threshold was 0.585 (precision 1.000, recall 0.025, F1 0.048, accuracy 0.578, balanced accuracy 0.512, MCC 0.119; TN 266, FP 0, FN 198, TP 5). The GCN threshold was 0.43 (precision 0.477, recall 0.828, F1 0.605, accuracy 0.533, balanced accuracy 0.568, MCC 0.156; TN 82, FP 184, FN 35, TP 168). These threshold results reuse OOF labels to choose the threshold and are selection-biased; AP and ROC-AUC remain the ranking comparisons.

The selected Transformer’s development OOF score had Spearman correlation **0.477** with PPI degree. AP by degree quartile was 0.369 (lowest), 0.471, 0.395, and 0.616 (highest). The high-degree performance advantage is substantial and should temper any interpretation of model scores as gene-specific biological evidence.

## Pseudo-labeling and independent reference

Development OOF scores did not support reliable positive or negative pseudo-label thresholds. The predeclared 0.95/0.05 confidence screen plus a maximum fold/seed SD of 0.05 yielded **zero** stable positive and zero stable negative candidates. The development-only calibration rule additionally required at least 10 OOF genes with empirical correctness ≥0.90 for a class; none of the three models had a supported threshold for either class. Unknown-gene mean score ranges were 0.367–0.469 for GCN, 0.462–0.583 for the sparse Transformer, and 0.304–0.750 for logistic regression. These weakly separated ranges do not justify relaxing the rule. Therefore conservative pseudo-training, cross-model-agreement retraining, and weighted iterative pseudo-labeling were stopped before retraining. This avoids creating provisional labels from scores that failed development calibration. The **18,411 genes remain unknown**.

The post-freeze descriptive COSMIC overlap found six matches among all 18,411 ranked unknown genes in the available NCG-reconciled COSMIC table, but **none among the top 500**. Because that reference table is NCG-scoped and does not cover the full expanded universe, these counts are not an independent validation metric. COSMIC was not used to alter the candidate scores or selection.

The top ten exploratory candidate scores are shown below. They remain unknown NCG labels; expression and degree columns are feature summaries, not supporting causal evidence. The vote count is the number of three model families with an ensemble probability ≥0.5; it is not a confidence probability.

| Rank | Gene | Entrez | Score | Mean log2(TPM+1) | STRING degree | Model votes |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TRIM28 | 10155 | 0.6067 | 8.029 | 354 | 2/3 |
| 2 | UBC | 7316 | 0.6035 | 9.335 | 569 | 1/3 |
| 3 | MAPT | 4137 | 0.6022 | 4.083 | 613 | 2/3 |
| 4 | ANLN | 54443 | 0.6016 | 4.351 | 569 | 2/3 |
| 5 | HDAC1 | 3065 | 0.6010 | 6.770 | 373 | 2/3 |
| 6 | GNB1 | 2782 | 0.5991 | 7.899 | 413 | 2/3 |
| 7 | ACTB | 60 | 0.5989 | 11.860 | 601 | 1/3 |
| 8 | CDC42 | 998 | 0.5984 | 5.910 | 290 | 2/3 |
| 9 | TP53BP1 | 7158 | 0.5958 | 3.911 | 354 | 2/3 |
| 10 | RBX1 | 9978 | 0.5957 | 5.944 | 326 | 2/3 |

These are ranking hypotheses only, not confirmed cancer drivers. The model is degree-sensitive, the reference overlap is limited, and no independent cohort supports biological validation here.

## Artifacts and verification

- Expanded table and features: `outputs/experiments/phase5_expansion_20261028_181232/data/expanded_gene_features_and_labels.csv`
- Expression and topology matrices: `data/expression_features.npy`, `data/topology_features.npy`
- Stable-ID crosswalk and label provenance: `data/gene_identifier_mapping.csv`, `data/label_provenance.csv`
- Canonical graph and bidirectional messages: `data/graph_edges_ge700.csv`, `data/graph_edge_index.npz`
- Cohort/sample manifest and audit: `audit/tcga_sample_case_manifest.csv`, `audit/initial_dataset_audit.md`, `audit/phase5_dataset_audit.json`; feature counts and graph stats: `audit/feature_availability.csv`, `audit/graph_validation_statistics.json`
- Source/license provenance: `audit/source_provenance.json`; detailed per-gene label provenance: `data/label_provenance_detailed.csv`
- Fold checkpoints and logs: `checkpoints/`, `logs/` at the dataset-run root. The selected final development refit is `training_replay_20261028_182511/checkpoints/final_selected.pt`.
- Fold metrics and summaries: `training_replay_20261028_182511/metrics/supervised_fold_seed_metrics.csv`, `supervised_summary.csv`, `development_oof_metrics_at_0_5.csv`, and `comparable_model_ap.csv`.
- Predictions and pseudo-label audit: `training_replay_20261028_182511/predictions/all_gene_scores_supervised.csv`, `ranked_unknown_candidates_with_model_agreement.csv`, `pseudo_label_candidates.csv`; `metrics/pseudo_label_eligibility.json`.
- Plots: `training_replay_20261028_182511/plots/development_oof_pr.png` and `development_oof_roc.png`.
- Integrity and leakage: `training_replay_20261028_182511/metrics/phase5_integrity_checks.json` passed **12/12** checks covering 18,998 unique IDs, finite features/scores, label masking, split coverage, edge counts, pseudo unknown counts, and final checkpoint prediction replay. `metrics/leakage_audit.json` records the target/holdout/feature controls. The separate `audit/expanded_graph_integrity_checks.json` passed **5/5** checks for bidirectional messages/weights, no self-edges, edge counts, and retention of the original 3,304-gene universe.

Several early build/aggregation attempts are preserved and recorded as superseded/failed in `metrics/experiment_registry.csv`. The final graph export contains both message directions, all fold checkpoints were reloaded to reconstruct OOF predictions, and the final selected checkpoint reproduces its all-gene scores within 1e-6. No previous phase artifacts were overwritten. No dependencies were installed and no data were downloaded.

## Reproduction

From PowerShell in `D:\Quantum`, create a new run (the scripts refuse to overwrite existing checkpoint artifacts):

```powershell
.\.venv\Scripts\python.exe src\build_phase5_expanded_dataset.py
$run = "outputs\experiments\phase5_expansion_YYYYMMDD_HHMMSS" # use the path printed by the builder
.\.venv\Scripts\python.exe src\complete_phase5_dataset_audit.py $run
.\.venv\Scripts\python.exe src\train_phase5_expanded_models.py $run
.\.venv\Scripts\python.exe src\audit_phase5_pseudolabels.py $run
.\.venv\Scripts\python.exe src\finalize_phase5_expansion.py $run $run
.\.venv\Scripts\python.exe src\export_phase5_supporting_audits.py $run $run
.\.venv\Scripts\python.exe src\verify_phase5_graph_integrity.py $run
```

## Conclusion and next step

The local feature/network expansion exceeded 5,000 genes by a wide margin but did not add verified labels. With the expanded network, all three supervised models performed weakly on the unchanged development genes; the sparse Transformer’s mean-fold AP is above the expanded GCN, but pooled OOF AP favors GCN and both are below prior smaller-universe results. Pseudo-labeling was appropriately stopped because no class threshold met the development calibration rule. The most useful next step is a genuinely independent, harmonized cohort or carefully reviewed additional verified labels—not further pseudo-label iteration on these scores. The current exposed holdout should remain unused for further tuning.
