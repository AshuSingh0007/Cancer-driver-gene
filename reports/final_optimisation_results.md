# Final optimization results

## Summary

The bounded development-only optimization is complete. **No candidate credibly improved on the saved expanded sparse graph Transformer by the primary ranking metric**, so it remains the final model. The aspirational **80% accuracy target was not reached**. The final copy is a replayable checkpoint from the completed Phase 5 development refit, not a newly trained improved checkpoint. No previous files were overwritten; all Phase 7 artifacts are timestamped under `outputs/experiments/final_optimization_20261028_233304`.

## Audit of prior results

Earlier metrics use different universes/splits and must remain separated. The initial NCG-only Logistic baseline report gives accuracy 0.647, AP 0.594 and ROC-AUC 0.675 on its single seed-42 80/20 split over a 2,756-gene NCG cohort; this is not comparable to later 18,998-gene runs. Phase 3 GCN heldout metrics were AP 0.6126, ROC-AUC 0.6313, balanced accuracy 0.6084, F1 0.5217, and accuracy 74/118=0.6271, but that heldout set was subsequently exposed in Phase 3 comparisons. It is not used as a clean test now.

Within Phase 5's expanded 18,998-gene universe, the saved Sparse Graph Transformer was chosen by mean fold-seed AP (0.5169 ± 0.0317); its pooled development OOF AP/ROC-AUC were 0.4912/0.5336. The GCN had the best pooled Phase 5 AP (0.4999) and ROC-AUC (0.5905), but at threshold 0.5 it predicted no OOF positives: accuracy 0.5672, balanced accuracy 0.5000, F1 0.000. The Transformer OOF accuracy was 0.5373, balanced accuracy 0.5163, and F1 0.4022. The Phase 5 fold/seed average ordinary accuracy was highest for GCN (0.5764 ± 0.0211); this accuracy was not accompanied by useful positive-class recall. At threshold 0.5, among Phase 5 pooled OOF models Logistic had the best F1 (0.4328) and the Transformer the best balanced accuracy (0.5163). In the new nested Logistic ablation, combined expression+network had the highest balanced accuracy (0.5251); the fixed-C Logistic had the highest F1 among those nested ablations (0.4367). Rankings differ by metric.

Features: Phase 5 Logistic used three expression summaries plus log degree and log weighted strength; GCN and sparse Transformer used graph topology features (log degree and log weighted strength) with the STRING graph. No label-derived predictors were used. The initial Phase 2 baseline Logistic used 16 NCG systems-level annotations; its accuracy cannot be compared to expression/network models. Phase 2/3 compared expression models (three cohort-summary features), graph models (STRING neighborhoods plus topology), and hybrids. Phase 3/4 explored graph architectures, layers, hidden width, class weighting, regularization, dropout and early stopping. The expanded universe has finite expression and topology inputs for all 18,998 genes; 1,922 genes are isolated at the retained STRING score cutoff, so graph message-passing context is absent for them. Phase 5's OOF validation fold also governed neural early stopping; it is a development metric, not a fully nested estimate. Current baseline comparisons are documented with this caveat.

## Phase 7 experiment and fair comparison

The new feature ablation used 469 labeled development genes (203 positives, 266 explicit zeros) and the fixed three outer gene folds. For each outer fold, a stratified three-fold inner CV selected Logistic Regression C and class weighting from C={.03,.1,1,10}, weights={unweighted, balanced}, with scaling fitted inside each training fold. Feature family was also selected using inner-fold AP. Outer fold labels were held until scoring. Heldout labels were excluded from fitting, inner tuning, feature selection, pseudo-labeling, threshold decisions, and model selection; no heldout evaluation was performed. All classifications use the fixed 0.5 threshold; no threshold was tuned for accuracy.

| Model | OOF AP | OOF ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Protocol note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| expanded_sparse_graph_transformer | 0.4912 | 0.5336 | 0.5373 | 0.5163 | 0.4562 | 0.3596 | 0.4022 | 0.0340 | same 3 folds; validation fold also governed early stopping |
| Nested LR: network_only | 0.4542 | 0.5040 | 0.5714 | 0.5096 | 0.5556 | 0.0493 | 0.0905 | 0.0495 | nested 3-fold outer CV; inner 3-fold AP search |
| expanded_gcn | 0.4999 | 0.5905 | 0.5672 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same 3 folds; validation fold also governed early stopping |
| Nested LR: inner-selected feature set | 0.4476 | 0.5065 | 0.5544 | 0.5109 | 0.4634 | 0.1872 | 0.2667 | 0.0284 | inner CV selects feature group/C/weight; outer fold untouched until scoring |
| Nested LR: Phase5 fixed C=1 balanced | 0.4473 | 0.5058 | 0.5160 | 0.5062 | 0.4400 | 0.4335 | 0.4367 | 0.0125 | nested outer folds; fixed Phase5 regularization |
| logistic_expression_topology | 0.4472 | 0.5059 | 0.5139 | 0.5038 | 0.4372 | 0.4286 | 0.4328 | 0.0075 | same 3 folds; validation fold also governed early stopping |
| Nested LR: expression_plus_network | 0.4526 | 0.5110 | 0.5480 | 0.5251 | 0.4706 | 0.3547 | 0.4045 | 0.0530 | nested 3-fold outer CV; inner 3-fold AP search |
| Nested LR: expression_only | 0.4351 | 0.5057 | 0.5437 | 0.5015 | 0.4368 | 0.1872 | 0.2621 | 0.0038 | nested 3-fold outer CV; inner 3-fold AP search |


The feature-only LR ablation suggests weak signal: expression-only pooled OOF AP 0.4351; network-only 0.4542; combined 0.4526. Combining expression with network did not increase AP over network-only. The nested feature-selection pipeline yielded AP 0.4476, ROC-AUC 0.5065, accuracy 0.5544, balanced accuracy 0.5109, precision 0.4634, recall 0.1872, F1 0.2667, and MCC 0.0284.

A comparable fixed Logistic reference using the same outer folds (C=1, balanced, expression+network) had AP 0.4473, ROC-AUC 0.5058, accuracy 0.5160, balanced accuracy 0.5062, precision 0.4400, recall 0.4335, F1 0.4367, and MCC 0.0125. The tuned inner-selected pipeline changes threshold metrics but does not materially improve ranking AP. This is not evidence of generalization beyond the development cohort. The fixed threshold is retained, and the reported results expose its precision/recall trade-off.

| Comparable nested Logistic pipeline | AP | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before: fixed C=1 balanced, expression+network | 0.4473 | 0.5058 | 0.5160 | 0.5062 | 0.4400 | 0.4335 | 0.4367 | 0.0125 |
| After: inner-selected features/C/class weight | 0.4476 | 0.5065 | 0.5544 | 0.5109 | 0.4634 | 0.1872 | 0.2667 | 0.0284 |
| Change | +0.0003 | +0.0007 | +0.0384 | +0.0047 | +0.0234 | -0.2463 | -0.1701 | +0.0159 |

This optimization comparison uses the same outer folds and the same 469 development genes; the selection procedure is nested for the after row. The accuracy increase comes with a large recall and F1 decrease and is far below 80%.

The Phase 5 sparse Transformer remains the selected candidate by mean fold-seed AP (0.5169) compared with the nested feature-selected Logistic fold AP mean (0.4765). This cross-protocol comparison is descriptive: Phase 5 selected epochs on the validation folds, whereas the new Logistic search used nested inner CV. The pooled AP of the old GCN is higher than the selected Transformer, and graph-only Logistic's ordinary accuracy is modestly higher than both, but its recall/F1 are poor. No single model wins all metrics. Therefore there is **no validated before/after performance gain**; the established Transformer checkpoint is preserved as the best current ranking model under the project's declared Phase 5 selection rule.

## Data, imbalance, pseudo-labels, and leakage

The only training targets remain the frozen NCG `NCG_oncogene` target: 254 positive and 333 explicit-zero labels overall; 469 are in development. Unlabeled genes were never treated as negatives. No additional local gene-label table suitable for training was found. Local reference data include NCBI crosswalk metadata, TCGA sample metadata, STRING aliases, and a restricted COSMIC reference already excluded from training. No data were downloaded and no dependencies installed. IntOGen release 2024.09 provides a public cohort-specific driver catalog, but its driver positives are not validated negatives and its methodology uses COSMIC CGC for calibration; a future use requires release freeze, gene-ID and tissue harmonization, overlap analysis, licensing/provenance review, and an independent evaluation plan before any labels are incorporated ([IntOGen downloads](https://www.intogen.org/download), [IntOGen FAQ](https://www.intogen.org/faq)). COSMIC requires registration/login for noncommercial downloads and a commercial licence for commercial use; local COSMIC content was not redistributed or used to train ([COSMIC downloads and terms](https://cancer.sanger.ac.uk/cosmic/download/cosmic)).

Phase 6's weighted self-training worsened AP slightly and materially worsened Brier/log loss, so pseudo-labeling was not repeated. Weak labels, a small development set, class uncertainty (explicit zero is not biological proof of non-driver), degree-driven visibility, transductive graph features, and selection variation remain limitations. The 80% target cannot be credibly pursued by threshold manipulation.

## Final model and outputs

- Best model before this optimization: **expanded sparse graph Transformer** by Phase 5 mean fold-seed AP; prior Phase 2 NCG-only Logistic had a higher 64.7% accuracy on a different cohort/split, which is not a comparable benchmark.
- Best model after this optimization: **expanded sparse graph Transformer**, unchanged; no new model credibly displaced it.
- 80% accuracy: **not achieved**. Highest Phase 5 average fold accuracy was 57.64% for GCN, with zero recall at the pooled 0.5 OOF threshold.
- Generalization evidence: no clean heldout or independent cohort remains. The previous holdout was exposed; only development CV results are reported for new work.
- Final checkpoint: `outputs/experiments/final_optimization_20261028_233304/checkpoints/best_model.pt`
- Final prediction scores: `outputs/experiments/final_optimization_20261028_233304/predictions/final_all_gene_predictions.csv`; unknown-only `predictions/final_18411_unknown_predictions.csv`.
- Model comparison, ablations, inner searches, logs, plots and checks: `outputs/experiments/final_optimization_20261028_233304/metrics/`, `logs/`, `plots/`.
- Checkpoint replay maximum score difference from Phase 5 export: 8.89e-08; integrity checks **9/9 passed**.

All candidate probabilities are model scores, not calibrated probabilities or confirmed biological labels.
