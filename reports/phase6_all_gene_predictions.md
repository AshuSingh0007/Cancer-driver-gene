# Phase 6: complete gene predictions

**Run:** `outputs/experiments/phase6_all_gene_predictions_20261028_222948`  
**Input:** frozen Phase 5 expanded universe and saved checkpoints/predictions. No previous artifact was overwritten.

## Results

- Genes processed: **18,998**; previously unknown genes receiving binary predictions: **18,411**.
- Unknown predicted positive: **5,193**; predicted negative: **13,218**.
- Verified labels preserved: **254 positive**, **333 explicit zero**. All **18,411** unknown rows are machine-generated predictions, not verified labels.
- Selected model: **expanded_sparse_graph_transformer**, selected by mean development-fold average precision, not confidence.

### Development out-of-fold model comparison

| Model | AP | ROC-AUC | Precision | Recall | F1 | Balanced accuracy | MCC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_expression_topology | 0.4472 | 0.5059 | 0.4372 | 0.4286 | 0.4328 | 0.5038 | 0.0075 | 0.2525 | 0.6985 |
| expanded_gcn | 0.4999 | 0.5905 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.2429 | 0.6789 |
| expanded_sparse_graph_transformer | 0.4912 | 0.5336 | 0.4562 | 0.3596 | 0.4022 | 0.5163 | 0.0340 | 0.2483 | 0.6897 |


Metrics use the existing 469-gene development OOF set and threshold 0.5 for classification metrics. Brier and log loss are reported as calibration diagnostics; these scores were not calibrated.

- Weighted self-training assigned **16006 pseudo-label observations across the three folds** at the prespecified 0.60/0.40 thresholds (genes may recur across fold fits), weight 0.1. Its pooled OOF AP was 0.4456 versus initial logistic 0.4472; ROC-AUC was 0.5054 versus 0.5059. It was **rejected** on the primary AP criterion. The final predictions continue to use the frozen selected supervised model; pseudo-label retraining was not used for final labels.
- Uncertainty categories among unknowns: {'LOW_OR_UNCERTAIN': 18406, 'MODERATE_MODEL_CONFIDENCE': 5}. These are heuristic confidence bins, not empirical correctness estimates.
- Heldout evaluation was not repeated. Prior heldout labels had been exposed in earlier project work, so they are not an untouched estimate.

## Audit of prior pseudo-label failure

Phase 5 found no model with at least 90% correctness on ten or more development OOF genes at a candidate class threshold, no stable predictions at the predeclared 0.95/0.05 extremes, and no robust cross-family extreme agreement. The models have modest OOF discrimination and disagree in ranking (GCN has the best pooled OOF ROC-AUC while the sparse graph Transformer has the best mean-fold AP). Thus confidence-based pseudo-labels were not justified. This phase separately tested one-round weighted Logistic Regression self-training with fold-specific predictions and 0.1 pseudo weights; its AP decreased slightly and Brier/log loss worsened, so this trial was rejected.

## Prediction and uncertainty policy

All unknowns receive a deterministic binary prediction using selected-model score >= 0.5. The score is not a calibrated biological probability. Agreement is the number of the three model families whose fold/seed mean score is >= 0.5. Confidence categories are heuristic, using distance from 0.5 and selected-model fold/seed spread; they are not empirical correctness probabilities. Predicted negatives are not claims that a gene cannot drive cancer. No genes were assigned invented features or external support.

## Deliverables

- `predictions/all_18998_gene_predictions.csv` and `predictions/unknown_18411_predictions.csv`
- `predictions/ranked_predicted_driver_candidates.csv`, `predictions/ranked_predicted_non_driver_candidates.csv`
- `metrics/evaluation_metrics.json`, `metrics/pseudo_label_comparison.json`, `metrics/pseudo_label_fold_history.csv`, `metrics/unknown_confidence_analysis.csv`
- `checkpoints/final_selected.pt`, `configs/phase6_protocol.json`, `metrics/integrity_checks.json`

The table keeps `original_label` separate from `predicted_binary_label`. Unknowns remain labelled `PREDICTED_POSITIVE` or `PREDICTED_NEGATIVE`; `is_pseudo_label` is false because these are final classifications, not retraining labels. Integrity checks: **9/9 passed**.
