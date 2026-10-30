# Phase 3 model development and validation

**Final Phase 3 run:** `outputs/experiments/phase3_gcn_20261028_164054/`  
**Starting-state audit:** `reports/phase3_starting_state.md`  
**Status:** Development-only hyperparameter search and graph-context diagnostics completed. One final refit was frozen from development CV, then evaluated against the previously used Phase 2 holdout. It is not a pristine independent test estimate because Phase 2 heldout results for multiple models had already been generated and reviewed.

## Starting point and leakage audit

Phase 2 completed expression residual MLP, expression Transformer, GCN, GraphSAGE, GAT, expression+GAT, expression+GraphSAGE, gated HMGAT-ET, concatenation HMGAT-ET, and a logistic expression+topology reference. Nine neural final checkpoints and fold-best checkpoints load. The split manifest, development OOF, heldout predictions, and per-fold metrics were present. The complete baseline metrics (including fold mean/SD classification summaries at fixed 0.5, heldout metrics, parameter counts) are recorded in `phase3_starting_state.md` and `metrics/phase2_full_comparison.csv`.

Code inspection confirmed scaling and class weights are fitted on training genes inside each fold, unknown labels do not enter supervised loss/scoring, threshold selection uses development OOF probabilities, and architecture selection uses CV AP. The graph and unlabeled features are transductively shared. Phase 2 did generate and expose heldout metrics for all candidate architectures; therefore, although heldout labels were not used by the selection code, this test set is not analyst-untouched. The Phase 3 search does not load heldout labels or Phase 2 test metrics while tuning. New final test figures are reported as exploratory reuse only.

## Controlled GCN search

The existing 3 development fold assignments were reused unchanged. Four predeclared GCN configurations were evaluated over seeds 17, 42, and 2025 (36 fold-seed fits), using early stopping on each fold's validation AP (max 70 epochs, patience 12). Search criterion: mean fold AP across the 3 folds and 3 seeds. Each fold fits a fresh scaler on training genes only. Weighted loss uses the train-fold negative/positive ratio; the unweighted variant uses positive weight 1. The tuning process reads only gene indices and development target indices; heldout target values and annotated rows are not accessed until the winning configuration and final checkpoint are frozen.

| Configuration | Hidden | Layers | Dropout | LR | Weight decay | Class loss | Parameters | CV fold AP mean ± SD | Pooled seed-averaged OOF AP | Mean fit seconds |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| GCN unweighted loss | 32 | 2 | 0.30 | 0.008 | 0.001 | Unweighted | 2,401 | 0.5692 ± 0.0399 | 0.5509 | 0.96 |
| GCN deeper moderate | 32 | 3 | 0.40 | 0.003 | 0.005 | Balanced | 3,521 | 0.5672 ± 0.0381 | 0.5639 | 1.32 |
| GCN wide regularized | 64 | 2 | 0.50 | 0.003 | 0.010 | Balanced | 6,785 | 0.5637 ± 0.0386 | 0.5405 | 1.16 |
| GCN baseline reproduction | 32 | 2 | 0.30 | 0.008 | 0.001 | Balanced | 2,401 | 0.5580 ± 0.0376 | 0.5575 | 0.82 |

The unweighted model was frozen because it had the highest predefined mean-fold AP. Its advantage over the deeper configuration is only 0.0021 AP, much smaller than fold/seed variability; pooled seed-averaged OOF AP instead favors the deeper model. This metric sensitivity means there is no robust evidence that the selected setting improves on the original GCN. The single seed-42 baseline reproduction exactly matches the saved Phase 2 fold APs, confirming split and baseline implementation alignment.

## Graph and degree diagnostics

A development-only identity-adjacency ablation retained the same centrality input features but removed messages from other nodes, using the selected configuration on the same folds and seeds. Full-graph mean-fold AP was 0.5692 ± 0.0399 versus 0.5295 ± 0.0168 for identity adjacency; pooled seed-averaged OOF AP was 0.5509 versus 0.5055. This suggests neighbor context contributes ranking signal under the transductive protocol, but does not establish causality or generalization to new components.

The model's development OOF score and degree Spearman correlation is 0.333. A degree-only ranking has AP 0.5062 and ROC-AUC 0.5455 on development labels. The highest-degree quartile has AP 0.723 (61/117 positives), while the lowest-degree quartile has AP 0.444 (47/118 positives). Thus performance is partly degree-associated; the model is not reducible to degree alone, but its high-degree performance advantage should be investigated in future work. Full strata are in `metrics/dev_degree_strata.csv`.

Component holdout remains infeasible: all components outside the giant component contain only 7 positive and 8 explicit-zero labels. No artificial component split was made. Alternative STRING edge thresholds were skipped to keep this bounded search on the frozen score-700 graph. Independent COSMIC evaluation was skipped: a future protocol must predefine target-role eligibility, multi-role handling, overlap exclusions, licensing, and analysis before scores are examined. COSMIC remains outside training and features.

## Frozen model and heldout evaluation

Final selection was development-only. Frozen configuration: `gcn_unweighted_loss`, hidden 32, 2 graph-convolution layers, dropout 0.30, AdamW LR 0.008, weight decay 0.001, no class reweighting, 16 refit epochs (median best epoch), seed 1041, 2,401 parameters. Final inference explicitly uses `eval()` mode. Decision threshold 0.475 was selected from pooled development OOF predictions by MCC.

Heldout cohort: 118 genes, 51 positive and 67 explicit-zero. Recomputed metrics: AP 0.6126, ROC-AUC 0.6313, precision 0.5854, recall 0.4706, F1 0.5217, balanced accuracy 0.6084, MCC 0.2256 (TN 50, FP 17, FN 27, TP 24). Compared with Phase 2 GCN AP 0.6030, the apparent delta is +0.0097; selection did not use this value. Because this same holdout was previously scored and viewed, this difference is not a confirmatory test of improvement. Development CV AP (0.5692 ± 0.0399 across fold-seed fits) is effectively unchanged from Phase 2's single-seed GCN 0.5693 ± 0.0517.

## Deliverables and traceability

- Final checkpoint: `outputs/experiments/phase3_gcn_20261028_164054/checkpoints/gcn_unweighted_loss_final.pt`.
- Frozen search settings/seeds: `configs/search.json`; selected configuration and metrics: `metrics/phase3_summary.json`.
- All 36 tuning fold-seed results and logs: `metrics/dev_search_folds.csv`, `logs/`; summary and seed-level stability: `metrics/dev_search_summary.csv`, `metrics/dev_search_seed_summary.csv`.
- Graph diagnostic: `metrics/dev_graph_dependence_folds.csv`, `metrics/dev_graph_dependence_summary.csv`, `metrics/degree_only_dev_baseline.json`, `metrics/dev_degree_strata.csv`.
- Registry includes completed, failed/superseded artifact attempts, and skipped analyses: `metrics/experiment_registry.csv` and `.json`.
- Stable-ID scores for all 3,304 genes: `predictions/all_gene_scores_with_ppi_context.csv`; ranked 2,717 unknown genes: `predictions/ranked_unknown_candidates_with_ppi_context.csv`. These retain known-positive / explicit-zero / unknown status and include Entrez, symbol, Ensembl ID, score, expression fields, and PPI degree/weighted strength. Scores are hypotheses, not confirmed cancer drivers.
- Four development-only training curve plots and graph message ablation plot: `plots/`; Phase 2 model comparison and heldout plot remain in the Phase 2 directory.
- Automated Phase 3 verification: 27/27 checks passed in `metrics/final_integrity_checks.json`. The verifier loaded all nine Phase 2 final neural checkpoints, all 36 search checkpoints, and the final Phase 3 checkpoint; regenerated all-gene predictions from the final checkpoint; checked IDs, unknown masks, split disjointness, and recomputed final heldout metrics.

The first Phase 3 attempt (`phase3_gcn_20261028_162533`) is preserved but its final predictions are invalidated: the refit model remained in training mode during inference, activating dropout. Its development search runs are reproducible, but do not use its final test scores. The corrected run is `phase3_gcn_20261028_164054`, with deterministic checkpoint-reproducing inference. Two graph-diagnostic export attempts failed at plot creation and were rerun successfully; all failed/superseded attempts are listed in the registry.

## Reproduction

See the root `README.md` for exact commands. All outputs are timestamped; Phase 2 artifacts remain unchanged. No new data was downloaded and no additional dependency was installed in Phase 3. CPU runtime for the corrected search/refit was about 95 seconds; final resident memory at completion was about 0.415 GiB (peak was not instrumented). PyTorch sparse invariant and NumPy read-only-array warnings appeared; all checkpoint/prediction recomputation and integrity checks passed. Treat this as exploratory model development; next scientifically justified work is external validation with a fresh data source or a newly reserved cohort, plus label provenance review and degree/graph sensitivity analysis.

