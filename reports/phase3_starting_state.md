# Phase 3 starting-state audit and Phase 2 benchmark

**Audit source:** saved Phase 2 code, config, metrics, split, prediction files, and checkpoints under `outputs/experiments/integrated_20261028_161519/`; task/input specs under `reports/` and `data/processed/integrated_model_inputs_20261028_155809/`.

## Completed models and artifacts

Completed models: expression residual MLP, expression Transformer, GCN, GraphSAGE, GAT, expression+GAT, expression+GraphSAGE, gated HMGAT-ET, concatenation HMGAT-ET, and logistic expression+topology reference. Nine neural models have final and fold-best `.pt` checkpoints; logistic has no neural checkpoint. The saved stratified split is `predictions/split_manifest.csv`; fixed development folds and predictions are in each model's `*_development_oof.csv`, with fold-level metrics in `metrics/*_cv_folds.csv`. Heldout predictions are in `predictions/*_heldout_test.csv`.

The Phase 2 config records seed 42, three development folds, 20% heldout fraction, max 70 epochs, patience 12, AdamW learning rate 0.008, weight decay 0.001, hidden setting 64 where applicable, dropout 0.30, and training-fold positive class weighting. The implementation uses model-specific graph/expression encoder sizes; parameter counts below were calculated from the saved architecture definitions. Per-model training time was not recorded. The full run took 117.97 seconds and observed peak process RSS was 0.503 GiB on CPU.

## Full comparison

Development AP and ROC-AUC are mean ± sample SD over three fixed folds. Development precision/recall/F1 are reported at a fixed 0.5 threshold per fold (mean ± sample SD), avoiding threshold tuning on each validation fold. Heldout classification metrics use each model's threshold selected from development OOF predictions. The selected original model was GCN, selected by development CV AP.

| Model | Params | CV AP mean ± SD | CV ROC-AUC mean ± SD | CV P / R / F1 at 0.5 (mean ± SD) | Test AP | Test ROC-AUC | Test P / R / F1 | Test BAcc | Test MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_gcn | 2401 | 0.569 ± 0.052 | 0.617 ± 0.028 | 0.512 ± 0.020 / 0.675 ± 0.047 / 0.581 ± 0.013 | 0.603 | 0.618 | 0.552 / 0.314 / 0.546 | 0.560 | 0.138 |
| graph_sage | 3521 | 0.569 ± 0.051 | 0.631 ± 0.022 | 0.545 ± 0.067 / 0.518 ± 0.220 / 0.502 ± 0.113 | 0.566 | 0.625 | 0.550 / 0.647 / 0.617 | 0.622 | 0.242 |
| expression_gat_concat | 34633 | 0.553 ± 0.024 | 0.620 ± 0.033 | 0.452 ± 0.022 / 0.955 ± 0.078 / 0.612 ± 0.007 | 0.528 | 0.607 | 0.466 / 0.941 / 0.458 | 0.560 | 0.179 |
| graph_gat | 2537 | 0.532 ± 0.009 | 0.620 ± 0.033 | 0.469 ± 0.039 / 0.681 ± 0.323 / 0.521 ± 0.118 | 0.494 | 0.594 | 0.536 / 0.725 / 0.610 | 0.624 | 0.249 |
| expression_sage_concat | 42753 | 0.527 ± 0.047 | 0.600 ± 0.040 | 0.329 ± 0.286 / 0.461 ± 0.409 / 0.381 ± 0.330 | 0.491 | 0.497 | 0.000 / 0.000 / 0.362 | 0.500 | 0.000 |
| expression_transformer | 19329 | 0.524 ± 0.016 | 0.624 ± 0.001 | 0.360 ± 0.314 / 0.422 ± 0.372 / 0.385 ± 0.334 | 0.528 | 0.583 | 0.453 / 0.667 / 0.506 | 0.527 | 0.056 |
| expression_residual | 8769 | 0.522 ± 0.032 | 0.579 ± 0.012 | 0.541 ± 0.057 / 0.434 ± 0.087 / 0.474 ± 0.036 | 0.598 | 0.664 | 0.458 / 0.863 / 0.468 | 0.543 | 0.110 |
| hmgat_et_gated | 49353 | 0.504 ± 0.017 | 0.583 ± 0.021 | 0.290 ± 0.251 / 0.667 ± 0.577 / 0.404 ± 0.350 | 0.410 | 0.439 | 0.000 / 0.000 / 0.362 | 0.500 | 0.000 |
| logistic_expression_topology | 7 | 0.491 ± 0.007 | 0.535 ± 0.034 | 0.461 ± 0.036 / 0.508 ± 0.041 / 0.483 ± 0.033 | 0.599 | 0.649 | 0.611 / 0.216 / 0.519 | 0.556 | 0.153 |
| hmgat_et_concat | 45193 | 0.489 ± 0.033 | 0.571 ± 0.071 | 0.293 ± 0.254 / 0.397 ± 0.531 / 0.290 ± 0.301 | 0.436 | 0.459 | 0.500 / 0.020 / 0.380 | 0.502 | 0.018 |

Full precision CSV/JSON is in this Phase 3 directory under `metrics/phase2_full_comparison.*`.

## Holdout exposure and leakage audit

Inspection of `src/train_integrated_models.py` confirms: (1) stratified 80/20 gene-level split is fixed with seed 42; (2) CV uses development labels only; (3) `StandardScaler` and positive class weight are fit from training nodes inside each fold; (4) unknown labels remain masked; (5) the architecture ranking and selected model use development CV AP; and (6) threshold selection uses development out-of-fold probabilities. The graph and unlabeled node features are shared across all genes (transductive setting), but heldout labels are not used in gradients or CV selection.

Important qualification: Phase 2 generated heldout predictions/metrics for every candidate model, and those metrics were subsequently viewed. The heldout labels did not select the Phase 2 winner in code, and the Phase 3 tuning code does not load prior test metrics or heldout labels during search. However, this holdout is not “untouched” from a project analyst-exposure perspective. New Phase 3 results are therefore exploratory reuse of a previously exposed holdout; independent external data or a newly reserved cohort is required for an uncontaminated final estimate.

## Method limitations confirmed from files

- Prediction unit is gene; target is pan-cancer NCG `NCG_oncogene`, with TCGA-BRCA cohort-summary expression context, not BRCA-specific driver discovery.
- 587 supervised labels (254 positive, 333 explicit zero) versus 2,717 masked unknowns. The explicit zero is source-defined, not proof of biological absence.
- Only three aggregate expression features per gene; patient-level variability is not available to this model.
- Transductive graph evaluation may benefit from shared graph neighborhoods; no disconnected-component generalization result is supportable because all outside-largest-component labels total 7 positive and 8 explicit zero.
- HMGAT-ET does not lead the saved comparison. Its Transformer operates on only three typed summary statistics; it is not justified as a preferred model.
- Prior model-specific runtime was not logged. COSMIC remains excluded; its differing roles, overlap, and redistribution terms require a predeclared separate validation design.

## Phase 3 experiment scope

No dataset rebuild is needed. Development tuning can reuse the frozen model input, split manifest, and folds. Meaningful low-cost work includes GCN width/depth/dropout/optimizer/class-weight/seed variation; graph-message versus identity-adjacency ablation; degree-stratified diagnostics; and expression-only/hybrid comparisons already available. Alternative edge thresholds are skipped in this bounded run to avoid rebuilding graph variants and confounding the predeclared score-700 analysis. Component-heldout and independent COSMIC evaluation remain skipped for the support/label-eligibility reasons documented in the Phase 3 results report.
