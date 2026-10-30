# Cancer driver-gene project: reproduction guide

## Current task

The model predicts the pan-cancer NCG `NCG_oncogene` source label at the gene level using TCGA-BRCA cohort-level expression summaries and STRING v12.5 graph context. It is not patient-level classification or BRCA-specific driver discovery. The 2,717 unknown-label genes are unlabeled and never used as negatives. COSMIC is excluded from features and training labels. Candidate scores are exploratory hypotheses, not confirmed drivers.

The strict Phase 3 run is `outputs/experiments/phase3_gcn_20261028_164054/`. Its report is `reports/phase3_model_development.md`; the Phase 2 starting-state audit is `reports/phase3_starting_state.md`.

## Environment

The validated environment is the project virtual environment on CPU. It includes Python, pandas, NumPy, scikit-learn, NetworkX, matplotlib, psutil, and CPU PyTorch. No quantum-computing packages or methods are used. Do not install packages or download data as part of routine reproduction.

## Reproduce Phase 3

From `D:\Quantum` in PowerShell:

```powershell
.\.venv\Scripts\python.exe src\run_phase3_gcn_search.py
```

The script reads frozen Phase 2 inputs and split manifest, reuses the existing three development fold assignments, and creates a new timestamped directory under `outputs/experiments/`. It evaluates four GCN configurations across three seeds and three development folds, chooses by mean development fold AP, refits on development labels, and evaluates the existing holdout once after checkpoint freeze. During search it reads only gene indices and development labels; the annotated feature table and heldout labels are loaded only after freeze. The existing holdout was already scored and reviewed in Phase 2, so the result is exploratory rather than an untouched test estimate.

After the run, set `$run` to the printed output directory name, then run:

```powershell
$run = "phase3_gcn_YYYYMMDD_HHMMSS"
.\.venv\Scripts\python.exe src\analyze_phase3_graph_dependence.py "outputs\experiments\$run"
.\.venv\Scripts\python.exe src\add_phase3_prediction_context.py "outputs\experiments\$run"
.\.venv\Scripts\python.exe src\plot_phase3_training_curves.py "outputs\experiments\$run"
.\.venv\Scripts\python.exe src\summarize_phase3_start.py "outputs\experiments\$run"
.\.venv\Scripts\python.exe src\summarize_degree_baseline.py "outputs\experiments\$run"
.\.venv\Scripts\python.exe src\build_phase3_registry.py "outputs\experiments\$run"
.\.venv\Scripts\python.exe src\verify_phase3_artifacts.py "outputs\experiments\$run"
```

`reports/phase3_starting_state.md` records the Phase 2 audit. `reports/phase3_model_development.md` documents the final run, full metrics, output paths, and limitations. The first Phase 3 inference attempt was invalidated due to dropout remaining active and is listed as failed/superseded in the registry; do not use its test predictions.

## Phase 4: sparse graph Transformer comparison

Phase 4 results are documented in `reports/phase4_transformer_results.md`. The final audited run is `outputs/experiments/phase4_transformer_20261028_172513/`. It compares a sparse edge-aware graph Transformer and an expression-MLP plus graph-Transformer gated hybrid on the same three development folds and three seeds. It reuses the validated Phase 2 inputs, fits scalers inside folds, does not index held-out targets during selection, and does not rerun the previously exposed holdout evaluation. The selected graph Transformer is exploratory; the small label set, fold variation, and degree dependence limit conclusions.

From PowerShell in `D:\Quantum`, start a fresh timestamped Phase 4 run with:

```powershell
.\.venv\Scripts\python.exe src\run_phase4_sparse_transformer.py
$run = "outputs\experiments\phase4_transformer_YYYYMMDD_HHMMSS"
.\.venv\Scripts\python.exe src\finalize_phase4_transformer.py $run
.\.venv\Scripts\python.exe src\analyze_phase4_degree.py $run
.\.venv\Scripts\python.exe src\check_phase4_integrity.py $run
```

The expression inputs are three unordered cohort summary statistics, so Phase 4 uses an MLP for that branch and does not impose an arbitrary sequence or cross-attention over them. No new dependencies or datasets are required.

## Phase 5: expanded gene universe

Phase 5 results are documented in `reports/phase5_expansion_results.md`. The final expanded set is `outputs/experiments/phase5_expansion_20261028_181232/`: 18,998 expression-featured, exact-ID-mapped STRING genes, including the former 3,304-gene cohort. It reuses only local TCGA, NCBI crosswalk, NCG, and STRING data. The verified label count did not increase; 18,411 genes remain unknown. Logistic regression, GCN, and sparse graph Transformer were retrained on the unchanged Phase 2 development folds. The expanded runs performed below prior smaller-universe references. No development-supported pseudo-label thresholds were found, so no pseudo labels were created or used. The prior heldout set was not re-evaluated.

For a new run, use the scripts below from PowerShell. The trainer refuses to reuse a directory that already contains fold checkpoints.

```powershell
.\.venv\Scripts\python.exe src\build_phase5_expanded_dataset.py
$run = "outputs\experiments\phase5_expansion_YYYYMMDD_HHMMSS" # use the printed run path
.\.venv\Scripts\python.exe src\complete_phase5_dataset_audit.py $run
.\.venv\Scripts\python.exe src\train_phase5_expanded_models.py $run
.\.venv\Scripts\python.exe src\audit_phase5_pseudolabels.py $run
.\.venv\Scripts\python.exe src\finalize_phase5_expansion.py $run $run
.\.venv\Scripts\python.exe src\export_phase5_supporting_audits.py $run $run
.\.venv\Scripts\python.exe src\verify_phase5_graph_integrity.py $run
```

Scores remain exploratory rankings, not confirmations of cancer-driver status. The independent COSMIC reference was not used for training or selection.
