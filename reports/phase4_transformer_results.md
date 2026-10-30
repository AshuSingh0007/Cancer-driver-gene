# Phase 4: sparse Transformer and expression–graph comparison

**Final audited run:** `outputs/experiments/phase4_transformer_20261028_172513/`  
**Status:** Completed on CPU; development-only comparison and final development refit. No holdout labels were read during search, no holdout evaluation was repeated, and no model was selected from holdout performance.

## Task, inputs, and protocol

The prediction task remains gene-level prediction of the pan-cancer NCG oncogene annotation from three TCGA-BRCA cohort summary features and STRING network context. It is not patient-level prediction or a claim of BRCA-specific driver status. The frozen candidate universe has 3,304 genes: 254 positive, 333 explicitly annotated zero, and 2,717 unknown. Unknown genes were never used in supervised loss or scored as negatives. COSMIC remains excluded from training and features; reference use and source restrictions are covered in the Phase 2/6 reports.

Inputs were reused from `data/processed/integrated_model_inputs_20261028_155809/`: three expression summaries, three graph topology summaries for the graph branches, and the frozen score-700 STRING graph with 24,606 undirected edges (49,212 directed edges). The 3,304 genes were split into 469 development labeled genes (203 positives, 266 zeros) and 118 previously exposed holdout genes (51 positives, 67 zeros); the rest are unknown. Three fixed Phase 2 development folds were reused unchanged. All feature standardizers were fitted on each fold's training genes only. The graph and node covariates are shared transductively; labels are masked to training indices for loss and validation indices for early stopping.

At run start, about 0.81 GiB RAM was available on a 7.68 GiB CPU-only machine; PyTorch 2.14.1+cpu was used, with no CUDA or graph-learning package. The whole Phase 4 training/refit process took 189 seconds. Each new model used 32 hidden dimensions, two sparse attention blocks, four heads, dropout 0.25, AdamW learning rate 0.003, weight decay 0.005, gradient clipping at 2, positive weighting from each training fold, maximum 40 epochs, and patience 8. It ran seeds 17, 42, and 2025 on each of three folds (18 fold/seed fits). Mean individual fit time was 10.28 seconds for the graph Transformer and 8.99 seconds for the hybrid.

## Models and results

Primary selection metric is average precision (AP), reported as mean ± sample SD across fold/seed validation runs. Pooled OOF AP is also shown because it can rank models differently. Historical Phase 2 baselines below used the same fixed three folds but one seed; Phase 3 GCN references used three seeds. These results are useful references, but differences in seed count mean the variances are not directly interchangeable.

| Model | Development fold AP mean ± SD | Pooled seed-averaged OOF AP | ROC-AUC statistic* | Parameters |
|---|---:|---:|---:|---:|
| Phase 4 sparse graph Transformer | **0.5898 ± 0.0518** | **0.5723** | 0.6261 | 17,913 |
| Phase 2 GCN | 0.5693 ± 0.0517 | — | 0.6170 | — |
| Phase 3 selected unweighted GCN | 0.5692 ± 0.0399 | 0.5509 | 0.6151 | 2,401 |
| Phase 2 GraphSAGE | 0.5687 ± 0.0505 | — | 0.6306 | — |
| Phase 3 deeper GCN | 0.5672 ± 0.0381 | 0.5639 | 0.6269 | 3,521 |
| Phase 3 wide GCN | 0.5637 ± 0.0386 | 0.5405 | 0.6084 | 6,785 |
| Phase 3 balanced baseline GCN | 0.5580 ± 0.0376 | 0.5575 | 0.6097 | 2,401 |
| Phase 4 expression + sparse graph Transformer gated fusion | 0.5555 ± 0.0373 | 0.5489 | 0.6168 | 22,218 |
| Phase 2 expression + GAT concatenation | 0.5527 ± 0.0241 | — | 0.6203 | — |
| Phase 2 GAT | 0.5320 ± 0.0095 | — | 0.6201 | — |
| Phase 2 expression residual MLP | 0.5221 ± 0.0324 | — | 0.5792 | — |
| Phase 2 expression Transformer | 0.5244 ± 0.0157 | — | 0.6244 | — |
| Phase 2 logistic expression/topology | 0.4912 ± 0.0074 | — | 0.5351 | — |

\* Phase 2 values are mean fold ROC-AUC across its three folds; Phase 3 and Phase 4 values are pooled/seed-averaged OOF ROC-AUC. The AP selection statistic is mean fold AP in all rows.

The sparse graph Transformer had the highest mean fold AP in this comparison. Its advantage over the Phase 3 GCN reference is about 0.0205 AP by fold mean, while the SD across its 9 fold/seed runs is 0.0518. Pooled seed-averaged OOF AP is 0.5723 versus 0.5509 for the selected GCN, a difference of 0.0214. This is promising exploratory evidence, not strong proof of a stable improvement: fold/seed variation is substantial, the positive set is small, and the graph model has more parameters. Phase 4 selected this model by the declared mean development AP rule and refit it for the median selected epoch (17) on all development labels.

The graph Transformer is sparse dot-product multi-head attention restricted to STRING edges, with edge confidence incorporated as a log-weight bias, residual updates, normalization, and a feed-forward block. It receives topology summaries (log degree, log weighted strength, weighted clustering) plus graph neighborhoods. It does not use dense all-pairs attention. The hybrid uses a residual MLP over the three expression summaries and the same sparse graph Transformer over expression plus topology, combined by a learned gate. Fusion underperformed both the graph-only Transformer and selected GCN in mean AP; there is no evidence here that expression improves the new graph model.

The existing expression Transformer was only a reference check: the three expression columns are mean, standard deviation, and fraction of cases with positive TPM. They are not a natural ordered sequence, so Phase 4 did not create arbitrary tokens or use cross-attention. The Phase 2 typed-statistic Transformer AP (0.5244) was close to its MLP reference (0.5221) and below graph methods. A more expressive Transformer is not justified by these three summary values alone.

## Threshold metrics and graph dependence

Threshold-independent metrics are the primary comparison above. For a descriptive classification view, Phase 4 selected threshold 0.625 by maximizing MCC over the selected model's pooled development OOF predictions. At that threshold the OOF results are precision 0.750, recall 0.177, F1 0.287, balanced accuracy 0.566, MCC 0.216 (TN 254, FP 12, FN 167, TP 36; 203 positives and 266 explicit zeros). Because the same OOF labels informed threshold choice, these threshold metrics are selection-biased; they are not independent estimates. The default 0.5 per-fold confusion metrics are also recorded in `metrics/dev_transformer_folds.csv`.

The new model's development OOF score has Spearman correlation 0.234 with PPI degree. Degree quartile AP ranges from 0.378 in the lowest-degree quartile (50/127 positives) to 0.735 in the highest (61/117 positives), with middle quartiles at 0.503 and 0.491. This resembles the prior Phase 3 degree sensitivity and warns that network visibility is uneven. These are stratified diagnostics, not proof that degree causes model performance. A component-heldout assessment remains statistically infeasible under prior counts (only 7 positive and 8 zero labels outside the giant component).

## Leakage and evaluation limitations

- The target array is memory-mapped; the experiment indexes target values only for development training and validation indices before freezing the selected configuration/checkpoint. It does not index heldout labels during selection. Annotation columns are loaded only after checkpoint freeze to annotate exported scores.
- Fold-specific scalers are fitted on each training fold. Unknown labels do not enter the supervised loss. Expression and graph features are visible for unlabeled and heldout nodes under the existing transductive graph protocol, but heldout labels are masked.
- COSMIC is not a training target, feature source, tuning source, or validation claim.
- The old holdout was scored and reviewed during Phase 2 and again in Phase 3. Phase 4 does not repeat the evaluation or claim pristine external validation. No independent cohort is available in the frozen project.
- The model was selected among Phase 4 additions; the existing GCN remains the essential reference. Given small sample size, overlapping variation, and degree sensitivity, treat the apparent AP gain as a hypothesis for further validation.
- Calibration was not attempted: 469 development labels and a small number of positives do not support a trustworthy separate calibration set without sacrificing already limited model-selection data. No extensive hyperparameter search was run.

## Frozen outputs and checks

The development-selected sparse graph Transformer checkpoint is `outputs/experiments/phase4_transformer_20261028_172513/checkpoints/sparse_graph_transformer_final.pt`. Its all-gene scores are in `predictions/all_gene_scores.csv`; `predictions/ranked_unknown_candidates.csv` contains only unknown-label candidates and keeps explicit positive/zero/unknown status in the all-gene export. These are ranking scores, not confirmed driver probabilities.

The 18 fold/seed checkpoints, logs, configs, fold metrics, model comparison, experiment registry, OOF predictions, plots, degree diagnostics, and JSON summary are all saved in the run directory. Checkpoint replay reproduced all 3,304 exported scores with max absolute difference 1.34e-7. Automated Phase 4 integrity checks passed 12/12, including stable IDs, finite scores, label counts, unknown export count, split separation, OOF coverage, checkpoint replay, and masked target handling. An earlier Phase 4 execution (`phase4_transformer_20261028_171933`) is preserved but superseded after an auditability correction that deferred loading annotation columns until after selection; its metrics matched the final rerun exactly.

## Reproduction

From PowerShell in `D:\Quantum`:

```powershell
.\.venv\Scripts\python.exe src\run_phase4_sparse_transformer.py
$run = "outputs\experiments\phase4_transformer_YYYYMMDD_HHMMSS"
.\.venv\Scripts\python.exe src\finalize_phase4_transformer.py $run
.\.venv\Scripts\python.exe src\analyze_phase4_degree.py $run
.\.venv\Scripts\python.exe src\check_phase4_integrity.py $run
```

No dependencies were installed, no data were downloaded, and raw inputs or prior experiment outputs were not overwritten.

## Recommended next step

Do not run another broad architecture sweep on this same small label set. First obtain a genuinely independent, predeclared cohort or a new label source with harmonized identifier and role rules, then evaluate the frozen sparse Transformer and GCN side by side. If external data cannot be obtained, focus on predeclared label-provenance review and graph/degree sensitivity rather than treating this AP difference as a confirmed gain.
