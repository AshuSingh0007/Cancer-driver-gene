"""Create starting-state audit and full Phase 2 comparison from saved outputs."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score,matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
from graph_models import NodeClassifier
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/experiments/integrated_20261028_161519"
OUT=Path(sys.argv[1]).resolve()
cmp=pd.read_csv(BASE/"metrics/model_comparison.csv")
inp=ROOT/"data/processed/integrated_model_inputs_20261028_155809"
target=np.load(inp/"target_values.npy",allow_pickle=False).astype(int)
split=pd.read_csv(BASE/"predictions/split_manifest.csv",usecols=["gene_index","partition"])
dev=split.loc[split.partition.eq("development"),"gene_index"].to_numpy(int)
fold_map={}
for f,(_,va) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=43).split(dev,target[dev]),start=1):
    fold_map.update({int(dev[i]):f for i in va})
rows=[]
for _,r in cmp.iterrows():
    name=r.model
    oof=pd.read_csv(BASE/f"predictions/{name}_development_oof.csv")
    if "fold" not in oof: oof["fold"]=oof.gene_index.map(fold_map)
    cls=[]
    for fold,part in oof.groupby("fold"):
        y=part.target.astype(int).to_numpy(); pred=(part.oof_probability.to_numpy()>=.5).astype(int)
        cls.append({"fold":int(fold),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),
          "f1":f1_score(y,pred,zero_division=0),"balanced_accuracy":balanced_accuracy_score(y,pred),"mcc":matthews_corrcoef(y,pred)})
    entry={"model":name,"cv_ap_mean":r.cv_average_precision_mean,"cv_ap_sd":r.cv_average_precision_sd,
        "cv_roc_auc_mean":r.cv_roc_auc_mean,"cv_roc_auc_sd":r.cv_roc_auc_sd}
    for met in ["precision","recall","f1","balanced_accuracy","mcc"]:
        entry[f"cv_{met}_mean_fixed_0_5"]=float(np.mean([x[met] for x in cls]))
        entry[f"cv_{met}_sd_fixed_0_5"]=float(np.std([x[met] for x in cls],ddof=1))
    for src,dst in [("test_average_precision","test_ap"),("test_roc_auc","test_roc_auc"),("test_precision","test_precision"),("test_recall","test_recall"),("test_macro_f1","test_f1"),("test_balanced_accuracy","test_balanced_accuracy"),("test_mcc","test_mcc"),("test_threshold","test_threshold")]:entry[dst]=float(r[src])
    if name=="logistic_expression_topology":
        entry["parameter_count"]=7;entry["parameters_note"]="6 coefficients plus intercept; sklearn model; no neural checkpoint"
    else:
        model=NodeClassifier(name)
        entry["parameter_count"]=sum(p.numel() for p in model.parameters())
        entry["parameters_note"]="instantiated from the saved Phase 2 architecture"
    entry["training_time_seconds_per_model"]="not recorded; total run 117.97 s"
    entry["heldout_n"]=int(r.test_n)
    rows.append(entry)
df=pd.DataFrame(rows).sort_values("cv_ap_mean",ascending=False)
df.to_csv(OUT/"metrics/phase2_full_comparison.csv",index=False)
summary=json.loads((BASE/"metrics/experiment_summary.json").read_text())
(OUT/"metrics/phase2_full_comparison.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

def f(v): return f"{v:.3f}"
tbl=["| Model | Params | CV AP mean ± SD | CV ROC-AUC mean ± SD | CV P / R / F1 at 0.5 (mean ± SD) | Test AP | Test ROC-AUC | Test P / R / F1 | Test BAcc | Test MCC |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for _,r in df.iterrows():
    cvpr=" / ".join(f"{f(r['cv_'+m+'_mean_fixed_0_5'])} ± {f(r['cv_'+m+'_sd_fixed_0_5'])}" for m in ["precision","recall","f1"])
    tpr=" / ".join(f(r[k]) for k in ["test_precision","test_recall","test_f1"])
    tbl.append(f"| {r.model} | {int(r.parameter_count)} | {f(r.cv_ap_mean)} ± {f(r.cv_ap_sd)} | {f(r.cv_roc_auc_mean)} ± {f(r.cv_roc_auc_sd)} | {cvpr} | {f(r.test_ap)} | {f(r.test_roc_auc)} | {tpr} | {f(r.test_balanced_accuracy)} | {f(r.test_mcc)} |")
text=f'''# Phase 3 starting-state audit and Phase 2 benchmark

**Audit source:** saved Phase 2 code, config, metrics, split, prediction files, and checkpoints under `outputs/experiments/integrated_20261028_161519/`; task/input specs under `reports/` and `data/processed/integrated_model_inputs_20261028_155809/`.

## Completed models and artifacts

Completed models: expression residual MLP, expression Transformer, GCN, GraphSAGE, GAT, expression+GAT, expression+GraphSAGE, gated HMGAT-ET, concatenation HMGAT-ET, and logistic expression+topology reference. Nine neural models have final and fold-best `.pt` checkpoints; logistic has no neural checkpoint. The saved stratified split is `predictions/split_manifest.csv`; fixed development folds and predictions are in each model's `*_development_oof.csv`, with fold-level metrics in `metrics/*_cv_folds.csv`. Heldout predictions are in `predictions/*_heldout_test.csv`.

The Phase 2 config records seed 42, three development folds, 20% heldout fraction, max 70 epochs, patience 12, AdamW learning rate 0.008, weight decay 0.001, hidden setting 64 where applicable, dropout 0.30, and training-fold positive class weighting. The implementation uses model-specific graph/expression encoder sizes; parameter counts below were calculated from the saved architecture definitions. Per-model training time was not recorded. The full run took 117.97 seconds and observed peak process RSS was 0.503 GiB on CPU.

## Full comparison

Development AP and ROC-AUC are mean ± sample SD over three fixed folds. Development precision/recall/F1 are reported at a fixed 0.5 threshold per fold (mean ± sample SD), avoiding threshold tuning on each validation fold. Heldout classification metrics use each model's threshold selected from development OOF predictions. The selected original model was GCN, selected by development CV AP.

{chr(10).join(tbl)}

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
'''
(ROOT/"reports/phase3_starting_state.md").write_text(text,encoding="utf-8")
print(f"Wrote {OUT/'metrics/phase2_full_comparison.csv'} and reports/phase3_starting_state.md")
