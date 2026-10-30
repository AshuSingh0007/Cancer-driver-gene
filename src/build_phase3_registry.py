"""Build a transparent registry of successful, failed, and skipped Phase 3 work."""
import json,sys
from pathlib import Path
import pandas as pd
run=Path(sys.argv[1]).resolve()
fold=pd.read_csv(run/"metrics/dev_search_folds.csv")
graph=pd.read_csv(run/"metrics/dev_graph_dependence_folds.csv")
rows=[]
for _,r in fold.iterrows():
 rows.append({"experiment":r.configuration,"kind":"gcn_hyperparameter_search","seed":int(r.seed),"fold":int(r.fold),"status":"completed","development_ap":float(r.average_precision),"heldout_used":False,"note":f"best_epoch={int(r.best_epoch)}; parameters={int(r.parameters)}; seconds={r.training_seconds:.3f}"})
for _,r in graph.iterrows():
 rows.append({"experiment":"gcn_selected_config_"+r["mode"],"kind":"graph_message_ablation","seed":int(r.seed),"fold":int(r.fold),"status":"completed","development_ap":float(r.validation_ap),"heldout_used":False,"note":f"best_epoch={int(r.best_epoch)}; seconds={r.seconds:.3f}"})
for name,note in [
 ("initial_final_inference_attempt","Invalidated: refit model remained in train mode, leaving dropout active during final prediction. Re-run after explicit eval() switch; do not use first directory heldout metrics."),
 ("intermediate_pre_freeze_label_table_load","Superseded: tuning accessed only development labels, but the full annotated CSV was loaded before freeze. Strict final rerun reads only gene indices and memory-maps targets until freeze."),
 ("graph_ablation_export_attempt_1","Training completed, artifact export failed due incorrect matplotlib.pyplot import; rerun completed."),
 ("graph_ablation_export_attempt_2","Training completed, plotting failed because DataFrame.mode resolved as method; rerun completed after column indexing fix.")]:
 rows.append({"experiment":name,"kind":"artifact_generation","seed":"","fold":"","status":"failed_and_superseded","development_ap":"","heldout_used":False,"note":note})
skips=json.loads((run/"metrics/phase3_summary.json").read_text())["skipped_experiments"]
for name,note in skips.items(): rows.append({"experiment":name,"kind":"planned_diagnostic","seed":"","fold":"","status":"skipped","development_ap":"","heldout_used":False,"note":note})
rows.append({"experiment":"selected_final_refit_and_heldout_scoring","kind":"final_evaluation","seed":1041,"fold":"all_development","status":"completed_after_freeze","development_ap":json.loads((run/"metrics/phase3_summary.json").read_text())["development_cv_average_precision_mean"],"heldout_used":True,"note":"One final checkpoint; prior holdout scores from Phase 2 were already exposed, so this is not a pristine independent test estimate."})
pd.DataFrame(rows).to_csv(run/"metrics/experiment_registry.csv",index=False)
(run/"metrics/experiment_registry.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

f=pd.read_csv(run/"metrics/dev_search_folds.csv")
seed=f.groupby(["configuration","seed"]).average_precision.agg(["mean","std"]).reset_index();seed.to_csv(run/"metrics/dev_search_seed_summary.csv",index=False)
print(f"Registry: {len(rows)} records ({int((pd.DataFrame(rows).status=='completed').sum())} completed fold fits, including diagnostics)")
