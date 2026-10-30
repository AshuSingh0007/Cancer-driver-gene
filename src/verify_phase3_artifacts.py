"""Verify Phase 3 model loading, identifiers, labels, predictions, and metrics."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,balanced_accuracy_score,matthews_corrcoef,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from graph_models import NodeClassifier,normalized_sparse_adjacency
from run_phase3_gcn_search import TunableGCN,INPUT,BASE
RUN=Path(sys.argv[1]).resolve();checks={}
def ck(k,v):
 checks[k]=bool(v)
 if not v:raise AssertionError(k)

summary=json.loads((RUN/"metrics/phase3_summary.json").read_text()); inp=json.loads((INPUT/"integrated_input_summary.json").read_text())
frame=pd.read_csv(INPUT/"gene_features_and_labels.csv"); y=np.load(INPUT/"target_values.npy",allow_pickle=False);mask=np.load(INPUT/"target_mask.npy",allow_pickle=False).astype(bool)
split=pd.read_csv(RUN/"predictions/split_manifest.csv"); test=split.loc[split.partition.eq("held_out_test"),"gene_index"].to_numpy(int); dev=split.loc[split.partition.eq("development"),"gene_index"].to_numpy(int)
ck("candidate_count_and_identifier_uniqueness",len(frame)==3304 and frame.gene_index.is_unique and frame.entrez.is_unique and frame.ensembl_gene_id_versioned.is_unique)
ck("unknown_labels_still_masked",np.all(y[~mask]==-1) and (frame.loc[frame.oncogene_status.eq("unknown"),"oncogene_label"].isna().all()))
ck("split_disjoint_and_known_only",set(dev).isdisjoint(test) and mask[dev].all() and mask[test].all())
cfg=json.loads((RUN/"configs/search.json").read_text()); ck("search_did_not_read_heldout_labels",cfg["test_labels_read_during_search"] is False and summary["test_scores_not_used_for_search_or_selection"] is True)
script=(ROOT/"src/run_phase3_gcn_search.py").read_text(encoding="utf-8")
freeze=script.index("# Freeze configuration using development results")
ck("training_search_memory_maps_targets","mmap_mode=\"r\"" in script and "target_values.npy" in script)
label_load=script.index('frame = pd.read_csv(INPUT / "gene_features_and_labels.csv")',freeze)
ck("feature_label_table_loaded_only_after_freeze",label_load>freeze)
for token in ["target[test].sum()","metrics(target[test],all_prob[test]"]:
 ck("test_label_access_after_model_freeze_"+str(token).replace("/","_"),script.index(token)>freeze)
ck("test_scores_never_used_for_selection", "test_average_precision" not in script and "test_roc_auc" not in script)

cmp2=pd.read_csv(BASE/"metrics/model_comparison.csv");full=pd.read_csv(RUN/"metrics/phase2_full_comparison.csv")
ck("phase2_metrics_match_saved_source",np.allclose(cmp2.sort_values("model").test_average_precision.to_numpy(),full.sort_values("model").test_ap.to_numpy()))
for name in cmp2.model:
 if name=="logistic_expression_topology":continue
 model=NodeClassifier(name); result=model.load_state_dict(torch.load(BASE/f"checkpoints/{name}/final.pt",map_location="cpu",weights_only=True)["model"]); ck("phase2_checkpoint_load_"+name,not result.missing_keys and not result.unexpected_keys)

ckpt=torch.load(ROOT/summary["final_checkpoint"],map_location="cpu",weights_only=True); model=TunableGCN(ckpt["config"]["hidden"],ckpt["config"]["layers"],ckpt["config"]["dropout"]);model.load_state_dict(ckpt["state_dict"]);model.eval()
mat=np.load(INPUT/"graph_edge_index.npz",allow_pickle=False);ei=torch.tensor(mat["edge_index"].copy(),dtype=torch.long);ew=torch.tensor(mat["edge_weight"].copy(),dtype=torch.float32);adj=normalized_sparse_adjacency(ei,ew,len(frame))
import networkx as nx
graph=nx.Graph();graph.add_nodes_from(range(len(frame)))
for u,v,w in zip(mat["edge_index"][0],mat["edge_index"][1],mat["edge_weight"]):
 if u<v:graph.add_edge(int(u),int(v),weight=float(w))
degree=dict(graph.degree());strength=dict(graph.degree(weight="weight"));clustering=nx.clustering(graph,weight="weight")
topo=np.asarray([[np.log1p(degree[i]),np.log1p(strength[i]),clustering[i]] for i in range(len(frame))],dtype=np.float32)
xs=torch.tensor(((topo-np.asarray(ckpt["scaler_mean"]))/np.asarray(ckpt["scaler_scale"])).astype(np.float32))
with torch.no_grad():prob=torch.sigmoid(model(xs,adj)).numpy()
pred=pd.read_csv(RUN/"predictions/all_gene_scores.csv"); ck("all_gene_prediction_count_and_scores",len(pred)==len(frame) and np.isfinite(pred.score).all() and pred.score.between(0,1).all())
ck("checkpoint_reproduces_all_gene_scores",np.allclose(prob,pred.score.to_numpy(),atol=1e-6))
context=pd.read_csv(RUN/"predictions/all_gene_scores_with_ppi_context.csv")
ranked=pd.read_csv(RUN/"predictions/ranked_unknown_candidates_with_ppi_context.csv")
ck("stable_id_and_score_fields",all(c in context for c in ["entrez","symbol","ensembl_gene_id_versioned","score","oncogene_status","partition","ppi_degree_score700"]))
ck("unknown_rank_file_preserves_unknowns",len(ranked)==2717 and ranked.oncogene_status.eq("unknown").all() and ranked.score.is_monotonic_decreasing)
held=pd.read_csv(RUN/"predictions/final_heldout_predictions.csv");m=summary["heldout_metrics"];yp=held.oncogene_label.astype(int).to_numpy();pp=held.score.to_numpy();thr=summary["development_threshold"]
pr=(pp>=thr).astype(int);tn,fp,fn,tp=confusion_matrix(yp,pr,labels=[0,1]).ravel()
recalc={"average_precision":average_precision_score(yp,pp),"roc_auc":roc_auc_score(yp,pp),"precision":precision_score(yp,pr,zero_division=0),"recall":recall_score(yp,pr,zero_division=0),"f1":f1_score(yp,pr,zero_division=0),"balanced_accuracy":balanced_accuracy_score(yp,pr),"mcc":matthews_corrcoef(yp,pr),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
ck("heldout_rows_and_class_counts",len(held)==summary["heldout_test_n"] and int(yp.sum())==summary["heldout_positive"] and int((1-yp).sum())==summary["heldout_explicit_zero"])
ck("heldout_metrics_recompute",all(abs(float(recalc[k])-float(m[k]))<1e-10 for k in recalc))
ck("final_checkpoint_parameter_count",sum(p.numel() for p in model.parameters())==summary["final_parameter_count"])
fold_ckpts=list((RUN/"checkpoints").glob("*_seed*_fold*.pt"));ck("all_search_checkpoints_load",len(fold_ckpts)==36)
for p in fold_ckpts:
 c=torch.load(p,map_location="cpu",weights_only=True);mod=TunableGCN(c["config"]["hidden"],c["config"]["layers"],c["config"]["dropout"]);mod.load_state_dict(c["state_dict"])
result={"status":"passed","checks_passed":len(checks),"checks":checks,"phase2_checkpoint_count_checked":9,"phase3_search_checkpoint_count_checked":len(fold_ckpts),"heldout_recomputed_metrics":recalc}
(RUN/"metrics/final_integrity_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
