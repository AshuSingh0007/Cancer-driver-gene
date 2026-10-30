"""Development-only GCN search, followed by one frozen-model test evaluation.

The tuning loop loads only the persisted development fold assignments and
never reads held-out labels. Test labels are loaded only after the winning
configuration and final-development checkpoint have been frozen.
"""
from __future__ import annotations
import csv, json, math, random, time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix)
from sklearn.preprocessing import StandardScaler

from graph_models import normalized_sparse_adjacency

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/integrated_model_inputs_20261028_155809"
BASE = ROOT / "outputs/experiments/integrated_20261028_161519"
SEED_VALUES = [17, 42, 2025]
MAX_EPOCHS, PATIENCE = 70, 12
CONFIGS = [
    {"name":"gcn_baseline_reproduction", "hidden":32, "layers":2, "dropout":0.30, "lr":0.008, "weight_decay":0.001, "class_weight":"balanced"},
    {"name":"gcn_wide_regularized", "hidden":64, "layers":2, "dropout":0.50, "lr":0.003, "weight_decay":0.010, "class_weight":"balanced"},
    {"name":"gcn_deeper_moderate", "hidden":32, "layers":3, "dropout":0.40, "lr":0.003, "weight_decay":0.005, "class_weight":"balanced"},
    {"name":"gcn_unweighted_loss", "hidden":32, "layers":2, "dropout":0.30, "lr":0.008, "weight_decay":0.001, "class_weight":"none"},
]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class TunableGCN(torch.nn.Module):
    def __init__(self, hidden, layers, dropout):
        super().__init__(); self.dropout = dropout
        dims = [3] + [hidden] * layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i+1]) for i in range(layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 32), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(32, 1))
    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            h = torch.sparse.mm(adj, layer(h))
            h = torch.nn.functional.gelu(self.norms[i](h))
            if i < len(self.layers)-1:
                h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)

def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {"average_precision":float(average_precision_score(y,p)), "roc_auc":float(roc_auc_score(y,p)),
        "precision":float(precision_score(y,pred,zero_division=0)), "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)), "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "mcc":float(matthews_corrcoef(y,pred)), "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    # Do not load annotation/target columns before model selection.
    node_index = pd.read_csv(INPUT / "gene_features_and_labels.csv", usecols=["gene_index"])
    topology_edges = np.load(INPUT / "graph_edge_index.npz")
    edge_index = topology_edges["edge_index"].astype(np.int64)
    edge_weight = topology_edges["edge_weight"].astype(np.float32)
    n = len(node_index)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for u,v,w in zip(edge_index[0],edge_index[1],edge_weight):
        if u < v: g.add_edge(int(u),int(v),weight=float(w))
    deg = dict(g.degree()); strength = dict(g.degree(weight="weight")); clust = nx.clustering(g,weight="weight")
    x = np.asarray([[math.log1p(deg[i]),math.log1p(strength[i]),clust[i]] for i in range(n)],dtype=np.float32)
    # Memory-map the target array; only development train/validation indices are
    # accessed until the final model and checkpoint are frozen.
    target = np.load(INPUT / "target_values.npy",allow_pickle=False,mmap_mode="r")
    mask = np.load(INPUT / "target_mask.npy",allow_pickle=False).astype(bool)
    # Deliberately load only gene_index and partition, not held-out labels.
    split = pd.read_csv(BASE / "predictions/split_manifest.csv",usecols=["gene_index","partition"])
    dev = split.loc[split.partition.eq("development"),"gene_index"].to_numpy(dtype=int)
    test = split.loc[split.partition.eq("held_out_test"),"gene_index"].to_numpy(dtype=int)
    fold_table = pd.read_csv(BASE / "predictions/graph_gcn_development_oof.csv",usecols=["gene_index","fold"])
    fold_for = dict(zip(fold_table.gene_index.astype(int),fold_table.fold.astype(int)))
    folds = {f:np.asarray([i for i in dev if fold_for[int(i)]==f],dtype=int) for f in (1,2,3)}
    assert len(dev)==sum(map(len,folds.values())) and len(test)>0 and set(dev).isdisjoint(test)
    # The tuning loop accesses target only at train/validation indices.
    out = ROOT / "outputs/experiments" / ("phase3_gcn_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    for d in ["configs","metrics","logs","predictions","checkpoints","plots"]: (out/d).mkdir(parents=True,exist_ok=False)
    split[["gene_index","partition"]].to_csv(out/"predictions/split_manifest.csv",index=False)
    (out/"configs/search.json").write_text(json.dumps({"selection":"development mean AP over fixed existing 3 folds and 3 random seeds",
        "seed_values":SEED_VALUES,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"test_labels_read_during_search":False,
        "fold_source":"Phase 2 graph_gcn_development_oof.csv; unchanged","configs":CONFIGS},indent=2),encoding="utf-8")

    records=[]; fold_rows=[]; oof_acc={c["name"]:{int(i):[] for i in dev} for c in CONFIGS}; epoch_values={c["name"]:[] for c in CONFIGS}
    for cfg in CONFIGS:
        for seed in SEED_VALUES:
            for fold in (1,2,3):
                val=folds[fold]; train=np.concatenate([folds[k] for k in (1,2,3) if k!=fold])
                sc=StandardScaler().fit(x[train]); xs=torch.tensor(sc.transform(x).astype(np.float32))
                ei=torch.tensor(edge_index,dtype=torch.long); ew=torch.tensor(edge_weight,dtype=torch.float32)
                adj=normalized_sparse_adjacency(ei,ew,n)
                run_seed=seed+fold*101+2; seed_all(run_seed)
                model=TunableGCN(cfg["hidden"],cfg["layers"],cfg["dropout"])
                params=sum(p.numel() for p in model.parameters())
                opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
                pos=int(target[train].sum()); neg=len(train)-pos
                pos_weight=(neg/max(pos,1)) if cfg["class_weight"]=="balanced" else 1.0
                best_ap=-1; best_state=None; best_epoch=0; stale=0; log=[]; t0=time.perf_counter()
                for epoch in range(1,MAX_EPOCHS+1):
                    model.train(); opt.zero_grad(set_to_none=True)
                    logits=model(xs,adj)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[train],torch.tensor(target[train],dtype=torch.float32),pos_weight=torch.tensor(pos_weight))
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
                    model.eval()
                    with torch.no_grad(): prob=torch.sigmoid(model(xs,adj)[val]).numpy()
                    ap=float(average_precision_score(target[val],prob))
                    log.append({"epoch":epoch,"train_bce":float(loss.detach()),"validation_ap":ap})
                    if ap>best_ap+1e-6:
                        best_ap=ap;best_epoch=epoch;stale=0;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                    else: stale+=1
                    if stale>=PATIENCE:break
                elapsed=time.perf_counter()-t0; model.load_state_dict(best_state); model.eval()
                with torch.no_grad(): probs=torch.sigmoid(model(xs,adj)).numpy()
                for i,p in zip(val,probs[val]): oof_acc[cfg["name"]][int(i)].append(float(p))
                epoch_values[cfg["name"]].append(best_epoch)
                stem=f"{cfg['name']}_seed{seed}_fold{fold}"
                pd.DataFrame(log).to_csv(out/f"logs/{stem}.csv",index=False)
                torch.save({"state_dict":model.state_dict(),"config":cfg,"seed":run_seed,"fold":fold,"best_epoch":best_epoch},out/f"checkpoints/{stem}.pt")
                fold_metric=metrics(target[val],probs[val])
                row={"configuration":cfg["name"],"seed":seed,"fold":fold,"best_epoch":best_epoch,"parameters":params,"training_seconds":elapsed,"train_n":len(train),"validation_n":len(val),"validation_positive":int(target[val].sum()),"validation_explicit_zero":int(len(val)-target[val].sum()),**fold_metric}
                fold_rows.append(row); records.append(row)
                print(f"{cfg['name']} seed={seed} fold={fold}: AP={best_ap:.4f} epoch={best_epoch} time={elapsed:.1f}s",flush=True)
    fold_df=pd.DataFrame(fold_rows); fold_df.to_csv(out/"metrics/dev_search_folds.csv",index=False)
    summary=[]
    oof_out=[]
    for cfg in CONFIGS:
        name=cfg["name"]; vals=[]
        for i in dev:
            ps=oof_acc[name][int(i)]; assert len(ps)==len(SEED_VALUES)
            p=float(np.mean(ps));oof_out.append({"configuration":name,"gene_index":int(i),"probability":p,"target":int(target[i]),"fold":fold_for[int(i)]})
            vals.append(p)
        yy=target[dev]; oo=np.asarray(vals)
        row={"configuration":name,"development_oof_ap_mean":float(fold_df.loc[fold_df.configuration.eq(name),"average_precision"].mean()),
             "development_oof_ap_sd":float(fold_df.loc[fold_df.configuration.eq(name),"average_precision"].std(ddof=1)),
             "pooled_seed_averaged_oof_ap":float(average_precision_score(yy,oo)),"pooled_seed_averaged_oof_roc_auc":float(roc_auc_score(yy,oo)),
             "parameter_count":int(fold_df.loc[fold_df.configuration.eq(name),"parameters"].iloc[0]),
             "mean_fold_seconds":float(fold_df.loc[fold_df.configuration.eq(name),"training_seconds"].mean()),
             "median_best_epoch":int(np.median(epoch_values[name])),**cfg}
        summary.append(row)
    summary_df=pd.DataFrame(summary).sort_values(["development_oof_ap_mean","pooled_seed_averaged_oof_ap"],ascending=False)
    summary_df.to_csv(out/"metrics/dev_search_summary.csv",index=False)
    pd.DataFrame(oof_out).to_csv(out/"predictions/development_oof_predictions.csv",index=False)
    best_name=summary_df.iloc[0].configuration;best_cfg=next(c for c in CONFIGS if c["name"]==best_name)
    # Freeze configuration using development results, then fit once on all development labels.
    final_epoch=max(1,int(np.median(epoch_values[best_name])))
    final_seed=SEED_VALUES[1]+999
    seed_all(final_seed); scaler=StandardScaler().fit(x[dev]); xs=torch.tensor(scaler.transform(x).astype(np.float32))
    ei=torch.tensor(edge_index,dtype=torch.long);ew=torch.tensor(edge_weight,dtype=torch.float32);adj=normalized_sparse_adjacency(ei,ew,n)
    final=TunableGCN(best_cfg["hidden"],best_cfg["layers"],best_cfg["dropout"]); opt=torch.optim.AdamW(final.parameters(),lr=best_cfg["lr"],weight_decay=best_cfg["weight_decay"])
    pos=int(target[dev].sum());neg=len(dev)-pos;pos_weight=(neg/max(pos,1)) if best_cfg["class_weight"]=="balanced" else 1.0
    for _ in range(final_epoch):
        final.train();opt.zero_grad(set_to_none=True); logits=final(xs,adj)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[dev],torch.tensor(target[dev].copy(),dtype=torch.float32),pos_weight=torch.tensor(pos_weight));loss.backward();torch.nn.utils.clip_grad_norm_(final.parameters(),2.0);opt.step()
    # threshold is selected only from pooled development OOF probabilities
    best_threshold=(0.5,-1.0)
    dfoof=pd.DataFrame(oof_out); probs_dev=dfoof.loc[dfoof.configuration.eq(best_name)].sort_values("gene_index").probability.to_numpy(); yy=target[dev]
    for threshold in np.linspace(.05,.95,181):
        pred=(probs_dev>=threshold).astype(int); score=matthews_corrcoef(yy,pred)
        if score>best_threshold[1]:best_threshold=(float(threshold),float(score))
    torch.save({"state_dict":final.state_dict(),"config":best_cfg,"seed":final_seed,"fit_epoch_count":final_epoch,
        "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"threshold_from_development_oof":best_threshold[0]},out/f"checkpoints/{best_name}_final.pt")
    final.eval()
    with torch.no_grad(): all_prob=torch.sigmoid(final(xs,adj)).numpy()
    # Held-out labels are first indexed only here, after architecture/configuration/checkpoint freeze.
    test_metrics=metrics(target[test],all_prob[test],best_threshold[0])
    frame = pd.read_csv(INPUT / "gene_features_and_labels.csv")
    predframe=frame.copy(); predframe["score"]=all_prob;predframe["partition"]=split.set_index("gene_index").partition.reindex(predframe.gene_index).to_numpy()
    predframe["score_interpretation"]="exploratory model score; not a confirmed driver probability"
    predframe.to_csv(out/"predictions/all_gene_scores.csv",index=False)
    predframe.loc[predframe.partition.eq("held_out_test")].assign(prediction=(all_prob[test]>=best_threshold[0]).astype(int)).to_csv(out/"predictions/final_heldout_predictions.csv",index=False)
    unknown=predframe.loc[predframe.oncogene_status.eq("unknown")].sort_values("score",ascending=False)
    unknown.to_csv(out/"predictions/ranked_unknown_candidates.csv",index=False)
    run_summary={"status":"completed","selected_configuration":best_cfg,"selection_metric":"development fold mean average precision across 3 fixed folds x 3 seeds",
        "development_cv_average_precision_mean":float(summary_df.iloc[0].development_oof_ap_mean),"development_cv_average_precision_sd":float(summary_df.iloc[0].development_oof_ap_sd),
        "development_pooled_seed_averaged_oof_ap":float(summary_df.iloc[0].pooled_seed_averaged_oof_ap),"development_threshold":best_threshold[0],"final_fit_epochs":final_epoch,
        "final_checkpoint":str((out/f"checkpoints/{best_name}_final.pt").relative_to(ROOT)),"final_fit_seed":final_seed,
        "final_parameter_count":sum(p.numel() for p in final.parameters()),"heldout_test_evaluated_after_freeze":True,
        "heldout_test_n":len(test),"heldout_positive":int(target[test].sum()),"heldout_explicit_zero":int((1-target[test]).sum()),"heldout_metrics":test_metrics,
        "test_scores_not_used_for_search_or_selection":True,"test_set_has_prior_phase2_scores":True,"elapsed_seconds":time.perf_counter()-start,
        "resident_set_gib_at_completion":__import__('psutil').Process().memory_info().rss/1024**3,
        "peak_process_rss_gib":"not instrumented for this run","device":"cpu","configurations_attempted":len(CONFIGS),
        "fold_seed_fits":len(records),"skipped_experiments":{"edge_threshold_ablation":"Skipped: requires rebuilding alternative STRING induced graphs and would confound the bounded GCN tuning run; current score-700 graph retained.",
          "component_holdout":"Statistically infeasible: outside largest component has only 7 positive and 8 explicit-zero genes.",
          "independent_COSMIC_test":"Not performed: target roles/eligibility and source overlap require a separately frozen external-validation protocol; COSMIC remains excluded from training."}}
    (out/"metrics/phase3_summary.json").write_text(json.dumps(run_summary,indent=2),encoding="utf-8")
    # Small plots are derived from development metrics only; heldout is not plotted into search comparisons.
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(summary_df.configuration,summary_df.development_oof_ap_mean,yerr=summary_df.development_oof_ap_sd)
    ax.set_ylabel("Development CV AP");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(out/"plots/dev_search_ap.png",dpi=160);plt.close(fig)
    print(json.dumps({"run_directory":str(out.relative_to(ROOT)),"selected_configuration":best_cfg,"cv_ap":run_summary["development_cv_average_precision_mean"],"cv_sd":run_summary["development_cv_average_precision_sd"],"test_metrics":test_metrics},indent=2),flush=True)

if __name__=="__main__":main()
