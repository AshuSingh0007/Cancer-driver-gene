"""Development-only graph-message and degree-dependence diagnostics."""
import json, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from graph_models import normalized_sparse_adjacency
from run_phase3_gcn_search import TunableGCN, seed_all, metrics, INPUT, BASE, ROOT

OUT=Path(sys.argv[1]).resolve(); cfg=json.loads((OUT/"metrics/phase3_summary.json").read_text())["selected_configuration"]
fold_df=pd.read_csv(BASE/"predictions/graph_gcn_development_oof.csv",usecols=["gene_index","fold"])
fold_map=dict(zip(fold_df.gene_index.astype(int),fold_df.fold.astype(int)))
split=pd.read_csv(BASE/"predictions/split_manifest.csv",usecols=["gene_index","partition"])
dev=split.loc[split.partition.eq("development"),"gene_index"].to_numpy(int)
frame=pd.read_csv(INPUT/"gene_features_and_labels.csv")
target=np.load(INPUT/"target_values.npy",allow_pickle=False).astype(int)
mat=np.load(INPUT/"graph_edge_index.npz"); edge_index=mat["edge_index"].astype(np.int64); edge_weight=mat["edge_weight"].astype(np.float32); n=len(frame)
g=__import__('networkx').Graph();g.add_nodes_from(range(n))
for u,v,w in zip(edge_index[0],edge_index[1],edge_weight):
    if u<v:g.add_edge(int(u),int(v),weight=float(w))
degree=np.asarray([g.degree(i) for i in range(n)])
clustering=__import__('networkx').clustering(g,weight="weight")
x=np.asarray([[np.log1p(g.degree(i)),np.log1p(g.degree(i,weight="weight")),clustering[i]] for i in range(n)],dtype=np.float32)
ei=torch.tensor(edge_index,dtype=torch.long); ew=torch.tensor(edge_weight,dtype=torch.float32); adj=normalized_sparse_adjacency(ei,ew,n)
empty_ei=torch.empty((2,0),dtype=torch.long);empty_ew=torch.empty(0,dtype=torch.float32);identity=normalized_sparse_adjacency(empty_ei,empty_ew,n)
seed_values=[17,42,2025]; rows=[]; predictions={"full_graph":{int(i):[] for i in dev},"identity_only":{int(i):[] for i in dev}}
for seed in seed_values:
  for fold in (1,2,3):
    va=np.asarray([i for i in dev if fold_map[int(i)]==fold],dtype=int);tr=np.asarray([i for i in dev if fold_map[int(i)]!=fold],dtype=int)
    sc=StandardScaler().fit(x[tr]);xs=torch.tensor(sc.transform(x).astype(np.float32))
    pos=int(target[tr].sum());pw=((len(tr)-pos)/max(pos,1)) if cfg["class_weight"]=="balanced" else 1.0
    for mode,adj_here in (("full_graph",adj),("identity_only",identity)):
      seed_all(seed+fold*101+2);model=TunableGCN(cfg["hidden"],cfg["layers"],cfg["dropout"]);opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
      best=-1;best_state=None;best_epoch=0;stale=0;t0=time.perf_counter()
      for epoch in range(1,71):
        model.train();opt.zero_grad(set_to_none=True);z=model(xs,adj_here)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z[tr],torch.tensor(target[tr],dtype=torch.float32),pos_weight=torch.tensor(pw));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        model.eval()
        with torch.no_grad():p=torch.sigmoid(model(xs,adj_here)[va]).numpy()
        ap=average_precision_score(target[va],p)
        if ap>best+1e-6:best=ap;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=12:break
      model.load_state_dict(best_state);model.eval()
      with torch.no_grad():p=torch.sigmoid(model(xs,adj_here)[va]).numpy()
      for i,pr in zip(va,p):predictions[mode][int(i)].append(float(pr))
      rows.append({"mode":mode,"seed":seed,"fold":fold,"validation_ap":float(best),"best_epoch":best_epoch,"seconds":time.perf_counter()-t0})
      print(f"{mode} seed={seed} fold={fold} AP={best:.4f}",flush=True)
pd.DataFrame(rows).to_csv(OUT/"metrics/dev_graph_dependence_folds.csv",index=False)
summary=[]
for mode in predictions:
  p=np.asarray([np.mean(predictions[mode][int(i)]) for i in dev]); y=target[dev]
  summary.append({"mode":mode,"pooled_seed_averaged_oof_ap":float(average_precision_score(y,p)),"pooled_seed_averaged_oof_roc_auc":float(roc_auc_score(y,p)),
      "fold_ap_mean":float(pd.DataFrame(rows).query("mode==@mode").validation_ap.mean()),"fold_ap_sd":float(pd.DataFrame(rows).query("mode==@mode").validation_ap.std(ddof=1)),
      "spearman_score_vs_degree":float(spearmanr(p,degree[dev]).statistic)})
summary_df=pd.DataFrame(summary);summary_df.to_csv(OUT/"metrics/dev_graph_dependence_summary.csv",index=False)
bins=pd.qcut(pd.Series(degree[dev]).rank(method="first"),q=4,labels=["Q1_low","Q2","Q3","Q4_high"])
degree_rows=[]
for mode in predictions:
  p=np.asarray([np.mean(predictions[mode][int(i)]) for i in dev]);
  for q in bins.unique():
    idx=np.flatnonzero(np.asarray(bins==q)); yy=target[dev[idx]];pp=p[idx]
    degree_rows.append({"mode":mode,"degree_quartile":str(q),"n":len(idx),"positive":int(yy.sum()),"explicit_zero":int(len(yy)-yy.sum()),
       "mean_degree":float(degree[dev[idx]].mean()),"mean_score":float(pp.mean()),"average_precision":float(average_precision_score(yy,pp)) if yy.sum() else None})
pd.DataFrame(degree_rows).to_csv(OUT/"metrics/dev_degree_strata.csv",index=False)
(OUT/"metrics/dev_graph_dependence.json").write_text(json.dumps({"evaluation":"development-only fixed folds, 3 seeds; identity adjacency removes neighbor messages while retaining centrality inputs",
 "results":summary,"interpretation":"This is a diagnostic ablation, not causal evidence. High-degree score correlation and stratified AP are descriptive."},indent=2),encoding="utf-8")
fig,ax=plt.subplots(figsize=(6,4));ax.bar(summary_df["mode"],summary_df.pooled_seed_averaged_oof_ap);ax.set_ylabel("Development pooled OOF AP");ax.set_title("GCN message-passing ablation");fig.tight_layout();fig.savefig(OUT/"plots/dev_graph_dependence.png",dpi=160);plt.close(fig)
print(summary_df.to_string(index=False))
