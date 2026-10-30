"""Development-only ranking check using STRING degree as a single score."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,networkx as nx
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.stats import spearmanr
from run_phase3_gcn_search import ROOT,INPUT,BASE
run=Path(sys.argv[1]).resolve(); frame=pd.read_csv(INPUT/"gene_features_and_labels.csv");y=np.load(INPUT/"target_values.npy",allow_pickle=False).astype(int)
split=pd.read_csv(BASE/"predictions/split_manifest.csv",usecols=["gene_index","partition"]);dev=split.loc[split.partition.eq("development"),"gene_index"].to_numpy(int)
mat=np.load(INPUT/"graph_edge_index.npz",allow_pickle=False);g=nx.Graph();g.add_nodes_from(range(len(frame)))
for u,v in zip(mat["edge_index"][0],mat["edge_index"][1]):
 if u<v:g.add_edge(int(u),int(v))
d=np.asarray([g.degree(i) for i in range(len(frame))]);result={"evaluation":"development labels only","score":"STRING v12.5 degree at predeclared score threshold >=700","n":len(dev),"positive":int(y[dev].sum()),"explicit_zero":int(len(dev)-y[dev].sum()),"average_precision":float(average_precision_score(y[dev],d[dev])),"roc_auc":float(roc_auc_score(y[dev],d[dev])),"spearman_degree_vs_target":float(spearmanr(d[dev],y[dev]).statistic)}
(run/"metrics/degree_only_dev_baseline.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
