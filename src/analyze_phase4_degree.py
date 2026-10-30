import json,sys
from pathlib import Path
import numpy as np,pandas as pd,networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1];RUN=Path(sys.argv[1]).resolve();base=ROOT/'data/processed/integrated_model_inputs_20261028_155809'
d=pd.read_csv(RUN/'predictions/development_oof_predictions.csv').query("model == 'sparse_graph_transformer'").sort_values('gene_index')
a=np.load(base/'graph_edge_index.npz');g=nx.Graph();g.add_nodes_from(range(3304));g.add_edges_from(zip(a['edge_index'][0],a['edge_index'][1]));deg=np.asarray([g.degree(int(i)) for i in d.gene_index]);q=pd.qcut(deg,4,labels=False,duplicates='drop');y=d.target.to_numpy();p=d.probability.to_numpy()
out={'spearman_oof_score_vs_degree':float(spearmanr(p,deg).statistic),'quartiles':[]}
for k in sorted(set(q)):
 m=np.asarray(q)==k;out['quartiles'].append({'quartile':int(k+1),'n':int(m.sum()),'positives':int(y[m].sum()),'mean_degree':float(deg[m].mean()),'average_precision':float(average_precision_score(y[m],p[m]))})
(RUN/'metrics/development_degree_diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
