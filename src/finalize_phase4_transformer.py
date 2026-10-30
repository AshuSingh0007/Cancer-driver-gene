"""Add development-only threshold diagnostics, plots, and checkpoint verification."""
import json, sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,confusion_matrix,matthews_corrcoef,precision_score,recall_score,f1_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
RUN=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else max((ROOT/'outputs/experiments').glob('phase4_transformer_*'),key=lambda p:p.stat().st_mtime)
INPUT=ROOT/'data/processed/integrated_model_inputs_20261028_155809'
oof=pd.read_csv(RUN/'predictions/development_oof_predictions.csv'); best=json.loads((RUN/'metrics/phase4_summary.json').read_text())['selected_new_model']; d=oof[oof.model.eq(best)].sort_values('gene_index')
y=d.target.to_numpy(int);p=d.probability.to_numpy(float)
thresholds=np.linspace(.05,.95,181); vals=[matthews_corrcoef(y,p>=t) for t in thresholds]; th=float(thresholds[int(np.argmax(vals))]); pred=(p>=th).astype(int)
tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
thm={'threshold_from_development_oof_by_mcc':th,'oof_precision':float(precision_score(y,pred,zero_division=0)),'oof_recall':float(recall_score(y,pred,zero_division=0)),'oof_f1':float(f1_score(y,pred,zero_division=0)),'oof_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'oof_mcc':float(matthews_corrcoef(y,pred)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'positive':int(y.sum()),'explicit_zero':int(len(y)-y.sum()),'interpretation':'threshold is development OOF optimized; apparent threshold metrics are selection-biased and not an independent estimate'}
(RUN/'metrics/development_threshold_metrics.json').write_text(json.dumps(thm,indent=2))
fig,ax=plt.subplots(figsize=(6,5));pr,rc,_=precision_recall_curve(y,p);ax.plot(rc,pr,label=f'{best} (AP={np.mean(pd.read_csv(RUN/"metrics/dev_transformer_folds.csv").query("model == @best").average_precision):.3f})');ax.set(xlabel='Recall',ylabel='Precision',title='Development out-of-fold precision–recall');ax.legend();fig.tight_layout();fig.savefig(RUN/'plots/development_oof_pr.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));fpr,tpr,_=roc_curve(y,p);ax.plot(fpr,tpr,label=f'{best}');ax.plot([0,1],[0,1],':',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Development out-of-fold ROC');ax.legend();fig.tight_layout();fig.savefig(RUN/'plots/development_oof_roc.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(4.5,4));ax.imshow([[tn,fp],[fn,tp]],cmap='Blues');ax.set_xticks([0,1],['Predicted 0','Predicted 1']);ax.set_yticks([0,1],['True 0','True 1']);
for (i,j),v in np.ndenumerate([[tn,fp],[fn,tp]]):ax.text(j,i,str(v),ha='center',va='center')
ax.set_title(f'Development OOF confusion (threshold={th:.3f})');fig.tight_layout();fig.savefig(RUN/'plots/development_oof_confusion.png',dpi=160);plt.close(fig)
# Training curves: validation AP vs epoch for every completed fold-seed fit.
logs=list((RUN/'logs').glob('*.csv'));fig,ax=plt.subplots(figsize=(8,5))
for path in logs:
    z=pd.read_csv(path); model=path.stem.split('_seed')[0];ax.plot(z.epoch,z.validation_ap,alpha=.3,label=model if model not in ax.get_legend_handles_labels()[1] else None)
ax.set(xlabel='Epoch',ylabel='Validation AP',title='Phase 4 validation curves (all completed fits)');ax.legend();fig.tight_layout();fig.savefig(RUN/'plots/training_curves.png',dpi=160);plt.close(fig)
# Reproduce final prediction vector from the frozen checkpoint and train-fitted scalers.
import sys as _sys;_sys.path.insert(0,str(ROOT/'src'))
from phase4_sparse_transformer import SparseGraphTransformer,SparseTransformerFusion
expr=np.load(INPUT/'expression_features_raw.npy',allow_pickle=False).astype(np.float32);dat=np.load(INPUT/'graph_edge_index.npz');ei=torch.tensor(dat['edge_index'].astype(np.int64));ew=torch.tensor(dat['edge_weight'].astype(np.float32))
device='cpu'; pack=torch.load(RUN/f'checkpoints/{best}_final.pt',map_location=device,weights_only=True)
import networkx as nx,math
g=nx.Graph();g.add_nodes_from(range(len(expr)));g.add_weighted_edges_from((int(a),int(b),float(w)) for a,b,w in zip(dat['edge_index'][0],dat['edge_index'][1],dat['edge_weight']) if a<b);deg=dict(g.degree());strength=dict(g.degree(weight='weight'));clust=nx.clustering(g,weight='weight');topo=np.asarray([[math.log1p(deg[i]),math.log1p(strength[i]),clust[i]] for i in range(len(expr))],np.float32)
xs=torch.tensor((expr-np.asarray(pack['expr_mean'],np.float32))/np.asarray(pack['expr_scale'],np.float32));ts=torch.tensor((topo-np.asarray(pack['topo_mean'],np.float32))/np.asarray(pack['topo_scale'],np.float32))
model=SparseGraphTransformer(3) if best=='sparse_graph_transformer' else SparseTransformerFusion();model.load_state_dict(pack['state_dict']);model.eval()
with torch.no_grad(): repl=torch.sigmoid(model(ts,ei,ew) if best=='sparse_graph_transformer' else model(xs,ts,ei,ew)).numpy()
saved=pd.read_csv(RUN/'predictions/all_gene_scores.csv').sort_values('gene_index').score.to_numpy(float);delta=float(np.max(np.abs(repl-saved)));assert delta<1e-6
checks={'checkpoint_loaded':True,'all_gene_prediction_count':len(repl),'max_absolute_prediction_difference':delta,'within_tolerance_1e-6':delta<1e-6,'unknown_labels_not_used_for_training':'verified by training loop: loss indexes development labeled genes only','heldout_labels_not_read':True}
(RUN/'metrics/checkpoint_prediction_reproduction.json').write_text(json.dumps(checks,indent=2))
# Extend registry to include audited prior phase references and explicit skips.
reg=pd.read_csv(RUN/'metrics/experiment_registry.csv')
extra=[]
for name in ['logistic_expression_topology','expression_residual','expression_transformer','graph_gcn','graph_sage','graph_gat','expression_gat_concat','expression_sage_concat','hmgat_et_gated','hmgat_et_concat']:
    extra.append({'model':name,'seed':'Phase2 seed 42','fold':'3 fixed folds','status':'completed_historical_reference','heldout_used':'previous Phase 2 holdout exposed','best_epoch':'','seconds':'','note':'See outputs/experiments/integrated_20261028_161519/metrics/model_comparison.csv; not retrained in Phase 4'})
for name,note in [('expression_cross_attention','Skipped: three unordered summary statistics do not define meaningful cross-attention tokens.'),('dense_all_pairs_graph_transformer','Skipped: unnecessary quadratic attention over 3,304 genes; sparse edge attention implemented.'),('external_independent_validation','Skipped: no untouched cohort available; previous holdout was exposed.'),('component_holdout','Skipped: prior audit found only 7 positive and 8 explicit-zero genes outside the giant component.')]:
    extra.append({'model':name,'seed':'','fold':'','status':'skipped_by_design','heldout_used':False,'best_epoch':'','seconds':'','note':note})
extra.append({'model':'phase4_transformer_20261028_171933','seed':'17,42,2025','fold':'3 fixed folds','status':'superseded_auditability_correction','heldout_used':False,'best_epoch':'','seconds':'','note':'Metrics matched final rerun; preserved, superseded because the first script version loaded annotation columns before model selection.'})
pd.concat([reg,pd.DataFrame(extra)],ignore_index=True,sort=False).drop_duplicates(['model','status'],keep='last').to_csv(RUN/'metrics/experiment_registry.csv',index=False)
print(json.dumps({'run':str(RUN.relative_to(ROOT)),'best_model':best,'threshold_metrics':thm,'checkpoint_checks':checks,'registry_rows':len(reg)+len(extra)},indent=2))
