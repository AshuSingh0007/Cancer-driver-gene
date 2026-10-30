"""Summarize, annotate, and integrity-check a Phase 5 expanded model run."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,confusion_matrix,matthews_corrcoef,precision_score,recall_score,f1_score,balanced_accuracy_score,accuracy_score,average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];BASE=Path(sys.argv[1]).resolve();TRAIN=Path(sys.argv[2]).resolve();mroot=ROOT/'data/processed/integrated_model_inputs_20261028_155809';
for q in ['plots','metrics','predictions']:(TRAIN/q).mkdir(exist_ok=True)
fold=pd.read_csv(TRAIN/'metrics/supervised_fold_seed_metrics.csv');oof=pd.read_csv(TRAIN/'predictions/development_oof_predictions.csv');summary=json.loads((TRAIN/'metrics/supervised_training_summary.json').read_text());
fold['accuracy']=(fold.tn+fold.tp)/fold.validation_n
names=list(fold.model.unique());metriccols=['average_precision','roc_auc','accuracy','precision','recall','f1','balanced_accuracy','mcc']
ag=[]
for name,g in fold.groupby('model'):
 row={'model':name,'fold_seed_fits':len(g)}
 for c in metriccols:row[c+'_mean']=float(g[c].mean());row[c+'_sd']=float(g[c].std(ddof=1)) if len(g)>1 else 0.
 row['confusion_tn_sum']=int(g.tn.sum());row['confusion_fp_sum']=int(g.fp.sum());row['confusion_fn_sum']=int(g.fn.sum());row['confusion_tp_sum']=int(g.tp.sum());ag.append(row)
agg=pd.DataFrame(ag);agg.to_csv(TRAIN/'metrics/threshold_0_5_fold_metrics_summary.csv',index=False)
oofrows=[];thrrows=[]
for name,g in oof.groupby('model'):
 y=g.target.to_numpy(int);p=g.probability.to_numpy(float);t=.5;pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 oofrows.append({'model':name,'oof_n':len(y),'positive':int(y.sum()),'explicit_zero':int(len(y)-y.sum()),'oof_ap':float(average_precision_score(y,p)),'oof_roc_auc':float(roc_auc_score(y,p)),'threshold_0_5':t,'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
 ts=np.linspace(.05,.95,181);ms=[matthews_corrcoef(y,p>=t) for t in ts];best=float(ts[int(np.argmax(ms))]);q=p>=best;tn2,fp2,fn2,tp2=confusion_matrix(y,q,labels=[0,1]).ravel()
 thrrows.append({'model':name,'threshold_mcc_selected_on_dev_oof':best,'threshold':best,'accuracy':float(accuracy_score(y,q)),'precision':float(precision_score(y,q,zero_division=0)),'recall':float(recall_score(y,q,zero_division=0)),'f1':float(f1_score(y,q,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'mcc':float(matthews_corrcoef(y,q)),'tn':int(tn2),'fp':int(fp2),'fn':int(fn2),'tp':int(tp2),'selection_note':'Development OOF threshold and labels are reused; optimistic, not independent.'})
pd.DataFrame(oofrows).to_csv(TRAIN/'metrics/development_oof_metrics_at_0_5.csv',index=False);pd.DataFrame(thrrows).to_csv(TRAIN/'metrics/development_oof_threshold_metrics.csv',index=False)
# Comparable historical AP references: same labeled development folds, distinct universe/graph context.
phase2=pd.read_csv(ROOT/'outputs/experiments/integrated_20261028_161519/metrics/model_comparison.csv')
phase3=pd.read_csv(ROOT/'outputs/experiments/phase3_gcn_20261028_164054/metrics/dev_search_summary.csv')
phase4=json.loads((ROOT/'outputs/experiments/phase4_transformer_20261028_172513/metrics/phase4_summary.json').read_text())
comp=[]
for r in phase2.itertuples():comp.append({'model':'Phase 2 '+r.model,'ap_mean':float(r.cv_average_precision_mean),'ap_sd':float(r.cv_average_precision_sd),'pooled_oof_ap':np.nan,'evaluation':'same fixed development genes/folds; original 3,304 universe; 1 seed'})
for r in phase3.itertuples():comp.append({'model':'Phase 3 '+r.configuration,'ap_mean':float(r.development_oof_ap_mean),'ap_sd':float(r.development_oof_ap_sd),'pooled_oof_ap':float(r.pooled_seed_averaged_oof_ap),'evaluation':'same fixed development genes/folds; original universe; 3 seeds'})
for r in phase4['new_models']:comp.append({'model':'Phase 4 '+r['model'],'ap_mean':r['development_cv_ap_mean'],'ap_sd':r['development_cv_ap_sd'],'pooled_oof_ap':r['pooled_seed_averaged_oof_ap'],'evaluation':'same fixed development genes/folds; original 3,304 universe; 3 seeds'})
for r in summary['models']:comp.append({'model':'Phase 5 expanded '+r['model'],'ap_mean':r['development_fold_ap_mean'],'ap_sd':r['development_fold_ap_sd'],'pooled_oof_ap':r['pooled_oof_ap'],'evaluation':'same old development genes/folds; expanded 18,998-gene STRING universe'})
pd.DataFrame(comp).sort_values('ap_mean',ascending=False).to_csv(TRAIN/'metrics/comparable_model_ap.csv',index=False)
# Out-of-fold degree diagnostic for the expanded Transformer.
feature=pd.read_csv(BASE/'data/expanded_gene_features_and_labels.csv',usecols=['gene_index','string_degree_ge700'])
g=oof[oof.model.eq('expanded_sparse_graph_transformer')].merge(feature,on='gene_index',validate='one_to_one')
from scipy.stats import spearmanr
q=pd.qcut(g.string_degree_ge700,4,labels=False,duplicates='drop');degree={'score_degree_spearman':float(spearmanr(g.probability,g.string_degree_ge700).statistic),'quartiles':[]}
for k in sorted(set(q)):
 z=np.asarray(q)==k;degree['quartiles'].append({'quartile':int(k+1),'n':int(z.sum()),'positive':int(g.target.to_numpy()[z].sum()),'mean_degree':float(g.string_degree_ge700.to_numpy()[z].mean()),'ap':float(average_precision_score(g.target.to_numpy()[z],g.probability.to_numpy()[z]))})
(TRAIN/'metrics/development_degree_diagnostic.json').write_text(json.dumps(degree,indent=2))
# Frozen descriptive reference only. COSMIC was not read until predictions/model selection were complete.
sc=pd.read_csv(TRAIN/'predictions/all_gene_scores_supervised.csv');cos= pd.read_csv(ROOT/'data/processed/phase3_task_20261028_145404/cosmic_reference_comparison.csv',dtype={'entrez':str})
cos=cos[cos.cosmic_role_in_cancer.notna()&cos.cosmic_role_in_cancer.astype(str).str.strip().ne('')].drop_duplicates('entrez')
unknown=sc[sc.label_status.eq('unknown')].sort_values('score',ascending=False).copy();unknown['entrez']=unknown.entrez.astype(str)
unknown=unknown.merge(cos[['entrez','cosmic_role_in_cancer','cosmic_tumour_types_somatic']],on='entrez',how='left',validate='one_to_one');unknown['cosmic_reference_match']=unknown.cosmic_role_in_cancer.notna();unknown.to_csv(TRAIN/'predictions/ranked_unknown_candidates_with_reference.csv',index=False)
reference={'source':'Existing Phase 3 COSMIC CGC v104 GRCh38 reference comparison, post-freeze only','training_or_tuning_use':False,'comparison_scope':'COSMIC rows already reconciled to the NCG reference table; does not cover all 18,411 unknown genes. Descriptive catalogue overlap only, not an independent performance estimate.','top_rank_overlap':{str(k):{'candidates':int(min(k,len(unknown))),'reference_matches':int(unknown.head(k).cosmic_reference_match.sum())} for k in [10,50,100,500]},'all_unknown_with_reference_match':int(unknown.cosmic_reference_match.sum())}
(TRAIN/'metrics/cosmic_postfreeze_reference_overlap.json').write_text(json.dumps(reference,indent=2))
# PR/ROC from selected model pooled OOF only.
winner=summary['selected_model'];d=oof[oof.model.eq(winner)].sort_values('gene_index');yy=d.target.to_numpy(int);pp=d.probability.to_numpy(float)
fig,ax=plt.subplots(figsize=(6,5));pr,rc,_=precision_recall_curve(yy,pp);ax.plot(rc,pr,label=f'{winner} (AP={average_precision_score(yy,pp):.3f})');ax.set(xlabel='Recall',ylabel='Precision',title='Phase 5 expanded development OOF PR');ax.legend();fig.tight_layout();fig.savefig(TRAIN/'plots/development_oof_pr.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));fpr,tpr,_=roc_curve(yy,pp);ax.plot(fpr,tpr,label=f'{winner} (AUC={roc_auc_score(yy,pp):.3f})');ax.plot([0,1],[0,1],':',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Phase 5 expanded development OOF ROC');ax.legend();fig.tight_layout();fig.savefig(TRAIN/'plots/development_oof_roc.png',dpi=160);plt.close(fig)
# Attach ensemble votes and pseudo status to candidate ranking.
ens=pd.read_csv(TRAIN/'predictions/unlabeled_model_ensemble.csv');pivot=ens.pivot(index='gene_index',columns='model',values='mean_probability').reset_index()
rename={'expanded_gcn':'gcn_ensemble_probability','expanded_sparse_graph_transformer':'sparse_transformer_ensemble_probability','logistic_expression_topology':'logistic_ensemble_probability'};pivot=pivot.rename(columns=rename)
rank=unknown.drop(columns=['cosmic_role_in_cancer','cosmic_tumour_types_somatic','cosmic_reference_match']).merge(pivot,on='gene_index',how='left',validate='one_to_one');cols=[x for x in rename.values() if x in rank];rank['model_positive_votes_at_0_5']=rank[cols].ge(.5).sum(axis=1);rank['pseudo_label_status']='unknown; no candidate passed predeclared confidence/stability rules';rank.to_csv(TRAIN/'predictions/ranked_unknown_candidates_with_model_agreement.csv',index=False)
# Check final selected checkpoint replays its exported score vector.
arr=np.load(BASE/'data/graph_edge_index.npz');ei=torch.tensor(arr['edge_index'].astype(np.int64));ew=torch.tensor(arr['edge_weight'].astype(np.float32));topo=np.load(BASE/'data/topology_features.npy',allow_pickle=False);ckpath=TRAIN/'checkpoints/final_selected.pt'
checkpoint_check={'model':winner,'reproduction_checked':False,'max_abs_score_delta':None}
if winner!='logistic_expression_topology':
 from phase5_models import ExpandedGCN
 from phase4_sparse_transformer import SparseGraphTransformer
 ck=torch.load(ckpath,map_location='cpu',weights_only=True);x=torch.tensor(((topo-np.asarray(ck['feature_mean'],np.float32))/np.asarray(ck['feature_scale'],np.float32)).astype(np.float32));m=ExpandedGCN(2,32,2,.30) if winner=='expanded_gcn' else SparseGraphTransformer(2,hidden=32,dropout=.25);m.load_state_dict(ck['state_dict']);m.eval()
 if winner=='expanded_gcn':
  from graph_models import normalized_sparse_adjacency
  adj=normalized_sparse_adjacency(ei,ew,len(topo))
 with torch.no_grad():pr=torch.sigmoid(m(x,adj) if winner=='expanded_gcn' else m(x,ei,ew)).numpy()
 checkpoint_check={'model':winner,'reproduction_checked':True,'all_gene_count':len(pr),'max_abs_score_delta':float(np.max(np.abs(pr-sc.sort_values('gene_index').score.to_numpy()))),'within_tolerance_1e-6':bool(np.max(np.abs(pr-sc.sort_values('gene_index').score.to_numpy()))<1e-6)}
 assert checkpoint_check['within_tolerance_1e-6']
(TRAIN/'metrics/checkpoint_reproduction.json').write_text(json.dumps(checkpoint_check,indent=2))
# Phase 5 registry records completed models and explicit non-runs / failures.
reg=[]
for r in fold.to_dict('records'):reg.append({'experiment':r['model'],'seed':r['seed'],'fold':r['fold'],'status':'completed_checkpoint_replayed','development_ap':r['average_precision'],'heldout_used':False,'note':'Training checkpoint completed and replayed; no holdout label access.'})
ps=json.loads((TRAIN/'metrics/pseudo_label_eligibility.json').read_text())
for name,note in [('conservative_pseudolabel_retrain','Not run: zero eligible pseudo-positive and pseudo-negative candidates at predeclared thresholds.'),('model_agreement_pseudolabel_retrain','Not run: zero candidates with strict three-family agreement.'),('weighted_iterative_pseudolabeling','Stopped before training: no initial confident candidates; no pseudo-label iteration can be justified.'),('independent_full_universe_COSMIC_validation','Not run as performance validation: the available reference table covers NCG rows only; post-freeze overlap reported descriptively.'),('holdout_retest','Not repeated: previously exposed heldout set; not a pristine evaluation.')]:reg.append({'experiment':name,'seed':'','fold':'','status':'not_run_or_stopped','development_ap':'','heldout_used':False,'note':note})
for path,reason in [('phase5_expansion_20261028_180839','Superseded: initial STRING stream parser delimiter mismatch.'),('phase5_expansion_20261028_180854','Superseded: old split manifest includes Entrez, initial merge created suffix mismatch.'),('phase5_expansion_20261028_180928','Superseded: graph summary attempted to index a gene-edge dictionary key as a three-column tuple.'),('phase5_expansion_20261028_181012','Superseded: canonical edge export was one-direction only; model message edges corrected in final dataset.'),('phase5_training_aggregate_attempt','Failed after fold training completed: aggregation dict key bug; saved checkpoints preserved and replayed.'),('phase5_replay_attempt_1','Failed after replay: prediction export omitted two feature columns; checkpoints unchanged and final replay succeeded.')]:reg.append({'experiment':path,'seed':'','fold':'','status':'failed_or_superseded','development_ap':'','heldout_used':False,'note':reason})
pd.DataFrame(reg).to_csv(TRAIN/'metrics/experiment_registry.csv',index=False)
# Machine-check key data and model artifacts.
ds=json.loads((BASE/'audit/phase5_dataset_audit.json').read_text());sct=sc.label_status.value_counts().to_dict();integrity={'status':'passed','checks':{
 'expanded_gene_count_matches':len(sc)==ds['candidate_universe_genes'],'unique_entrez_gene_ids':sc.entrez.is_unique,'all_expression_values_finite':bool(np.isfinite(sc[['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive']].to_numpy()).all()),'all_probability_scores_finite_0_1':bool(sc.score.notna().all() and sc.score.between(0,1).all()),'label_counts_preserved':sct.get('positive',0)==254 and sct.get('explicit_zero',0)==333,'unknown_never_training_target':summary['unknown_targets_used_for_loss'] is False,'old_holdout_not_read':summary['heldout_labels_read_during_selection'] is False,'development_fold_coverage':len(oof[oof.model.eq(winner)])==len(pd.read_csv(BASE/'data/split_manifest.csv').query("partition == 'development'")),'canonical_edge_unique':pd.read_csv(BASE/'data/graph_edges_ge700.csv').shape[0]==ds['STRING_graph']['canonical_gene_edges'],'directed_graph_edge_count':np.load(BASE/'data/graph_edge_index.npz')['edge_index'].shape[1]==ds['STRING_graph']['directed_message_edges'],'pseudo_unknown_count':int((sc.label_status.eq('unknown')).sum())==ds['label_status_counts']['unknown'],'checkpoint_reproduces_scores':checkpoint_check.get('within_tolerance_1e-6',False)
}}
integrity['checks']={k:bool(v) for k,v in integrity['checks'].items()};integrity['checks_passed']=sum(integrity['checks'].values());assert all(integrity['checks'].values()),integrity
(TRAIN/'metrics/phase5_integrity_checks.json').write_text(json.dumps(integrity,indent=2))
print(json.dumps({'training_output':str(TRAIN.relative_to(ROOT)),'winner':winner,'oof_metrics':oofrows,'threshold_metrics':thrrows,'degree':degree,'cosmic':reference,'integrity':integrity},indent=2))
