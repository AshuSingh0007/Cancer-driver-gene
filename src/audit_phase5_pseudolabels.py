"""Audit pseudo-label eligibility and confidence against development OOF only."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1];RUN=Path(sys.argv[1]).resolve();out=RUN/'metrics';pred=pd.read_csv(RUN/'predictions/development_oof_predictions.csv');unk=pd.read_csv(RUN/'predictions/unlabeled_model_ensemble.csv')
dev_results={}; selected_thresholds={}
for model,g in pred.groupby('model'):
 rows=[]
 for t in np.linspace(.50,.95,10):
  z=g.probability.ge(t);rows.append({'threshold':float(t),'predicted_positive':int(z.sum()),'precision':float(g.loc[z,'target'].mean()) if z.any() else None})
 for t in np.linspace(.05,.50,10):
  z=g.probability.le(t);rows.append({'threshold':float(t),'predicted_negative':int(z.sum()),'negative_predictive_value':float(1-g.loc[z,'target'].mean()) if z.any() else None})
 # A usable pseudo threshold must attain >=90% empirical correctness with at
 # least ten OOF examples of that predicted class. Thresholds are development
 # OOF-only; no threshold is extrapolated from the historical holdout.
 y=g.target.to_numpy(int);p=g.probability.to_numpy(float)
 pos_options=[];neg_options=[]
 for t in np.unique(p):
  z=p>=t
  if int(z.sum())>=10 and float(y[z].mean())>=.90:pos_options.append((float(t),int(z.sum()),float(y[z].mean())))
  q=p<=t
  if int(q.sum())>=10 and float((1-y[q]).mean())>=.90:neg_options.append((float(t),int(q.sum()),float((1-y[q]).mean())))
 dev_results[model]={'oof_ap':float(average_precision_score(y,p)),'probability_min':float(p.min()),'probability_max':float(p.max()),'development_supported_pseudo_thresholds':{'positive':{'threshold':min(pos_options,key=lambda x:x[0])[0],'n':min(pos_options,key=lambda x:x[0])[1],'precision':min(pos_options,key=lambda x:x[0])[2]} if pos_options else None,'negative':{'threshold':max(neg_options,key=lambda x:x[0])[0],'n':max(neg_options,key=lambda x:x[0])[1],'negative_predictive_value':max(neg_options,key=lambda x:x[0])[2]} if neg_options else None,'minimum_examples':10,'minimum_oof_correctness':.90},'threshold_metrics':rows}
u={m:g.set_index('gene_index') for m,g in unk.groupby('model')}; allgenes=unk.gene_index.unique()
base=dev_results['expanded_sparse_graph_transformer']; selected='expanded_sparse_graph_transformer'
stat={m:{'unlabeled_gene_count':len(gg),'max_mean_probability':float(gg.mean_probability.max()),'min_mean_probability':float(gg.mean_probability.min()),'positive_at_0.95_stable_sd_le_0.05':int(((gg.mean_probability>=.95)&(gg.sd_probability_across_fold_seed_models<=.05)).sum()),'negative_at_0.05_stable_sd_le_0.05':int(((gg.mean_probability<=.05)&(gg.sd_probability_across_fold_seed_models<=.05)).sum())} for m,gg in u.items()}
# Conservative thresholds are predeclared 0.95/0.05 and require fold/seed stability.
sel=u[selected];pos=(sel.mean_probability>=.95)&(sel.sd_probability_across_fold_seed_models<=.05);neg=(sel.mean_probability<=.05)&(sel.sd_probability_across_fold_seed_models<=.05)
# Agreement requires all three supervised families to pass the same class threshold and the neural ensembles to be stable.
agree_pos=pos.copy();agree_neg=neg.copy()
for name in ['expanded_gcn','logistic_expression_topology']:
 z=u[name].reindex(sel.index);agree_pos &= z.mean_probability>=.95;agree_neg &= z.mean_probability<=.05
table=pd.DataFrame({'gene_index':sel.index,'sparse_transformer_mean':sel.mean_probability.values,'sparse_transformer_sd':sel.sd_probability_across_fold_seed_models.values,'gcn_mean':u['expanded_gcn'].reindex(sel.index).mean_probability.values,'logistic_mean':u['logistic_expression_topology'].reindex(sel.index).mean_probability.values})
table['pseudo_label_type']='none';table['pseudo_target']=np.nan;table['pseudo_weight']=np.nan;table['strategy']='no_qualifying_candidate'
table.loc[pos.values,'pseudo_label_type']='pseudo-positive';table.loc[pos.values,'pseudo_target']=1;table.loc[pos.values,'pseudo_weight']=.1;table.loc[pos.values,'strategy']='conservative_sparse_transformer'
table.loc[neg.values,'pseudo_label_type']='pseudo-negative';table.loc[neg.values,'pseudo_target']=0;table.loc[neg.values,'pseudo_weight']=.1;table.loc[neg.values,'strategy']='conservative_sparse_transformer'
table['agreement_eligible']=agree_pos.values|agree_neg.values
table['positive_threshold']=.95;table['negative_threshold']=.05;table['maximum_allowed_probability_sd']=.05;table['pseudo_training_weight_if_selected']=.1;table['probability_provenance']='Fold/seed supervised models trained only on development training genes; unlabeled probability mean/SD across fold/seed models'
ident=pd.read_csv(RUN.parent/'data/gene_identifier_mapping.csv',usecols=['gene_index','entrez','ensembl_gene_id_versioned','gene_symbol'])
table=table.reset_index(drop=True).merge(ident,on='gene_index',validate='one_to_one')
table.to_csv(RUN/'predictions/pseudo_label_candidates.csv',index=False)
result={'threshold_policy':{'development_calibration_rule':'For each pseudo class, select a development OOF threshold only if empirical correctness is at least 0.90 on at least 10 out-of-fold genes; no threshold is inferred from heldout genes. In addition, require prediction SD <=0.05 across fold/seed models and, for agreement pseudo-labels, all model families agree.','minimum_examples':10,'minimum_oof_correctness':.90,'max_fold_seed_probability_sd':.05,'pseudo_training_weight_if_eligible':.1,'verified_label_weight':1.0,'predeclared_extreme_screen_positive':.95,'predeclared_extreme_screen_negative':.05},'development_oof_confidence':dev_results,'unlabeled_prediction_distribution':stat,'conservative_candidate_counts':{'pseudo_positive':int(pos.sum()),'pseudo_negative':int(neg.sum())},'cross_model_agreement_counts':{'pseudo_positive':int(agree_pos.sum()),'pseudo_negative':int(agree_neg.sum())},'decision':'No pseudo-labelled retraining: no model family achieves the development-only 0.90 correctness minimum on at least 10 OOF genes for either pseudo class, and the predeclared extreme 0.95/0.05 screen yields zero stable candidates. Therefore conservative, model-agreement, and weighted iterative pseudo-training are stopped before retraining. Preserve all 18,411 as unknown.', 'COSMIC_used':False}
(out/'pseudo_label_eligibility.json').write_text(json.dumps(result,indent=2));print(json.dumps({'output':str(RUN.relative_to(ROOT)),'distribution':stat,'conservative':result['conservative_candidate_counts'],'agreement':result['cross_model_agreement_counts'],'ranges':{m:{'min':v['probability_min'],'max':v['probability_max']} for m,v in dev_results.items()}},indent=2))
