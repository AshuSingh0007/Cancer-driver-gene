"""Bounded nested-CV feature ablation and LR tuning for Phase 7 optimization."""
from __future__ import annotations
import json,time,shutil
from pathlib import Path
import numpy as np,pandas as pd,joblib,torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import (average_precision_score,roc_auc_score,accuracy_score,precision_score,
 recall_score,f1_score,balanced_accuracy_score,matthews_corrcoef,confusion_matrix,brier_score_loss,
 log_loss,roc_curve,precision_recall_curve,auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/experiments/phase5_expansion_20261028_181232'
TR=BASE/'training_replay_20261028_182511'
DATA=BASE/'data'
OUT=ROOT/'outputs/experiments'/f"final_optimization_{time.strftime('%Y%m%d_%H%M%S')}"
FEATS={
 'expression_only':['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive'],
 'network_only':['log_degree_ge700','log_weighted_strength_ge700'],
 'expression_plus_network':['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive','log_degree_ge700','log_weighted_strength_ge700']}

def score_metrics(y,p):
 z=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {'n':int(len(y)),'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),
  'accuracy':float(accuracy_score(y,z)),'precision':float(precision_score(y,z,zero_division=0)),
  'recall':float(recall_score(y,z,zero_division=0)),'f1':float(f1_score(y,z,zero_division=0)),
  'balanced_accuracy':float(balanced_accuracy_score(y,z)),'mcc':float(matthews_corrcoef(y,z)),
  'brier_score':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
  'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 for d in ['metrics','predictions','checkpoints','configs','logs','plots']: (OUT/d).mkdir()
 df=pd.read_csv(DATA/'expanded_gene_features_and_labels.csv').sort_values('gene_index').reset_index(drop=True)
 target=np.load(DATA/'target_values.npy',allow_pickle=False).astype(int)
 mask=np.load(DATA/'target_mask.npy',allow_pickle=False).astype(bool)
 dev=np.flatnonzero(df.partition.eq('development').to_numpy())
 assert len(dev)==469 and int((target[dev]==1).sum())==203 and int((target[dev]==0).sum())==266
 fold=df.development_fold.to_numpy(int)
 # Target values are indexed only at development folds for model selection; heldout labels remain untouched.
 labels=target[dev]; ymap={int(i):int(target[i]) for i in dev}
 feature_arrays={k:df[v].to_numpy(np.float64) for k,v in FEATS.items()}
 params={'model__C':[.03,.1,1.,10.],'model__class_weight':[None,'balanced']}
 outer_rows=[];oof_rows=[];search_rows=[];model_predictions={}
 for group,columns in FEATS.items():
  x=df[columns].to_numpy(np.float64); pred_oof={}; rows=[]
  for k in (1,2,3):
   va=dev[fold[dev]==k]; tr=dev[fold[dev]!=k]
   ytr=np.asarray([ymap[int(i)] for i in tr]); yv=np.asarray([ymap[int(i)] for i in va])
   inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=1700+k)
   pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(solver='liblinear',max_iter=2000,random_state=410+k))])
   gs=GridSearchCV(pipe,params,scoring='average_precision',cv=inner,n_jobs=1,refit=True,return_train_score=True)
   gs.fit(x[tr],ytr); p=gs.predict_proba(x[va])[:,1]
   pred_oof.update({int(i):float(q) for i,q in zip(va,p)})
   met=score_metrics(yv,p); record={'feature_set':group,'outer_fold':k,'best_inner_ap':float(gs.best_score_),**met,
      'best_C':float(gs.best_params_['model__C']),'best_class_weight':str(gs.best_params_['model__class_weight']),
      'train_n':len(tr),'validation_n':len(va)}
   rows.append(record);outer_rows.append(record)
   for ix,(candidate,cvscore,sd,rank) in enumerate(zip(gs.cv_results_['params'],gs.cv_results_['mean_test_score'],gs.cv_results_['std_test_score'],gs.cv_results_['rank_test_score'])):
    search_rows.append({'feature_set':group,'outer_fold':k,'candidate_C':candidate['model__C'],'candidate_class_weight':str(candidate['model__class_weight']),'inner_mean_ap':float(cvscore),'inner_ap_sd':float(sd),'inner_rank':int(rank)})
   for i,t,q in zip(va,yv,p):oof_rows.append({'feature_set':group,'gene_index':int(i),'target':int(t),'probability':float(q),'outer_fold':k})
  model_predictions[group]=pred_oof
 # Recompute existing model metrics from the same fixed gene OOF predictions; they had fold-based early stopping.
 old=pd.read_csv(TR/'predictions/development_oof_predictions.csv')
 old_summary=[];old_pred={};phase5fit=pd.read_csv(TR/'metrics/supervised_fold_seed_metrics.csv')
 for name,g in old.groupby('model'):
  g=g.sort_values('gene_index'); ix=g.gene_index.to_numpy(int); pp=g.probability.to_numpy(float); yy=g.target.to_numpy(int)
  old_pred[name]={int(i):float(p) for i,p in zip(ix,pp)}
  m=score_metrics(yy,pp); fits=phase5fit[phase5fit.model.eq(name)]
  old_summary.append({'model':name,'feature_set':'Phase5 saved inputs','protocol':'same 3 folds; validation fold also governed early stopping','mean_fold_ap':float(fits.average_precision.mean()),'fold_ap_sd':float(fits.average_precision.std(ddof=1)),**m})
 # Add an exact fixed Phase 5 LR reference under the nested outer protocol for clean within-family comparison.
 fixed_rows=[];fixed_oof=[];x=df[FEATS['expression_plus_network']].to_numpy(float)
 for k in (1,2,3):
  va=dev[fold[dev]==k];tr=dev[fold[dev]!=k]
  yt=np.asarray([ymap[int(i)] for i in tr]);yv=np.asarray([ymap[int(i)] for i in va])
  model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=42))])
  model.fit(x[tr],yt);p=model.predict_proba(x[va])[:,1];fixed_rows.append({'outer_fold':k,**score_metrics(yv,p)})
  fixed_oof.extend({'gene_index':int(i),'target':int(t),'probability':float(q),'outer_fold':k} for i,t,q in zip(va,yv,p))
 fdf=pd.DataFrame(fixed_oof);fm=score_metrics(fdf.target.to_numpy(int),fdf.probability.to_numpy(float));
 outer=pd.DataFrame(outer_rows);outer.to_csv(OUT/'metrics/nested_outer_fold_metrics.csv',index=False)
 pd.DataFrame(search_rows).to_csv(OUT/'metrics/inner_search_results.csv',index=False)
 oofdf=pd.DataFrame(oof_rows);oofdf.to_csv(OUT/'predictions/feature_ablation_oof_predictions.csv',index=False)
 comparison=[]
 for group,g in oofdf.groupby('feature_set'):
  m=score_metrics(g.target.to_numpy(int),g.probability.to_numpy(float));aps=[average_precision_score(gg.target,gg.probability) for _,gg in g.groupby('outer_fold')]
  comparison.append({'model':f'Nested LR: {group}','feature_set':group,'protocol':'nested 3-fold outer CV; inner 3-fold AP search','mean_fold_ap':float(np.mean(aps)),'fold_ap_sd':float(np.std(aps,ddof=1)),**m})
 comparison.append({'model':'Nested LR: Phase5 fixed C=1 balanced','feature_set':'expression_plus_network','protocol':'nested outer folds; fixed Phase5 regularization','mean_fold_ap':float(np.mean([r['average_precision'] for r in fixed_rows])),'fold_ap_sd':float(np.std([r['average_precision'] for r in fixed_rows],ddof=1)),**fm})
 # Properly nested feature-family selection: choose features/config from inner CV only, then score outer fold.
 selected_oof=[]; selected_fold_records=[]
 for k in (1,2,3):
  candidates=[r for r in outer_rows if int(r['outer_fold'])==k]
  chosen=max(candidates,key=lambda r:r['best_inner_ap']); group=chosen['feature_set']
  selected_fold_records.append({'outer_fold':k,'selected_feature_set':group,'inner_cv_average_precision':chosen['best_inner_ap'],'best_C':chosen['best_C'],'best_class_weight':chosen['best_class_weight']})
  for i in dev[fold[dev]==k]: selected_oof.append({'gene_index':int(i),'target':ymap[int(i)],'probability':model_predictions[group][int(i)],'outer_fold':k,'selected_feature_set':group})
 selected_df=pd.DataFrame(selected_oof);selected_df.to_csv(OUT/'predictions/nested_selected_logistic_oof_predictions.csv',index=False)
 pd.DataFrame(selected_fold_records).to_csv(OUT/'metrics/nested_feature_selection_by_fold.csv',index=False)
 sm=score_metrics(selected_df.target.to_numpy(int),selected_df.probability.to_numpy(float));saps=[average_precision_score(g.target,g.probability) for _,g in selected_df.groupby('outer_fold')]
 comparison.append({'model':'Nested LR: inner-selected feature set','feature_set':'selected within each outer training fold','protocol':'inner CV selects feature group/C/weight; outer fold untouched until scoring','mean_fold_ap':float(np.mean(saps)),'fold_ap_sd':float(np.std(saps,ddof=1)),**sm})
 comparison += old_summary
 cdf=pd.DataFrame(comparison).sort_values('mean_fold_ap',ascending=False);cdf.to_csv(OUT/'metrics/model_comparison.csv',index=False)
 # Choose final logistic feature group only from development CV; keep the feature-selection CV explicit.
 final_group_search=[]; final_models={}
 for group,columns in FEATS.items():
  pipe=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(solver='liblinear',max_iter=2000,random_state=42))])
  gs=GridSearchCV(pipe,params,scoring='average_precision',cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=779),n_jobs=1,refit=True,return_train_score=True).fit(df[columns].to_numpy(float)[dev],labels)
  final_group_search.append({'feature_set':group,'mean_inner_cv_ap':float(gs.best_score_),'best_params':gs.best_params_});final_models[group]=gs
 final_group_search.sort(key=lambda r:r['mean_inner_cv_ap'],reverse=True)
 bestgroup=str(final_group_search[0]['feature_set']);xbest=df[FEATS[bestgroup]].to_numpy(float);finalgs=final_models[bestgroup]
 joblib.dump(finalgs.best_estimator_,OUT/'checkpoints/optimized_logistic_pipeline.joblib')
 (OUT/'configs/optimized_model.json').write_text(json.dumps({'model':'LogisticRegression','feature_set':bestgroup,'features':FEATS[bestgroup],'best_inner_params':finalgs.best_params_,'best_inner_cv_ap':float(finalgs.best_score_),'selection':'highest full-development inner-CV AP among expression/network/combined LR ablations','outer_protocol':'reuse fixed Phase5 development folds; inner stratified 3-fold hyperparameter selection; heldout labels excluded from training and selection','grid':{'C':params['model__C'],'class_weight':[None,'balanced']},'threshold':0.5,'preprocessing':'StandardScaler fit within each inner training fold by Pipeline; final scaler fit on all development labels only'},indent=2),encoding='utf-8')
 # Model-ready candidate scores from optimized pipeline; predictions are not ground truth.
 p_all=finalgs.best_estimator_.predict_proba(xbest)[:,1]
 pred=pd.DataFrame({'gene_id':df.entrez.astype(str),'gene_symbol':df.gene_symbol,'ensembl_gene_id':df.ensembl_gene_id_versioned,'partition':df.partition,'original_label':np.where(mask,target.astype(object),None),'prediction_probability':p_all,'predicted_label_at_0_5':(p_all>=.5).astype(int),'model':'optimized_logistic_'+bestgroup,'label_source':np.where(mask,np.where(target==1,'VERIFIED_POSITIVE','VERIFIED_EXPLICIT_ZERO'),np.where(p_all>=.5,'PREDICTED_POSITIVE','PREDICTED_NEGATIVE'))})
 pred.to_csv(OUT/'predictions/all_gene_predictions_optimized_logistic.csv',index=False)
 # Plots: same dev OOF protocol within nested LR ablations and saved Phase5 comparator scores.
 fig,ax=plt.subplots(figsize=(7,6))
 for group,g in oofdf.groupby('feature_set'):
  fpr,tpr,_=roc_curve(g.target,g.probability);ax.plot(fpr,tpr,label=f"{group} (AUC={auc(fpr,tpr):.3f})")
 ax.plot([0,1],[0,1],'k--',lw=1);ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Nested development ROC');ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/roc_development_nested.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,6))
 for group,g in oofdf.groupby('feature_set'):
  pr,re,_=precision_recall_curve(g.target,g.probability);ax.plot(re,pr,label=f"{group} (AP={average_precision_score(g.target,g.probability):.3f})")
 ax.set(xlabel='Recall',ylabel='Precision',title='Nested development precision-recall');ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/precision_recall_development_nested.png',dpi=160);plt.close(fig)
 # Confusion matrix, pooled outer OOF, threshold was fixed in advance at 0.5.
 bestoof=selected_df;cm=confusion_matrix(bestoof.target,(bestoof.probability>=.5).astype(int),labels=[0,1]);pd.DataFrame(cm,index=['actual_0','actual_1'],columns=['predicted_0','predicted_1']).to_csv(OUT/'metrics/selected_nested_confusion_matrix.csv')
 fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap='Blues');ax.set_xticks([0,1],['Predicted 0','Predicted 1']);ax.set_yticks([0,1],['Actual 0','Actual 1']);
 for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
 ax.set_title('Nested OOF confusion matrix: inner-selected logistic pipeline');fig.tight_layout();fig.savefig(OUT/'plots/selected_confusion_matrix.png',dpi=160);plt.close(fig)
 (OUT/'metrics/optimization_summary.json').write_text(json.dumps({'nested_feature_selected_logistic_oof_metrics':sm,'nested_selected_fold_ap_mean':float(np.mean(saps)),'selected_feature_groups_by_outer_fold':selected_fold_records,'final_logistic_feature_group_by_full_development_inner_cv':bestgroup,'final_logistic_inner_cv_search':final_group_search,'all_gene_count':len(df),'development_labeled_count':len(dev),'heldout_labels_used_for_fit_or_selection':False,'unknown_genes_used_as_negative':False,'external_labels_added':0,'label_source_audit':'local NCG only for target; COSMIC excluded from training/tuning; local reference directory contained NCBI crosswalk, TCGA metadata, STRING aliases, no additional independent gene-level label table'},indent=2),encoding='utf-8')
 # Reproducibility record and report handled below.
 print(json.dumps({'output':str(OUT),'nested_selected_logistic_metrics':sm,'comparison':cdf[['model','mean_fold_ap','average_precision','roc_auc','accuracy','balanced_accuracy','f1','mcc']].to_dict('records'),'final_logistic_feature_set':bestgroup,'final_params':finalgs.best_params_},indent=2))

if __name__=='__main__':main()
