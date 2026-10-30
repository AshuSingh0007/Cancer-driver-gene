"""Recreate Phase 5 OOF metrics from saved fold checkpoints without retraining."""
from __future__ import annotations
import json,sys,time,random
from pathlib import Path
import numpy as np,pandas as pd,torch,psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,balanced_accuracy_score,matthews_corrcoef,confusion_matrix
from graph_models import normalized_sparse_adjacency
from phase4_sparse_transformer import SparseGraphTransformer
from phase5_models import ExpandedGCN

ROOT=Path(__file__).resolve().parents[1];SEEDS=[17,42,2025]
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def met(y,p):
 z=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'precision':float(precision_score(y,z,zero_division=0)),'recall':float(recall_score(y,z,zero_division=0)),'f1':float(f1_score(y,z,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,z)),'mcc':float(matthews_corrcoef(y,z)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def main():
 torch.set_num_threads(2);t0=time.perf_counter();base=Path(sys.argv[1]).resolve();src=base/'checkpoints';logs=base/'logs'
 out=base/('training_replay_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(exist_ok=False)
 for d in ['configs','metrics','checkpoints','predictions','plots']:(out/d).mkdir()
 ddir=base/'data';cols=['gene_index','entrez','ensembl_gene_id_versioned','gene_symbol','mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive','string_degree_ge700','string_weighted_strength_ge700','log_degree_ge700','log_weighted_strength_ge700','partition','development_fold']
 f=pd.read_csv(ddir/'expanded_gene_features_and_labels.csv',usecols=cols).sort_values('gene_index').reset_index(drop=True);n=len(f)
 expr=f[['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive']].to_numpy(np.float32);topo=f[['log_degree_ge700','log_weighted_strength_ge700']].to_numpy(np.float32);both=np.column_stack([expr,topo]).astype(np.float32)
 target=np.load(ddir/'target_values.npy',mmap_mode='r',allow_pickle=False);dev=f.loc[f.partition.eq('development'),'gene_index'].to_numpy(dtype=int,copy=True);held=f.loc[f.partition.eq('held_out_test'),'gene_index'].to_numpy(dtype=int,copy=True);unl=f.loc[f.partition.eq('unlabeled'),'gene_index'].to_numpy(dtype=int,copy=True);extra=f.loc[f.partition.eq('additional_verified_development'),'gene_index'].to_numpy(dtype=int,copy=True)
 foldarr=f.development_fold.to_numpy(int);folds={k:dev[foldarr[dev]==k] for k in (1,2,3)}
 a=np.load(ddir/'graph_edge_index.npz');ei=torch.tensor(a['edge_index'].astype(np.int64));ew=torch.tensor(a['edge_weight'].astype(np.float32));adj=normalized_sparse_adjacency(ei,ew,n)
 predarr={m:np.full((3,3,n),np.nan,np.float32) for m in ['expanded_gcn','expanded_sparse_graph_transformer']};larr=np.full((1,3,n),np.nan,np.float32);rows=[]
 for fold in (1,2,3):
  va=folds[fold];tr=np.concatenate([folds[k] for k in (1,2,3) if k!=fold]+([extra] if len(extra) else []));yv=np.array(target[va],np.int8,copy=True);ytr=np.array(target[tr],int,copy=True)
  pack=pd.read_pickle(src/f'logistic_fold{fold}.pkl');lr=pack['model'];lm=np.asarray(pack['scaler_mean']);ls=np.asarray(pack['scaler_scale']);lp=lr.predict_proba((both-lm)/ls)[:,1].astype(np.float32);larr[0,fold-1]=lp
  rows.append({'model':'logistic_expression_topology','seed':42,'fold':fold,'best_epoch':0,'parameters':int(lr.coef_.size+lr.intercept_.size),'training_seconds':np.nan,'train_n':len(tr),'validation_n':len(va),**met(yv,lp[va])})
  for si,seed in enumerate(SEEDS):
   for name in ('expanded_gcn','expanded_sparse_graph_transformer'):
    pth=src/f'{name}_seed{seed}_fold{fold}.pt';ck=torch.load(pth,map_location='cpu',weights_only=True)
    mean=np.asarray(ck['feature_mean'],np.float32);scale=np.asarray(ck['feature_scale'],np.float32);x=torch.tensor(((topo-mean)/scale).astype(np.float32))
    model=ExpandedGCN(2,32,2,.30) if name=='expanded_gcn' else SparseGraphTransformer(2,hidden=32,dropout=.25)
    model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad():pp=torch.sigmoid(model(x,adj) if name=='expanded_gcn' else model(x,ei,ew)).numpy().astype(np.float32)
    predarr[name][si,fold-1]=pp
    log=pd.read_csv(logs/f'{name}_seed{seed}_fold{fold}.csv');time_fit=float(0 if len(log)==0 else np.nan)
    rows.append({'model':name,'seed':seed,'fold':fold,'best_epoch':int(ck['best_epoch']),'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time_fit,'train_n':len(tr),'validation_n':len(va),'positive_weight':float(ck['positive_weight']),**met(yv,pp[va])})
    del model,x,pp
   print(f'replayed fold {fold} seed {seed}',flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'metrics/supervised_fold_seed_metrics.csv',index=False);oof=[];summ=[]
 for name in ['logistic_expression_topology','expanded_gcn','expanded_sparse_graph_transformer']:
  yp=[]
  for i in dev:
   fold=int(foldarr[i]);vals=[float(larr[0,fold-1,i])] if name.startswith('logistic') else [float(predarr[name][s,fold-1,i]) for s in range(3)]
   p=float(np.mean(vals));yp.append(p);oof.append({'model':name,'gene_index':int(i),'entrez':str(f.at[int(i),'entrez']),'target':int(target[i]),'probability':p,'fold':fold})
  yd=np.asarray(target[dev],int);yp=np.asarray(yp);dm=df[df.model.eq(name)]
  summ.append({'model':name,'development_fold_ap_mean':float(dm.average_precision.mean()),'development_fold_ap_sd':float(dm.average_precision.std(ddof=1)),'pooled_oof_ap':float(average_precision_score(yd,yp)),'pooled_oof_roc_auc':float(roc_auc_score(yd,yp)),'fit_count':int(len(dm)),'parameters':int(dm.parameters.iloc[0]),'median_best_epoch':int(dm.best_epoch.median())})
 pd.DataFrame(oof).to_csv(out/'predictions/development_oof_predictions.csv',index=False);sdf=pd.DataFrame(summ).sort_values('development_fold_ap_mean',ascending=False);sdf.to_csv(out/'metrics/supervised_summary.csv',index=False)
 np.savez_compressed(out/'metrics/fold_model_predictions.npz',expanded_gcn=predarr['expanded_gcn'],expanded_sparse_graph_transformer=predarr['expanded_sparse_graph_transformer'],logistic=larr)
 # Save a cross-fold/seed prediction summary for the previously unlabeled candidates.
 rows=[]
 for name in ('logistic_expression_topology','expanded_gcn','expanded_sparse_graph_transformer'):
  mat=np.stack([larr[0,f-1,unl] for f in (1,2,3)]) if name.startswith('logistic') else predarr[name][:,:,unl].reshape(-1,len(unl))
  avg=mat.mean(0);sd=mat.std(0,ddof=1)
  rows.extend({'model':name,'gene_index':int(g),'mean_probability':float(avg[j]),'sd_probability_across_fold_seed_models':float(sd[j]),'n_models':int(mat.shape[0])} for j,g in enumerate(unl))
 pd.DataFrame(rows).to_csv(out/'predictions/unlabeled_model_ensemble.csv',index=False)
 # Select among supervised methods on development folds only and refit once.
 winner=str(sdf.iloc[0].model);epochs=max(1,int(sdf.iloc[0].median_best_epoch));fitidx=np.concatenate([dev,extra]) if len(extra) else dev
 ck_path=out/'checkpoints/final_selected.pt'
 if winner=='logistic_expression_topology':
  sc=StandardScaler().fit(both[fitidx]);model=LogisticRegression(class_weight='balanced',max_iter=500,solver='lbfgs',C=1.).fit(sc.transform(both[fitidx]),np.asarray(target[fitidx],int));prob=model.predict_proba(sc.transform(both))[:,1];pd.to_pickle({'model':model,'mean':sc.mean_,'scale':sc.scale_},ck_path.with_suffix('.pkl'))
 else:
  seed_all(3042);sc=StandardScaler().fit(topo[fitidx]);x=torch.tensor(sc.transform(topo).astype(np.float32));model=ExpandedGCN(2,32,2,.30) if winner=='expanded_gcn' else SparseGraphTransformer(2,hidden=32,dropout=.25);opt=torch.optim.AdamW(model.parameters(),lr=.008 if winner=='expanded_gcn' else .003,weight_decay=.001 if winner=='expanded_gcn' else .005);yy=torch.tensor(np.array(target[fitidx],np.float32,copy=True));pw=1. if winner=='expanded_gcn' else max(1.,(len(fitidx)-int(yy.sum()))/max(int(yy.sum()),1))
  for _ in range(epochs):
   model.train();opt.zero_grad(set_to_none=True);lg=model(x,adj) if winner=='expanded_gcn' else model(x,ei,ew);loss=torch.nn.functional.binary_cross_entropy_with_logits(lg[fitidx],yy,pos_weight=torch.tensor(pw));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  model.eval()
  with torch.no_grad():prob=torch.sigmoid(model(x,adj) if winner=='expanded_gcn' else model(x,ei,ew)).numpy()
  torch.save({'state_dict':model.state_dict(),'model':winner,'seed':3042,'fit_epochs':epochs,'feature_mean':sc.mean_.tolist(),'feature_scale':sc.scale_.tolist(),'positive_weight':pw},ck_path)
 # Annotation fields are read only after development-based selection is frozen.
 ann=pd.read_csv(ddir/'expanded_gene_features_and_labels.csv',usecols=['gene_index','label_status','target_value','label_provenance'])
 scores=f[['gene_index','entrez','ensembl_gene_id_versioned','gene_symbol','mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive','string_degree_ge700','string_weighted_strength_ge700','partition']].copy();scores['score']=prob;scores=scores.merge(ann,on='gene_index',validate='one_to_one');scores['score_interpretation']='exploratory; not confirmed driver probability';scores.to_csv(out/'predictions/all_gene_scores_supervised.csv',index=False);scores[scores.label_status.eq('unknown')].sort_values('score',ascending=False).to_csv(out/'predictions/ranked_unknown_candidates_supervised.csv',index=False)
 summ={'status':'completed_from_saved_fold_checkpoints','selected_model':winner,'selected_fit_epochs':epochs,'selection_rule':'highest mean fold AP among logistic, GCN, and sparse graph Transformer; development only','models':summ,'development_genes':len(dev),'heldout_genes':len(held),'heldout_labels_read_during_selection':False,'heldout_evaluation_repeated':False,'unknown_genes':len(unl),'unknown_targets_used_for_loss':False,'available_ram_gib_at_replay_start':psutil.virtual_memory().available/1024**3,'available_ram_gib_at_end':psutil.virtual_memory().available/1024**3,'replay_and_final_refit_elapsed_seconds':time.perf_counter()-t0,'training_checkpoint_source':'../checkpoints from the completed Phase 5 fold/seed training run; replay does not overwrite them'}
 (out/'metrics/supervised_training_summary.json').write_text(json.dumps(summ,indent=2),encoding='utf8');(out/'configs/replay_protocol.json').write_text(json.dumps({'folds':[1,2,3],'seeds':SEEDS,'saved_fold_checkpoints_replayed':True,'target_index_access':'development training/validation genes only; heldout labels are not indexed','final_refit':'selected method refit on development labels after CV selection','raw_dataset_path':str(ddir.relative_to(ROOT))},indent=2),encoding='utf8')
 print(json.dumps({'output':str(out.relative_to(ROOT)),'selected_model':winner,'summary':summ['models'],'elapsed':summ['replay_and_final_refit_elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
