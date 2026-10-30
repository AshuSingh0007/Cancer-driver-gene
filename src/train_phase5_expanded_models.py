"""Train comparable supervised references on the expanded gene universe."""
from __future__ import annotations
import json,random,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,balanced_accuracy_score,matthews_corrcoef,confusion_matrix
from graph_models import normalized_sparse_adjacency
from phase4_sparse_transformer import SparseGraphTransformer
from phase5_models import ExpandedGCN

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[17,42,2025];MAX_EPOCHS=25;PATIENCE=5
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def metric(y,p):
 z=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'precision':float(precision_score(y,z,zero_division=0)),'recall':float(recall_score(y,z,zero_division=0)),'f1':float(f1_score(y,z,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,z)),'mcc':float(matthews_corrcoef(y,z)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def main():
 torch.set_num_threads(2);t0=time.perf_counter();run=Path(sys.argv[1]).resolve();data=run/'data'
 if (run/'checkpoints').exists() and any((run/'checkpoints').glob('*.pt')):
  raise FileExistsError('Phase 5 fold checkpoints already exist; use a fresh dataset run directory to avoid overwriting them.')
 for d in ['configs','metrics','logs','checkpoints','predictions','plots']:(run/d).mkdir(exist_ok=True)
 # Deliberately load no annotation columns. Index target values only at development train/validation indices.
 use=['gene_index','entrez','ensembl_gene_id_versioned','gene_symbol','mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive','log_degree_ge700','log_weighted_strength_ge700','partition','development_fold']
 f=pd.read_csv(data/'expanded_gene_features_and_labels.csv',usecols=use).sort_values('gene_index').reset_index(drop=True)
 expr=f[['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive']].to_numpy(np.float32)
 topo=f[['log_degree_ge700','log_weighted_strength_ge700']].to_numpy(np.float32)
 both=np.column_stack([expr,topo]).astype(np.float32)
 y=np.load(data/'target_values.npy',mmap_mode='r',allow_pickle=False)
 dev=f.loc[f.partition.eq('development'),'gene_index'].to_numpy(dtype=int,copy=True)
 held=f.loc[f.partition.eq('held_out_test'),'gene_index'].to_numpy(dtype=int,copy=True)
 unl=f.loc[f.partition.eq('unlabeled'),'gene_index'].to_numpy(dtype=int,copy=True)
 extra=f.loc[f.partition.eq('additional_verified_development'),'gene_index'].to_numpy(dtype=int,copy=True)
 fold_arr=f.development_fold.to_numpy(dtype=int);folds={k:dev[fold_arr[dev]==k] for k in (1,2,3)}
 assert sum(len(v) for v in folds.values())==len(dev) and set(dev).isdisjoint(held) and not set(unl)&set(held)
 arr=np.load(data/'graph_edge_index.npz');ei=torch.tensor(arr['edge_index'].astype(np.int64));ew=torch.tensor(arr['edge_weight'].astype(np.float32));n=len(f)
 assert ei.shape[1]==2*int(json.loads((run/'audit/phase5_dataset_audit.json').read_text())['STRING_graph']['canonical_gene_edges'])
 # Sparse normalized graph from bidirectional score-thresholded edges.
 adj=normalized_sparse_adjacency(ei,ew,n)
 (run/'configs/training.json').write_text(json.dumps({'models':['logistic_expression_topology','expanded_gcn','expanded_sparse_graph_transformer'],'seeds':SEEDS,'folds':[1,2,3],'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'development_genes':len(dev),'extra_verified_training_genes':len(extra),'heldout_gene_count_not_used_during_search':len(held),'unlabeled_candidates':len(unl),'graph_protocol':'transductive; graph/features visible at all nodes; heldout labels never indexed during development training or model selection','feature_sets':{'logistic':'3 TCGA-BRCA expression summaries + 2 STRING topology summaries','expanded_gcn':'2 STRING topology summaries','expanded_sparse_graph_transformer':'2 STRING topology summaries'},'class_weighting':{'logistic':'balanced','expanded_gcn':'none (Phase 3 selected reference loss)','expanded_sparse_graph_transformer':'training fold negative/positive ratio'},'device':'cpu','available_ram_gib_start':psutil.virtual_memory().available/1024**3},indent=2),encoding='utf8')
 # (model, seed, fold, node) scores; only development OOF positions are metrics-bearing.
 predarr={name:np.full((len(SEEDS),3,n),np.nan,dtype=np.float32) for name in ['expanded_gcn','expanded_sparse_graph_transformer']}
 logitarr=np.full((1,3,n),np.nan,dtype=np.float32); rows=[]; logs=[]; unknown_preds={name:[] for name in ['logistic_expression_topology','expanded_gcn','expanded_sparse_graph_transformer']}
 for fold in (1,2,3):
  va=folds[fold];tr=np.concatenate([folds[k] for k in (1,2,3) if k!=fold]+([extra] if len(extra) else []));yt=np.array(y[tr],dtype=np.float32,copy=True);yv=np.array(y[va],dtype=np.int8,copy=True)
  sc=StandardScaler().fit(both[tr]);xall=torch.tensor(sc.transform(both).astype(np.float32));lr=LogisticRegression(class_weight='balanced',max_iter=500,solver='lbfgs',C=1.0)
  t=time.perf_counter();lr.fit(sc.transform(both[tr]),yt.astype(int));lp=lr.predict_proba(sc.transform(both))[:,1].astype(np.float32);logitarr[0,fold-1]=lp
  row={'model':'logistic_expression_topology','seed':42,'fold':fold,'best_epoch':0,'parameters':int(lr.coef_.size+lr.intercept_.size),'training_seconds':time.perf_counter()-t,'train_n':len(tr),'validation_n':len(va),**metric(yv,lp[va])};rows.append(row)
  logistic_fold_ap=row['average_precision']
  unknown_preds['logistic_expression_topology'].append(lp[unl]);
  pd.to_pickle({'model':lr,'scaler_mean':sc.mean_,'scaler_scale':sc.scale_},run/f'checkpoints/logistic_fold{fold}.pkl')
  seed_models={name:[] for name in predarr}
  for seed_idx,seed in enumerate(SEEDS):
   for name in predarr:
    runseed=seed+fold*101+2;seed_all(runseed)
    st=StandardScaler().fit(topo[tr]);xt=torch.tensor(st.transform(topo).astype(np.float32))
    if name=='expanded_gcn':model=ExpandedGCN(2,32,2,.30);opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.001)
    else:model=SparseGraphTransformer(2,hidden=32,dropout=.25);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.005)
    pw=1.0 if name=='expanded_gcn' else max(1.,(len(tr)-int(yt.sum()))/max(int(yt.sum()),1))
    ytens=torch.tensor(yt);best=-1.;state=None;bestep=0;stale=0;log=[];t=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
     model.train();opt.zero_grad(set_to_none=True)
     lg=model(xt,adj) if name=='expanded_gcn' else model(xt,ei,ew)
     loss=torch.nn.functional.binary_cross_entropy_with_logits(lg[tr],ytens,pos_weight=torch.tensor(pw));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
     model.eval()
     with torch.no_grad(): pp=torch.sigmoid(model(xt,adj) if name=='expanded_gcn' else model(xt,ei,ew)).numpy()
     ap=float(average_precision_score(yv,pp[va]));log.append({'epoch':epoch,'train_bce':float(loss.detach()),'validation_ap':ap})
     if ap>best+1e-6:best=ap;bestep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
     else:stale+=1
     if stale>=PATIENCE:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():pp=torch.sigmoid(model(xt,adj) if name=='expanded_gcn' else model(xt,ei,ew)).numpy().astype(np.float32)
    predarr[name][seed_idx,fold-1]=pp;unknown_preds[name].append(pp[unl]);seed_models[name].append(pp)
    torch.save({'state_dict':state,'model':name,'seed':runseed,'fold':fold,'best_epoch':bestep,'feature_mean':st.mean_.tolist(),'feature_scale':st.scale_.tolist(),'positive_weight':pw},run/f'checkpoints/{name}_seed{seed}_fold{fold}.pt')
    pd.DataFrame(log).to_csv(run/f'logs/{name}_seed{seed}_fold{fold}.csv',index=False)
    row={'model':name,'seed':seed,'fold':fold,'best_epoch':bestep,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-t,'train_n':len(tr),'validation_n':len(va),'positive_weight':pw,**metric(yv,pp[va])};rows.append(row)
    print(f'{name} seed={seed} fold={fold} AP={best:.4f} epoch={bestep} sec={row["training_seconds"]:.1f}',flush=True)
  print(f'logistic fold={fold} AP={logistic_fold_ap:.4f}',flush=True)
 # Export metrics/predictions based only on development OOF predictions.
 df=pd.DataFrame(rows);df.to_csv(run/'metrics/supervised_fold_seed_metrics.csv',index=False)
 summary=[];oof=[]
 for name in ['logistic_expression_topology','expanded_gcn','expanded_sparse_graph_transformer']:
  devp=[]
  for i in dev:
   if name=='logistic_expression_topology': vals=[float(logitarr[0,fold-1,i]) for fold in (1,2,3) if i in folds[fold]]
   else: vals=[float(predarr[name][s,fold-1,i]) for s in range(3) for fold in (1,2,3) if i in folds[fold]]
   assert len(vals)>0
   p=float(np.mean(vals));devp.append(p);oof.append({'model':name,'gene_index':int(i),'entrez':str(f.loc[f.gene_index.eq(i),'entrez'].iloc[0]),'target':int(y[i]),'probability':p,'fold':int(fold_arr[i])})
  yd=np.array(y[dev],dtype=np.int8);devp=np.asarray(devp);dfm=df[df.model.eq(name)]
  summary.append({'model':name,'development_fold_ap_mean':float(dfm.average_precision.mean()),'development_fold_ap_sd':float(dfm.average_precision.std(ddof=1)),'pooled_oof_ap':float(average_precision_score(yd,devp)),'pooled_oof_roc_auc':float(roc_auc_score(yd,devp)),'fold_count':int(dfm.fold.nunique()),'fit_count':int(len(dfm)),'parameters':int(dfm.parameters.iloc[0]),'median_best_epoch':int(dfm.best_epoch.median())})
 pd.DataFrame(oof).to_csv(run/'predictions/development_oof_predictions.csv',index=False)
 pd.DataFrame(summary).sort_values('development_fold_ap_mean',ascending=False).to_csv(run/'metrics/supervised_summary.csv',index=False)
 # All trained base-model predictions for the unlabeled universe support stable
 # pseudo-label auditing; no unknown labels are assumed.
 outrows=[]
 for name,arrs in unknown_preds.items():
  mat=np.stack(arrs).astype(np.float32);mean=mat.mean(axis=0);sd=mat.std(axis=0,ddof=1)
  for j,i in enumerate(unl):outrows.append({'model':name,'gene_index':int(i),'mean_probability':float(mean[j]),'sd_probability_across_fold_seed_models':float(sd[j]),'n_models':int(mat.shape[0])})
 pd.DataFrame(outrows).to_csv(run/'predictions/unlabeled_model_ensemble.csv',index=False)
 # Compact full prediction tensor is needed to generate fold-local pseudo labels
 # without accidentally using validation labels to train their pseudo-labeler.
 np.savez_compressed(run/'metrics/fold_model_predictions.npz',expanded_gcn=predarr['expanded_gcn'],expanded_sparse_graph_transformer=predarr['expanded_sparse_graph_transformer'],logistic=logitarr)
 # Freeze one model using development-CV selection; final score export is not a test evaluation.
 bestname=str(pd.DataFrame(summary).sort_values('development_fold_ap_mean',ascending=False).iloc[0].model)
 cfg={'selected_model':bestname,'selected_by':'highest development fold AP mean among expanded Phase 5 models','epochs':max(1,int(df[df.model.eq(bestname)].best_epoch.median())),'seed':3042}
 if bestname=='logistic_expression_topology':
  sc=StandardScaler().fit(both[dev]);final=LogisticRegression(class_weight='balanced',max_iter=500,solver='lbfgs',C=1.0).fit(sc.transform(both[dev]),np.array(y[dev],int));prob=final.predict_proba(sc.transform(both))[:,1];pd.to_pickle({'model':final,'scaler_mean':sc.mean_,'scaler_scale':sc.scale_},run/'checkpoints/final_selected.pkl')
 else:
  st=StandardScaler().fit(topo[dev]);xt=torch.tensor(st.transform(topo).astype(np.float32));seed_all(cfg['seed'])
  final=ExpandedGCN(2,32,2,.30) if bestname=='expanded_gcn' else SparseGraphTransformer(2,hidden=32,dropout=.25)
  opt=torch.optim.AdamW(final.parameters(),lr=.008 if bestname=='expanded_gcn' else .003,weight_decay=.001 if bestname=='expanded_gcn' else .005);ydev=torch.tensor(np.array(y[dev],np.float32,copy=True));pw=1. if bestname=='expanded_gcn' else max(1.,(len(dev)-int(ydev.sum()))/max(int(ydev.sum()),1))
  for _ in range(cfg['epochs']):
   final.train();opt.zero_grad(set_to_none=True);lg=final(xt,adj) if bestname=='expanded_gcn' else final(xt,ei,ew);loss=torch.nn.functional.binary_cross_entropy_with_logits(lg[dev],ydev,pos_weight=torch.tensor(pw));loss.backward();torch.nn.utils.clip_grad_norm_(final.parameters(),2);opt.step()
  final.eval()
  with torch.no_grad():prob=torch.sigmoid(final(xt,adj) if bestname=='expanded_gcn' else final(xt,ei,ew)).numpy()
  torch.save({'state_dict':final.state_dict(),'model':bestname,'seed':cfg['seed'],'fit_epochs':cfg['epochs'],'feature_mean':st.mean_.tolist(),'feature_scale':st.scale_.tolist(),'positive_weight':pw},run/'checkpoints/final_selected.pt')
 # Load label columns only after model selection/final checkpoint freeze for annotations.
 ann=pd.read_csv(data/'expanded_gene_features_and_labels.csv',usecols=['gene_index','label_status','target_value','label_provenance'])
 score=f[['gene_index','entrez','ensembl_gene_id_versioned','gene_symbol','mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive','string_degree_ge700','string_weighted_strength_ge700','partition']].copy();score['score']=prob;score=score.merge(ann,on='gene_index',validate='one_to_one');score['score_interpretation']='exploratory ranking score; not a confirmed driver probability';score.to_csv(run/'predictions/all_gene_scores_supervised.csv',index=False)
 score[score.label_status.eq('unknown')].sort_values('score',ascending=False).to_csv(run/'predictions/ranked_unknown_candidates_supervised.csv',index=False)
 report={'status':'completed','selected_model':bestname,'selection_rule':'highest development mean fold AP among three expanded supervised models','selected_fit_epochs':cfg['epochs'],'final_checkpoint':str((run/'checkpoints/final_selected.pt').relative_to(ROOT)) if bestname!='logistic_expression_topology' else str((run/'checkpoints/final_selected.pkl').relative_to(ROOT)),'development_genes':len(dev),'heldout_genes_not_evaluated':len(held),'heldout_labels_read_during_selection':False,'unknown_targets_used_in_loss':False,'unlabeled_gene_count':len(unl),'models':summary,'available_ram_gib_at_start':json.loads((run/'configs/training.json').read_text())['available_ram_gib_start'],'available_ram_gib_end':psutil.virtual_memory().available/1024**3,'elapsed_seconds':time.perf_counter()-t0,'test_evaluation_repeated':False,'training_protocol':'Old Phase 2 development folds are unchanged. The previously exposed heldout genes are excluded from training and model selection; no final holdout metric was computed in Phase 5.'}
 (run/'metrics/supervised_training_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps({'run':str(run.relative_to(ROOT)),'selected_model':bestname,'models':summary,'elapsed_seconds':report['elapsed_seconds'],'free_ram_gib_end':report['available_ram_gib_end']},indent=2),flush=True)

if __name__=='__main__':main()
