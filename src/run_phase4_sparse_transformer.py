"""Development-only Phase 4 sparse graph Transformer comparison."""
from __future__ import annotations
import json, math, random, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch, networkx as nx, psutil
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from graph_models import NodeClassifier
from phase4_sparse_transformer import SparseGraphTransformer, SparseTransformerFusion

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data/processed/integrated_model_inputs_20261028_155809'
PHASE2=ROOT/'outputs/experiments/integrated_20261028_161519'
PHASE3=ROOT/'outputs/experiments/phase3_gcn_20261028_164054'
SEEDS=[17,42,2025]; EPOCHS=40; PATIENCE=8

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def scores(y,p):
    yh=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(average_precision=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),precision=float(precision_score(y,yh,zero_division=0)),recall=float(recall_score(y,yh,zero_division=0)),f1=float(f1_score(y,yh,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,yh)),mcc=float(matthews_corrcoef(y,yh)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main():
    torch.set_num_threads(2); started=time.perf_counter()
    out=ROOT/'outputs/experiments'/('phase4_transformer_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    for d in ['configs','metrics','logs','predictions','checkpoints','plots']: (out/d).mkdir(parents=True,exist_ok=False)
    # Load only feature inputs plus masked target array. Test targets are not indexed by this script.
    # During model selection read identifiers only; annotation columns are read
    # only after the development-selected checkpoint is frozen below.
    frame=pd.read_csv(INPUT/'gene_features_and_labels.csv',usecols=['gene_index','ensembl_gene_id_versioned','entrez','gene_symbol'])
    expr=np.load(INPUT/'expression_features_raw.npy',allow_pickle=False).astype(np.float32)
    dat=np.load(INPUT/'graph_edge_index.npz'); ei=dat['edge_index'].astype(np.int64); ew=dat['edge_weight'].astype(np.float32)
    n=len(expr); g=nx.Graph(); g.add_nodes_from(range(n))
    g.add_weighted_edges_from((int(a),int(b),float(w)) for a,b,w in zip(ei[0],ei[1],ew) if a<b)
    deg=dict(g.degree()); strength=dict(g.degree(weight='weight')); clust=nx.clustering(g,weight='weight')
    topo=np.asarray([[math.log1p(deg[i]),math.log1p(strength[i]),clust[i]] for i in range(n)],np.float32)
    target=np.load(INPUT/'target_values.npy',mmap_mode='r',allow_pickle=False)
    split=pd.read_csv(PHASE2/'predictions/split_manifest.csv',usecols=['gene_index','partition'])
    dev=split.loc[split.partition.eq('development'),'gene_index'].to_numpy(dtype=int,copy=True)
    foldtab=pd.read_csv(PHASE2/'predictions/graph_gcn_development_oof.csv',usecols=['gene_index','fold'])
    fmap=dict(zip(foldtab.gene_index.astype(int),foldtab.fold.astype(int)))
    folds={f:np.asarray([i for i in dev if fmap.get(int(i))==f],int) for f in (1,2,3)}
    assert sum(map(len,folds.values()))==len(dev) and not set(dev)&set(split.loc[split.partition.eq('held_out_test'),'gene_index'])
    edge=torch.tensor(ei,dtype=torch.long); weight=torch.tensor(ew,dtype=torch.float32)
    types=['sparse_graph_transformer','expression_sparse_transformer_fusion']
    (out/'configs/experiment.json').write_text(json.dumps({'models':types,'seeds':SEEDS,'fixed_folds':[1,2,3], 'max_epochs':EPOCHS,'patience':PATIENCE,'selection':'mean validation AP across fixed development folds and seeds','primary_metric':'average precision','device':'cpu','torch':torch.__version__,'cuda':torch.cuda.is_available(),'available_ram_gib_at_start':psutil.virtual_memory().available/1024**3,'heldout_labels_accessed':False,'transformer_note':'Sparse dot-product multi-head attention only along STRING edges; expression inputs are three summary statistics, not a sequence. Fusion uses an MLP expression branch.'},indent=2),encoding='utf8')
    folds_rows=[]; oof={name:{int(i):[] for i in dev} for name in types}; epochs_by={name:[] for name in types}; registry=[]
    for name in types:
      for seed in SEEDS:
       for fold in (1,2,3):
        va=folds[fold]; tr=np.concatenate([folds[k] for k in (1,2,3) if k!=fold]);
        se=StandardScaler().fit(expr[tr]); st=StandardScaler().fit(topo[tr]); xs=torch.from_numpy(se.transform(expr).astype(np.float32)); ts=torch.from_numpy(st.transform(topo).astype(np.float32))
        runseed=seed+fold*101+2; seed_all(runseed)
        model=SparseGraphTransformer(3) if name==types[0] else SparseTransformerFusion()
        model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.005)
        pos=int(np.asarray(target[tr]).sum()); pw=max(1.,(len(tr)-pos)/max(pos,1)); ytr=torch.tensor(np.array(target[tr],dtype=np.float32,copy=True))
        best=-1.; best_epoch=0; best_state=None; stale=0; log=[]; t0=time.perf_counter()
        for epoch in range(1,EPOCHS+1):
          model.train(); optimizer.zero_grad(set_to_none=True)
          logits=model(ts,edge,weight) if name==types[0] else model(xs,ts,edge,weight)
          loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[tr],ytr,pos_weight=torch.tensor(pw)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step()
          model.eval()
          with torch.no_grad():
            pred=torch.sigmoid(model(ts,edge,weight) if name==types[0] else model(xs,ts,edge,weight))[va].numpy()
          ap=float(average_precision_score(np.asarray(target[va]),pred)); log.append({'epoch':epoch,'train_bce':float(loss.detach()),'validation_ap':ap})
          if ap>best+1e-6:
            best=ap;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
          else: stale+=1
          if stale>=PATIENCE: break
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad(): fullp=torch.sigmoid(model(ts,edge,weight) if name==types[0] else model(xs,ts,edge,weight)).numpy()
        elapsed=time.perf_counter()-t0
        for i,p in zip(va,fullp[va]): oof[name][int(i)].append(float(p))
        stem=f'{name}_seed{seed}_fold{fold}'; torch.save({'state_dict':best_state,'name':name,'seed':runseed,'fold':fold,'best_epoch':best_epoch,'expr_mean':se.mean_.tolist(),'expr_scale':se.scale_.tolist(),'topo_mean':st.mean_.tolist(),'topo_scale':st.scale_.tolist()},out/f'checkpoints/{stem}.pt')
        pd.DataFrame(log).to_csv(out/f'logs/{stem}.csv',index=False)
        row={'model':name,'seed':seed,'fold':fold,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':elapsed,'train_n':len(tr),'validation_n':len(va),'validation_positive':int(np.asarray(target[va]).sum()),**scores(np.asarray(target[va]),fullp[va])};folds_rows.append(row);epochs_by[name].append(best_epoch)
        registry.append({'model':name,'seed':seed,'fold':fold,'status':'completed','heldout_used':False,'best_epoch':best_epoch,'seconds':elapsed})
        print(f'{name} seed={seed} fold={fold} AP={best:.4f} epoch={best_epoch} t={elapsed:.1f}s',flush=True)
    folds_df=pd.DataFrame(folds_rows);folds_df.to_csv(out/'metrics/dev_transformer_folds.csv',index=False)
    summary=[]; oofrows=[]
    for name in types:
      y=[];p=[]
      for i in dev:
        ps=oof[name][int(i)]; assert len(ps)==len(SEEDS); y.append(int(target[i]));p.append(float(np.mean(ps)));oofrows.append({'model':name,'gene_index':int(i),'probability':p[-1],'target':y[-1],'fold':fmap[int(i)]})
      df=folds_df[folds_df.model.eq(name)]
      summary.append({'model':name,'development_cv_ap_mean':float(df.average_precision.mean()),'development_cv_ap_sd':float(df.average_precision.std(ddof=1)),'pooled_seed_averaged_oof_ap':float(average_precision_score(y,p)),'pooled_seed_averaged_oof_roc_auc':float(roc_auc_score(y,p)),'median_best_epoch':int(np.median(epochs_by[name])),'parameters':int(df.parameters.iloc[0]),'mean_fit_seconds':float(df.training_seconds.mean())})
    pd.DataFrame(oofrows).to_csv(out/'predictions/development_oof_predictions.csv',index=False)
    new=pd.DataFrame(summary).sort_values('development_cv_ap_mean',ascending=False);new.to_csv(out/'metrics/new_model_summary.csv',index=False)
    old=pd.read_csv(PHASE2/'metrics/model_comparison.csv')[['model','cv_average_precision_mean','cv_average_precision_sd','cv_roc_auc_mean']].rename(columns={'cv_average_precision_mean':'development_cv_ap_mean','cv_average_precision_sd':'development_cv_ap_sd','cv_roc_auc_mean':'mean_fold_roc_auc'})
    old['source']='Phase 2 existing, same fixed folds, one seed; historical exposed holdout excluded from comparison'
    p3=pd.read_csv(PHASE3/'metrics/dev_search_summary.csv'); p3=p3.rename(columns={'configuration':'model','development_oof_ap_mean':'development_cv_ap_mean','development_oof_ap_sd':'development_cv_ap_sd','pooled_seed_averaged_oof_roc_auc':'mean_fold_roc_auc'})[['model','development_cv_ap_mean','development_cv_ap_sd','mean_fold_roc_auc']];p3['source']='Phase 3 existing GCN, same folds x 3 seeds'
    new['mean_fold_roc_auc']=new.pooled_seed_averaged_oof_roc_auc;new['source']='Phase 4; same folds x 3 seeds'
    pd.concat([old,p3,new],ignore_index=True,sort=False).sort_values('development_cv_ap_mean',ascending=False).to_csv(out/'metrics/model_comparison.csv',index=False)
    pd.DataFrame(registry).to_csv(out/'metrics/experiment_registry.csv',index=False)
    split.to_csv(out/'predictions/split_manifest.csv',index=False)
    # Select winner based on mean dev AP among newly trained models; no test labels/metrics are read.
    best_name=str(new.iloc[0].model); selected_epochs=max(1,int(np.median(epochs_by[best_name])))
    se=StandardScaler().fit(expr[dev]);st=StandardScaler().fit(topo[dev]);xs=torch.from_numpy(se.transform(expr).astype(np.float32));ts=torch.from_numpy(st.transform(topo).astype(np.float32))
    seed_all(3042);final=SparseGraphTransformer(3) if best_name==types[0] else SparseTransformerFusion();opt=torch.optim.AdamW(final.parameters(),lr=.003,weight_decay=.005)
    pos=int(np.asarray(target[dev]).sum());pw=max(1.,(len(dev)-pos)/max(pos,1));yt=torch.tensor(np.array(target[dev],dtype=np.float32,copy=True))
    for _ in range(selected_epochs):
      final.train();opt.zero_grad(set_to_none=True);lg=final(ts,edge,weight) if best_name==types[0] else final(xs,ts,edge,weight);loss=torch.nn.functional.binary_cross_entropy_with_logits(lg[dev],yt,pos_weight=torch.tensor(pw));loss.backward();torch.nn.utils.clip_grad_norm_(final.parameters(),2.);opt.step()
    final.eval()
    with torch.no_grad(): allp=torch.sigmoid(final(ts,edge,weight) if best_name==types[0] else final(xs,ts,edge,weight)).numpy()
    checkpoint=out/f'checkpoints/{best_name}_final.pt';torch.save({'state_dict':final.state_dict(),'name':best_name,'seed':3042,'fit_epochs':selected_epochs,'expr_mean':se.mean_.tolist(),'expr_scale':se.scale_.tolist(),'topo_mean':st.mean_.tolist(),'topo_scale':st.scale_.tolist()},checkpoint)
    annotations=pd.read_csv(INPUT/'gene_features_and_labels.csv',usecols=['gene_index','oncogene_status','oncogene_label','tsg_status','tsg_label'])
    pred=frame.merge(annotations,on='gene_index',validate='one_to_one');pred['score']=allp;pred['score_interpretation']='exploratory ranking score; not a confirmed driver probability';pred['partition']=pred.gene_index.map(dict(zip(split.gene_index,split.partition)));pred.to_csv(out/'predictions/all_gene_scores.csv',index=False)
    pred[pred.oncogene_status.eq('unknown')].sort_values('score',ascending=False).to_csv(out/'predictions/ranked_unknown_candidates.csv',index=False)
    summaryjson={'status':'completed','selected_new_model':best_name,'selection':'highest Phase 4 mean development fold AP across 3 fixed folds x 3 seeds; labels only accessed for development indices during fitting and selection; annotations loaded after checkpoint freeze','selected_fit_epochs':selected_epochs,'final_checkpoint':str(checkpoint.relative_to(ROOT)),'heldout_evaluation':'not repeated; historical holdout exposed in Phase 2 and Phase 3','phase2_reference_cv_ap':.5693408535721584,'phase3_reference_cv_ap':.5692474322452539,'new_models':summary,'elapsed_seconds':time.perf_counter()-started,'device':'cpu','available_ram_gib_start':json.loads((out/'configs/experiment.json').read_text())['available_ram_gib_at_start'],'unknown_labels_used_in_loss':False,'all_scores_generated_for':n,'target_access_protocol':'target_values is memory-mapped; only training and validation development indices are indexed until model freeze; label annotations are read after freeze for prediction-table annotation'}
    (out/'metrics/phase4_summary.json').write_text(json.dumps(summaryjson,indent=2),encoding='utf8')
    print(json.dumps({'output':str(out.relative_to(ROOT)),'winner':best_name,'new_models':summary,'elapsed_seconds':summaryjson['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__': main()
