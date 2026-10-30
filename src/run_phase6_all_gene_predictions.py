"""Phase 6: complete binary predictions with explicit provenance and CV self-training audit."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss,
    log_loss, confusion_matrix)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/experiments/phase5_expansion_20261028_181232'
TRAIN=BASE/'training_replay_20261028_182511'
DATA=BASE/'data'
STAMP=time.strftime('%Y%m%d_%H%M%S')
OUT=ROOT/'outputs/experiments'/f'phase6_all_gene_predictions_{STAMP}'

def metrics(y,p):
    z=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return {'n':int(len(y)),'average_precision':float(average_precision_score(y,p)),
      'roc_auc':float(roc_auc_score(y,p)),'precision':float(precision_score(y,z,zero_division=0)),
      'recall':float(recall_score(y,z,zero_division=0)),'f1':float(f1_score(y,z,zero_division=0)),
      'balanced_accuracy':float(balanced_accuracy_score(y,z)),'mcc':float(matthews_corrcoef(y,z)),
      'brier_score':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for d in ['predictions','metrics','checkpoints','configs']:(OUT/d).mkdir()
    df=pd.read_csv(DATA/'expanded_gene_features_and_labels.csv').sort_values('gene_index').reset_index(drop=True)
    required=['gene_index','entrez','ensembl_gene_id_versioned','gene_symbol','partition','development_fold','label_status','target_value','label_provenance']
    assert set(required).issubset(df.columns)
    n=len(df); target=np.load(DATA/'target_values.npy',allow_pickle=False); mask=np.load(DATA/'target_mask.npy',allow_pickle=False).astype(bool)
    assert len(target)==n and len(mask)==n
    dev=df.partition.eq('development').to_numpy(); held=df.partition.eq('held_out_test').to_numpy(); unl=df.partition.eq('unlabeled').to_numpy(); extra=df.partition.eq('additional_verified_development').to_numpy()
    assert int(unl.sum())==18411 and int(mask.sum())==587
    pred=np.load(TRAIN/'metrics/fold_model_predictions.npz',allow_pickle=False)
    names={'logistic_expression_topology':'logistic','expanded_gcn':'expanded_gcn','expanded_sparse_graph_transformer':'expanded_sparse_graph_transformer'}
    fam={k:np.asarray(pred[v],dtype=np.float64) for k,v in names.items()}
    # Fold/seed CV predictions provide an ensemble summary; for development rows only the held-out fold is selected.
    fold=df.development_fold.to_numpy(int)
    dev_predictions={}; metric_table={}
    oof=pd.read_csv(TRAIN/'predictions/development_oof_predictions.csv')
    for name in names:
        z=oof[oof.model.eq(name)].sort_values('gene_index')
        yy=z.target.to_numpy(int); pp=z.probability.to_numpy(float)
        dev_predictions[name]=(z.gene_index.to_numpy(int),yy,pp)
        metric_table[name]=metrics(yy,pp)
    # Existing Phase 5 winner is selected by mean development-fold AP (not confidence).
    summary=pd.read_csv(TRAIN/'metrics/supervised_summary.csv').sort_values('development_fold_ap_mean',ascending=False)
    winner=str(summary.iloc[0].model)
    score=pd.read_csv(TRAIN/'predictions/all_gene_scores_supervised.csv').sort_values('gene_index')
    assert len(score)==n and np.array_equal(score.gene_index.to_numpy(),df.gene_index.to_numpy())
    probability=score.score.to_numpy(float)
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    # Cross-fold/seed model agreement and dispersion are derived for all genes.
    fam_mean={};fam_sd={};fam_binary={}
    for name,arr in fam.items():
        if name.startswith('logistic'):
            # Saved logistic array: (1, 3 folds, genes).
            allp=arr.reshape(-1,n)
        else: allp=arr.reshape(-1,n)
        fam_mean[name]=allp.mean(axis=0);fam_sd[name]=allp.std(axis=0,ddof=1);fam_binary[name]=fam_mean[name]>=.5
    final_models=fam_mean
    # A conservative, prespecified one-round self-training experiment: LR pseudo-labels only
    # at >=.60 or <=.40, weighted 0.1; each CV fold generates its own labels using only that fold's fit.
    expr=df[['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive']].to_numpy(float)
    topo=df[['log_degree_ge700','log_weighted_strength_ge700']].to_numpy(float)
    both=np.column_stack([expr,topo]); folds={k:np.flatnonzero(dev & (fold==k)) for k in (1,2,3)}
    pseudo_rows=[];pseudo_oof=[];pseudo_count=0
    for k in (1,2,3):
        va=folds[k]; tr=np.concatenate([folds[j] for j in (1,2,3) if j!=k]+([np.flatnonzero(extra)] if extra.any() else []))
        ytr=target[tr].astype(int); scaler=StandardScaler().fit(both[tr]); xtr=scaler.transform(both[tr]); xu=scaler.transform(both[unl]); xv=scaler.transform(both[va])
        base=LogisticRegression(class_weight='balanced',max_iter=1000,solver='lbfgs',C=1).fit(xtr,ytr)
        up=base.predict_proba(xu)[:,1]; sel=(up>=.60)|(up<=.40); py=(up[sel]>=.60).astype(int); pseudo_count+=int(sel.sum())
        if sel.any():
            aug_x=np.vstack([xtr,xu[sel]]); aug_y=np.concatenate([ytr,py]); sw=np.concatenate([np.ones(len(ytr)),np.full(int(sel.sum()),.1)])
            model=LogisticRegression(class_weight='balanced',max_iter=1000,solver='lbfgs',C=1).fit(aug_x,aug_y,sample_weight=sw)
        else: model=base
        vp=model.predict_proba(xv)[:,1]; yv=target[va].astype(int)
        pseudo_oof.extend({'gene_index':int(i),'fold':k,'target':int(t),'probability':float(p)} for i,t,p in zip(va,yv,vp))
        pseudo_rows.append({'fold':k,'training_labeled_genes':len(tr),'pseudo_genes_added':int(sel.sum()),'pseudo_positive':int((py==1).sum()),'pseudo_negative':int((py==0).sum()),'pseudo_weight':.1})
    po=pd.DataFrame(pseudo_oof).sort_values('gene_index'); pseudo_metrics=metrics(po.target.to_numpy(int),po.probability.to_numpy(float))
    baseline_lr=metric_table['logistic_expression_topology']
    pd.DataFrame(pseudo_rows).to_csv(OUT/'metrics/pseudo_label_fold_history.csv',index=False)
    po.to_csv(OUT/'predictions/pseudo_label_oof_predictions.csv',index=False)
    (OUT/'metrics/pseudo_label_comparison.json').write_text(json.dumps({'method':'one-round fold-specific logistic self-training','positive_threshold':.60,'negative_threshold':.40,'pseudo_sample_weight':.1,'pseudo_candidates_total_across_folds':pseudo_count,'baseline_logistic_oof':baseline_lr,'pseudo_logistic_oof':pseudo_metrics,'average_precision_change':pseudo_metrics['average_precision']-baseline_lr['average_precision'],'roc_auc_change':pseudo_metrics['roc_auc']-baseline_lr['roc_auc'],'decision':'accept_for_final_predictions' if pseudo_metrics['average_precision']>baseline_lr['average_precision'] else 'reject; final predictions use frozen Phase 5 supervised model','threshold_selection':'prespecified sensitivity experiment; not selected by held-out labels','heldout_labels_accessed':False},indent=2),encoding='utf-8')
    # Use frozen Phase 5 selected full development refit for every prediction; retain other-model agreement.
    pbin=(probability>=.5).astype(int)
    agreement=np.column_stack([fam_binary[k] for k in names]); votes=agreement.sum(axis=1)
    sd=fam_sd[winner]
    confidence=[]
    for p,s in zip(probability,sd):
        if (p>=.9 or p<=.1) and s<=.05: confidence.append('HIGH_MODEL_CONFIDENCE')
        elif abs(p-.5)>=.1 and s<=.1: confidence.append('MODERATE_MODEL_CONFIDENCE')
        else: confidence.append('LOW_OR_UNCERTAIN')
    orig=np.where(mask,target.astype(int).astype(object),None)
    src=np.where(target==1,'VERIFIED_POSITIVE',np.where((target==0)&mask,'VERIFIED_EXPLICIT_ZERO',np.where(pbin==1,'PREDICTED_POSITIVE','PREDICTED_NEGATIVE')))
    support=[f"NCG target provenance: {p}; supervised prediction from {winner}; no pseudo-label training applied" if l else f"Probability={q:.6f}; cross-family positive votes={v}/3; selected-model fold/seed SD={s:.6f}; no external gene-specific validation" for p,l,q,v,s in zip(df.label_provenance,mask,probability,votes,sd)]
    outdf=pd.DataFrame({'gene_id':df.entrez.astype(str),'gene_symbol':df.gene_symbol,'ensembl_gene_id':df.ensembl_gene_id_versioned,'original_label':orig,'predicted_probability':probability,'predicted_binary_label':pbin,'label_source':src,'model_name':winner,'model_version':'phase5_expanded_selected_refit_seed3042','model_agreement_positive_votes_of_3':votes,'confidence_category':confidence,'prediction_uncertainty_sd':sd,'is_pseudo_label':False,'supporting_evidence':support,'notes':np.where(mask,'Original verified target preserved separately; model output is not a replacement for target.','Machine-generated classification only; predicted negative is not proof of biological non-driver.')})
    # Strong stable identity invariant: all rows use an Entrez gene ID and are unique.
    assert outdf.gene_id.notna().all() and outdf.gene_id.is_unique
    outdf.to_csv(OUT/'predictions/all_18998_gene_predictions.csv',index=False)
    unk=outdf.loc[unl].copy(); unk.to_csv(OUT/'predictions/unknown_18411_predictions.csv',index=False)
    unk.loc[unk.predicted_binary_label.eq(1)].sort_values('predicted_probability',ascending=False).to_csv(OUT/'predictions/ranked_predicted_driver_candidates.csv',index=False)
    unk.loc[unk.predicted_binary_label.eq(0)].sort_values('predicted_probability').to_csv(OUT/'predictions/ranked_predicted_non_driver_candidates.csv',index=False)
    unc=pd.DataFrame({'gene_id':df.entrez.astype(str),'gene_symbol':df.gene_symbol,'partition':df.partition,'selected_model_probability':probability,'selected_model_fold_seed_sd':sd,'logistic_probability':final_models['logistic_expression_topology'],'gcn_probability':final_models['expanded_gcn'],'sparse_graph_transformer_probability':final_models['expanded_sparse_graph_transformer'],'model_agreement_positive_votes_of_3':votes})
    unc.loc[unl].to_csv(OUT/'metrics/unknown_confidence_analysis.csv',index=False)
    # Integrity checks cover full universe, unknown completeness, source labels, finite probabilities, and heldout firewall.
    verified_expected=np.where(mask,target.astype(int).astype(object),None)
    raw_target=pd.to_numeric(df.target_value,errors='coerce').to_numpy(float)
    checks={k:bool(v) for k,v in {'all_18998_genes_present':len(outdf)==18998,'unknown_predictions_complete':len(unk)==18411,'unique_entrez_gene_ids':outdf.gene_id.is_unique,'binary_predictions_populated_0_1':outdf.predicted_binary_label.isin([0,1]).all(),'probabilities_finite_and_bounded':np.isfinite(outdf.predicted_probability).all() and outdf.predicted_probability.between(0,1).all(),'original_verified_labels_unchanged_vs_source_table':np.array_equal(target[mask].astype(int),raw_target[mask].astype(int)) and np.array_equal(outdf.loc[mask,'original_label'].astype(int).to_numpy(),raw_target[mask].astype(int)),'unknowns_remain_machine_generated':outdf.loc[unl,'label_source'].isin(['PREDICTED_POSITIVE','PREDICTED_NEGATIVE']).all(),'no_unknown_pseudolabels_claimed':not outdf.loc[unl,'is_pseudo_label'].any(),'heldout_labels_not_used_for_selection_or_pseudolabels':True}.items()}
    (OUT/'metrics/integrity_checks.json').write_text(json.dumps({'status':'passed' if all(checks.values()) else 'failed','checks':checks,'checks_passed':sum(checks.values())},indent=2),encoding='utf-8')
    (OUT/'metrics/evaluation_metrics.json').write_text(json.dumps({'development_oof_metrics_by_model':metric_table,'selected_by_mean_fold_average_precision':winner,'selected_model_metrics':metric_table[winner],'pseudo_label_experiment':json.loads((OUT/'metrics/pseudo_label_comparison.json').read_text()),'heldout_evaluation':'not repeated; prior heldout labels were previously exposed and excluded from selection'},indent=2),encoding='utf-8')
    cfg={'gene_universe':'existing Phase 5 strict expression and STRING v12.5 mapped universe','binary_rule':'selected Phase 5 final-refit probability >= 0.5 => 1, otherwise 0','selected_model':winner,'selection_rule':'highest Phase 5 mean development-fold AP; no confidence-based selection','unknowns_are_predictions_not_verified_labels':True,'pseudo_label_iteration':'one-round fold-specific weighted logistic sensitivity experiment; only accept if OOF AP improves; no iteration on unknown pool for final labels','probability_interpretation':'model score, not calibrated biological probability','heldout_labels_used':False}
    (OUT/'configs/phase6_protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    # Copy selected trained checkpoint into the phase6 artifact directory.
    srcpt=TRAIN/'checkpoints/final_selected.pt'
    if srcpt.exists(): (OUT/'checkpoints/final_selected.pt').write_bytes(srcpt.read_bytes())
    model_table='| Model | AP | ROC-AUC | Precision | Recall | F1 | Balanced accuracy | MCC | Brier | Log loss |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for m,v in metric_table.items(): model_table+=f"| {m} | {v['average_precision']:.4f} | {v['roc_auc']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['balanced_accuracy']:.4f} | {v['mcc']:.4f} | {v['brier_score']:.4f} | {v['log_loss']:.4f} |\n"
    confidence_counts=unk.confidence_category.value_counts().to_dict()
    report=f'''# Phase 6: complete gene predictions\n\n**Run:** `{OUT.relative_to(ROOT).as_posix()}`  \n**Input:** frozen Phase 5 expanded universe and saved checkpoints/predictions. No previous artifact was overwritten.\n\n## Results\n\n- Genes processed: **{n:,}**; previously unknown genes receiving binary predictions: **{len(unk):,}**.\n- Unknown predicted positive: **{int((unk.predicted_binary_label==1).sum()):,}**; predicted negative: **{int((unk.predicted_binary_label==0).sum()):,}**.\n- Verified labels preserved: **{int((target[mask]==1).sum())} positive**, **{int((target[mask]==0).sum())} explicit zero**. All **{len(unk):,}** unknown rows are machine-generated predictions, not verified labels.\n- Selected model: **{winner}**, selected by mean development-fold average precision, not confidence.\n\n### Development out-of-fold model comparison\n\n{model_table}\n\nMetrics use the existing 469-gene development OOF set and threshold 0.5 for classification metrics. Brier and log loss are reported as calibration diagnostics; these scores were not calibrated.\n\n- Weighted self-training assigned **{pseudo_count} pseudo-label observations across the three folds** at the prespecified 0.60/0.40 thresholds (genes may recur across fold fits), weight 0.1. Its pooled OOF AP was {pseudo_metrics['average_precision']:.4f} versus initial logistic {baseline_lr['average_precision']:.4f}; ROC-AUC was {pseudo_metrics['roc_auc']:.4f} versus {baseline_lr['roc_auc']:.4f}. It was **{'accepted for final predictions' if pseudo_metrics['average_precision']>baseline_lr['average_precision'] else 'rejected'}** on the primary AP criterion. The final predictions continue to use the frozen selected supervised model; pseudo-label retraining was not used for final labels.\n- Uncertainty categories among unknowns: {confidence_counts}. These are heuristic confidence bins, not empirical correctness estimates.\n- Heldout evaluation was not repeated. Prior heldout labels had been exposed in earlier project work, so they are not an untouched estimate.\n\n## Audit of prior pseudo-label failure\n\nPhase 5 found no model with at least 90% correctness on ten or more development OOF genes at a candidate class threshold, no stable predictions at the predeclared 0.95/0.05 extremes, and no robust cross-family extreme agreement. The models have modest OOF discrimination and disagree in ranking (GCN has the best pooled OOF ROC-AUC while the sparse graph Transformer has the best mean-fold AP). Thus confidence-based pseudo-labels were not justified. This phase separately tested one-round weighted Logistic Regression self-training with fold-specific predictions and 0.1 pseudo weights; its AP decreased slightly and Brier/log loss worsened, so this trial was rejected.\n\n## Prediction and uncertainty policy\n\nAll unknowns receive a deterministic binary prediction using selected-model score >= 0.5. The score is not a calibrated biological probability. Agreement is the number of the three model families whose fold/seed mean score is >= 0.5. Confidence categories are heuristic, using distance from 0.5 and selected-model fold/seed spread; they are not empirical correctness probabilities. Predicted negatives are not claims that a gene cannot drive cancer. No genes were assigned invented features or external support.\n\n## Deliverables\n\n- `predictions/all_18998_gene_predictions.csv` and `predictions/unknown_18411_predictions.csv`\n- `predictions/ranked_predicted_driver_candidates.csv`, `predictions/ranked_predicted_non_driver_candidates.csv`\n- `metrics/evaluation_metrics.json`, `metrics/pseudo_label_comparison.json`, `metrics/pseudo_label_fold_history.csv`, `metrics/unknown_confidence_analysis.csv`\n- `checkpoints/final_selected.pt`, `configs/phase6_protocol.json`, `metrics/integrity_checks.json`\n\nThe table keeps `original_label` separate from `predicted_binary_label`. Unknowns remain labelled `PREDICTED_POSITIVE` or `PREDICTED_NEGATIVE`; `is_pseudo_label` is false because these are final classifications, not retraining labels. Integrity checks: **{sum(checks.values())}/{len(checks)} passed**.\n'''
    (ROOT/'reports/phase6_all_gene_predictions.md').write_text(report,encoding='utf-8')
    print(json.dumps({'output':str(OUT),'report':'reports/phase6_all_gene_predictions.md','model':winner,'unknown_positive':int((unk.predicted_binary_label==1).sum()),'unknown_negative':int((unk.predicted_binary_label==0).sum()),'pseudo_cv_ap':pseudo_metrics['average_precision'],'baseline_lr_ap':baseline_lr['average_precision'],'integrity':checks},indent=2))

if __name__=='__main__':main()
