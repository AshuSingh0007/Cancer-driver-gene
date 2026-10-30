"""Integrity checks for a completed Phase 4 transformer output directory."""
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=Path(sys.argv[1]).resolve()
pred=pd.read_csv(RUN/'predictions/all_gene_scores.csv')
oof=pd.read_csv(RUN/'predictions/development_oof_predictions.csv')
split=pd.read_csv(RUN/'predictions/split_manifest.csv')
cfg=json.loads((RUN/'metrics/phase4_summary.json').read_text())
repro=json.loads((RUN/'metrics/checkpoint_prediction_reproduction.json').read_text())
statuses={str(k):int(v) for k,v in pred.oncogene_status.value_counts().to_dict().items()}
dev=set(split.loc[split.partition.eq('development'),'gene_index'])
held=set(split.loc[split.partition.eq('held_out_test'),'gene_index'])
unlab=set(split.loc[split.partition.eq('unlabeled'),'gene_index'])
checks={
 'all_gene_rows_3304':len(pred)==3304,
 'gene_index_unique_and_complete':pred.gene_index.is_unique and set(pred.gene_index)==set(range(3304)),
 'stable_ensembl_ids_complete':pred.ensembl_gene_id_versioned.notna().all(),
 'all_scores_finite_and_bounded':pred.score.notna().all() and pred.score.between(0,1).all(),
 'label_status_counts_expected':statuses=={'unknown':2717,'explicit_zero':333,'positive':254},
 'unknown_candidate_export_count':len(pd.read_csv(RUN/'predictions/ranked_unknown_candidates.csv'))==2717,
 'split_partitions_disjoint':not(dev&held or dev&unlab or held&unlab),
 'oof_rows_cover_development_once_per_model':all((oof[oof.model.eq(m)].gene_index.nunique()==len(dev) and set(oof[oof.model.eq(m)].gene_index)==dev) for m in oof.model.unique()),
 'heldout_scores_not_used_for_model_selection':cfg['heldout_evaluation'].startswith('not repeated'),
 'checkpoint_reproduction_tolerance':repro['within_tolerance_1e-6'],
 'unknown_labels_not_used_in_loss':cfg['unknown_labels_used_in_loss'] is False,
 'heldout_target_not_read_during_selection':json.loads((RUN/'configs/experiment.json').read_text())['heldout_labels_accessed'] is False,
}
assert all(checks.values()),checks
out={'status':'passed','checks_passed':int(sum(bool(v) for v in checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'label_status_counts':statuses}
(RUN/'metrics/data_integrity_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
