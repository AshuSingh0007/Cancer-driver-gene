"""Export Phase 3 ranked scores with stable identifiers and graph context."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=Path(sys.argv[1]).resolve()
INPUT=ROOT/"data/processed/integrated_model_inputs_20261028_155809"
score=pd.read_csv(RUN/"predictions/all_gene_scores.csv")
mat=np.load(INPUT/"graph_edge_index.npz",allow_pickle=False)
ei=mat["edge_index"]; ew=mat["edge_weight"]; n=len(score)
degree=np.bincount(ei[0],minlength=n)
strength=np.bincount(ei[0],weights=ew,minlength=n)
score["ppi_degree_score700"]=degree
score["ppi_weighted_strength_score700"]=strength
score["target_status_interpretation"]=score["oncogene_status"].map({"positive":"known_positive","explicit_zero":"known_source_zero_not_biological_negative","unknown":"unknown_unlabeled"})
score["prediction_interpretation"]="exploratory score; not a confirmed cancer driver"
score.to_csv(RUN/"predictions/all_gene_scores_with_ppi_context.csv",index=False)
score.loc[score.oncogene_status.eq("unknown")].sort_values("score",ascending=False).to_csv(RUN/"predictions/ranked_unknown_candidates_with_ppi_context.csv",index=False)
score.loc[score.partition.eq("held_out_test")].to_csv(RUN/"predictions/final_heldout_with_ppi_context.csv",index=False)
print(f"Wrote stable-ID scores for {len(score)} nodes and {int(score.oncogene_status.eq('unknown').sum())} unknowns")
