import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=Path(sys.argv[1]).resolve();data=BASE/'data';a=np.load(data/'graph_edge_index.npz');e=a['edge_index'];w=a['edge_weight'];half=e.shape[1]//2;edge=pd.read_csv(data/'graph_edges_ge700.csv');old=pd.read_csv(ROOT/'data/processed/integrated_model_inputs_20261028_155809/gene_features_and_labels.csv',usecols=['entrez']).entrez.astype(str);new=set(pd.read_csv(data/'gene_identifier_mapping.csv',usecols=['entrez']).entrez.astype(str))
checks={'directed_messages_are_exactly_double_canonical_edges':e.shape[1]==2*len(edge),'reverse_direction_present_for_each_edge':bool(np.array_equal(e[:,half:],e[::-1,:half])),'reverse_weights_are_equal':bool(np.array_equal(w[half:],w[:half])),'no_gene_self_edges':bool((edge.source_index!=edge.target_index).all()),'all_original_3304_genes_retained':set(old)<=new}
result={'status':'passed' if all(checks.values()) else 'failed','checks_passed':sum(checks.values()),'checks':checks}
(BASE/'audit/expanded_graph_integrity_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert all(checks.values())
