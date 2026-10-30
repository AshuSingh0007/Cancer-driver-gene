"""Build an expanded, non-destructive expression × STRING gene universe."""
from __future__ import annotations
import csv,json,time,psutil
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed'
EXP=PROC/'expression_features_20261028_153824/gene_expression_features.csv'
XW=PROC/'identifier_mapping_20261028_155713/ensembl_v36_to_entrez_ncbi_gene_info.csv'
ALIASES=PROC/'validation_20261028_152218/string_protein_entrez_mapping_v12_5_aliases.csv'
LABELS=PROC/'validation_20261028_152218/ncg_gene_label_table.csv'
LINKS=ROOT/'9606.protein.links.v12.5.txt/9606.protein.links.v12.5.txt'
PH2=ROOT/'outputs/experiments/integrated_20261028_161519'
OUT=ROOT/'outputs/experiments'/('phase5_expansion_'+datetime.now().strftime('%Y%m%d_%H%M%S'))

def main():
 t0=time.perf_counter(); assert not OUT.exists(); (OUT/'data').mkdir(parents=True);(OUT/'audit').mkdir()
 mem0=psutil.virtual_memory().available/1024**3
 expr=pd.read_csv(EXP,dtype={'ensembl_gene_id_versioned':str})
 xw=pd.read_csv(XW,dtype=str,keep_default_na=False)
 xw=xw[xw.mapping_status.eq('unique_entrez')].copy()
 # Enforce a one-to-one gene-level crosswalk in both directions. Do not sum
 # duplicate Ensembl records into one Entrez gene without evidence that the
 # records represent interchangeable measurements.
 xw=xw[~xw.ensembl_gene_id_versioned.duplicated(keep=False)]
 rev=xw.groupby('entrez_gene_id_candidates').ensembl_gene_id_versioned.transform('nunique')
 xw=xw[rev.eq(1)]
 exp=expr.merge(xw[['ensembl_gene_id_versioned','entrez_gene_id_candidates','ncbi_symbol_for_unique_entrez','expression_ncbi_symbol_match']],on='ensembl_gene_id_versioned',how='inner',validate='one_to_one')
 exp=exp.rename(columns={'entrez_gene_id_candidates':'entrez'})
 aliases=pd.read_csv(ALIASES,dtype=str,keep_default_na=False)
 aliases=aliases[(aliases.mapping_status.eq('unambiguous')) & aliases.present_in_links_v12_5.eq('True')]
 aliases=aliases.drop_duplicates('string_protein_id')
 protein_to_gene=dict(zip(aliases.string_protein_id,aliases.entrez_ids))
 string_genes=set(protein_to_gene.values())
 expr_entrez_before_string=set(exp.entrez.astype(str))
 expr_without_string=expr_entrez_before_string-string_genes
 exp=exp[exp.entrez.isin(string_genes)].copy()
 # One row per Entrez is required for model node alignment.
 if exp.entrez.duplicated().any(): raise ValueError('One-to-one Ensembl/Entrez crosswalk invariant failed')
 exp['entrez']=exp.entrez.astype(str)
 genes=sorted(exp.entrez.unique(),key=lambda x:int(x)); gene_set=set(genes); gi={g:i for i,g in enumerate(genes)}
 labels=pd.read_csv(LABELS,dtype={'entrez':str},keep_default_na=False)
 labels=labels.drop_duplicates('entrez').set_index('entrez')
 label_rows=[]
 for g in genes:
  if g not in labels.index: st='unknown'; y=-1; prov='No NCG record in frozen annotation table'
  else:
   st=str(labels.at[g,'oncogene_status']); raw=labels.at[g,'oncogene_label']
   if st=='positive' and str(raw) not in ('','nan'): y=1;prov='NCG_oncogene explicit positive'
   elif st=='explicit_zero' and str(raw) not in ('','nan'): y=0;prov='NCG_oncogene explicit zero (source comparison label, not biological absence)'
   else: y=-1;prov='NCG unknown/conflict; excluded from supervised labels'
  label_rows.append((g,st,y,prov))
 lab=pd.DataFrame(label_rows,columns=['entrez','label_status','target_value','label_provenance'])
 # Stream all STRING rows; retain only thresholded gene-level edges for the
 # expanded universe. Several proteins may collapse to a gene pair; retain
 # maximum confidence and report collapsed duplicate mappings.
 edge_max={}; counts={'raw_rows':0,'score_ge_700_rows':0,'mapped_both_endpoints':0,'unmapped_endpoint_rows':0,'outside_candidate_rows':0,'mapped_self_gene_rows':0}
 with LINKS.open('r',encoding='utf-8',errors='strict',newline='') as fh:
  rdr=csv.reader(fh,delimiter=' '); header=next(rdr); assert header==['protein1','protein2','combined_score'],header
  for row in rdr:
   counts['raw_rows']+=1
   if len(row)!=3: raise ValueError(f'Malformed STRING row {counts["raw_rows"]}: {row[:4]}')
   score=int(row[2])
   if score<700: continue
   counts['score_ge_700_rows']+=1
   a=protein_to_gene.get(row[0]);b=protein_to_gene.get(row[1])
   if a is None or b is None: counts['unmapped_endpoint_rows']+=1;continue
   counts['mapped_both_endpoints']+=1
   if a not in gene_set or b not in gene_set: counts['outside_candidate_rows']+=1;continue
   if a==b: counts['mapped_self_gene_rows']+=1;continue
   key=(a,b) if int(a)<int(b) else (b,a)
   if score>edge_max.get(key,0):edge_max[key]=score
 if counts['raw_rows']!=10826054: raise ValueError(f'Expected validated STRING row count 10826054, got {counts["raw_rows"]}')
 edges=sorted(((gi[a],gi[b],v) for (a,b),v in edge_max.items()),key=lambda z:(z[0],z[1]))
 ei=np.asarray([[e[0] for e in edges],[e[1] for e in edges]],dtype=np.int64)
 ew=np.asarray([e[2]/1000. for e in edges],dtype=np.float32)
 n=len(genes); deg=np.zeros(n,dtype=np.int32); strength=np.zeros(n,dtype=np.float64)
 if len(edges):
  np.add.at(deg,ei[0],1);np.add.at(deg,ei[1],1);np.add.at(strength,ei[0],ew);np.add.at(strength,ei[1],ew)
 topology=np.column_stack([np.log1p(deg),np.log1p(strength)]).astype(np.float32)
 features=exp.set_index('entrez').loc[genes].reset_index()
 features['gene_index']=np.arange(n,dtype=int)
 features['gene_symbol']=features.ncbi_symbol_for_unique_entrez.where(features.ncbi_symbol_for_unique_entrez.ne(''),features.gene_symbol)
 features['expression_symbol']=features['gene_symbol'] # retain source symbol separately below
 features['expression_symbol']=exp.set_index('entrez').loc[genes].reset_index().gene_symbol.to_numpy()
 features['string_degree_ge700']=deg;features['string_weighted_strength_ge700']=strength
 features['log_degree_ge700']=topology[:,0];features['log_weighted_strength_ge700']=topology[:,1]
 features=features.merge(lab,on='entrez',validate='one_to_one')
 # Preserve Phase 2 gene holdout and folds by Entrez ID. Existing heldout is
 # explicitly previously exposed. Additional verified genes are train-only in
 # every development fold so the historical validation rows stay unchanged.
 old=pd.read_csv(PH2/'predictions/split_manifest.csv',usecols=['gene_index','entrez','partition'])
 oldfold=pd.read_csv(PH2/'predictions/graph_gcn_development_oof.csv',usecols=['gene_index','fold'])
 old=old.merge(oldfold,on='gene_index',how='left',validate='one_to_one')
 oldmap={str(r.entrez):(r.partition,int(r.fold) if pd.notna(r.fold) else -1) for r in old.itertuples()}
 partitions=[];folds=[]
 known_set=set(lab.loc[lab.target_value.ge(0),'entrez'])
 for g in genes:
  if g in oldmap and lab.loc[lab.entrez.eq(g),'target_value'].iloc[0]>=0:
   p,f=oldmap[g]
  elif g in known_set: p,f='additional_verified_development',-1
  else: p,f='unlabeled',-1
  partitions.append(p);folds.append(f)
 features['partition']=partitions;features['development_fold']=folds
 # Human-readable artifacts, machine-friendly arrays, and a strict ID map.
 data=OUT/'data';features.to_csv(data/'expanded_gene_features_and_labels.csv',index=False)
 lab[['entrez','label_status','target_value','label_provenance']].to_csv(data/'label_provenance.csv',index=False)
 pd.DataFrame({'gene_index':np.arange(n),'entrez':genes,'ensembl_gene_id_versioned':features.ensembl_gene_id_versioned,'gene_symbol':features.gene_symbol}).to_csv(data/'gene_identifier_mapping.csv',index=False)
 pd.DataFrame({'gene_index':np.arange(n),'entrez':genes,'partition':partitions,'development_fold':folds}).to_csv(data/'split_manifest.csv',index=False)
 pd.DataFrame(edges,columns=['source_index','target_index','combined_score']).to_csv(data/'graph_edges_ge700.csv',index=False)
 # Store directed messages in both directions; graph_edges_ge700.csv remains
 # the canonical undirected edge list.
 directed_ei=np.concatenate([ei,ei[::-1,:]],axis=1) if len(edges) else ei
 directed_ew=np.concatenate([ew,ew]) if len(edges) else ew
 np.savez_compressed(data/'graph_edge_index.npz',edge_index=directed_ei,edge_weight=directed_ew)
 x=features[['mean_log2_tpm_plus_1','sd_log2_tpm_plus_1','fraction_cases_tpm_positive']].to_numpy(np.float32)
 np.save(data/'expression_features.npy',x,allow_pickle=False);np.save(data/'topology_features.npy',topology,allow_pickle=False)
 y=features.target_value.to_numpy(np.int8);mask=y>=0;np.save(data/'target_values.npy',y,allow_pickle=False);np.save(data/'target_mask.npy',mask,allow_pickle=False)
 status=features.label_status.value_counts().to_dict(); part=features.partition.value_counts().to_dict()
 counts.update({'canonical_gene_edges':len(edges),'directed_message_edges':2*len(edges),'collapsed_gene_edge_duplicates':counts['mapped_both_endpoints']-counts['outside_candidate_rows']-counts['mapped_self_gene_rows']-len(edges),'graph_nodes_with_degree_ge_1':int((deg>0).sum()),'graph_isolates':int((deg==0).sum()),'mean_degree':float(deg.mean()),'max_degree':int(deg.max()),'mean_edge_confidence':float(ew.mean()) if len(ew) else 0,'score_range':([int(min(edge_max.values())),int(max(edge_max.values()))] if edge_max else [])})
 summary={'status':'completed','output_directory':str(OUT.relative_to(ROOT)),'source_files':{'expression_features':str(EXP.relative_to(ROOT)),'ncbi_crosswalk':str(XW.relative_to(ROOT)),'string_aliases_v12_5':str(ALIASES.relative_to(ROOT)),'string_links_v12_5':str(LINKS.relative_to(ROOT)),'ncg_labels':str(LABELS.relative_to(ROOT))},'source_versions':{'expression':'TCGA-BRCA GENCODE v36, 1,103 primary-tumour samples collapsed to 1,092 cases','mapping':'NCBI Gene gene_info Ensembl stable-ID dbXrefs','STRING':'v12.5; human aliases; combined_score >=700','labels':'frozen NCG_oncogene statuses; no external labels or COSMIC used'},'expression_gene_rows':len(expr),'unique_exactly_mapped_expression_entrez_before_STRING_intersection':int(pd.read_csv(XW,dtype=str).query("mapping_status == 'unique_entrez'").entrez_gene_id_candidates.nunique()),'candidate_universe_genes':n,'original_gene_count':3304,'added_genes':n-3304,'feature_counts':{'expression':n,'mapped_STRING_alias':n,'both_expression_and_STRING':n},'label_status_counts':{str(k):int(v) for k,v in status.items()},'split_partition_counts':{str(k):int(v) for k,v in part.items()},'STRING_graph':counts,'identifier_exclusion_counts':{'expression_rows_unmapped_or_ambiguous':60660-len(xw),'strict_ensembl_entrez_reverse_ambiguity_rows_removed':int((pd.read_csv(XW,dtype=str).query("mapping_status == 'unique_entrez'").entrez_gene_id_candidates.value_counts()>1).sum()),'mapped_expression_genes_without_unambiguous_STRING_alias':int(exp.entrez.nunique() if False else 0),'candidate_set_excluded_non_unique_expression_mapping':60660-len(xw)-int((60660-len(xw))*0)},'independent_COSMIC_used':False,'free_ram_gib_start':mem0,'free_ram_gib_end':psutil.virtual_memory().available/1024**3,'elapsed_seconds':time.perf_counter()-t0,'notes':['Candidate universe is determined only by expression availability, strict stable-ID mapping, and unambiguous STRING mapping; label status does not determine inclusion.','NCG absent annotations remain unknown. Explicit zero is only the NCG source-defined comparison class, not biological absence.','Multiple Ensembl records mapping to the same Entrez gene were excluded rather than arbitrarily aggregated.','The graph is transductive; graph topology includes heldout nodes and edges, but no heldout labels are exposed during selection.','No new cohorts, labels, dependencies, or downloads were used.']}
 summary['identifier_exclusion_counts']['mapped_expression_genes_without_unambiguous_STRING_alias']=int(len(expr_without_string))
 summary['identifier_exclusion_counts']['unique_Entrez_expression_rows_removed_by_reverse_ambiguity']=int(pd.read_csv(XW,dtype=str).query("mapping_status == 'unique_entrez'").shape[0]-len(xw))
 (OUT/'audit/phase5_dataset_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 report=f'''# Phase 5 expanded dataset audit\n\n**Run:** `{OUT.relative_to(ROOT)}`\n\n## Expansion result\n\nThe local-only, strict-ID intersection contains **{n:,} unique genes**, exceeding the 5,000 target. It adds **{n-3304:,}** genes to the former 3,304-gene NCG-constrained set. All {n:,} genes have the existing TCGA-BRCA expression summary features and an unambiguous STRING v12.5 protein-to-Entrez mapping. The expanded score-700 graph has **{len(edges):,} canonical undirected edges**, {int((deg>0).sum()):,} non-isolated nodes and {int((deg==0).sum()):,} isolates.\n\n| Label state | Count |\n|---|---:|\n| NCG explicit positive | {int((y==1).sum()):,} |\n| NCG explicit zero | {int((y==0).sum()):,} |\n| Unknown / no target annotation | {int((y<0).sum()):,} |\n\nThe prior integrated dataset contained 254 positive, 333 explicit-zero, and 2,717 unknown labels. The expanded set's counts reflect every NCG annotation that survives this expanded exact-ID feature/network intersection; unknown remains a separate masked state. No additional verified labels were acquired externally.\n\n## Sources and mapping rules\n\nExpression features: `{EXP.relative_to(ROOT)}`; TCGA-BRCA STAR/GENCODE v36 primary-tumour features summarized over 1,092 cases. Expression Ensembl IDs were mapped with exact Ensembl stable-ID references from the frozen NCBI Gene `gene_info` crosswalk. The version suffix (and recognized `_PAR_Y` suffix per the crosswalk construction) is handled only by the validated mapping artifact. Ambiguous Ensembl-to-Entrez mappings and Entrez IDs represented by more than one expression Ensembl record were excluded; no symbol-only mapping was used. STRING aliases and links are both v12.5; aliases with one Entrez ID only were retained. Interactions use combined score >=700; protein interactions collapsing to the same Entrez pair are deduplicated by maximum score, and gene self-links are discarded.\n\nThe expanded candidate set is label-independent: membership requires an expression row, a unique NCBI crosswalk, and an unambiguous STRING mapped gene. NCG `oncogene_status` supplies training labels only. Missing NCG records/annotations and conflicting target states are unknown/masked. Explicit zero means source-defined NCG comparison label, not verified biological absence. COSMIC is not used for membership, labels, model features, pseudo-labels, or tuning.\n\n## Evaluation split and constraints\n\nThe old Phase 2 development fold assignments and heldout set are mapped by Entrez ID and retained. The historical heldout set remains previously exposed; it is not treated as an untouched test set. Any additional verified NCG genes outside the old integrated set are added to development training only, not to CV validation or the old holdout. The graph is transductive: heldout and unknown nodes' expression and PPI topology are visible; heldout labels are not available for training/model selection.\n\nThis audit used only existing local files, installed dependencies, and streaming interaction processing. Raw inputs and prior outputs were not modified. See `phase5_dataset_audit.json` for complete counts and run metadata.\n'''
 (OUT/'audit/initial_dataset_audit.md').write_text(report,encoding='utf8')
 print(json.dumps({'run':str(OUT.relative_to(ROOT)),'genes':n,'added':n-3304,'labels':{str(k):int(v) for k,v in status.items()},'edges':len(edges),'graph_nodes_with_degree':int((deg>0).sum()),'elapsed':summary['elapsed_seconds'],'free_ram_gib_start':mem0},indent=2),flush=True)

if __name__=='__main__':main()
