"""Create the Phase 3 breast-evidence sensitivity cohort and task summary."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / ("phase3_task_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
VALIDATION = max((p for p in (ROOT / "data" / "processed").glob("validation_*") if (p / "ncg_gene_label_table.csv").exists()), key=lambda p: p.stat().st_mtime)

def status(series):
    values = set(pd.to_numeric(series, errors="coerce").dropna().astype(int))
    if values == {1}: return "positive", 1
    if values == {0}: return "explicit_zero", 0
    if values == {0, 1}: return "conflict", None
    return "unknown", None

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    ann = pd.read_csv(ROOT / "NCG_cancerdrivers_annotation_supporting_evidence.tsv", sep="\t", low_memory=False)
    site = ann.primary_site.astype("string").str.strip().str.casefold()
    cancer = ann.cancer_type.astype("string").str.strip().str.casefold()
    selected = site.eq("breast") | cancer.str.contains("breast", regex=False, na=False)
    rows = ann.loc[selected].copy()
    records=[]
    for entrez, g in rows.groupby("entrez", dropna=True):
        symbols=sorted(set(g.symbol.dropna().astype(str).str.strip().str.upper()))
        on_status, on_label=status(g.NCG_oncogene)
        tsg_status, tsg_label=status(g.NCG_tsg)
        records.append({"entrez":int(entrez),"symbol":";".join(symbols),"breast_evidence_rows":len(g),
            "primary_sites":";".join(sorted(set(g.primary_site.dropna().astype(str)))),
            "cancer_types":";".join(sorted(set(g.cancer_type.dropna().astype(str)))),
            "oncogene_status":on_status,"oncogene_label":on_label,"tsg_status":tsg_status,"tsg_label":tsg_label,
            "symbol_conflict":len(symbols)>1})
    labels=pd.DataFrame(records)
    labels.to_csv(OUT / "ncg_breast_evidence_gene_labels.csv",index=False)
    cosmic_path=ROOT / "Cosmic_CancerGeneCensus_Tsv_v104_GRCh38" / "Cosmic_CancerGeneCensus_v104_GRCh38.tsv" / "Cosmic_CancerGeneCensus_v104_GRCh38.tsv"
    cosmic=pd.read_csv(cosmic_path,sep="\t",dtype=str)
    c_map={r.GENE_SYMBOL.strip().upper():r for r in cosmic.itertuples(index=False)}
    comparisons=[]
    for r in labels.itertuples(index=False):
        c=c_map.get(r.symbol.split(";")[0])
        if c is None: continue
        comparisons.append({"entrez":r.entrez,"symbol":r.symbol,"ncg_breast_context_oncogene_status":r.oncogene_status,
            "cosmic_role_in_cancer":c.ROLE_IN_CANCER,"cosmic_tumour_types_somatic":c.TUMOUR_TYPES_SOMATIC,
            "interpretation":"Reference overlap only; not a training label or independent validation."})
    pd.DataFrame(comparisons).to_csv(OUT / "cosmic_reference_comparison.csv",index=False)
    summary={"prediction_unit":"gene","primary_task":"Predict the source-defined pan-cancer NCG_oncogene annotation among the 3,347 NCG cohort genes.",
        "primary_label_status_counts": {k:int(v) for k,v in pd.read_csv(VALIDATION / "ncg_gene_label_table.csv").oncogene_status.value_counts().items()},
        "secondary_breast_evidence_sensitivity_task":{"inclusion":"At least one NCG supporting-evidence row with primary_site exactly breast or cancer_type containing breast.",
            "evidence_rows":int(len(rows)),"candidate_genes":int(len(labels)),"oncogene_status_counts":{k:int(v) for k,v in labels.oncogene_status.value_counts().items()},
            "oncogene_supervised_n":int(labels.oncogene_label.notna().sum()),"oncogene_positive":int(labels.oncogene_label.eq(1).sum()),
            "oncogene_explicit_zero":int(labels.oncogene_label.eq(0).sum()),"oncogene_conflict":int(labels.oncogene_status.eq("conflict").sum()),
            "cosmic_symbol_overlap":int(len(comparisons))},
        "scope_warning":"Primary target is pan-cancer annotation prediction with a BRCA expression modality. It is not BRCA-specific driver discovery. Breast-evidence subset is a sensitivity cohort, not an independent validation set.",
        "unlabeled_policy":"Unknown NCG labels remain unknown and are not treated as negatives.",
        "cosmic_policy":"COSMIC CGC v104 GRCh38 is descriptive reference only; no role enters labels or features.",
        "sample_linkage":"Not available; no BRCA expression features joined or generated.",
        "output_directory":str(OUT.relative_to(ROOT))}
    (OUT / "phase3_task_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
