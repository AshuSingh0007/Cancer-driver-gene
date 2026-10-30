"""Assertions for aligned Ensembl/Entrez features, graph edges, and targets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = max((p for p in (ROOT / "data/processed").glob("integrated_model_inputs_*") if (p / "integrated_input_summary.json").exists()), key=lambda p: p.stat().st_mtime)


def main():
    frame = pd.read_csv(INPUT / "gene_features_and_labels.csv")
    x = np.load(INPUT / "expression_features_raw.npy", allow_pickle=False)
    y = np.load(INPUT / "target_values.npy", allow_pickle=False)
    mask = np.load(INPUT / "target_mask.npy", allow_pickle=False)
    graph = np.load(INPUT / "graph_edge_index.npz")
    summary = json.loads((INPUT / "integrated_input_summary.json").read_text(encoding="utf-8"))
    ei, ew = graph["edge_index"], graph["edge_weight"]
    checks = {
        "contiguous_unique_gene_index": frame.gene_index.tolist() == list(range(len(frame))) and frame.entrez.is_unique and frame.ensembl_gene_id_versioned.is_unique,
        "feature_matrix_row_alignment": x.shape == (len(frame), 3) and np.isfinite(x).all(),
        "label_mask_alignment": len(y) == len(mask) == len(frame) and np.array_equal(mask, np.isfinite(frame.oncogene_label.to_numpy(dtype=float))),
        "unknowns_are_masked_not_negative": np.all(y[~mask] == -1) and frame.loc[~mask, "oncogene_status"].eq("unknown").all(),
        "known_target_status_consistency": frame.loc[mask, "oncogene_status"].isin(["positive", "explicit_zero"]).all(),
        "graph_index_bounds_and_no_self_edges": ei.shape[0] == 2 and ei.min() >= 0 and ei.max() < len(frame) and not np.any(ei[0] == ei[1]),
        "graph_is_symmetric_with_matching_weights": ei.shape[1] % 2 == 0 and np.array_equal(ei[:, :ei.shape[1]//2][::-1], ei[:, ei.shape[1]//2:]) and np.allclose(ew[:len(ew)//2], ew[len(ew)//2:]),
        "summary_matches_files": summary["gene_count"] == len(frame) and summary["known_label_count"] == int(mask.sum()) and summary["directed_message_edges"] == ei.shape[1],
    }
    checks = {k: bool(v) for k, v in checks.items()}
    result = {"input_directory": str(INPUT.relative_to(ROOT)), "checks": checks, "passed": all(checks.values())}
    path = INPUT / "alignment_integrity_checks.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
