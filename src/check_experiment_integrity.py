"""Check integrated model inputs and completed experiment artifacts."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(
    (p for p in (ROOT / "outputs/experiments").glob("integrated_*")
     if (p / "metrics/experiment_summary.json").exists()), key=lambda p: p.stat().st_mtime)
INPUT = max((p for p in (ROOT / "data/processed").glob("integrated_model_inputs_*")
             if (p / "integrated_input_summary.json").exists()), key=lambda p: p.stat().st_mtime)
summary = json.loads((INPUT / "integrated_input_summary.json").read_text(encoding="utf-8"))
frame = pd.read_csv(INPUT / "gene_features_and_labels.csv")
expr = np.load(INPUT / "expression_features_raw.npy", allow_pickle=False)
target = np.load(INPUT / "target_values.npy", allow_pickle=False)
mask = np.load(INPUT / "target_mask.npy", allow_pickle=False)
edges = np.load(INPUT / "graph_edge_index.npz")
edge_index, edge_weight = edges["edge_index"], edges["edge_weight"]
checks = {}
def check(name, condition):
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)

n = int(summary["gene_count"])
check("aligned_row_counts", len(frame) == len(expr) == len(target) == len(mask) == n)
check("contiguous_gene_index", frame.gene_index.tolist() == list(range(n)))
check("expression_shape_and_finiteness", expr.shape == tuple(summary["expression_feature_shape"]) and np.isfinite(expr).all())
check("target_mask_matches_labels", np.array_equal(mask.astype(bool), target >= 0))
check("unknown_labels_masked", np.all(target[~mask.astype(bool)] == -1))
check("known_target_values_binary", set(np.unique(target[mask.astype(bool)])) == {0, 1})
check("label_counts_match_summary", int(mask.sum()) == summary["known_label_count"] and
      int((target == 1).sum()) == summary["positive_count"] and int((target == 0).sum()) == summary["explicit_zero_count"])
check("edge_array_shapes", edge_index.shape[0] == 2 and edge_index.shape[1] == len(edge_weight))
check("edge_index_bounds", edge_index.min() >= 0 and edge_index.max() < n)
check("edge_weights_finite_range", np.isfinite(edge_weight).all() and edge_weight.min() >= 0 and edge_weight.max() <= 1)
check("no_self_edges", not np.any(edge_index[0] == edge_index[1]))
edge_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
check("edge_reverse_pairs", len(edge_pairs) == len(edge_weight) and
      all((v, u) in edge_pairs for u, v in edge_pairs))

exp = json.loads((RUN / "metrics/experiment_summary.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(RUN / "metrics/model_comparison.csv")
split = pd.read_csv(RUN / "predictions/split_manifest.csv")
check("run_completed", exp["run_status"] == "completed")
check("heldout_partition_disjoint", not split.loc[split.partition == "development", "gene_index"].isin(split.loc[split.partition == "held_out_test", "gene_index"]).any())
check("partition_contains_only_known_labels", (split.loc[split.partition.isin(["development", "held_out_test"]), "oncogene_status"] != "unknown").all())
check("all_models_have_test_predictions", len(comparison) == 10 and all((RUN / "predictions" / f"{m}_heldout_test.csv").is_file() for m in comparison.model))
check("all_neural_checkpoints_exist", all((RUN / "checkpoints" / m / "final.pt").is_file() for m in comparison.model if m != "logistic_expression_topology"))
check("selected_model_is_cv_best", exp["best_validation_model"] == comparison.sort_values("cv_average_precision_mean", ascending=False).iloc[0].model)
check("test_set_size_matches_metrics", int((split.partition == "held_out_test").sum()) == int(exp["best_model_test_metrics"]["test_n"]))

out = {"status": "passed", "input_directory": str(INPUT.relative_to(ROOT)), "run_directory": str(RUN.relative_to(ROOT)),
       "checks_passed": len(checks), "checks": checks}
(RUN / "metrics/experiment_integrity_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
