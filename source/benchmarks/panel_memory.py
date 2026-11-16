"""Audit full metric-panel memory, beyond the neighborhood-only benchmark.

Synthetic float32 matrices; no Challenge observations or hidden targets. Dimensions
follow public validation-panel limits. Includes probe training, excludes file I/O.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = {"T2": (20000, 8393, 5374), "T3": (7000, 2320, 2483)}


def worker(task, mode):
    import resource
    import numpy as np
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from score_h5ad import _load_task_metrics

    metrics, panel = _load_task_metrics(task)
    if mode == "legacy":
        def legacy(X, coords, k=15):
            _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
            dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return dense[idx].mean(1)
        panel.neighborhood_mmd.__globals__["_knn_neighborhood_pb"] = legacy
    rng = np.random.default_rng(20261116)
    pred_n, true_n, ref_n = CASES[task]
    def expression(n, shift):
        X = rng.random((n, 500), dtype=np.float32)
        X[X < 0.8] = 0
        X[:, :50] += np.float32(shift)
        return X
    pred_X, true_X, ref_X = expression(pred_n, 0.4), expression(true_n, 0.5), expression(ref_n, 0)
    pred_C, true_C = rng.normal(size=(pred_n, 3)), rng.normal(size=(true_n, 3))
    # A deterministic two-class target ensures the real frozen classifier is exercised.
    true_ct = np.where(true_X[:, 100] > 0, "type_a", "type_b")
    pred_ct = np.repeat("unused", pred_n)
    started = time.perf_counter()
    scores = getattr(panel, "score_task" + task[1] + "_v2")(
        pred_X, pred_C, pred_ct, true_X, true_C, true_ct, ref_X, seed=0
    )
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = rss if sys.platform == "darwin" else rss * 1024
    encoded = json.dumps(scores, sort_keys=True, allow_nan=False).encode()
    print(json.dumps({"task": task, "mode": mode, "prediction_cells": pred_n,
                      "truth_cells": true_n, "reference_cells": ref_n, "genes": 500,
                      "peak_rss_mib": peak / 1024**2, "seconds": elapsed,
                      "metric_count": len(scores), "output_sha256": hashlib.sha256(encoded).hexdigest()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", choices=["legacy", "batched"])
    ap.add_argument("--task", choices=list(CASES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "panel_results.json")
    args = ap.parse_args()
    if args.worker:
        if not args.task:
            ap.error("--worker requires --task")
        worker(args.task, args.worker)
        return
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    runs, summaries = [], []
    for task in CASES:
        for repeat in range(args.repeats):
            pair = []
            for mode in (["legacy", "batched"] if repeat % 2 == 0 else ["batched", "legacy"]):
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                            "--worker", mode, "--task", task],
                                           env=env, text=True, capture_output=True, check=True)
                row = json.loads(completed.stdout)
                row["repeat"] = repeat
                runs.append(row)
                pair.append(row)
                print(json.dumps(row), flush=True)
            if pair[0]["output_sha256"] != pair[1]["output_sha256"]:
                raise RuntimeError("Full panel changed: " + task)
        old = [r for r in runs if r["task"] == task and r["mode"] == "legacy"]
        new = [r for r in runs if r["task"] == task and r["mode"] == "batched"]
        old_rss = statistics.median(r["peak_rss_mib"] for r in old)
        new_rss = statistics.median(r["peak_rss_mib"] for r in new)
        summaries.append({"task": task, "legacy_peak_rss_mib": old_rss,
                          "batched_peak_rss_mib": new_rss,
                          "peak_rss_reduction_percent": 100 * (1 - new_rss / old_rss),
                          "legacy_seconds": statistics.median(r["seconds"] for r in old),
                          "batched_seconds": statistics.median(r["seconds"] for r in new),
                          "identical_output_hashes": True})
    report = {"scope": "Synthetic complete T2/T3 metric panels including probe training, excluding file I/O",
              "dimensions_source": "https://virtualembryo.ai/challenge/panels/index.json",
              "upstream_commit": "cc3859ab2a2305efc4544815d3f44f660efa77d4",
              "python": sys.version, "platform": platform.platform(), "threads": 1,
              "summary": summaries, "runs": runs}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
