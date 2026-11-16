"""Compare old/new neighborhood aggregation in isolated subprocesses (synthetic data).

Run: python benchmarks/neighborhood_memory.py --cells 20000 --genes 500 --repeats 3
Peak RSS is the whole child process, not just NumPy allocations. Timing covers neighbor fitting,
querying, and averaging, but not data generation/imports. This is not a full-scoring benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


def worker(args):
    import resource
    import numpy as np
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from score_h5ad import _load_task_metrics

    metrics, _ = _load_task_metrics("T2")
    rng = np.random.default_rng(args.seed)
    X = rng.random((args.cells, args.genes), dtype=np.float32)
    X[X < 0.8] = 0
    coords = rng.normal(size=(args.cells, 3)).astype(np.float32)
    if args.storage != "dense":
        X = getattr(sparse, args.storage + "_matrix")(X)
    started = time.perf_counter()
    if args.worker == "legacy":
        _, idx = NearestNeighbors(n_neighbors=args.k).fit(coords).kneighbors(coords)
        dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        result = dense[idx].mean(1)
    else:
        result = metrics._knn_neighborhood_pb(X, coords, args.k)
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    digest = hashlib.sha256(result.tobytes(order="C")).hexdigest()
    print(json.dumps({"mode": args.worker, "storage": args.storage, "cells": args.cells,
                     "genes": args.genes, "k": args.k, "seed": args.seed,
                     "elapsed_seconds": elapsed, "peak_rss_bytes": rss_bytes,
                     "output_sha256": digest, "output_dtype": str(result.dtype)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=20000)
    ap.add_argument("--genes", type=int, default=500)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results.json")
    ap.add_argument("--worker", choices=["legacy", "batched"])
    ap.add_argument("--storage", choices=["dense", "csr", "csc"], default="dense")
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    rows = []
    for storage in ("dense", "csr", "csc"):
        for repeat in range(args.repeats):
            order = ("legacy", "batched") if repeat % 2 == 0 else ("batched", "legacy")
            pair = []
            for mode in order:
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                       "--storage", storage, "--cells", str(args.cells), "--genes", str(args.genes),
                       "--k", str(args.k), "--seed", str(args.seed)]
                run = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
                row = json.loads(run.stdout)
                row["repeat"] = repeat
                rows.append(row)
                pair.append(row)
                print(f"{storage} {mode}: {row['peak_rss_bytes']/1024**2:.1f} MiB, "
                      f"{row['elapsed_seconds']:.3f} s", flush=True)
            if pair[0]["output_sha256"] != pair[1]["output_sha256"]:
                raise RuntimeError(f"Output mismatch for {storage}, repeat {repeat}")
    summary = []
    for storage in ("dense", "csr", "csc"):
        old = [r for r in rows if r["storage"] == storage and r["mode"] == "legacy"]
        new = [r for r in rows if r["storage"] == storage and r["mode"] == "batched"]
        old_rss = statistics.median(r["peak_rss_bytes"] for r in old)
        new_rss = statistics.median(r["peak_rss_bytes"] for r in new)
        summary.append({"storage": storage, "legacy_peak_rss_mib": old_rss / 1024**2,
                        "batched_peak_rss_mib": new_rss / 1024**2,
                        "peak_rss_reduction_percent": 100 * (1 - new_rss / old_rss),
                        "legacy_seconds": statistics.median(r["elapsed_seconds"] for r in old),
                        "batched_seconds": statistics.median(r["elapsed_seconds"] for r in new),
                        "identical_output_hashes": True})
    report = {"scope": "Neighborhood aggregation only; synthetic fixtures; fresh process per run",
              "upstream_commit": "cc3859ab2a2305efc4544815d3f44f660efa77d4",
              "platform": platform.platform(), "python": sys.version,
              "versions": {p: importlib.metadata.version(p) for p in
                           ("numpy", "scipy", "scikit-learn", "anndata")},
              "threads": 1, "summary": summary, "runs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
