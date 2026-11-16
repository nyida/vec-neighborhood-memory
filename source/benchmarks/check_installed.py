"""Check installed-wheel CLI/API parity on the upstream public 150-cell samples.

Usage: python benchmarks/check_installed.py --installed-root /path/to/pip-target --out report.json
Run from a checkout containing the original data/ samples. No samples are copied into the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

API_SCRIPT = r'''
import json, os
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import score_h5ad
import veckit
spec = json.loads(os.environ["VEC_CHECK_SPEC"])
expected_root = Path(os.environ["VEC_INSTALLED_ROOT"]).resolve()
assert Path(veckit.__file__).resolve().is_relative_to(expected_root), veckit.__file__
if os.environ["VEC_CHECK_MODE"] == "legacy":
    original = score_h5ad._load_task_metrics
    def legacy(X, coords, k=15):
        _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
        dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return dense[idx].mean(1)
    def load(task):
        metrics, panel = original(task)
        panel.neighborhood_mmd.__globals__["_knn_neighborhood_pb"] = legacy
        return metrics, panel
    score_h5ad._load_task_metrics = load
result = veckit.score(**spec)
print(json.dumps(result, sort_keys=True))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--installed-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    installed = args.installed_root.resolve()
    data = ROOT / "data"
    specs = [
        {"task": "T2", "input": str(data / "sample_heart_9.25.h5ad"),
         "target": str(data / "sample_heart_9.5.h5ad"),
         "reference": str(data / "sample_heart_9.25.h5ad"), "seed": 0},
        {"task": "T3", "input": str(data / "sample_wt.h5ad"),
         "target": str(data / "sample_mab21l2_ko.h5ad"),
         "wt": str(data / "sample_wt.h5ad"), "seed": 0},
    ]
    env = dict(os.environ, PYTHONPATH=str(installed), VEC_INSTALLED_ROOT=str(installed))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    checks = []
    with tempfile.TemporaryDirectory() as work:
        for spec in specs:
            env["VEC_CHECK_SPEC"] = json.dumps(spec)
            outputs = {}
            for mode in ("legacy", "batched"):
                env["VEC_CHECK_MODE"] = mode
                result = subprocess.run([sys.executable, "-c", API_SCRIPT], cwd=work, env=env,
                                        capture_output=True, text=True, check=True)
                outputs[mode] = json.dumps(json.loads(result.stdout), sort_keys=True)
            assert outputs["legacy"] == outputs["batched"], spec["task"] + " metric mismatch"
            cmd = [sys.executable, str(installed / "score_h5ad.py")]
            for key, value in spec.items():
                cmd.extend(["--" + key.replace("_", "-"), str(value)])
            report_file = Path(work) / (spec["task"] + ".json")
            cmd.extend(["--out", str(report_file)])
            result = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, check=True)
            cli = json.dumps(json.loads(result.stdout), sort_keys=True)
            saved = json.dumps(json.loads(report_file.read_text()), sort_keys=True)
            assert cli == outputs["batched"] == saved, spec["task"] + " CLI/API mismatch"
            checks.append({"task": spec["task"], "legacy_vs_batched_api": "identical",
                           "cli_vs_api": "identical", "saved_json_vs_stdout": "identical",
                           "metric_count": len(json.loads(cli)["metrics"])})
    used_files = {Path(value) for spec in specs for key, value in spec.items()
                  if key in ("input", "target", "reference", "wt")}
    report = {"checks": checks, "fixtures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in sorted(used_files)},
              "scope": "Installed wheel, existing public upstream samples; no hidden challenge data"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
