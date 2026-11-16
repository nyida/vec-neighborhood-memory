"""Preserve neighborhood means and downstream scores while bounding temporary memory.

Synthetic fixtures only. The legacy helper freezes the implementation at upstream
cc3859ab2a2305efc4544815d3f44f660efa77d4; it is an equivalence oracle, not a new metric.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_h5ad import _load_task_metrics


def legacy_neighborhood_pb(X, coords, k=15):
    _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return X[idx].mean(1)


@pytest.fixture
def metrics():
    return _load_task_metrics("T2")[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.bool_])
@pytest.mark.parametrize("storage", ["dense", "fortran", "strided", "csr", "csc", "csr_array"])
@pytest.mark.parametrize("k", [1, 7, 41])
def test_means_match_legacy_exactly(metrics, dtype, storage, k):
    rng = np.random.default_rng(108)
    raw = rng.integers(0, 9, size=(41, 26)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        raw *= dtype(0.137)
    X = raw[:, ::2]
    if storage == "dense":
        X = X.copy(order="C")
    elif storage == "fortran":
        X = X.copy(order="F")
    elif storage == "csr":
        X = sparse.csr_matrix(X)
    elif storage == "csc":
        X = sparse.csc_matrix(X)
    elif storage == "csr_array":
        X = sparse.csr_array(X)
    coords = rng.normal(size=(41, 3))
    expected = legacy_neighborhood_pb(X, coords, k)
    actual = metrics._knn_neighborhood_pb(X, coords, k)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_matches_independent_neighbor_average(metrics):
    # Distinct distances avoid tie-policy ambiguities; this verifies the mathematical result.
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [4., 0., 0.], [10., 0., 0.]])
    X = np.array([[1., 7.], [3., 9.], [5., 11.], [13., 15.]], dtype=np.float32)
    expected = np.array([[2., 8.], [2., 8.], [4., 10.], [9., 13.]], dtype=np.float32)
    np.testing.assert_array_equal(metrics._knn_neighborhood_pb(X, coords, 2), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("storage", ["coo", "csr", "csc", "lil", "dok", "dia", "bsr"])
def test_sparse_duplicate_summation_order_matches_legacy(metrics, dtype, storage):
    # Positive, finite values spanning scales expose rounding changes when duplicates
    # are coalesced. Construct noncanonical CSR/CSC directly so duplicates survive.
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 31, 600), rng.integers(0, 17, 600)
    data = np.exp(rng.uniform(-15, 15, 600)).astype(dtype)
    coo = sparse.coo_matrix((data, (rows, cols)), shape=(31, 17))
    if storage in ("csr", "csc"):
        major, minor, count = (rows, cols, 31) if storage == "csr" else (cols, rows, 17)
        order = np.argsort(major, kind="stable")
        indptr = np.r_[0, np.cumsum(np.bincount(major, minlength=count))]
        X = getattr(sparse, storage + "_matrix")(
            (data[order], minor[order], indptr), shape=coo.shape)
        assert not X.has_canonical_format
    else:
        X = coo.asformat(storage)
    coords = rng.normal(size=(31, 3))
    np.testing.assert_array_equal(metrics._knn_neighborhood_pb(X, coords, 7),
                                  legacy_neighborhood_pb(X, coords, 7))


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_coincident_coordinates_keep_existing_neighbor_policy(metrics, storage):
    rng = np.random.default_rng(88)
    coords = np.repeat(rng.normal(size=(8, 3)), 4, axis=0)
    X = rng.random((32, 17), dtype=np.float32)
    if storage != "dense":
        X = getattr(sparse, storage + "_matrix")(X)
    np.testing.assert_array_equal(
        metrics._knn_neighborhood_pb(X, coords, 6), legacy_neighborhood_pb(X, coords, 6)
    )


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_multiple_batches_and_partial_final_batch(metrics, storage):
    rng = np.random.default_rng(112)
    X = rng.random((1201, 512), dtype=np.float32)
    X[X < 0.75] = 0
    if storage != "dense":
        X = getattr(sparse, storage + "_matrix")(X)
    coords = rng.normal(size=(1201, 3))
    # 1170 rows fit per 16 MiB batch at k=7, so the final 31 rows exercise the boundary.
    np.testing.assert_array_equal(
        metrics._knn_neighborhood_pb(X, coords, 7), legacy_neighborhood_pb(X, coords, 7)
    )


def test_sparse_densification_is_limited_to_selected_batches(metrics, monkeypatch):
    rng = np.random.default_rng(404)
    X = sparse.csr_matrix(rng.random((1201, 512), dtype=np.float32))
    coords = rng.normal(size=(1201, 3))
    calls = []
    original = sparse.csr_matrix.toarray

    def observed(self, *args, **kwargs):
        calls.append(self.shape)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", observed)
    actual = metrics._knn_neighborhood_pb(X, coords, 7)
    assert actual.shape == X.shape
    assert len(calls) == 2
    assert all(rows * genes * X.dtype.itemsize <= 16 * 1024**2 for rows, genes in calls)
    assert sum(rows for rows, _ in calls) == 1201 * 7
    assert X.shape not in calls


def test_neighborhood_larger_than_budget_still_processes_one_cell(metrics):
    # One neighborhood exceeds the budget. The documented bound permits one such neighborhood.
    rng = np.random.default_rng(21)
    X = rng.random((2, 1_050_000), dtype=np.float64)
    coords = np.array([[0., 0., 0.], [1., 0., 0.]])
    np.testing.assert_array_equal(
        metrics._knn_neighborhood_pb(X, coords, 2), legacy_neighborhood_pb(X, coords, 2)
    )


def test_invalid_neighbor_count_still_raises(metrics):
    with pytest.raises(ValueError):
        metrics._knn_neighborhood_pb(np.ones((3, 4)), np.zeros((3, 3)), 4)


@pytest.mark.parametrize("task", ["T2", "T3"])
@pytest.mark.parametrize("seed", [0, 17])
def test_full_metric_panels_are_unchanged(task, seed, monkeypatch):
    metrics, panel = _load_task_metrics(task)
    rng = np.random.default_rng(seed)
    ref_X = rng.uniform(0.1, 3., size=(67, 40)).astype(np.float32)
    true_X = rng.uniform(0.1, 3., size=(73, 40)).astype(np.float32)
    pred_X = rng.uniform(0.1, 3., size=(79, 40)).astype(np.float32)
    true_X[:, :10] += 0.8
    pred_X[:, :10] += 0.6
    true_C, pred_C = rng.normal(size=(73, 3)), rng.normal(size=(79, 3))
    true_ct = np.where(np.arange(73) % 2, "type_a", "type_b")
    pred_ct = np.repeat("unused", 79)
    probe = metrics.train_frozen_probe(true_X, true_ct)
    scorer = getattr(panel, "score_task" + task[1] + "_v2")
    args = (pred_X, pred_C, pred_ct, true_X, true_C, true_ct, ref_X)
    actual = scorer(*args, probe=probe, seed=seed)
    # T3 reuses T2's neighborhood_mmd; patch the function's actual globals in either task.
    monkeypatch.setitem(panel.neighborhood_mmd.__globals__, "_knn_neighborhood_pb", legacy_neighborhood_pb)
    expected = scorer(*args, probe=probe, seed=seed)
    assert actual.keys() == expected.keys()
    for key in actual:
        if actual[key] is None or expected[key] is None:
            assert actual[key] is expected[key]
        else:
            np.testing.assert_equal(actual[key], expected[key], err_msg=key)
