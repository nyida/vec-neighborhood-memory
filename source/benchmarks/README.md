# Bounded-memory spatial neighborhood averaging

The Task 2/3 local scorer computes a mean expression profile for each cell's 15 nearest
spatial neighbors. Previously it gathered all neighborhoods into an array shaped
`(cells, neighbors, genes)` before reducing it. For 20,000 cells and 500 float32 genes,
that temporary alone holds 600,000,000 bytes (572.2 MiB).

The change in `T2/metrics.py` gathers and reduces one batch at a time. Each gathered
batch is bounded to approximately 16 MiB, except when one neighborhood is larger than
that budget. CSR/CSC inputs use selected-row batch densification. Other sparse formats
retain legacy full densification because converting COO to CSR can change the rounding
of duplicate-entry sums. Neighbor selection, its existing self-neighbor convention, and the averaging
formula are preserved. The final dense means and neighbor index remain in memory;
this is not an out-of-core scorer or a 16 MiB limit on the whole process.

## Measured result

Synthetic, nonnegative expression: 20,000 cells, 500 genes, 15 neighbors, 80% zeros.
Three fresh processes per implementation per storage format, alternating execution
order, with numerical-library thread counts set to one. Values below are medians.

| Input format | Original peak RSS | Batched peak RSS | Reduction | Original time | Batched time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 827.3 MiB | 271.4 MiB | 67.2% | 1.674 s | 0.769 s |
| CSR | 856.2 MiB | 261.8 MiB | 69.4% | 1.697 s | 0.478 s |
| CSC | 856.1 MiB | 273.1 MiB | 68.1% | 1.733 s | 0.627 s |

Every original/batched pair produced the same SHA-256 hash of the output array.
Peak RSS includes interpreter/library memory and fixture creation. Timing covers the
neighbor query and aggregation, excluding imports and fixture generation. These are
component measurements, not full-pipeline speedups or competition-score improvements.
Raw runs, versions, seeds and hashes are in `results.json`. Timing depends on hardware
and load; the recorded run used a shared Linux host.

## Complete-panel audit and limits

A deeper audit on November 16 measured complete Task 2/3 metric panels, including
frozen-probe training but excluding file I/O, on synthetic dense float32 inputs.
Task 2 used 20,000 prediction cells, 8,393 truth cells and 5,374 reference cells;
Task 3 used 7,000 prediction cells, 2,320 truth cells and 2,483 reference cells.
Both had 500 genes. These dimensions were chosen using public validation-board
metadata, not hidden observations. Three fresh processes per implementation/task,
alternating order, produced these medians:

| Panel | Original peak RSS | Batched peak RSS | Change | Original time | Batched time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task 2 | 877.4 MiB | 767.4 MiB | 12.5% lower | 8.207 s | 8.853 s |
| Task 3 | 686.9 MiB | 702.7 MiB | 2.3% higher | 4.025 s | 4.597 s |

All 23 Task 2 and 24 Task 3 reported metrics had identical JSON hashes in every
paired run. The component gain does **not** imply a comparable reduction for the
complete scorer: other calculations still allocate large temporary arrays, and
process peak memory depends on allocation reuse. The Task 3 case had no overall
memory improvement. Median total times were higher in these shared-host runs;
no full-panel speedup is claimed. Raw runs are in `panel_results.json`.

These panel measurements preceded the COO-only compatibility correction; their
dense-input calculation is unchanged by that correction. The corrected package
was then checked by the expanded tests and rebuilt-wheel verification.

```bash
python benchmarks/panel_memory.py --repeats 3
```

## Reproduce

Use upstream commit `cc3859ab2a2305efc4544815d3f44f660efa77d4` plus this patch.

```bash
python -m pip install -e . pytest build
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python benchmarks/neighborhood_memory.py --cells 20000 --genes 500 --repeats 3
```

The benchmark uses `resource.getrusage`, and runs on Linux or macOS. Its benchmark data
is generated in memory and includes no Challenge data. The benchmark raises an error
if any original/batched output hash differs.

To check the installed wheel, use an absolute installation directory outside the source
checkout. The script also runs its subprocesses outside the checkout and verifies the
imported package path, preventing the source tree from masking a packaging problem.

```bash
python -m build --wheel --outdir dist
python -m pip install --no-deps --target /absolute/path/veckit-wheel-check dist/veckit-0.1.1-py3-none-any.whl
python benchmarks/check_installed.py --installed-root /absolute/path/veckit-wheel-check --out benchmarks/installed_check.json
```

Use a fresh target directory. This locally built wheel is for verification, not an
official package release. `check_installed.py` uses the small sample files already
published in upstream `data/`; it writes no biological values into its report.

## Validation and scope

- All 104 tests passed: 100 new cases and the four existing upstream tests.
- Exact array equality checks cover float32, float64, integer and boolean inputs;
  C/F/strided arrays; CSR/CSC matrices and CSR arrays; neighbor counts; duplicate
  coordinates; multiple batches; and the one-neighborhood-over-budget case.
- Duplicate-entry summation tests cover COO/CSR/CSC/LIL/DOK/DIA/BSR with float32
  and float64. The audit caught and corrected a COO-to-CSR rounding regression.
- A sparse-allocation test checks that only batch-selected rows are densified.
- Complete Task 2 and Task 3 metric panels match the old helper on two synthetic seeds.
- On the existing public samples, the installed wheel preserves all 23 Task 2 and
  24 Task 3 reported metrics, and CLI/stdout/saved JSON agree with the Python API.
- No hidden test data or competition submissions were used. Equivalence on these
  fixtures is not a claim of validation on the private evaluation server.

The implementation, tests and documentation were developed with AI coding assistance.
All reported results were executed; organizer acceptance and an award are not implied.
