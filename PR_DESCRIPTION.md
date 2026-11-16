Spatial neighborhood averaging currently gathers an entire `cells × neighbors × genes` array. At 20,000 cells, 15 neighbors and 500 float32 genes, this temporary alone is 572 MiB. This change gathers and reduces bounded batches. CSR/CSC inputs are densified after batch selection; other sparse formats retain legacy densification to preserve duplicate summation order. It preserves the neighbor query, self-neighbor convention, and averaging formula.

Three isolated runs per format measured median whole-process peak RSS reductions of 67.2% for dense, 69.4% for CSR and 68.1% for CSC input. Output hashes were identical in every original/batched pair. These are synthetic component benchmarks, not full-pipeline performance claims.

Validation: all 104 tests passed, including 100 new regression/equivalence cases. This includes a regression test for COO duplicate summation, caught during audit. Complete T2/T3 panels matched the old helper on two synthetic seeds. Rebuilt-wheel checks on upstream public samples preserved all 23 T2 and 24 T3 metrics, with CLI/API/file-output parity. The raw benchmark and reproduction commands are included in `benchmarks/`.

A broader synthetic full-panel benchmark measured 12.5% lower Task 2 peak RSS but 2.3% higher Task 3 peak RSS. Both preserved every reported metric; neither showed a median runtime improvement. This targets one allocation, with no universal whole-scorer memory or speed claim. See `benchmarks/README.md` and `panel_results.json` for dimensions, raw runs and scope.

The output matrix and neighbor index remain in memory. A single neighborhood can exceed the nominal 16 MiB batch budget. The implementation, tests, and documentation were developed with AI coding assistance.
