# Community Contribution Award filing

Status: prepared; not yet filed or approved.

**Title:** Reduce memory use in Task 2/3 spatial-neighborhood scoring

**Category:** build — evaluation tooling and code

**Contributor:** Nyida Gyal

**Public contribution URL:** Pending publication

**Description:**

This contribution reduces the memory needed to compute spatial-neighborhood expression profiles in the public `veckit` local scorer. The original implementation gathers all cells' neighborhoods at once, creating a large three-dimensional temporary. The patch processes bounded batches and densifies CSR/CSC inputs only after selecting each batch, preserving the existing neighbor query and averaging formula. Other sparse formats retain legacy densification to preserve duplicate-entry summation order; this compatibility point was caught and corrected during audit.

In a reproducible synthetic benchmark with 20,000 cells, 500 genes and 15 neighbors, median whole-process peak memory fell from 827.3 to 271.4 MiB for dense input, from 856.2 to 261.8 MiB for CSR, and from 856.1 to 273.1 MiB for CSC. All original/batched output hashes matched. There were three fresh-process runs per implementation and input format. These are component benchmarks, not full-pipeline or leaderboard results.

The contribution includes the patch, 100 new regression/equivalence tests, reproducible benchmarks, raw measurements, and an installed-wheel verification script. All 104 tests passed, including the four existing upstream tests. On the upstream public examples, all 23 Task 2 and 24 Task 3 metrics matched the original implementation, and CLI/API/saved outputs agreed.

A broader synthetic full-panel audit found 12.5% lower Task 2 peak RSS, but 2.3% higher Task 3 peak RSS; other scorer allocations still matter. All panel metrics were unchanged. Full-panel median runtimes were higher in these shared-host measurements. We do not claim universal whole-scorer memory savings or speedups.

The work is intended to make local scoring more practical on machines with limited RAM. It was developed with AI coding assistance. No hidden Challenge data or competition submissions were used. Community adoption has not yet been measured.

**Relationship to prior work:**

The checked upstream issues and pull requests already cover MMD distance memory, sparse core-expression metrics, Moran self-neighbors, concurrency, JSON export, gene matching, and input/output path collisions. This change targets the separate three-dimensional gather in `T2/metrics.py::_knn_neighborhood_pb`, reused by Task 3. On November 16, the rendered public registry showed 15 reviewed/accepted entries and 700 pending entries. Its resource titles and available descriptions were surveyed for neighborhood, batching, memory and scorer overlap. Related VEC Profile and VEC Sparse Semantics Lab documentation describe profiling and input preflight respectively. No exact duplicate was identified in this survey, which is not an exhaustive source-code audit of every listed project. The upstream memory PR #2 does not modify this function or its file.

**Filing page:** https://virtualembryo.ai/challenge/community

Publication and filing are separate steps. A public repository or resource listing does not establish award acceptance. The organizer determines acceptance and award amount.
