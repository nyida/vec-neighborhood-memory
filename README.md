# Virtual Embryo contribution: bounded-memory spatial averaging

Prepared for Nyida Gyal, November 16, 2026.

This package contains a tested contribution to the public `aristoteleo/veckit` local
scorer for the Virtual Embryo Challenge. It reduces temporary memory when averaging
spatial neighborhoods, a calculation used by both Task 2 and Task 3.

## Result

On synthetic 20,000-cell, 500-gene inputs, median peak process memory fell by 67–69%
across dense, CSR and CSC formats. Output hashes matched in every comparison. All
104 tests passed after an audit corrected a sparse COO duplicate-summation edge case.
Installed-wheel CLI/API checks also preserved all reported Task 2
and Task 3 metrics on upstream public sample data.

The 67–69% figure is a component result. A broader synthetic full-panel audit measured
12.5% less peak memory for Task 2 and 2.3% more for Task 3, with identical metric outputs.
The tested full panels showed no overall speedup. These are local measurements,
not results from the private server or leaderboard gains.
The code has not been accepted upstream and no award has been granted.

## Contents

- `contribution.patch`: apply to the pinned upstream repository.
- `source/`: changed source file, new tests, benchmark scripts, results, and explanation.
- `PR_DESCRIPTION.md`: prepared pull-request description.
- `AWARD_SUBMISSION.md`: prepared community-award filing, with publication still pending.
- `requirements-validation.txt`: main package versions used for verification.
- `RECHECK.md`: findings, caveats and award assessment from the deeper audit.
- `LICENSE`: upstream MIT license; the contribution is provided under the same license.

No biological data, private credentials, or user account files are included.

## Apply and verify

```bash
git clone https://github.com/aristoteleo/veckit.git
cd veckit
git checkout cc3859ab2a2305efc4544815d3f44f660efa77d4
git apply /path/to/contribution.patch
python -m venv .venv
. .venv/bin/activate
python -m pip install -r /path/to/requirements-validation.txt
python -m pip install -e .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python benchmarks/neighborhood_memory.py --cells 20000 --genes 500 --repeats 3
```

The upstream clone supplies its existing small public fixtures for the four upstream
tests. Our new tests and memory benchmark generate their own synthetic fixtures.
`source/benchmarks/README.md` explains the benchmark limits and the installed-wheel check.

## Publication and award filing

Publish the contribution in a public repository or upstream pull request, replace the
pending public URL in `AWARD_SUBMISSION.md`, and file it through the signed-in form at
https://virtualembryo.ai/challenge/community. The community award does not require a
registered competition team or an upstream merge. Existing contribution overlap and
the organizer's funding availability remain relevant to selection.

The implementation, tests, and documentation were developed with AI coding assistance.
