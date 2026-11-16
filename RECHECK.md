# Independent recheck — November 16, 2026

## Technical finding

The main optimization is useful and preserves the tested scoring results, but its
benefit must be stated at the correct scope. It avoids one large neighborhood
gather; it does not make the whole scorer bounded-memory.

The audit found a compatibility defect in the first version: converting COO to
CSR can coalesce duplicate sparse entries in a different order and change
floating-point rounding. A positive, finite float32 fixture reproduced a maximum
absolute neighborhood-mean difference of 0.000244140625. The corrected code keeps
legacy full densification for formats other than CSR/CSC, then batches the
neighborhood gather. CSR/CSC retain selected-row densification.

The expanded suite has **104 passing cases**, including 14 additional sparse-format
and duplicate-entry cases. The rebuilt wheel preserves all **23 T2 and 24 T3**
metrics on the upstream public mini samples; CLI, Python API and saved JSON agree.
The upstream default branch still points to the pinned commit
`cc3859ab2a2305efc4544815d3f44f660efa77d4` at the time of this check.

## Performance finding

| Measurement | Finding |
| --- | --- |
| Neighborhood component, 20,000 cells × 500 genes | 67–69% lower peak process memory across dense/CSR/CSC inputs |
| Complete synthetic T2 panel | 877.4 → 767.4 MiB; 12.5% lower |
| Complete synthetic T3 panel | 686.9 → 702.7 MiB; 2.3% higher |
| Complete-panel output comparison | Identical hashes for all metric outputs in every old/new pair |
| Complete-panel runtime | No median speedup observed; timings varied on the shared host |

Full-panel measurements include probe training and exclude file I/O. Three fresh
processes per implementation per task were used. See `source/benchmarks/README.md`
for dimensions and `panel_results.json` for every run. The full-panel measurements
preceded the sparse-format correction, which leaves the measured dense path's
calculation unchanged. The final code was checked by tests and rebuilt-wheel runs.

The component benchmark is not a prediction of overall RAM savings on every
workload. Other unmodified metrics create large temporary arrays. No hidden data,
leaderboard submission, biological performance improvement or adoption is claimed.

## Award assessment

This is a plausible contribution for organizer review, not a secured or predictable
award. The organizer explicitly accepts public tools and fixes, including standalone
repositories without an upstream merge. Selection depends on originality,
usefulness, quality, reach, openness, impact and remaining funds. Up to $200 can be
awarded per contribution; an accepted contribution's actual amount is determined by
the organizers. Filing remains rolling until the fund is committed or December 11,
2026. Payment is after the Challenge closes; a quick decision is not promised.

The live rendered community registry contained **15 reviewed/accepted entries and
700 pending entries** on November 16. These are contributions, not distinct people.
The counts do not reveal an acceptance rate, a rejection count, or the remaining
budget. Many pending entries are separate translations or closely related tools.

The registry's titles and available descriptions were surveyed for relevant
overlap. VEC Profile profiles local commands; VEC Sparse Semantics Lab checks
numeric ingestion; upstream PR #2 changes other expression-metric code. No exact
duplicate of this neighborhood-gather patch was identified. This is a scoped
survey, not an exhaustive source review of 715 projects.

The Challenge is on NeurIPS's official 2026 competition list. This particular
recognition is an organizer-issued community contribution award. Its modest scope
should be named accurately; it does not establish a leaderboard placement, paper
acceptance, best-paper award, or conference speaking invitation.

**Current status:** code and application draft prepared; public publication and
award filing pending. GitHub sign-in was not completed. No organizer acceptance,
committed award amount, or payment exists for this contribution.

## Sources checked

- Award criteria: https://virtualembryo.ai/challenge/community
- Live contribution registry: https://virtualembryo.ai/challenge/resources
- Prize categories and organizer: https://virtualembryo.ai/challenge/prizes
- Official NeurIPS listing: https://blog.neurips.cc/2026/07/28/neurips-2026-competitions-announced/
- Public panel dimensions: https://virtualembryo.ai/challenge/panels/index.json
- Upstream pull requests: https://github.com/aristoteleo/veckit/pulls
- Existing memory PR: https://github.com/aristoteleo/veckit/pull/2
- Related profiler: https://github.com/random-guy-05/vec-profile
- Related preflight tool: https://github.com/startupcompliancekit/vec-sparse-semantics-lab

The code, audit and documentation were produced with AI coding assistance.
