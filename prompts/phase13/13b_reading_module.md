# 13b — `harness/reading.py`: the reader arm's crop-and-score path 🟢 PHI-free to build

> **Prerequisite: 13a (D-13.5 config identity) must be merged.** Every reader is its own arm; without
> distinct identities they average into one number.
>
> Building and testing this is PHI-free (synthetic fixtures). **Running it on real renders is not —
> Arnav runs that, and it emits PHI-free aggregates only.**

---

## Goal

One module that takes the frozen GT boxes, crops each one from its render, hands the crop to any
reader, and scores the returned string — so a recognizer with no detector of its own becomes
testable, and reading quality is measured without detection errors contaminating it.

## Context

- `harness/contract.py` — `GTToken`, `OCRWord`, `OCROutput`, and the **frozen** `normalize()`.
- `harness/metrics.py:61` — `score()`; read the docstring on the Found axis
  (`found_count`/`omission_count`) vs the Added axis (`added_count`). They are never averaged.
- `harness/matching.py` — **not used by this arm**; the box is given, so there is nothing to match.
- `harness/runners/base.py` — the `Runner` ABC. A reader is a *different* interface (crop in,
  string out); decide whether it subclasses, sits beside it, or is a separate protocol.
- `ground_truth/gt_schema.py` — how `gt.csv` rows are loaded (`load_gt()`), including that a blank
  image contributes zero rows.
- `tests/synthetic.py` + `tests/conftest.py` — the synthetic fixture factory. All tests use it.
- plan.md, PHASE 13, "Step 7 in detail" — the spec this implements, including the fairness rules.

## Constraints

- **Reuse the frozen `normalize()` and the existing metrics.** Do not write a second scoring path;
  a reader's string must be judged by exactly the same rules as an end-to-end engine's string.
- **Crop preprocessing is pinned as module-level constants and is identical for every reader** —
  padding, target height, interpolation. This is the single easiest way to silently rig the
  comparison, so it must be impossible for a reader to override it.
- **Detection metrics are undefined for this arm.** The module must not emit box recall/precision,
  and must make it structurally hard for a caller to report them.
- Map the two axes honestly: reader returns empty → **omission**; reader returns text on a box that
  has none → **added**. Never collapse them into one accuracy number.
- **PHI:** the module reads renders and GT — so it writes results to a file and returns aggregates.
  No token text, no crop, no pixel value in stdout or in any return value that reaches a terminal.
- Tests run on **synthetic fixtures only** (fake images, fake tokens like `CMFN-00421`).

## Steps

1. Enter **Plan Mode**. The plan must enumerate: the reader interface signature, the exact
   preprocessing constants and their values with a one-line justification each, the module's public
   functions, every file to create or edit, and the test list.
2. Implement `harness/reading.py` and `tests/test_reading.py`.
3. Include a **stub reader** in the tests (returns a canned string) so the scoring path is proven
   without any model dependency.

## Output format

Plan first (no code). After approval: the new files in full, then the test output.

## Done when

- `pytest tests/test_reading.py` passes, covering at minimum: exact match, case/format normalization,
  empty return → omission, invented text → added, a single-character token (`L`), and a blank image
  contributing zero rows.
- Full `pytest` still green; `ruff check .` clean.
- The module has no import of `matching.py`.

## Finally

Spawn one fresh-context subagent to review the diff against plan.md's "Step 7 in detail" fairness
rules. Scope: can any reader alter its own crop preprocessing, can detection metrics leak out of this
arm, and does any code path put token text on stdout? Correctness and constraint violations only.
