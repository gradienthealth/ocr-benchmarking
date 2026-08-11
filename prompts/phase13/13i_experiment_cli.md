# 13i — `scripts/run_experiment.py`: the driver 🟡 build PHI-free, run is PHI-touching

> **Prerequisites: 13a, 13b, 13e, and at least one reader (13c).** This is the thin CLI that runs the
> test plan end to end; it wires existing pieces and adds no scoring logic of its own.

---

## Goal

One command that selects arms, verifies the frozen `gt.csv` hash, runs the harness, and writes the
three report tables — with PHI-free aggregates only on stdout.

## Context

- plan.md **PHASE 13** — the six-step test plan this executes, in order.
- `harness/harness.py` — `run_harness()`; `harness/aggregate.py`; `harness/report.py`.
- `harness/reading.py` — the reader arm path (13b).
- `ground_truth/gt.csv.sha256` = `afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e`
  and `ground_truth/gt_set.sha256` = `e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea`
  — `gt_v1`, 199 images, 2351 rows.
- `ground_truth/gt_schema.py` — `load_gt()`; a blank image contributes zero rows.

## Constraints

- **Verify the `gt.csv` sha256 first and abort on mismatch.** A changed `gt.csv` invalidates every
  previously reported number (rule #8); the driver must refuse to run rather than quietly produce
  incomparable results. Print the hash it verified.
- **Stamp every result with the artifact identity** — `gt_v1`, both hashes, image count — so no table
  can be mistaken for a future 307- or 966-image build.
- **PHI-free stdout only**: counts, rates, hashes, shapes. Never a token, never a path that reveals
  a UID, never a crop.
- **Iteration order is free** — metrics are order-independent means (D-1.1). Do not add an ordering
  requirement or a "never re-sort" rule.
- **Three tables, never blended**: end-to-end (Arm A), detector-only (Arm B), reader (Arm C). Arm C
  gets oracle boxes and is not comparable to Arm A.
- **Do NOT modify** `matching.py`, `metrics.py`, `normalize()`, or any runner. This is wiring.
- Every generative arm runs the **negative control before its accuracy numbers are reported**, and the
  driver should make that ordering hard to skip.
- Resume/partial-run behaviour: if a run dies halfway, it must not emit a table that looks complete.

## Steps

1. Enter **Plan Mode**: enumerate the CLI surface (flags, arm selection, output paths), the abort
   conditions, and the exact table set. Order of operations matters here — state it.
2. Implement `scripts/run_experiment.py` + tests over synthetic fixtures, including: hash mismatch
   aborts, a partial run does not emit a complete-looking table, and arm selection produces exactly
   the tables selected.
3. Write the command list Arnav runs for the real pass, in test-plan order, with the negative control
   first.

## Output format

Plan first. Then the diff, test output, and a sample of each of the three tables rendered from
synthetic data.

## Done when

- `pytest` green; `ruff check .` clean.
- A deliberately wrong hash aborts the run with a clear message.
- Sample tables show the `gt_v1` identity stamp.
- `git diff --stat` shows no change to `matching.py`, `metrics.py`, or `contract.py`.

## Finally

Spawn one fresh-context subagent. Scope: can any code path print token text or a UID-bearing path,
can a run proceed past a hash mismatch, and can a partial run produce a table that reads as complete?
Correctness only.
