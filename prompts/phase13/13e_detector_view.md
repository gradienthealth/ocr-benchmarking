# 13e — Arm B: the detector-only report view 🟢 PHI-free

> **Prerequisite: 13a (config identity) merged.** Independent of 13b–13d — can run in parallel.
>
> Cheapest arm in the project: **no new models and no new inference.** It is a second view of runs
> that already happen.

---

## Goal

A detector-only table that scores boxes against GT at IoU ≥ 0.5 and ignores strings entirely — so a
loss can be attributed to *finding* the text rather than *reading* it.

## Context

- `harness/metrics.py:61` — `score()` already separates the Found axis
  (`found_count`/`omission_count`, real GT text located) from the Added axis (`added_count`,
  predictions covering no GT token). Detection recall/precision falls out of these counts; the
  string comparison is what gets switched off.
- `harness/matching.py` — the IoU ≥ 0.5 matcher that produces those counts.
- `harness/aggregate.py`, `harness/report.py` — where the new table is assembled and rendered.
- `ground_truth/gt_summary.txt` — the per-stratum image counts.

## Constraints

- **Do NOT re-run any engine and do NOT add an inference path.** If this prompt makes you load a
  model, you have misread it.
- **Do NOT modify** `matching.py`, `metrics.py`, or `normalize()`. Read from what `score()` already
  returns; if a needed count genuinely is not exposed, stop and report that rather than editing the
  frozen scoring path.
- **`ct_scout` and `mg_tomo` have zero text-bearing images.** They have no recall denominator, so a
  per-image recall mean over them reads a meaningless saturated 100%. The table must either exclude
  them explicitly or print `n/a` with the reason — never a silent 100%.
- The blank control set is **64** images, not 66.
- Detection precision on blank controls is the **hallucination floor** — surface it, do not average
  it into a detection score.
- This table is reported **separately** from the end-to-end and reader tables. Three questions, three
  tables, never blended.

## Steps

1. Read `metrics.py`'s return dict and confirm which detection counts are already available; report
   that list before writing code.
2. Add the detector-only aggregation + report table.
3. Add tests over synthetic fixtures, including: a stratum with zero text-bearing images renders
   `n/a` rather than 100%, and a blank-control image's predictions land in the floor, not in recall.

## Output format

First: the list of counts `score()` already exposes and whether they suffice. Then the diff, then
test output, then a sample rendered table from synthetic data.

## Done when

- `pytest` green; `ruff check .` clean.
- The rendered sample table shows per-stratum detection recall/precision, `n/a` for the two
  zero-denominator strata, and the hallucination floor as its own line.
- `git diff --stat` shows **no** change to `matching.py`, `metrics.py`, or `contract.py`.

## Finally

Spawn one fresh-context subagent to check one thing specifically: can a zero-denominator stratum
reach the report as a real number, and can a blank-control image's invented boxes be counted as
detection recall anywhere? Correctness only.
