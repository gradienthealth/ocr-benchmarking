# 13h — RESUME HERE (paused 2026-08-10, mid-flow)

> **Claude: raise this before Arnav starts writing anything.** He stopped for the day with
> the tuning slice fully annotated and exactly one command left to run. Do not begin other
> Phase 13 work without telling him this is outstanding.

---

## The one command

```
bash experiments/run_sweep_dev_v1.sh
```

Tens of minutes: 13 configurations (docTR 5, PP-OCRv6 6, EasyOCR 2) over 22 images, CPU-only.
It does **not** freeze anything — that is deliberate, see "the decision" below.

## Where things stand

**Done.** `dev_v1`, the D-13.4 tuning slice, is drawn, downloaded, rendered, seeded,
human-annotated, and built:

```
22 images · 471 rows · KEEP 394 / PHI 77 · 5 blank images (invention floor)
dev_v1.csv sha256   1733f78acc03318d2c832fda2b07ee19d47accb916e49b707e7bf7364dfea7df
dev_v1_set.sha256   897059501160f36d4311d8ea037945971abaa5a9e3fcdcd47969d72d8eb85abe
```

`gt_v1`'s three artifacts were verified byte-identical after the dev build — the whole point
of 13h's `build_gt.py` sibling-naming fix, confirmed on real data rather than only in tests.

**Merged to `main`:** 13a, 13b, 13e, 13h (+ the `--review-dir` and blank-strata follow-ups).
**Not merged:** 13c — and **13i is gated on it**, so 13c is the natural parallel track.

## The decision waiting for him

Which metric selects the winning config:

- `false_redaction_rate` — CLAUDE.md §1's stated headline metric
- `keep_exact_match_rate` — the reading-quality framing that has since become the PM priority

Both are computed for **every** trial, so this does not need answering before the run: the
winner can be re-selected from `experiments/sweep_dev_v1/sweep_report.json` without
re-running a single engine. Present both sides neutrally and let him pick, then:

```
bash experiments/run_sweep_dev_v1.sh --freeze
```

Freezing writes the winners into `experiments/tuned_configs.json`, which is what makes
`build_arm("<engine>:tuned")` constructible. Until then it raises `ArmNotFrozenError` by
design.

## What to check in the report before freezing anything

1. **Did anything actually beat stock?** A config must beat stock on the pooled metric AND
   regress no stratum beyond the tolerance. "Nothing beat stock" is a real, reportable
   outcome — the stock arms stand and no tuned config gets frozen.
2. **`ct_secondary_capture` is n=1.** Its per-stratum regression guard is not meaningful;
   never let a tuned config be justified on that stratum's delta
   (`experiments/dev_v1_dropped.md`).
3. **The invention floor** comes from 5 blank images. If it reads as noise, say so rather
   than tuning against it.
4. **Latency per config** — raising `text_det_limit_side_len` costs compute, so a tuned win
   can be partly a compute win. It is in the table for that reason.

## Loose end

`ground_truth/dev_v1.csv.sha256`, `dev_v1_set.sha256`, `dev_v1_summary.txt` and
`text_presence_dev_v1.csv` are PHI-free and committable, and they are the provenance for
every tuned number that follows. They were **not** committed — offer to.

## Hard-won details from the annotation pass (do not relearn these)

- **Long commands must not be pasted.** Three separate commands truncated mid-flag in his
  terminal; one of them silently applied 1 of 4 `--exclude` files and still printed a
  plausible summary. Every multi-flag invocation now lives in a `bash experiments/*.sh`
  wrapper. Keep doing that.
- **`ct_axial` is exhausted** — gt_v1 uses all 66 of the manifest's 66, so the dev slice's
  blank controls come from `ct_scout` + `mg_tomo` instead.
- **Two out-of-scope document pages** (`75ea0cd7`, `cf23f1c1`) were dropped; they were caught
  by per-stratum token density running 9–12x above `gt_v1`. That density check is a cheap
  PHI-free way to spot an out-of-scope frame in any future slice.
- **A gcloud progress line naming a series UID reached the session** during the tar download.
  Flagged to him at the time; copy only the script's own count summary in future.
