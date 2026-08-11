# 13l — The final report 🟡 written from PHI-free aggregates only

> **Renumbered 13j → 13l on 2026-08-10** to make room for the verifier arm (13j) and cloud runners
> (13k), which plan.md PHASE 13 invokes as test-plan steps 3 and 4. This stays the terminus.
>
> **Prerequisite: 13i has produced real numbers.** If the verifier arm (13j) or a cloud arm (13k) was
> run, their tables belong here too; if they were skipped, say so explicitly rather than leaving the
> reader to wonder. This is the deliverable the whole project exists to produce — the document Cal and
> Gradient read.
>
> Claude never sees a token, a crop, or `gt.csv`. It writes the report **from the aggregate tables
> Arnav pastes in**.

---

## Goal

A stratified findings report that ranks the engines on reading quality, states plainly what was and
was not measured, and cannot be misread as a claim it does not support.

## Context

- plan.md **PHASE 13** — the six-step test plan and what each step establishes.
- plan.md **§6 STATUS BOARD** — "Findings the report must carry". Every bullet there changes how a
  number is reported; work through them explicitly.
- `harness/report.py` — existing table/chart rendering.
- The artifact: **`gt_v1`, 199 images, 2351 rows, 102 with text, 97 blank, KEEP 1973 / PHI 378**,
  `gt.csv.sha256 afc65989…` — every number in the report is stamped with this.

## What the report must state, not bury

- **Reading quality is the ranking signal.** False redaction is secondary. Do not rank on a blended
  score.
- **Hallucination and omission are reported separately, never averaged.** Found axis and Added axis
  stay in separate columns.
- **Three tables, three questions**: end-to-end, detector-only, reader (oracle boxes). Reader numbers
  are not comparable to end-to-end ones — say so *in the table caption*, not in a footnote.
- **The blank control is 64 images, not 66** — two `ct_axial` frames turned out to carry text and left
  the control set. That is the negative control working, and it belongs in the report.
- **`ct_scout` and `mg_tomo` have zero text-bearing images** — no recall denominator, so any per-image
  recall over them is a meaningless saturated 100%. Print `n/a` with the reason.
- **`mg_2d` is 40-of-44 text-bearing but only 123 rows, and is 233 series in the full manifest** — a
  pooled dataset-wide rate must be reweighted by the `share` column first. `gt_v2` is deliberately
  non-proportional; an unweighted pooled number is wrong.
- **Latency is CPU-only** (D-9.1) and not representative of GPU-served production numbers. Flag it
  wherever latency appears.
- **Confidence is not comparable across engines**: docTR's and PP-OCRv6's are the same family, but
  PP-OCR's is a *line-inherited recognition* score, and a generative arm's is a decode probability.
  Never present them in one sortable column without saying this.
- **The field was swept and no new end-to-end engine met the contract** (5-agent panel, 2026-08-09).
  Say it explicitly — otherwise the small lineup reads as an unexamined default. Include why the
  usual benchmarks don't apply: OCRBench v2 / OmniDocBench / HF leaderboards score document parsing,
  not sparse overlay tokens; ICDAR2015 Incidental Scene Text and TextOCR are the closest analogs.
- **Stock vs tuned** (13h): if tuning happened on a dev slice, say so and disclose the sweep size. An
  undisclosed tuned number is an optimistic bound presented as a measurement.
- **Sample size honesty**: 199 images. Give per-stratum n beside every per-stratum rate, and do not
  report a rate for a stratum too small to support one.

## Constraints

- **Never re-identify**: no UIDs, no dates, no institution names, no `image_id`-to-source mapping, no
  token text anywhere in the report (CLAUDE.md §8 — re-identification by linkage).
- Claude writes the report **only** from the PHI-free aggregate tables pasted into the session. If a
  number is needed that is not in those tables, ask for it — never estimate it.
- No engine is declared "the winner" in code. The harness emits numbers; the keep/replace decision is
  downstream and human (plan.md §5).
- Do not call the dataset de-identified or anonymous — it is **pseudonymized and still PHI**.

## Steps

1. Ask Arnav for the aggregate tables (all three arms + the negative-control counts + timings).
2. Draft the report to `results/FINAL_REPORT.md`: headline, method, the three tables, per-stratum
   breakdown, limitations, and a "what this does not show" section.
3. Include a **limitations** section that states the sample size, CPU-only latency, oracle-box caveat,
   single-annotator ground truth, and any unresolved self-agreement number (D-10.7).

## Output format

The report file, plus a ≤10-line executive summary printed in the session for pasting into Slack.

## Done when

- `results/FINAL_REPORT.md` exists, every table carries the `gt_v1` hash stamp and per-stratum n,
  and the limitations section is present.
- A grep of the report finds no UID, no date, no token text.
- Every bullet in "What the report must state" is addressed somewhere in the document.

## Finally

Spawn one fresh-context subagent to read the report as a skeptical reviewer. Scope: which sentences
claim more than the numbers support, is any rate reported without its denominator, and could any
reader mistake the oracle-box table for end-to-end performance? Report the specific sentences.
