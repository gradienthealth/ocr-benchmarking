# OCR engine benchmark report

| Field               | Value                          |
| ------------------- | ------------------------------ |
| Engine (model_name) | cloud-ocr-demo                 |
| Tier                | step-4-cloud-fallback+verifier |
| Exact version       | cloud-ocr-2026.07-demo         |
| Run hash            | e83ea9da9cdf                   |
| Set hash            | 29085d550351                   |
| Generated at        | 2026-08-04 17:18:34            |
| Images scored       | 5                              |

## Headline

| Metric                | Value  | Axis                                                    |
| --------------------- | ------ | ------------------------------------------------------- |
| False-redaction rate  | 20.00% | headline (plan.md) — valid KEEP tokens wrongly redacted |
| KEEP exact-match rate | 80.00% | primary reading quality — KEEP tokens read byte-exact   |

> **Unreconciled ranking:** plan.md designates **false-redaction rate** as THE headline; the current PM focus ranks engines by **OCR reading quality** (KEEP exact-match), with false-redaction demoted to secondary. The docs are **not yet reconciled**, so both are shown side by side here and are **never averaged** into a single score.

## Found vs Added

Two separate axes — **never averaged**. *Found*/*Omission* is the omission axis (real GT text the engine did / didn't read); *Added* is the hallucination axis (invented text — the dangerous one).

| Stratum           | Found | Omission | Added | GT total |
| ----------------- | ----- | -------- | ----- | -------- |
| ct_scout          | 0     | 1        | 2     | 1        |
| mammo             | 1     | 0        | 0     | 1        |
| secondary_capture | 2     | 0        | 0     | 2        |
| us_header         | 2     | 0        | 0     | 2        |
| Overall           | 5     | 1        | 2     | 6        |

![Found vs Added by stratum](report_cloud_verified_found_vs_added.png)

## Per-stratum breakout

| Stratum           | Images | FR rate | KEEP exact | Found | Omission | Added |
| ----------------- | ------ | ------- | ---------- | ----- | -------- | ----- |
| ct_scout          | 2      | 100.00% | 0.00%      | 0     | 1        | 2     |
| mammo             | 1      | 0.00%   | 100.00%    | 1     | 0        | 0     |
| secondary_capture | 1      | 0.00%   | 100.00%    | 2     | 0        | 0     |
| us_header         | 1      | 0.00%   | 100.00%    | 2     | 0        | 0     |
| Overall           | 5      | 20.00%  | 80.00%     | 5     | 1        | 2     |

![False-redaction rate by stratum](report_cloud_verified_fr_by_stratum.png)

## Latency & cost

Primary (model) and verifier (re-read) time are reported **separately** — never merged.

| Timer              | Mean (s) | Median (s) | p95 (s) |
| ------------------ | -------- | ---------- | ------- |
| Primary (model)    | 0.0061   | 0.0061     | 0.0061  |
| Verifier (re-read) | 0.0008   | 0.0000     | 0.0070  |

| Cost         | Value     |
| ------------ | --------- |
| Total        | $0.007500 |
| Mean / image | $0.001500 |

## Negative-control hallucination floor

Every prediction on a confirmed-blank frame is a pure hallucination; this is the floor.

| Metric                           | Value  |
| -------------------------------- | ------ |
| Control images                   | 1      |
| Hallucinated words (floor count) | 2      |
| Hallucinations / image           | 2.0000 |

![Hallucination floor](report_cloud_verified_neg_control.png)

## Diagnostic appendix

> **DIAGNOSTIC ONLY — never a ranking signal.** CER/WER are for debugging read quality; engines are ranked by false-redaction / reading-quality above, not by these. `n` is the number of images that produced a CER (an image with no matched pairs has none). CER/WER are **macro means** over those images.

| Stratum           | CER (mean) | WER (mean) | n | Char-censor rate |
| ----------------- | ---------- | ---------- | - | ---------------- |
| ct_scout          | —          | —          | 0 | 100.00%          |
| mammo             | 0.0000     | 0.0000     | 1 | 0.00%            |
| secondary_capture | 0.0000     | 0.0000     | 1 | 0.00%            |
| us_header         | 0.0000     | 0.0000     | 1 | 0.00%            |
| Overall           | 0.0000     | 0.0000     | 3 | 20.41%           |
