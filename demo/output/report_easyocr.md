# OCR engine benchmark report

| Field               | Value               |
| ------------------- | ------------------- |
| Engine (model_name) | easyocr             |
| Tier                | step-1-local        |
| Exact version       | 1.7.2               |
| Run hash            | 82cf3c60c4ac        |
| Set hash            | a18e1b320476        |
| Generated at        | 2026-07-31 20:43:19 |
| Images scored       | 3                   |

## Headline

| Metric                | Value   | Axis                                                    |
| --------------------- | ------- | ------------------------------------------------------- |
| False-redaction rate  | 100.00% | headline (plan.md) — valid KEEP tokens wrongly redacted |
| KEEP exact-match rate | 0.00%   | primary reading quality — KEEP tokens read byte-exact   |

> **Unreconciled ranking:** plan.md designates **false-redaction rate** as THE headline; the current PM focus ranks engines by **OCR reading quality** (KEEP exact-match), with false-redaction demoted to secondary. The docs are **not yet reconciled**, so both are shown side by side here and are **never averaged** into a single score.

## Found vs Added

Two separate axes — **never averaged**. *Found*/*Omission* is the omission axis (real GT text the engine did / didn't read); *Added* is the hallucination axis (invented text — the dangerous one).

| Stratum        | Found | Omission | Added | GT total |
| -------------- | ----- | -------- | ----- | -------- |
| synth_ct_axial | 0     | 1        | 1     | 1        |
| synth_xr_chest | 0     | 2        | 2     | 2        |
| Overall        | 0     | 3        | 3     | 3        |

![Found vs Added by stratum](report_easyocr_found_vs_added.png)

## Per-stratum breakout

| Stratum        | Images | FR rate | KEEP exact | Found | Omission | Added |
| -------------- | ------ | ------- | ---------- | ----- | -------- | ----- |
| synth_ct_axial | 2      | 100.00% | 0.00%      | 0     | 1        | 1     |
| synth_xr_chest | 1      | 100.00% | 0.00%      | 0     | 2        | 2     |
| Overall        | 3      | 100.00% | 0.00%      | 0     | 3        | 3     |

![False-redaction rate by stratum](report_easyocr_fr_by_stratum.png)

## Latency & cost

Primary (model) and verifier (re-read) time are reported **separately** — never merged.

| Timer              | Mean (s) | Median (s) | p95 (s) |
| ------------------ | -------- | ---------- | ------- |
| Primary (model)    | 2.2478   | 1.2910     | 6.5333  |
| Verifier (re-read) | —        | —          | —       |

| Cost         | Value     |
| ------------ | --------- |
| Total        | $0.000000 |
| Mean / image | $0.000000 |

## Negative-control hallucination floor

Every prediction on a confirmed-blank frame is a pure hallucination; this is the floor.

| Metric                           | Value  |
| -------------------------------- | ------ |
| Control images                   | 1      |
| Hallucinated words (floor count) | 0      |
| Hallucinations / image           | 0.0000 |

![Hallucination floor](report_easyocr_neg_control.png)

## Diagnostic appendix

> **DIAGNOSTIC ONLY — never a ranking signal.** CER/WER are for debugging read quality; engines are ranked by false-redaction / reading-quality above, not by these. `n` is the number of images that produced a CER (an image with no matched pairs has none). CER/WER are **macro means** over those images.

| Stratum        | CER (mean) | WER (mean) | n | Char-censor rate |
| -------------- | ---------- | ---------- | - | ---------------- |
| synth_ct_axial | —          | —          | 0 | 100.00%          |
| synth_xr_chest | —          | —          | 0 | 100.00%          |
| Overall        | —          | —          | 0 | 100.00%          |
