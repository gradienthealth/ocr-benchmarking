# OCR engine benchmark report

| Field               | Value               |
| ------------------- | ------------------- |
| Engine (model_name) | pp-ocrv6_medium     |
| Tier                | step-1-local        |
| Exact version       | 3.7.0               |
| Run hash            | 5ba0ff87b045        |
| Set hash            | a18e1b320476        |
| Generated at        | 2026-07-31 20:43:19 |
| Images scored       | 3                   |

## Headline

| Metric                | Value   | Axis                                                    |
| --------------------- | ------- | ------------------------------------------------------- |
| False-redaction rate  | 0.00%   | headline (plan.md) — valid KEEP tokens wrongly redacted |
| KEEP exact-match rate | 100.00% | primary reading quality — KEEP tokens read byte-exact   |

> **Unreconciled ranking:** plan.md designates **false-redaction rate** as THE headline; the current PM focus ranks engines by **OCR reading quality** (KEEP exact-match), with false-redaction demoted to secondary. The docs are **not yet reconciled**, so both are shown side by side here and are **never averaged** into a single score.

## Found vs Added

Two separate axes — **never averaged**. *Found*/*Omission* is the omission axis (real GT text the engine did / didn't read); *Added* is the hallucination axis (invented text — the dangerous one).

| Stratum        | Found | Omission | Added | GT total |
| -------------- | ----- | -------- | ----- | -------- |
| synth_ct_axial | 1     | 0        | 0     | 1        |
| synth_xr_chest | 2     | 0        | 0     | 2        |
| Overall        | 3     | 0        | 0     | 3        |

![Found vs Added by stratum](report_pp-ocrv6_medium_found_vs_added.png)

## Per-stratum breakout

| Stratum        | Images | FR rate | KEEP exact | Found | Omission | Added |
| -------------- | ------ | ------- | ---------- | ----- | -------- | ----- |
| synth_ct_axial | 2      | 0.00%   | 100.00%    | 1     | 0        | 0     |
| synth_xr_chest | 1      | 0.00%   | 100.00%    | 2     | 0        | 0     |
| Overall        | 3      | 0.00%   | 100.00%    | 3     | 0        | 0     |

![False-redaction rate by stratum](report_pp-ocrv6_medium_fr_by_stratum.png)

## Latency & cost

Primary (model) and verifier (re-read) time are reported **separately** — never merged.

| Timer              | Mean (s) | Median (s) | p95 (s) |
| ------------------ | -------- | ---------- | ------- |
| Primary (model)    | 3.8590   | 3.2448     | 7.5149  |
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

![Hallucination floor](report_pp-ocrv6_medium_neg_control.png)

## Diagnostic appendix

> **DIAGNOSTIC ONLY — never a ranking signal.** CER/WER are for debugging read quality; engines are ranked by false-redaction / reading-quality above, not by these. `n` is the number of images that produced a CER (an image with no matched pairs has none). CER/WER are **macro means** over those images.

| Stratum        | CER (mean) | WER (mean) | n | Char-censor rate |
| -------------- | ---------- | ---------- | - | ---------------- |
| synth_ct_axial | 0.0000     | 0.0000     | 1 | 0.00%            |
| synth_xr_chest | 0.0000     | 0.0000     | 1 | 0.00%            |
| Overall        | 0.0000     | 0.0000     | 2 | 0.00%            |
