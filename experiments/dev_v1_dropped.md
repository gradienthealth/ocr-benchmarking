# dev_v1 — images dropped from the tuning slice

PHI-free: hashed `image_id`, stratum, and reason. No token text, no UID.

Disclosed because a slice that silently shrinks reads as "we tuned on the whole draw" when
we did not. Every number the sweep reports is over the images listed as REMAINING below.

| image_id | stratum | reason | dropped |
|---|---|---|---|
| `75ea0cd7` | mg_2d | Renders as a **full-page document**, not a burned-in overlay. Out of scope: this project scores short clinical tokens (patient IDs, accession numbers) burned into pixels, and one document page's token count would dominate the pooled metrics of a 24-image slice — pulling the tuned thresholds toward document OCR, the opposite of the target. | 2026-08-10 |
| `cf23f1c1` | ct_secondary_capture | Same: a full-page document. | 2026-08-10 |

**How they were caught, for next time.** Both were `accepted` unedited, and the giveaway was
per-stratum token DENSITY against `gt_v1`: `mg_2d` ran 36 tokens/image against gt_v1's 3
(12x), `ct_secondary_capture` 127 against 14 (9x), with 1 text-fix across 284 kept seeds.
Density-vs-gt_v1 is a cheap PHI-free check worth running on any future slice before building
it — an out-of-scope frame shows up as an order-of-magnitude density outlier long before
anyone opens an image.

**It was briefly marked `accepted` before being dropped**, which would have written Tesseract's
raw seeds — inventions included — into `dev_v1` as human-confirmed ground truth. The review
record was deleted rather than left in place; `build_gt`'s orphan gate would have caught it
either way, which is the gate doing its job.

## Slice after drops

- **22 images** (draw was 26; 2 lost to `AmbiguousFrameAxisError` at render, 2 dropped here)
- `mg_2d` 8 -> 7, `ct_secondary_capture` 2 -> 1
- blank controls unchanged: 3 (`ct_scout` 2, `mg_tomo` 1), all confirmed zero-token
- all 10 strata still represented

**`ct_secondary_capture` is now n=1.** Its per-stratum regression guard in the sweep is
therefore not meaningful — one image cannot show that a threshold is destroying a stratum.
Report any `ct_secondary_capture` delta as indicative only, and do not let a tuned config be
chosen on the strength of it.

## Note for the report

A `mg_2d`-labelled series that renders as a document page is worth a second look at some
point: the stratum label came from Gradient's manifest, so either the label or the series is
unusual. Not chased here — it does not affect `gt_v1`, which does not contain this series.
