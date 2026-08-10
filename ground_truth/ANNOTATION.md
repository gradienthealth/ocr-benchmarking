# ANNOTATION.md — what to annotate in the 10d review UI 🟢 PHI-FREE

> **This file is PHI-FREE and committed. Keep it that way.** Every example below is a generic
> vendor-overlay string or a synthetic placeholder — never a real token off a real frame. If you
> need to ask about a specific token, describe its *class* ("a patient ID like 1456-whatever"),
> never its value.

`review_gt.py` documents *how the tool works*. This file is the other half: the conventions the
annotation itself has to follow. Arnav is the **sole annotator** (D-10.7) and `gt.csv` is
**frozen + hashed** (CLAUDE.md rule #8), so there is no second pass to catch drift and an
inconsistently annotated GT can only be discarded and redone. Read this before a session, not
after.

Pinned as **D-10.9** (2026-08-07). It also resolves the box-tightness question left open in
plan.md Phase 9 ("how tightly should GT boxes be drawn" — a scoring decision, not a cosmetic one).

---

## 1. Token granularity — one box per whitespace-separated run

**Whitespace is the only splitter.** Never split on `.`, `/`, `:`, or `-`; never merge across a
space.

| On the pixels | Boxes | Text |
| --- | --- | --- |
| `12.3` | 1 | `12.3` |
| `12/32/12` | 1 | `12/32/12` |
| `1.2mm` | 1 | `1.2mm` |
| `1.2 mm` | 2 | `1.2` · `mm` |
| `soft/32`, `soft/ss40` | 1 | as printed |
| `Angle: 0.0` | 2 | `Angle:` · `0.0` |
| `DFOV 19.3cm` | 2 | `DFOV` · `19.3cm` |
| `DFOV:19.3cm` | 1 | `DFOV:19.3cm` |

**Why this is forced, not chosen.** Matching is one-to-one by IoU ≥ 0.5 against each engine's
**word** boxes (`harness/matching.py`), and no engine in the lineup splits on punctuation —
docTR, PaddleOCR with `return_word_box=True`, and Tesseract level 5 all split on whitespace only.
So a GT box holding `Angle: 0.0` as one unit scores an engine that reads it *correctly* as
1 omission **plus** 2 hallucinations: a localization artifact masquerading as a reading failure,
which is exactly the defect logged against EasyOCR's line-level output (plan.md Phase 9).

**In practice this means the seed's splitting is already the convention.** `seed_tesseract.py`
emits `level == 5` (word level), so accept its cuts. Intervene only where Tesseract merged two
real words into one box, or split one word across two — a real error it makes on overlay text.

## 2. Transcribe what is printed — never what it should say

- **Punctuation stays**, attached as printed: `Angle:` keeps its colon. The frozen `normalize()`
  (`harness/contract.py:121`) strips surrounding whitespace and applies NFC — *nothing else*. Any
  "drop the trailing colon" habit is an invisible extra rule the engines don't share.
- **Case is preserved**: `mm` ≠ `MM`, `l` ≠ `L`. `normalize()` is case-sensitive on purpose.
- **Don't correct the content.** If a mammo view label reads `MLOE` on the pixels, the question is
  what the *glyphs* show, not what a valid view name would be. A stray appended character is a
  classic Tesseract failure and should be fixed **to the pixels**; a real suffix should be kept.
  If the glyph can't be resolved by eye, defer (§6) — don't guess.
- The box text is the yardstick every engine is scored against. A wrong GT string makes every
  engine that read the token *correctly* look wrong, all at once.

## 3. Box tightness — tight to the glyphs

Draw hand-added boxes **tight around the ink**, and leave the seed's boxes tight. Don't pad for
comfort.

This is a scoring decision: EasyOCR's boxes measure ~1.9× taller than the text they bound (26px
around 14px of glyph), which lands its IoU at 0.41–0.50 against a tight GT — at or below the
matcher's threshold. `review_gt.py --summary` reports the GT box-geometry distribution, and that
number is what settles whether `iou_thr` stays at 0.5. A loose annotation convention would move
every engine's IoU and quietly hide the problem instead of measuring it.

## 4. `PHI` vs `KEEP`

**The rule:** does this text identify or re-link the patient?

**`PHI`** — patient ID, MRN, accession number, patient or physician name, **any** date (full dates
survive in this dataset and keep it PHI under Safe Harbor), and **hospital / institution name**.
The institution isn't in the enumerated 18 as an individual's name, but DICOM's Basic Application
Level Confidentiality Profile removes `InstitutionName` as standard practice and it re-links
against the surviving `StudyDate` + UIDs (CLAUDE.md §8). Not worth litigating — the redactor
blacks it out either way.

**`KEEP`** — everything else burned into the pixels that a redactor should preserve:

- measurements — `12.3`, `1.2mm`, `19.3cm`
- acquisition / reconstruction parameters — `DFOV`, `soft/ss40`, `soft/32`, `Angle: 0.0`, window
  and brightness settings, kVp/mAs
- laterality and view markers — `L`, `R`, `MLO`, `CC`, `RMLO`, `XCCL`
- vendor and technique text

Cryptic ≠ identifying. Scanner parameters are KEEP however opaque they look.

**What the label actually drives.** `keep_exact_match` and `false_redaction_count` only — both
denominators are KEEP-only (`harness/metrics.py:100-125`). The reading-quality axes you rank
engines on (Found vs. Added, hallucination floor, CER/WER) count every token regardless of label.
So a label you get wrong shifts a secondary metric; it cannot corrupt the ranking.

**The one outcome to avoid: a whole image left all-`PHI`.** Then `keep_total == 0` and the image
contributes nothing to the false-redaction metric at all. The seeder hardcodes `label = "PHI"` on
every token because Tesseract cannot know better (D-10c.2) — that placeholder is not information,
and leaving it is a silent zero, not a neutral default.

## 5. The efficient pass

On vendor-overlay strata (`ct_secondary_capture`, ultrasound) nearly every token is KEEP:

1. **`K`** — bulk-set every **visible** box to KEEP.
2. **`L`** on the two or three identifier-shaped tokens to flip them back to PHI.
3. **`A`** to save and advance.

`P` is the reverse bulk for an ID-banner frame where most tokens really are PHI. Both act on
visible boxes only — labelling what you can't see is guessing. `Ctrl+Z` undoes.

## 6. When you can't decide — defer, don't guess

**`D`** → pick a reason:

- `unreadable` — the glyphs can't be resolved at this render.
- `ambiguous_token` — legible, but you can't tell what it is or how to label it.
- `possible_non_blank_control` — a `ct_axial` frame that appears to carry text. Flag it; a control
  frame with text leaves the control set (D-10.5).
- `needs_cal` — a policy question, not a reading question.

A deferred image is a distinct state from unreviewed, so nothing gets lost. **Open for Cal:**
whether an ambiguous token should default `PHI` or `KEEP` outright; until that's answered, defer
it rather than pick one.

## 7. Things that have already gone wrong — don't repeat them

- **No length floor on tokens.** A single `L` or `R` is real content, and a misread `L` that gets
  redacted **is** a false redaction — the headline metric. One real frame's entire content was
  `R` and `L`.
- **The confidence gate is a view filter, not a delete.** A token hidden by the slider stays in
  the frozen seed on disk; an over-filtered stratum is fixed by moving the slider, not by
  re-seeding. Deleting a box, by contrast, sticks.
- **Don't sweep away faint text.** Deletion is the dominant gesture (`mg_2d` seeds ~364 boxes per
  frame of which ~1.5 are real), which is exactly why a real box lost in a sweep is the easy
  mistake. A missing GT box turns every engine's correct read into a hallucination.
- **Exchange `image_id`s, never ordinals.** If you hand a batch of labels back for anything,
  pair each with its `image_id` — "the first 10 in the folder" has silently permuted before,
  because file managers natural-sort and `ls | sort` doesn't.
