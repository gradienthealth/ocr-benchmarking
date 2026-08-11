# 13h — Stock vs tuned + D-13.4 (where tuning happens) 🟡 build PHI-free, sweep is PHI-touching

> **Prerequisite: 13a (config identity).** Stock and tuned share a library version, so without 13a
> they average into one number — this prompt is the reason 13a exists.
>
> **D-13.4 is unresolved and must be settled in this session before any sweep runs.**

---

## Goal

Each end-to-end engine measured twice — stock config and tuned config — with the tuning done
somewhere that does not fit the frozen scored set, so the bake-off ranks *engines* rather than our
configuration of them.

## Context

- plan.md **PHASE 13, "Step 6 in detail"** — what stock and tuned mean per engine, written out.
  Read it before planning; it already specifies the knobs and the mislabelling traps.
- **Trap 1** in that section — do not tune on the scored set. `gt.csv` is small and frozen; sweeping
  against it and reporting the best result makes the tuned number an optimistic bound, not a
  measurement.
- `harness/runners/run_doctr.py`, `run_paddle_v6.py`, `run_easyocr.py` — the three arms.
- `ground_truth/gt_summary.txt` — stratum/vendor counts, for judging whether a dev slice can span
  them.

## Constraints — read these before designing the sweep

- **PP-OCRv6: `return_word_box=True`, `enable_mkldnn=False`, and the orientation/unwarp stages stay
  set in BOTH arms.** They are correctness requirements, not tuning: word-level output is
  contract-required, oneDNN crashes on this stack, and orientation moves the pixels the boxes are
  expressed in. "Stock" here means **stock thresholds**, not "PaddleOCR out of the box" — label it
  that way in the report or the finding is misdescribed.
- **docTR stock = pretrained defaults with no override** (D-9.2), which is what the current runner
  already is. Tuned = architecture override and/or detector postprocess thresholds.
- **EasyOCR is already non-default** (0.2 on both `text_threshold` and `low_text` vs library defaults
  0.7/0.4). Its stock/tuned pair answers how much of its behaviour is EasyOCR and how much is our
  threshold choice — include it even though it is the floor.
- **No length floors anywhere.** Single-character tokens (`L`, `R`) are in scope, and a misread `L`
  that gets redacted is a real false redaction.
- A gate calibrated on one stratum can delete real text on another: `conf>=60, len>=2` tuned on CT
  deleted **29–47% of genuine burned-in text** on ultrasound. Any threshold must be checked
  per-stratum, never accepted on a pooled number.

## Steps

1. Enter **Plan Mode**. Present **D-13.4** as an explicit decision with these options and a
   recommendation:
   (a) hold out a dev slice for tuning and report only on the rest — must span strata *and* vendors
   or the tuning fits one scanner;
   (b) tune on synthetic fixtures only;
   (c) report the tuned number explicitly labelled an upper bound, with the sweep size disclosed.
   State the cost of each in one line. **Stop and wait for my answer.**
2. Implement the tuned variants as distinct config identities (13a). No new engines.
3. Write the sweep script Arnav runs, emitting PHI-free aggregates only, with the sweep size recorded
   in the output so an upper-bound claim is auditable.

## Output format

Plan + the D-13.4 options first. Then the diff, then test output, then the sweep command.

## Done when

- `pytest` green; `ruff check .` clean.
- Six identities exist and aggregate separately (3 engines × stock/tuned).
- D-13.4 has a written resolution ready to paste into plan.md §4's decision table.
- The sweep script records how many configurations were tried.

## Finally

Spawn one fresh-context subagent. Scope: can any tuned run's numbers be reported without disclosing
that tuning happened, do stock and tuned share an identity anywhere, and does any correctness-required
PP-OCR setting differ between the two arms? Correctness only.
