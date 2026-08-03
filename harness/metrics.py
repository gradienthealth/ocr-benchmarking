"""Phase 5 scoring — turn one image's `MatchResult` into raw metric ingredients. 🟢 PHI-free.

Matching (Phase 4) decided *what the engine did with each GT token* by location.
Scoring decides *whether the matched text is correct* and packages the result as the
**raw, unaveraged per-image ingredients** the six scored metrics are built from. All
averaging — rates, mean/median/p95 — happens later at Phase 6 aggregation, never here.

Two hard structural rules from CLAUDE.md, enforced by the shape of the returned dict
(not merely by comment):

- **Hallucination and omission are never averaged into one "accuracy."** They live in
  two disjoint groups of keys: the Found side (real GT text the engine did / didn't
  read) and the Added side (invented text). See `found_count`/`omission_count` vs
  `added_count`.
- **False-redaction rate is the headline metric.** Its per-image ingredients are
  `false_redaction_count` (numerator) over `keep_total` (denominator — KEEP tokens
  ONLY, never all GT). CER/WER and confidence are DIAGNOSTIC only: they live under a
  nested `"diagnostic"` key and share no key name with the ranking group, so a refactor
  cannot accidentally promote them into the ranking signal.

Engine-agnostic: this module sees only the frozen `contract.py` types via the
`MatchResult` it is handed, plus the frozen `normalize()`. Nothing engine-specific.
"""

from __future__ import annotations

import jiwer

from harness.contract import GTToken, normalize
from harness.matching import MatchResult

KEEP = "KEEP"
PHI = "PHI"


def is_allowed(text: str, allowlist: set[str]) -> bool:
    """True iff `normalize(text)` is an exact member of `allowlist`.

    Exact-match-after-`normalize()` — NOT fuzzy, NOT substring, NOT pattern. This is the
    allowlist gate of the redaction pipeline: a token is kept (not blacked out) only if
    OCR reads it into a known-good value. `allowlist` is a set of literal known-good
    token strings supplied by the caller; `is_allowed` is scope-agnostic and does not
    care whether that set is per-image or global (Phase 6's concern).
    """
    return normalize(text) in allowlist


def _first_gt(matched: MatchResult) -> GTToken | None:
    """Any GT token on this image — for carrying modality/stratum/vendor through.

    None on a blank negative control (zero GT tokens): its stratum metadata is not
    recoverable from the MatchResult and is supplied by Phase 6 from the manifest.
    """
    for m in matched.matches:
        return m.gt
    for t in matched.omissions:
        return t
    return None


def score(
    matched: MatchResult,
    allowlist: set[str],
    elapsed: float,
    cost: float,
) -> dict:
    """Score one image's `MatchResult` into raw, unaveraged per-image ingredients.

    Returns a flat dict of RANKING/headline keys plus one nested `"diagnostic"` dict.
    The two never share a key name (see module docstring); a test pins that invariant.

    Ranking keys:
      keep_total / keep_exact_match_count / keep_exact_match_all
          KEEP-token read fidelity, post-`normalize()` byte-exact.
      false_redaction_count  — THE HEADLINE. A KEEP token is falsely redacted when it
          is NOT (matched AND read byte-exact AND on the allowlist): the redaction
          pipeline blacks out anything it can't confirm, destroying good data. The rate
          is `false_redaction_count / keep_total`, computed at Phase 6. Denominator is
          KEEP tokens ONLY.
      found_count / omission_count / gt_total  — the Found (omission) axis: real GT text
          the engine did / did not read. NEVER blended with the Added axis.
      added_count  — the Added (hallucination) axis: predictions covering no GT token,
          the dangerous "invented text." NEVER blended with the Found axis.
      negative_control / negative_control_floor_count  — on a confirmed-blank frame
          (zero GT tokens) EVERY prediction is a pure hallucination; this is the
          hallucination-floor signal Phase 6 aggregates over the control set.
      box_free / iou_thr  — run metadata carried through (box-free is ranked separately;
          iou_thr belongs in run metadata, D-4.2).
      elapsed / cost  — carried through UNAGGREGATED (percentiles happen at Phase 6).
      modality / stratum / vendor  — carried through for Phase 6 stratified breakout;
          None on a blank control (no GT token to read them from).

    Diagnostic-only (nested, never a ranking signal): CER/WER, char-level censoring
    counts, and matched-word confidence values.

    PHI tokens are bookkept as correctly-redacted context: they are counted in
    gt_total/found/omission but never enter the KEEP exact-match or false-redaction
    numbers — those denominators are KEEP-only.
    """
    keep_matches = [m for m in matched.matches if m.gt.label == KEEP]
    keep_omissions = [t for t in matched.omissions if t.label == KEEP]
    keep_total = len(keep_matches) + len(keep_omissions)

    # A KEEP token is correctly kept iff it was matched, read byte-exact, AND the read
    # value is on the allowlist. Anything short of that gets blacked out = false redaction.
    keep_exact_match_count = 0
    false_redaction_count = 0
    keep_chars_total = 0
    keep_chars_censored = 0
    for m in keep_matches:
        gt_norm = normalize(m.gt.token_text)
        keep_chars_total += len(gt_norm)
        exact = normalize(m.word.text) == gt_norm
        if exact:
            keep_exact_match_count += 1
        kept = exact and is_allowed(m.word.text, allowlist)
        if not kept:
            false_redaction_count += 1
            keep_chars_censored += len(gt_norm)
    for t in keep_omissions:
        # Omitted KEEP token: never read, so never confirmed -> redacted -> false.
        gt_norm = normalize(t.token_text)
        keep_chars_total += len(gt_norm)
        false_redaction_count += 1
        keep_chars_censored += len(gt_norm)

    found_count = len(matched.matches)
    omission_count = len(matched.omissions)
    gt_total = found_count + omission_count
    added_count = len(matched.hallucinations)

    # Negative control = an image with zero GT tokens. Every prediction on it is a pure
    # hallucination; that count is the floor signal (== added_count here, but labelled
    # separately so Phase 6 can aggregate it over just the control set).
    negative_control = gt_total == 0
    negative_control_floor_count = added_count if negative_control else 0

    first = _first_gt(matched)

    # --- diagnostic-only (CER/WER, censoring chars, confidences) --------------------
    # jiwer raises on an empty reference; guard when there are no matched pairs.
    ref = " ".join(normalize(m.gt.token_text) for m in matched.matches)
    hyp = " ".join(normalize(m.word.text) for m in matched.matches)
    cer = jiwer.cer(ref, hyp) if ref else None
    wer = jiwer.wer(ref, hyp) if ref else None

    return {
        # --- KEEP read fidelity ---
        "keep_total": keep_total,
        "keep_exact_match_count": keep_exact_match_count,
        "keep_exact_match_all": keep_total > 0 and keep_exact_match_count == keep_total,
        # --- headline ---
        "false_redaction_count": false_redaction_count,
        # --- Found (omission) axis — never blended with Added ---
        "found_count": found_count,
        "omission_count": omission_count,
        "gt_total": gt_total,
        # --- Added (hallucination) axis — never blended with Found ---
        "added_count": added_count,
        # --- negative-control hallucination floor ---
        "negative_control": negative_control,
        "negative_control_floor_count": negative_control_floor_count,
        # --- run metadata carried through ---
        "box_free": matched.box_free,
        "iou_thr": matched.iou_thr,
        "elapsed": elapsed,
        "cost": cost,
        "modality": first.modality if first else None,
        "stratum": first.stratum if first else None,
        "vendor": first.vendor if first else None,
        # --- DIAGNOSTIC ONLY — never a ranking signal (see module docstring) ---
        "diagnostic": {
            "cer": cer,
            "wer": wer,
            "keep_chars_total": keep_chars_total,
            "keep_chars_censored": keep_chars_censored,
            "matched_confidences": [m.word.confidence for m in matched.matches],
        },
    }
