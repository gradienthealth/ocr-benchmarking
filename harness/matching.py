"""Location-based matching of OCR predictions to ground-truth tokens. 🟢 PHI-free.

Phase 4 of the harness: decide, per ground-truth token, what the engine did with
*that token* — matched, omitted, or (for predictions) hallucinated — **by location
only**, never by character-error-rate. Matching finds the pairing; whether the
matched text is actually *correct* is Phase 5's job (`metrics.py`). That is why
`Match.word` carries the engine's raw, un-normalized text: `normalize()` is used
here only to make comparisons on the box-free path, never stored.

Engine-agnostic on purpose: this module sees only the frozen `contract.py` types
(`OCRWord`/`OCROutput`/`GTToken`) and depends on nothing else but the stdlib, so
every engine flows through the exact same matching logic and the comparison stays
fair.

Resolved decisions baked in (plan.md Phase 4, resolved 2026-07-08):
- D-4.1 — greedy one-to-one assignment by descending IoU (not Hungarian).
- D-4.2 — `iou_thr` is a parameter defaulting to 0.5; changing it later is a
  "results not comparable" event (like a gt.csv/engine-version change) and must
  be logged in run metadata. The value is recorded on every `MatchResult`.
- D-4.3 — box-free matching is a plain whole-token substring check on
  `normalize()`'d text (see `match()`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from harness.contract import GTToken, OCROutput, OCRWord, normalize

# (x0, y0, x1, y1), axis-aligned, TOP-LEFT origin, in PIXELS of the fed image —
# the shared convention of OCRWord.bbox and GTToken.bbox (see contract.py).
BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two axis-aligned pixel boxes in [0.0, 1.0]."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        # Two zero-area (degenerate) boxes: no overlap is decidable — call it 0.
        return 0.0
    return inter / union


@dataclass
class Match:
    """One GT token paired with one prediction."""

    gt: GTToken
    word: OCRWord
    # The raw prediction exactly as the engine emitted it — un-normalized. On the
    # box-free path this is the whole blob word, since no per-token prediction exists.

    iou: float | None
    # None on the box-free path: a substring match carries no location info.


@dataclass
class MatchResult:
    """What the engine did with each GT token on one image."""

    matches: list[Match]
    omissions: list[GTToken]  # GT tokens no prediction covered
    hallucinations: list[OCRWord]  # predictions covering no GT token (the dangerous axis)
    box_free: bool
    # True → pairing came from the substring path: lower-confidence, no location,
    # ranked separately from boxed matches — never mixed into the same ranking.

    iou_thr: float
    # Threshold this result was computed with; belongs in run metadata (D-4.2).


# Content-based sort keys (never input index) covering EVERY field, so ties on
# bbox alone — or bbox+text — still order identically under input shuffling.
def _word_key(w: OCRWord) -> tuple:
    return (
        w.bbox is None,
        w.bbox or (0.0, 0.0, 0.0, 0.0),
        w.text,
        w.confidence is None,
        w.confidence or 0.0,
    )


def _gt_key(t: GTToken) -> tuple:
    return (t.bbox, t.token_text, dataclasses.astuple(t))


def match(out: OCROutput, gt: list[GTToken], iou_thr: float = 0.5) -> MatchResult:
    """Pair predictions with GT tokens by location (or substring when box-free).

    Boxed path: greedy one-to-one assignment by descending IoU (D-4.1); a GT token
    with no overlapping prediction >= `iou_thr` is an OMISSION; a prediction
    overlapping no GT token is a HALLUCINATION (on a blank image, every prediction
    is a hallucination).

    Box-free path (`out.box_free`): is `normalize(gt.token_text)` a substring of
    the `normalize()`'d blob? Matches are flagged `box_free`, carry no location
    info, and are ranked separately.

    Order-independent: shuffling `out.words` or `gt` yields an equal result —
    assignment ties and output ordering are broken by content, never input index.
    """
    if out.box_free:
        # D-4.3: plain whole-token substring — the intentionally weaker/lower-confidence
        # path; token-boundary-aware matching deliberately deferred (plan.md D-4.3).
        # Words join in content order, not input order, so a shuffled multi-word
        # blob yields the same result (real box-free outputs are one word anyway).
        blob_word = OCRWord(
            text=" ".join(w.text for w in sorted(out.words, key=_word_key)),
            bbox=None,
            confidence=None,
        )
        blob = normalize(blob_word.text)
        matches = [
            Match(gt=t, word=blob_word, iou=None)
            for t in gt
            if normalize(t.token_text) in blob
        ]
        omissions = [t for t in gt if normalize(t.token_text) not in blob]
        # Hallucinations are undetectable box-free: with no per-word structure there
        # is no "prediction overlapping no GT token" to point at. Scored separately.
        return MatchResult(
            matches=sorted(matches, key=lambda m: _gt_key(m.gt)),
            omissions=sorted(omissions, key=_gt_key),
            hallucinations=[],
            box_free=True,
            iou_thr=iou_thr,
        )

    # Boxed path. Candidate pairs at or above threshold, greedily assigned
    # one-to-one by descending IoU (D-4.1). Ties broken by box/text content so the
    # assignment is identical regardless of input order.
    candidates: list[tuple[float, GTToken, OCRWord]] = []
    for t in gt:
        for w in out.words:
            if w.bbox is None:
                continue  # boxless word in a boxed output: can match nothing by location
            overlap = iou(t.bbox, w.bbox)
            if overlap >= iou_thr:
                candidates.append((overlap, t, w))
    candidates.sort(key=lambda c: (-c[0], _gt_key(c[1]), _word_key(c[2])))

    matches = []
    taken_gt: set[int] = set()  # id()-keyed: tokens/words may compare equal yet be
    taken_words: set[int] = set()  # distinct slots, and OCRWord is not hashable-by-slot
    for overlap, t, w in candidates:
        if id(t) in taken_gt or id(w) in taken_words:
            continue
        taken_gt.add(id(t))
        taken_words.add(id(w))
        matches.append(Match(gt=t, word=w, iou=overlap))

    omissions = [t for t in gt if id(t) not in taken_gt]
    hallucinations = [w for w in out.words if id(w) not in taken_words]

    return MatchResult(
        matches=sorted(matches, key=lambda m: _gt_key(m.gt)),
        omissions=sorted(omissions, key=_gt_key),
        hallucinations=sorted(hallucinations, key=_word_key),
        box_free=False,
        iou_thr=iou_thr,
    )
