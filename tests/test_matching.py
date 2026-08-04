"""Phase 4 tests for harness/matching.py — SYNTHETIC fixtures only, zero PHI.

Every image and token below comes from tests/synthetic.py (fake tokens like
"CMFN", "GRDN1234") or is hand-built with obviously-fake strings. Each Phase 4
acceptance criterion (plan.md:480-483) has its own named test.
"""

from __future__ import annotations

import dataclasses
import random

from harness.contract import GTToken, OCROutput, OCRWord, normalize
from harness.matching import iou, match
from tests.synthetic import (
    SYNTHETIC_RAW_RESPONSE,
    SYNTHETIC_VERSION,
    box_free_output,
    hallucination_output,
    make_synthetic_image,
    misread_output,
    omission_output,
    perfect_output,
)

_TOKENS = ("CMFN", "GRDN1234", "ACC-0001")


def _gt(tokens=_TOKENS) -> list[GTToken]:
    _, gt = make_synthetic_image(tokens)
    return gt


def _boxed_output(words: list[OCRWord], model_name: str = "synthetic-handmade") -> OCROutput:
    return OCROutput(
        words=words,
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name=model_name,
        version=SYNTHETIC_VERSION,
    )


# --- iou() -------------------------------------------------------------------


def test_iou_hand_computed_exact():
    # a=(0,0,10,10), b=(5,0,15,10): intersection 5x10=50, union 100+100-50=150.
    assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == 50.0 / 150.0


def test_iou_identical_boxes_is_one():
    assert iou((3.0, 4.0, 20.0, 30.0), (3.0, 4.0, 20.0, 30.0)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0
    # Touching edges share zero area: still 0.
    assert iou((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)) == 0.0


def test_iou_zero_area_boxes_is_zero():
    assert iou((5.0, 5.0, 5.0, 5.0), (5.0, 5.0, 5.0, 5.0)) == 0.0


# --- boxed path: the four verdict cases ---------------------------------------


def test_perfect_overlap_matched():
    gt = _gt()
    result = match(perfect_output(gt), gt)
    assert len(result.matches) == len(gt)
    assert result.omissions == []
    assert result.hallucinations == []
    assert result.box_free is False
    assert all(m.iou == 1.0 for m in result.matches)
    assert {m.gt.token_text for m in result.matches} == set(_TOKENS)


def test_no_overlap_omission():
    gt = _gt()
    result = match(omission_output(gt, index=1), gt)
    assert [t.token_text for t in result.omissions] == [gt[1].token_text]
    assert len(result.matches) == len(gt) - 1
    assert result.hallucinations == []


def test_extra_prediction_hallucination():
    gt = _gt()
    result = match(hallucination_output(gt), gt)
    assert len(result.matches) == len(gt)
    assert result.omissions == []
    assert [w.text for w in result.hallucinations] == ["ZZZZ-FAKE"]


def test_blank_image_every_prediction_hallucinated():
    # Negative control: gt=[] (blank frame), engine still emits a word.
    result = match(hallucination_output(gt=[]), gt=[])
    assert result.matches == []
    assert result.omissions == []
    assert len(result.hallucinations) == 1
    assert result.hallucinations[0].text == "ZZZZ-FAKE"


def test_order_independence():
    gt = _gt()
    out = hallucination_output(gt)  # matches AND a hallucination in one output
    baseline = match(out, gt)
    for seed in (1, 2, 3):
        shuffled_words = list(out.words)
        random.Random(seed).shuffle(shuffled_words)
        shuffled_gt = list(gt)
        random.Random(seed + 100).shuffle(shuffled_gt)
        assert match(_boxed_output(shuffled_words, out.model_name), shuffled_gt) == baseline


def test_order_independence_identical_gt_boxes():
    # Regression (fresh review, Phase 4): two GT tokens sharing one bbox but with
    # different text must resolve the assignment tie by content, not input order.
    base = _gt(("CMFN",))[0]
    t1 = dataclasses.replace(base, token_text="CMFN")
    t2 = dataclasses.replace(base, token_text="GRDN1234")
    pred = OCRWord(text="CMFN", bbox=base.bbox, confidence=0.9)
    a = match(_boxed_output([pred]), [t1, t2])
    b = match(_boxed_output([pred]), [t2, t1])
    assert a == b
    assert len(a.matches) == 1 and len(a.omissions) == 1


# --- boxed path: pairing details ----------------------------------------------


def test_raw_text_preserved():
    # A misread still pairs by location; matching never touches text correctness,
    # and the paired text is the engine's raw string (Phase 5 judges it).
    gt = _gt()
    result = match(misread_output(gt, index=0), gt)
    assert result.omissions == []
    assert result.hallucinations == []
    by_gt = {m.gt.token_text: m.word.text for m in result.matches}
    assert by_gt["CMFN"] == "CMEN"  # mutate_token("CMFN")

    # Raw means raw: surrounding whitespace survives un-normalized.
    padded = _boxed_output([OCRWord(text="  CMFN  ", bbox=gt[0].bbox, confidence=0.9)])
    m = match(padded, [gt[0]]).matches[0]
    assert m.word.text == "  CMFN  "
    assert normalize(m.word.text) == "CMFN"  # normalization is comparison-only


def test_greedy_one_to_one():
    # Two predictions over one GT box: the higher-IoU one wins the assignment,
    # the loser overlaps no *remaining* GT token -> hallucination (D-4.1).
    gt_tok = _gt(("CMFN",))[0]
    x0, y0, x1, y1 = gt_tok.bbox
    exact = OCRWord(text="CMFN", bbox=(x0, y0, x1, y1), confidence=0.9)
    shifted = OCRWord(text="CMFN?", bbox=(x0 + (x1 - x0) * 0.2, y0, x1, y1), confidence=0.8)
    result = match(_boxed_output([shifted, exact]), [gt_tok])
    assert len(result.matches) == 1
    assert result.matches[0].word == exact
    assert result.matches[0].iou == 1.0
    assert result.hallucinations == [shifted]
    assert result.omissions == []


def test_below_threshold_is_omission_and_hallucination():
    # Overlap exists but IoU < iou_thr: the GT token is omitted AND the weakly
    # overlapping prediction is a hallucination — no partial credit by location.
    gt_tok = _gt(("CMFN",))[0]
    x0, y0, x1, y1 = gt_tok.bbox
    weak = OCRWord(text="CMFN", bbox=(x1 - (x1 - x0) * 0.1, y0, x1 + (x1 - x0), y1), confidence=0.9)
    assert 0.0 < iou(gt_tok.bbox, weak.bbox) < 0.5
    result = match(_boxed_output([weak]), [gt_tok])
    assert result.matches == []
    assert result.omissions == [gt_tok]
    assert result.hallucinations == [weak]


def test_iou_thr_is_respected_as_parameter():
    # D-4.2: the threshold is a knob, not a constant. The same weak overlap that
    # fails at the default matches when the caller lowers iou_thr.
    gt_tok = _gt(("CMFN",))[0]
    x0, y0, x1, y1 = gt_tok.bbox
    weak = OCRWord(text="CMFN", bbox=(x0 + (x1 - x0) * 0.7, y0, x1, y1), confidence=0.9)
    overlap = iou(gt_tok.bbox, weak.bbox)
    assert 0.1 < overlap < 0.5
    default = match(_boxed_output([weak]), [gt_tok])
    lowered = match(_boxed_output([weak]), [gt_tok], iou_thr=0.1)
    assert default.matches == [] and default.iou_thr == 0.5
    assert len(lowered.matches) == 1 and lowered.iou_thr == 0.1


# --- box-free path -------------------------------------------------------------


def test_box_free_substring_hit():
    gt = _gt()
    result = match(box_free_output(gt), gt)
    assert result.box_free is True
    assert len(result.matches) == len(gt)
    assert result.omissions == []
    assert result.hallucinations == []
    # No location info anywhere on this path.
    assert all(m.iou is None for m in result.matches)
    assert all(m.word.bbox is None for m in result.matches)


def test_box_free_substring_miss():
    gt = _gt()
    blob_without_last = box_free_output(gt[:-1])  # "ACC-0001" absent from the blob
    result = match(blob_without_last, gt)
    assert result.box_free is True
    assert {m.gt.token_text for m in result.matches} == {"CMFN", "GRDN1234"}
    assert [t.token_text for t in result.omissions] == ["ACC-0001"]
    assert all(m.iou is None for m in result.matches)


def test_box_free_normalize_applied():
    # Comparison runs on normalize()'d text: NFC-composing the blob's decomposed
    # accent and stripping the padded GT token both still hit.
    decomposed = "CAFE\u0301-01"  # E + U+0301 combining acute (decomposed)
    composed = "CAF\u00c9-01"  # U+00C9 (composed)
    assert decomposed != composed  # sanity: distinct strings until normalized
    gt_tok = _gt(("CMFN",))[0]
    gt_composed = [
        dataclasses.replace(gt_tok, token_text=composed),
        dataclasses.replace(gt_tok, token_text="  CMFN"),
    ]
    blob = OCROutput(
        words=[OCRWord(text=f"header {decomposed} CMFN", bbox=None, confidence=None)],
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name="synthetic-boxfree",
        version=SYNTHETIC_VERSION,
        box_free=True,
    )
    result = match(blob, gt_composed)
    assert result.box_free is True
    assert len(result.matches) == 2
    assert result.omissions == []
    # Raw blob text is exposed un-normalized on the matched word.
    assert all(m.word.text == f"header {decomposed} CMFN" for m in result.matches)


def test_box_free_order_independence():
    # Regression (fresh review, Phase 4): a multi-word box-free blob must yield
    # the same result regardless of out.words order (blob joins in content order).
    gt = _gt(("CMFN", "GRDN1234"))
    words = [OCRWord(text=t.token_text, bbox=None, confidence=None) for t in gt]

    def blob(ws: list[OCRWord]) -> OCROutput:
        return OCROutput(
            words=ws,
            raw_response=SYNTHETIC_RAW_RESPONSE,
            model_name="synthetic-boxfree",
            version=SYNTHETIC_VERSION,
            box_free=True,
        )

    a = match(blob(list(words)), gt)
    b = match(blob(list(reversed(words))), gt)
    assert a == b
    assert len(a.matches) == 2 and a.omissions == []
