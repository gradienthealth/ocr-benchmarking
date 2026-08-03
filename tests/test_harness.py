"""Phase 6 tests for harness/harness.py + harness/cost.py — SYNTHETIC fixtures only, zero PHI.

Every image and token comes from tests/synthetic.py (fake tokens like "CMFN", "XZQV-777");
no real patient IDs appear here. The tests pin the design's correctness properties:

- the latency timer wraps ONLY `run_func` (a sleeping fake proves the verifier is excluded),
- the verifier re-reads ONLY sub-threshold words and, being box-free, swaps the string while
  keeping the primary's box,
- cost formulas match the confirmed D-6.4 constants,
- the end-to-end loop produces correct per-stratum + overall numbers, with the blank control
  landing in its real stratum.
"""

from __future__ import annotations

import dataclasses
import math
import time

import pytest

from harness.cost import PRICES, estimate_cost, priced
from harness.harness import ImageRef, apply_verifier, run_harness
from tests import synthetic
from tests.synthetic import (
    box_free_output,
    make_synthetic_image,
    misread_output,
    perfect_output,
)


# --- helpers -------------------------------------------------------------------


def _ref(image_id, image, stratum, vendor, gt, path="/synthetic/x.png"):
    """Build an ImageRef from a synthetic image + its GT (dimensions from the PIL image)."""
    w, h = image.size
    modality = gt[0].modality if gt else "CT"
    frame_idx = gt[0].frame_idx if gt else 0
    return ImageRef(
        id=image_id,
        path=path,
        w=w,
        h=h,
        stratum=stratum,
        modality=modality,
        vendor=vendor,
        frame_idx=frame_idx,
    )


def _single(tokens=("CMFN", "ACC-0001", "GRDN1234"), label="KEEP"):
    """One synthetic image relabelled to `label`, with its ImageRef and gt-by-id map."""
    image, raw = make_synthetic_image(tokens)
    gt = [dataclasses.replace(t, label=label) for t in raw]
    img = _ref("synth-000", image, "synth_ct_axial", "FakeVendorA", gt)
    return img, gt


# --- end-to-end ----------------------------------------------------------------


def test_end_to_end_perfect_scene_numbers(scene):
    """Perfect reads over the fixed scene: grouping, counts, headline, blank-in-stratum."""
    refs, gt_by_id = [], {}
    for s in scene.images:
        gt = [dataclasses.replace(t, label="KEEP") for t in s.gt]  # exercise KEEP metrics
        refs.append(_ref(s.image_id, s.image, s.stratum, s.vendor, gt))
        gt_by_id[s.image_id] = gt

    def run(img):
        return perfect_output(gt_by_id[img.id])

    agg = run_harness(refs, run, gt_by_id, allowlist=set(scene.allowlist))

    ov = agg["overall"]
    assert agg["n_images"] == 7
    assert ov["found_count"] == 10  # 2+1+2+1+2+2+0 GT tokens, all read
    assert ov["gt_total"] == 10
    assert ov["omission_count"] == 0
    assert ov["added_count"] == 0
    assert ov["keep_total"] == 10
    # Exactly one drawn token (XZQV-777) is off the allowlist -> one false redaction even
    # though it was read perfectly. That is the allowlist gate doing its job.
    assert ov["false_redaction_count"] == 1
    assert ov["false_redaction_rate"] == pytest.approx(1 / 10)
    assert ov["keep_exact_match_count"] == 10

    # Two strata, no None bucket -> the blank control landed in its real stratum.
    assert set(agg["per_stratum"]) == {"synth_ct_axial", "synth_xr_chest"}
    ct = agg["per_stratum"]["synth_ct_axial"]
    xr = agg["per_stratum"]["synth_xr_chest"]
    assert ct["n_images"] == 4  # images 0,1,5 + the blank control 6
    assert ct["keep_total"] == 5
    assert ct["false_redaction_count"] == 1  # XZQV-777 lives on image 5 (ct)
    assert ct["found_count"] == 5
    assert xr["n_images"] == 3
    assert xr["keep_total"] == 5
    assert xr["false_redaction_count"] == 0

    # Negative-control floor: one blank frame, zero hallucinations under perfect reading.
    assert agg["negative_control"]["n_control_images"] == 1
    assert agg["negative_control"]["floor_count"] == 0

    assert agg["model_name"] == "synthetic-perfect"


# --- timer wraps ONLY run_func -------------------------------------------------


def test_timer_wraps_only_run_func_not_verifier():
    """A sleeping run_func and a slower sleeping verifier: primary latency excludes the verifier."""
    img, gt = _single(("CMFN", "ACC-0001"))
    s1, s2 = 0.02, 0.10  # verifier is deliberately much slower than the model call

    def run(i):
        time.sleep(s1)
        return misread_output(gt, index=0)  # index 0 has confidence 0.61 (< threshold)

    def verifier(i, w):
        time.sleep(s2)
        return w.text

    agg = run_harness(
        [img], run, {img.id: gt},
        allowlist=set(), verifier_func=verifier, conf_threshold=0.7,
    )

    lat = agg["overall"]["latency"]
    vlat = agg["overall"]["verifier_latency"]
    assert lat["mean"] >= s1
    # If the verifier's time leaked into the primary timer, elapsed would be >= s1 + s2.
    assert lat["mean"] < s1 + s2
    assert vlat["mean"] >= s2


# --- verifier: re-reads sub-threshold only, keeps box, flips the decision ------


def test_apply_verifier_only_rereads_subthreshold_and_keeps_box():
    img, gt = _single(("CMFN", "ACC-0001"))
    out = misread_output(gt, index=0)  # word0: text mutated, conf 0.61; word1: exact, conf 0.99
    calls = []

    def verifier(image, w):
        calls.append(w)
        return "FIXED"

    new = apply_verifier(out, img, verifier, 0.7)

    # Only the low-confidence word was re-read.
    assert len(calls) == 1
    assert calls[0].confidence == 0.61
    # Box-free swap: the string changed; bbox and confidence are preserved.
    assert new.words[0].text == "FIXED"
    assert new.words[0].bbox == out.words[0].bbox
    assert new.words[0].confidence == out.words[0].confidence
    # The high-confidence word is untouched.
    assert new.words[1].text == out.words[1].text
    assert new.words[1].bbox == out.words[1].bbox
    # Provenance preserved.
    assert new.raw_response is out.raw_response
    assert new.model_name == out.model_name
    assert new.box_free == out.box_free


def test_none_confidence_word_never_reread():
    img, gt = _single(("CMFN",))
    out = box_free_output(gt)  # single blob word: confidence None, bbox None

    def boom(image, w):
        raise AssertionError("a None-confidence word must never be re-read")

    new = apply_verifier(out, img, boom, 0.9)
    assert new.words[0].text == out.words[0].text


def test_verifier_flips_false_redaction_to_zero():
    img, gt = _single(("CMFN", "ACC-0001", "GRDN1234"))
    allowlist = {t.token_text for t in gt}  # all three are on the allowlist

    def run(i):
        return misread_output(gt, index=0)  # "CMFN" -> one char off; falls off the allowlist

    without = run_harness([img], run, {img.id: gt}, allowlist=allowlist)
    assert without["overall"]["false_redaction_count"] == 1

    def verifier(image, w):
        return "CMFN"  # re-read restores the correct value

    with_verifier = run_harness(
        [img], run, {img.id: gt},
        allowlist=allowlist, verifier_func=verifier, conf_threshold=0.7,
    )
    # The re-read makes the KEEP token exact + allowlisted again -> kept, not redacted.
    assert with_verifier["overall"]["false_redaction_count"] == 0
    assert with_verifier["overall"]["keep_exact_match_count"] == 3


def test_verifier_inactive_without_threshold():
    img, gt = _single(("CMFN", "ACC-0001"))

    def run(i):
        return misread_output(gt, index=0)

    # verifier_func present but conf_threshold None -> arm is off, no verifier time recorded.
    agg = run_harness(
        [img], run, {img.id: gt},
        allowlist=set(), verifier_func=lambda i, w: "X", conf_threshold=None,
    )
    assert agg["overall"]["verifier_latency"] is None


# --- cost formulas match D-6.4 -------------------------------------------------


def _cref(w, h):
    return ImageRef(
        id="x", path="p", w=w, h=h,
        stratum="s", modality="CT", vendor="v", frame_idx=0,
    )


def test_cost_claude_megapixel_tokens():
    @priced("claude")
    def run(img):
        return None

    tokens = math.ceil((640 * 480) / 750)
    assert estimate_cost(_cref(640, 480), run) == tokens * PRICES["claude"]


def test_cost_gpt4o_tiles_plus_base():
    @priced("gpt4o")
    def run(img):
        return None

    tiles = math.ceil(1024 / 512) * math.ceil(1024 / 512)  # 2 x 2 = 4
    tokens = 85 + 170 * tiles
    assert estimate_cost(_cref(1024, 1024), run) == tokens * PRICES["gpt4o"]


def test_cost_cloud_is_flat_per_image():
    @priced("cloud")
    def run(img):
        return None

    assert estimate_cost(_cref(100, 100), run) == 0.0015
    assert estimate_cost(_cref(9999, 9999), run) == 0.0015  # flat regardless of size


def test_cost_self_hosted_and_untagged_are_zero():
    @priced("self_hosted")
    def hosted(img):
        return None

    def untagged(img):
        return None

    assert estimate_cost(_cref(640, 480), hosted) == 0.0
    assert estimate_cost(_cref(640, 480), untagged) == 0.0  # no .pricing -> self_hosted


def test_priced_rejects_unknown_rule():
    with pytest.raises(ValueError):
        priced("nonsense")
