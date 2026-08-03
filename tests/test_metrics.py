"""Phase 5 tests for harness/metrics.py — SYNTHETIC fixtures only, zero PHI.

Every image and token comes from tests/synthetic.py (fake tokens like "CMFN",
"GRDN1234"); no real patient IDs / accession numbers appear here. MatchResults are
built by pairing a synthetic OCROutput with its GT via harness.matching.match(),
never hand-assembled, so the scorer is tested against real matcher output.

These tests pin the structural invariants from CLAUDE.md / the task constraints:
false-redaction is the headline (denominator = KEEP tokens only), Found and Added
axes never blend, and CER/WER + confidence are DIAGNOSTIC only — structurally walled
off from the ranking keys, not merely documented as such.
"""

from __future__ import annotations

import dataclasses

from harness.contract import normalize
from harness.matching import match
from harness.metrics import is_allowed, score
from tests.synthetic import (
    hallucination_output,
    make_synthetic_image,
    misread_output,
    omission_output,
    perfect_output,
)

_TOKENS = ("CMFN", "GRDN1234", "ACC-0001")


def _keep_gt(tokens=_TOKENS):
    """Synthetic GT relabelled all-KEEP (make_synthetic_image defaults to PHI)."""
    _, gt = make_synthetic_image(tokens)
    return [dataclasses.replace(t, label="KEEP") for t in gt]


def _allowlist(gt) -> set[str]:
    """This scenario's own allowlist: the normalized KEEP token values."""
    return {normalize(t.token_text) for t in gt if t.label == "KEEP"}


def _score(out, gt, allowlist=None, elapsed=0.1, cost=0.0) -> dict:
    allowlist = _allowlist(gt) if allowlist is None else allowlist
    return score(match(out, gt), allowlist, elapsed=elapsed, cost=cost)


# --- is_allowed(): exact-match-after-normalize, no fuzzing ---------------------


def test_is_allowed_exact_member():
    assert is_allowed("CMFN", {"CMFN", "GRDN1234"}) is True


def test_is_allowed_normalizes_input():
    # NFC + strip is applied to the input before the membership check.
    assert is_allowed("  CMFN  ", {"CMFN"}) is True
    assert is_allowed("CAFÉ", {"CAFÉ"}) is True  # decomposed vs composed


def test_is_allowed_is_case_sensitive():
    assert is_allowed("cmfn", {"CMFN"}) is False


def test_is_allowed_is_not_substring_or_fuzzy():
    assert is_allowed("CMF", {"CMFN"}) is False       # prefix, not exact
    assert is_allowed("CMFNX", {"CMFN"}) is False      # superstring, not exact
    assert is_allowed("CMEN", {"CMFN"}) is False       # one char off, not fuzzy


def test_is_allowed_empty_allowlist():
    assert is_allowed("CMFN", set()) is False


# --- headline: false redaction (denominator = KEEP tokens only) ----------------


def test_all_correct_no_false_redaction():
    gt = _keep_gt()
    r = _score(perfect_output(gt), gt)
    assert r["false_redaction_count"] == 0
    assert r["keep_total"] == len(gt)
    assert r["keep_exact_match_count"] == len(gt)
    assert r["keep_exact_match_all"] is True


def test_one_misread_keep_off_allowlist_is_one_false_redaction():
    gt = _keep_gt()
    # index 0 "CMFN" -> "CMEN": off the allowlist, so that one KEEP token is redacted.
    r = _score(misread_output(gt, index=0), gt)
    assert r["false_redaction_count"] == 1
    assert r["keep_exact_match_count"] == len(gt) - 1
    assert r["keep_exact_match_all"] is False
    assert r["keep_total"] == len(gt)


def test_false_redaction_denominator_is_keep_only():
    # Mixed labels: a misread on a PHI token must NOT count as false redaction, and
    # keep_total counts KEEP tokens only. PHI is correctly-redacted bookkeeping.
    _, raw = make_synthetic_image(_TOKENS)
    gt = [
        dataclasses.replace(raw[0], label="KEEP"),
        dataclasses.replace(raw[1], label="PHI"),
        dataclasses.replace(raw[2], label="KEEP"),
    ]
    r = _score(misread_output(gt, index=1), gt)  # misread the PHI token
    assert r["keep_total"] == 2
    assert r["false_redaction_count"] == 0
    assert r["keep_exact_match_all"] is True
    assert r["found_count"] == 3  # all three still found by location


# --- Found (omission) axis vs Added (hallucination) axis — never blended --------


def test_omission_only_hits_found_side_not_added_side():
    gt = _keep_gt()
    r = _score(omission_output(gt, index=1), gt)
    # Found side reflects the miss...
    assert r["omission_count"] == 1
    assert r["found_count"] == len(gt) - 1
    # ...and the Added side is untouched.
    assert r["added_count"] == 0


def test_hallucination_only_hits_added_side_not_found_side():
    gt = _keep_gt()
    r = _score(hallucination_output(gt), gt)
    # Added side reflects the invented token...
    assert r["added_count"] == 1
    # ...and the Found side is untouched (all real tokens found, nothing omitted).
    assert r["omission_count"] == 0
    assert r["found_count"] == len(gt)
    # A pure invented token is not a false redaction (that axis is KEEP-read fidelity).
    assert r["false_redaction_count"] == 0


# --- negative-control hallucination floor --------------------------------------


def test_blank_control_prediction_raises_floor():
    gt = []  # confirmed-blank frame: zero GT tokens
    r = _score(hallucination_output(gt=gt), gt, allowlist=set())
    assert r["negative_control"] is True
    assert r["negative_control_floor_count"] > 0
    assert r["keep_total"] == 0
    assert r["false_redaction_count"] == 0


def test_non_control_reports_zero_floor():
    gt = _keep_gt()
    r = _score(perfect_output(gt), gt)
    assert r["negative_control"] is False
    assert r["negative_control_floor_count"] == 0


# --- carried-through metadata (unaggregated) -----------------------------------


def test_elapsed_cost_and_stratum_carried_through():
    gt = _keep_gt()
    r = _score(perfect_output(gt), gt, elapsed=1.23, cost=0.05)
    assert r["elapsed"] == 1.23
    assert r["cost"] == 0.05
    assert r["modality"] == gt[0].modality
    assert r["stratum"] == gt[0].stratum
    assert r["vendor"] == gt[0].vendor
    assert r["iou_thr"] == 0.5
    assert r["box_free"] is False


# --- structural wall: diagnostic keys never overlap ranking keys ---------------


def test_diagnostic_keys_never_overlap_ranking_keys():
    gt = _keep_gt()
    r = _score(misread_output(gt, index=0), gt)
    ranking_keys = set(r) - {"diagnostic"}
    diagnostic_keys = set(r["diagnostic"])
    # The two groups are structurally disjoint — a refactor cannot silently pull a
    # diagnostic signal into ranking without a key collision this test would catch.
    assert ranking_keys.isdisjoint(diagnostic_keys)


def test_cer_wer_and_confidence_absent_from_ranking():
    gt = _keep_gt()
    r = _score(misread_output(gt, index=0), gt)
    ranking_keys = set(r) - {"diagnostic"}
    for banned in ("cer", "wer", "confidence", "matched_confidences"):
        assert banned not in ranking_keys
    # They DO live in the diagnostic sub-dict.
    assert "cer" in r["diagnostic"]
    assert "wer" in r["diagnostic"]
    assert "matched_confidences" in r["diagnostic"]


def test_diagnostic_carries_cer_wer_and_confidences():
    gt = _keep_gt()
    r = _score(misread_output(gt, index=0), gt)
    d = r["diagnostic"]
    # One char wrong out of the matched text -> CER strictly between 0 and 1.
    assert 0.0 < d["cer"] < 1.0
    assert d["keep_chars_censored"] > 0
    assert d["keep_chars_total"] >= d["keep_chars_censored"]
    # Confidence values are carried through verbatim (may include None), one per match.
    assert len(d["matched_confidences"]) == len(gt)


def test_diagnostic_cer_none_when_no_matches():
    gt = []  # blank control: no matched pairs, so CER/WER are undefined
    r = _score(hallucination_output(gt=gt), gt, allowlist=set())
    assert r["diagnostic"]["cer"] is None
    assert r["diagnostic"]["wer"] is None
