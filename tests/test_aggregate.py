"""Phase 6 tests for harness/aggregate.py — PHI-free, purely numeric rows.

Aggregation consumes only the numeric/string ingredients a score row carries (never token
text or pixels), so these rows are built as plain dicts with a factory. The tests pin the
CLAUDE.md invariants that live at the aggregation layer: false-redaction is the KEEP-denominated
headline, the Found and Added axes stay in disjoint keys (never blended), latency is reported
mean/median/p95 and split primary vs. verifier, the negative-control floor is over blank rows
only, and the whole rollup is order-independent.
"""

from __future__ import annotations

import random
import statistics

import pytest

from harness.aggregate import aggregate, detection_view


def _mkrow(**over) -> dict:
    """A score row with harness-stamped fields — every key aggregate() reads, with defaults."""
    row = dict(
        keep_total=0,
        false_redaction_count=0,
        keep_exact_match_count=0,
        found_count=0,
        omission_count=0,
        gt_total=0,
        added_count=0,
        elapsed=0.1,
        verifier_elapsed=None,
        cost=0.0,
        negative_control=False,
        negative_control_floor_count=0,
        stratum="synth_ct_axial",
        model_name="fake-engine",
        # Non-empty by default: aggregate() now rejects a blank version (D-8.4).
        version="0.0.0-fake",
        # Non-empty by default for the same reason, one dimension over (D-13.5): rows that
        # all carry a blank config_hash collapse to ONE identity and would be averaged.
        config_id="fake-stock",
        config_hash="0000fakehash",
        # None by default: no verifier ran. verifier_elapsed=None is what aggregate() checks
        # to decide whether a row's verifier identity is even relevant (D-9.1) — a row with
        # a real verifier_elapsed but blank identity is what gets rejected, not this default.
        verifier_model_name=None,
        verifier_version=None,
    )
    row.update(over)
    return row


# --- grouping + headline false-redaction rate ----------------------------------


def test_per_stratum_grouping_and_false_redaction_rate():
    rows = [
        _mkrow(stratum="A", keep_total=4, false_redaction_count=1, keep_exact_match_count=3,
               found_count=2, omission_count=1, gt_total=3, added_count=1),
        _mkrow(stratum="A", keep_total=2, false_redaction_count=0, keep_exact_match_count=2,
               found_count=2, gt_total=2),
        _mkrow(stratum="B", keep_total=0, false_redaction_count=0, added_count=2,
               gt_total=0, negative_control=True, negative_control_floor_count=2),
    ]
    agg = aggregate(rows)

    assert set(agg["per_stratum"]) == {"A", "B"}
    a = agg["per_stratum"]["A"]
    assert a["keep_total"] == 6
    assert a["false_redaction_count"] == 1
    assert a["false_redaction_rate"] == pytest.approx(1 / 6)
    assert a["keep_exact_match_count"] == 5
    assert a["keep_exact_match_rate"] == pytest.approx(5 / 6)
    assert a["found_count"] == 4
    assert a["omission_count"] == 1
    assert a["added_count"] == 1

    ov = agg["overall"]
    assert ov["keep_total"] == 6
    assert ov["false_redaction_count"] == 1
    assert ov["false_redaction_rate"] == pytest.approx(1 / 6)
    assert ov["found_count"] == 4
    assert ov["omission_count"] == 1
    assert ov["added_count"] == 3  # 1 + 0 + 2
    assert ov["gt_total"] == 5


def test_false_redaction_rate_none_when_no_keep_tokens():
    b = aggregate([_mkrow(stratum="B", keep_total=0)])["per_stratum"]["B"]
    assert b["false_redaction_rate"] is None
    assert b["keep_exact_match_rate"] is None


# --- Found and Added axes never blended ----------------------------------------


def test_found_and_added_axes_stay_in_disjoint_keys():
    ov = aggregate([_mkrow(added_count=3, omission_count=2, found_count=5)])["overall"]
    # Both axes are present as independent keys...
    assert ov["added_count"] == 3
    assert ov["omission_count"] == 2
    assert ov["found_count"] == 5
    # ...and no key blends hallucination with omission into one number.
    for banned in ("accuracy", "error_rate", "hallucination_omission", "combined", "avg_error"):
        assert banned not in ov


# --- latency: mean / median / p95, split primary vs verifier -------------------


def test_latency_mean_median_p95_on_known_values():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lat = aggregate([_mkrow(elapsed=v) for v in values])["overall"]["latency"]
    assert lat["mean"] == pytest.approx(5.5)
    assert lat["median"] == pytest.approx(5.5)
    expected_p95 = statistics.quantiles(values, n=100, method="exclusive")[94]
    assert lat["p95"] == pytest.approx(expected_p95)


def test_latency_single_value_degenerates():
    lat = aggregate([_mkrow(elapsed=0.42)])["overall"]["latency"]
    assert lat == {"mean": 0.42, "median": 0.42, "p95": 0.42}


def test_verifier_latency_split_and_none_filtering():
    # Both rows share one verifier identity (a real gated run's rows always do, D-9.1) even
    # though only one of them actually has a recorded verifier_elapsed — that row alone is
    # what the blank-verifier-identity check inspects.
    rows = [
        _mkrow(elapsed=0.1, verifier_elapsed=0.5,
               verifier_model_name="fake-verifier", verifier_version="0.0.0-fake"),
        _mkrow(elapsed=0.2, verifier_elapsed=None,
               verifier_model_name="fake-verifier", verifier_version="0.0.0-fake"),
    ]
    ov = aggregate(rows)["overall"]
    assert ov["latency"]["mean"] == pytest.approx(0.15)  # both primary calls
    assert ov["verifier_latency"]["mean"] == pytest.approx(0.5)  # None row filtered out


def test_verifier_latency_none_when_no_verifier_ran():
    assert aggregate([_mkrow(), _mkrow()])["overall"]["verifier_latency"] is None


# --- negative-control floor ----------------------------------------------------


def test_negative_control_floor_over_blank_rows_only():
    rows = [
        _mkrow(negative_control=True, negative_control_floor_count=3),
        _mkrow(negative_control=True, negative_control_floor_count=1),
        _mkrow(negative_control=False, added_count=9),  # not a control -> ignored by the floor
    ]
    nc = aggregate(rows)["negative_control"]
    assert nc["n_control_images"] == 2
    assert nc["floor_count"] == 4
    assert nc["hallucinations_per_image"] == pytest.approx(2.0)


def test_negative_control_none_when_no_control_rows():
    nc = aggregate([_mkrow()])["negative_control"]
    assert nc == {"floor_count": 0, "n_control_images": 0, "hallucinations_per_image": None}


# --- cost, model_name, empty input, order-independence -------------------------


def test_cost_sum_and_mean():
    c = aggregate([_mkrow(cost=0.001), _mkrow(cost=0.003)])["overall"]["cost"]
    assert c["sum"] == pytest.approx(0.004)
    assert c["mean"] == pytest.approx(0.002)


def test_model_name_carried_through():
    assert aggregate([_mkrow(model_name="engineX")])["model_name"] == "engineX"


# --- version provenance: rule #9 enforced where the merge happens -------------
# D-8.3 (never blend two (model_name, version) pairs) had no test before D-8.4 was added;
# both guards are pinned here, including the hole the second one exists to close.


def test_version_carried_through():
    assert aggregate([_mkrow(version="2.3.4")])["version"] == "2.3.4"


def test_refuses_to_blend_two_engine_versions():
    """Same engine, two builds, one aggregate() call — the rule #9 failure."""
    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate([_mkrow(version="1.0.0"), _mkrow(version="1.1.0")])


def test_refuses_to_blend_two_model_names():
    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate([_mkrow(model_name="engineA"), _mkrow(model_name="engineB")])


# --- config provenance: the D-13.5 collision ----------------------------------
# The identity guard above keys on (model_name, version, verifier_*). Two arms of the SAME
# engine at the SAME version — docTR stock vs tuned vs parseq, PP-OCRv6 stock vs tuned
# thresholds — satisfy every one of those and were averaged into one meaningless row.


def test_refuses_to_blend_two_configs_of_one_engine_version():
    """The headline D-13.5 bug: same engine, same version, different config.

    Everything the pre-D-13.5 guard could see is identical here — one model_name, one
    version, no verifier — so the batch sailed through and produced a single averaged row
    that belonged to neither arm.
    """
    stock = _mkrow(config_id="stock", config_hash="aaaaaaaaaaaa")
    tuned = _mkrow(config_id="tuned", config_hash="bbbbbbbbbbbb")
    assert stock["model_name"] == tuned["model_name"]
    assert stock["version"] == tuned["version"]  # indistinguishable before D-13.5
    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate([stock, tuned])


def test_refuses_to_blend_when_only_the_hash_differs():
    """A relabelled arm is still a different arm.

    The hash is what actually protects the aggregation: if someone reuses a label but a
    knob moved, `config()` changes and the digest changes. Catching this is the whole
    reason the digest — not just the human label — sits in the guard tuple.
    """
    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate([
            _mkrow(config_id="stock", config_hash="aaaaaaaaaaaa"),
            _mkrow(config_id="stock", config_hash="cccccccccccc"),
        ])


def test_identical_configs_still_aggregate_together():
    """Guard against over-tightening: one arm's rows must still merge into one result."""
    agg = aggregate([_mkrow(), _mkrow(), _mkrow()])
    assert agg["n_images"] == 3


def test_blank_config_hash_rejected_even_though_every_row_agrees():
    """The same hole D-8.4 closed for versions, one dimension over.

    Rows that all carry config_hash "" form ONE tuple, so the blend check is perfectly
    satisfied — while the batch carries no config provenance at all and may be two arms
    concatenated. Hence a separate, earlier check.
    """
    rows = [_mkrow(config_hash=""), _mkrow(config_hash="")]
    assert len({r["config_hash"] for r in rows}) == 1  # the blend check sees no problem
    with pytest.raises(ValueError, match="no `config_hash`"):
        aggregate(rows)


def test_config_identity_carried_through():
    agg = aggregate([_mkrow(config_id="parseq", config_hash="dddddddddddd")])
    assert agg["config_id"] == "parseq"
    assert agg["config_hash"] == "dddddddddddd"


def test_blank_version_rejected_even_though_every_row_agrees():
    """The hole a mixed-pair check alone cannot see (D-8.4).

    Rows that all carry version "" form ONE (model_name, version) pair, so the D-8.3
    check is perfectly satisfied — while the batch in fact carries no provenance at all
    and may be two engine builds concatenated. Hence a separate, earlier check.
    """
    rows = [_mkrow(version=""), _mkrow(version="")]
    assert len({(r["model_name"], r["version"]) for r in rows}) == 1  # D-8.3 sees no problem
    with pytest.raises(ValueError, match="no engine version"):
        aggregate(rows)


@pytest.mark.parametrize("blank", ["", "   ", "\t", None])
def test_blank_version_values_are_rejected(blank):
    with pytest.raises(ValueError, match="no engine version"):
        aggregate([_mkrow(version=blank)])


def test_missing_version_key_is_rejected():
    """A row assembled without the key at all — not just with a blank value."""
    row = _mkrow()
    del row["version"]
    with pytest.raises(ValueError, match="no engine version"):
        aggregate([row])


def test_one_unversioned_row_among_good_ones_is_rejected():
    """A single bad row fails the batch; it is not dropped or tolerated."""
    with pytest.raises(ValueError, match="no engine version"):
        aggregate([_mkrow(version="1.0.0"), _mkrow(version=""), _mkrow(version="1.0.0")])


def test_blank_version_error_reports_counts_not_row_contents():
    """PHI-safety of the error path: aggregate rows are numeric, but the message must
    still stay a count/index — never a dumped row (CLAUDE.md §7, PHI never to stdout)."""
    with pytest.raises(ValueError) as exc:
        aggregate([_mkrow(version="", stratum="synth_ct_axial")])
    msg = str(exc.value)
    assert "1 of 1 rows" in msg and "row index 0" in msg
    assert "synth_ct_axial" not in msg  # no row payload in the message


def test_verifier_identity_carried_through():
    agg = aggregate([
        _mkrow(verifier_elapsed=0.1, verifier_model_name="qwen3-vl", verifier_version="1.0.0"),
    ])
    assert agg["verifier_model_name"] == "qwen3-vl"
    assert agg["verifier_version"] == "1.0.0"


def test_verifier_identity_none_when_no_verifier_ran():
    assert aggregate([_mkrow(), _mkrow()])["verifier_model_name"] is None


def test_refuses_to_blend_two_verifier_versions():
    """Same primary engine, same verifier model, two verifier builds — rule #9 for the
    second model in a gated run (D-9.1)."""
    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate([
            _mkrow(verifier_elapsed=0.1, verifier_model_name="qwen3-vl", verifier_version="1.0.0"),
            _mkrow(verifier_elapsed=0.1, verifier_model_name="qwen3-vl", verifier_version="1.1.0"),
        ])


def test_refuses_to_blend_gated_and_ungated_rows():
    """A primary-only row and a gated row must never average into one aggregate — plan.md
    Phase 11 measures primary-only vs. primary+verifier separately, and this is the same
    mixed-pair mechanism enforcing it structurally."""
    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate([
            _mkrow(verifier_elapsed=None),
            _mkrow(verifier_elapsed=0.1, verifier_model_name="qwen3-vl", verifier_version="1.0.0"),
        ])


def test_blank_verifier_identity_rejected_even_though_every_row_agrees():
    """The D-8.4 hole, replayed for the verifier axis: an all-blank gated batch is one
    consistent (verifier_model_name, verifier_version) pair to the mixed-pair check alone."""
    rows = [
        _mkrow(verifier_elapsed=0.1, verifier_model_name="", verifier_version=""),
        _mkrow(verifier_elapsed=0.2, verifier_model_name="", verifier_version=""),
    ]
    with pytest.raises(ValueError, match="ran a verifier"):
        aggregate(rows)


def test_non_gated_rows_never_trigger_the_verifier_identity_check():
    # verifier_elapsed=None (the default) -> no verifier ran -> blank identity is expected,
    # not an error.
    agg = aggregate([_mkrow(), _mkrow()])
    assert agg["verifier_model_name"] is None
    assert agg["verifier_version"] is None


def test_empty_rows_no_crash():
    agg = aggregate([])
    assert agg["n_images"] == 0
    assert agg["model_name"] is None
    assert agg["per_stratum"] == {}
    assert agg["overall"]["false_redaction_rate"] is None
    assert agg["overall"]["latency"] is None
    assert agg["overall"]["keep_total"] == 0
    assert agg["negative_control"]["hallucinations_per_image"] is None


# --- diagnostic rollup: CER/WER quarantined, never a ranking signal ------------


def _diag(**over) -> dict:
    d = dict(cer=None, wer=None, keep_chars_total=0, keep_chars_censored=0)
    d.update(over)
    return d


def test_diagnostic_cer_wer_macro_mean_and_char_sums():
    rows = [
        _mkrow(diagnostic=_diag(cer=0.2, wer=0.4, keep_chars_total=10, keep_chars_censored=2)),
        _mkrow(diagnostic=_diag(cer=0.4, wer=0.6, keep_chars_total=30, keep_chars_censored=4)),
    ]
    diag = aggregate(rows)["overall"]["diagnostic"]
    assert diag["cer_mean"] == pytest.approx(0.3)  # macro mean of 0.2, 0.4
    assert diag["wer_mean"] == pytest.approx(0.5)
    assert diag["n_cer_images"] == 2
    assert diag["keep_chars_total"] == 40
    assert diag["keep_chars_censored"] == 6
    assert diag["char_censor_rate"] == pytest.approx(6 / 40)


def test_diagnostic_ignores_none_cer_images_in_mean_but_counts_base():
    # An image with no matched pairs has cer/wer None (jiwer can't score it); it must not
    # drag the mean toward zero, and n_cer_images reports the real base.
    rows = [
        _mkrow(diagnostic=_diag(cer=0.5, wer=0.5, keep_chars_total=4, keep_chars_censored=1)),
        _mkrow(diagnostic=_diag(cer=None, wer=None, keep_chars_total=0, keep_chars_censored=0)),
    ]
    diag = aggregate(rows)["overall"]["diagnostic"]
    assert diag["cer_mean"] == pytest.approx(0.5)
    assert diag["wer_mean"] == pytest.approx(0.5)
    assert diag["n_cer_images"] == 1  # only the one that produced a value


def test_diagnostic_all_none_and_zero_chars():
    diag = aggregate([_mkrow(diagnostic=_diag())])["overall"]["diagnostic"]
    assert diag["cer_mean"] is None
    assert diag["wer_mean"] is None
    assert diag["n_cer_images"] == 0
    assert diag["char_censor_rate"] is None  # zero denominator


def test_diagnostic_robust_to_rows_without_a_diagnostic_key():
    # _mkrow() carries no diagnostic; the rollup must degrade gracefully, not crash.
    diag = aggregate([_mkrow(), _mkrow()])["overall"]["diagnostic"]
    assert diag == {
        "cer_mean": None,
        "wer_mean": None,
        "n_cer_images": 0,
        "keep_chars_total": 0,
        "keep_chars_censored": 0,
        "char_censor_rate": None,
    }


def test_cer_wer_are_not_top_level_ranking_keys():
    # CER/WER live ONLY under the nested diagnostic dict — never promoted to a ranking key.
    ov = aggregate([_mkrow(diagnostic=_diag(cer=0.1, wer=0.1))])["overall"]
    for banned in ("cer", "wer", "cer_mean", "wer_mean"):
        assert banned not in ov
    assert "cer_mean" in ov["diagnostic"]


def test_aggregation_is_order_independent():
    # One shared verifier identity across all four rows (D-9.1) — only the second row has a
    # recorded verifier_elapsed, but the mixed-pair guard keys on identity, not elapsed, so
    # every row in one batch must agree on identity regardless of which ones actually ran it.
    verifier_id = dict(verifier_model_name="fake-verifier", verifier_version="0.0.0-fake")
    rows = [
        _mkrow(stratum="A", keep_total=3, false_redaction_count=1, elapsed=0.3, **verifier_id),
        _mkrow(stratum="B", keep_total=1, elapsed=0.1, verifier_elapsed=0.2, **verifier_id),
        _mkrow(stratum="A", keep_total=2, added_count=1, elapsed=0.5, **verifier_id),
        _mkrow(stratum="B", negative_control=True, negative_control_floor_count=2, elapsed=0.2,
               **verifier_id),
    ]
    baseline = aggregate(rows)
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    assert aggregate(shuffled) == baseline


# --- detector-only view (Phase 13e) --------------------------------------------
# Boxes vs GT at the matcher's IoU bar, strings ignored. The invariant under test is the
# one the 2026-08-06 `ct_scout` sweep violated: a stratum with no text-bearing images has
# NO recall denominator, and must read n/a — never a saturated 100% — while predictions on
# blank frames stay in the hallucination floor instead of being averaged into a rate.


def _det_row(**over) -> dict:
    """A score row carrying the two keys the detector view reads beyond the counts."""
    row = _mkrow(box_free=False, iou_thr=0.5)
    row.update(over)
    return row


def test_detection_rates_are_derived_from_the_counts_score_already_exposes():
    rows = [
        _det_row(stratum="A", gt_total=4, found_count=3, omission_count=1, added_count=1),
        _det_row(stratum="A", gt_total=2, found_count=2, omission_count=0, added_count=0),
    ]
    g = aggregate(rows)["detection"]["per_stratum"]["A"]
    assert g["gt_total"] == 6 and g["found_count"] == 5
    assert g["detection_recall"] == pytest.approx(5 / 6)
    # precision denominator = every prediction = found + added (greedy 1:1 assignment)
    assert g["predicted_count"] == 6
    assert g["detection_precision"] == pytest.approx(5 / 6)
    assert g["recall_na_reason"] is None and g["precision_na_reason"] is None


def test_zero_text_stratum_is_na_with_a_reason_never_a_saturated_100_percent():
    # ct_scout / mg_tomo in the real set: every image blank, so nothing to find.
    rows = [
        _det_row(stratum="mg_tomo", gt_total=0, negative_control=True,
                 negative_control_floor_count=3, added_count=3),
        _det_row(stratum="mg_tomo", gt_total=0, negative_control=True),
        _det_row(stratum="us_ge", gt_total=2, found_count=2, added_count=1),
    ]
    g = aggregate(rows)["detection"]["per_stratum"]["mg_tomo"]
    assert g["n_text_images"] == 0 and g["n_control_images"] == 2
    assert g["detection_recall"] is None
    assert g["detection_precision"] is None
    assert "no boxed text-bearing images" in g["recall_na_reason"]
    assert "no boxed text-bearing images" in g["precision_na_reason"]
    # and the rate it is NOT allowed to become:
    assert g["detection_recall"] != 1.0


def test_blank_control_boxes_land_in_the_floor_not_in_recall_or_precision():
    rows = [
        _det_row(stratum="ct_axial", gt_total=2, found_count=1, omission_count=1,
                 added_count=1),
        _det_row(stratum="ct_axial", gt_total=0, negative_control=True,
                 negative_control_floor_count=7, added_count=7),
    ]
    g = aggregate(rows)["detection"]["per_stratum"]["ct_axial"]
    # recall: the blank image contributes to neither side of the fraction
    assert g["gt_total"] == 2 and g["found_count"] == 1
    assert g["detection_recall"] == pytest.approx(0.5)
    # precision: the blank image's 7 boxes are NOT in the denominator
    assert g["added_count"] == 1 and g["predicted_count"] == 2
    assert g["detection_precision"] == pytest.approx(0.5)
    # they are here instead
    assert g["floor_boxes"] == 7
    assert g["floor_boxes_per_image"] == pytest.approx(7.0)


def test_box_free_rows_are_excluded_from_detection_and_counted():
    # matching.py returns hallucinations=[] on the box-free path unconditionally, so a
    # box-free row would contribute added_count=0 and read as perfect precision.
    rows = [
        _det_row(stratum="A", box_free=True, gt_total=5, found_count=5),
        _det_row(stratum="A", gt_total=2, found_count=1, omission_count=1, added_count=3),
    ]
    g = aggregate(rows)["detection"]["per_stratum"]["A"]
    assert g["n_box_free_excluded"] == 1
    assert g["gt_total"] == 2 and g["found_count"] == 1  # box-free row's 5/5 not counted
    assert g["detection_precision"] == pytest.approx(1 / 4)


def test_all_box_free_run_is_na_with_the_box_free_reason():
    rows = [_det_row(stratum="A", box_free=True, gt_total=3, found_count=3)]
    g = aggregate(rows)["detection"]["overall"]
    assert g["detection_recall"] is None and g["detection_precision"] is None
    assert "box-free" in g["recall_na_reason"]
    assert "box-free" in g["precision_na_reason"]


def test_detection_iou_threshold_is_carried_and_none_when_rows_disagree():
    same = [_det_row(stratum="A"), _det_row(stratum="A")]
    assert aggregate(same)["detection"]["iou_thr"] == 0.5
    mixed = [_det_row(stratum="A"), _det_row(stratum="A", iou_thr=0.75)]
    assert aggregate(mixed)["detection"]["iou_thr"] is None


def test_detection_view_does_not_leak_into_the_end_to_end_ranking_keys():
    # Three questions, three tables: the detector numbers live under their own key and are
    # never promoted into the overall stats the engines are ranked on.
    agg = aggregate([_det_row(stratum="A", gt_total=2, found_count=2, added_count=1)])
    for banned in ("detection_recall", "detection_precision", "predicted_count"):
        assert banned not in agg["overall"]
    assert "detection_recall" in agg["detection"]["overall"]


def test_empty_group_na_reasons_do_not_invent_blank_controls():
    # A group with no rows at all has no text-bearing images AND no controls; the reason
    # must not assert controls that don't exist.
    g = aggregate([])["detection"]["overall"]
    assert g["recall_na_reason"] == "no images in this group"
    assert g["precision_na_reason"] == "no images in this group"


def test_box_free_controls_are_excluded_from_the_detector_floor_and_flagged():
    # A box-free frame reports no boxes because it structurally cannot, not because none
    # were invented — counting it would dilute the floor toward zero. So the detector floor
    # is boxed-only and can legitimately differ from the run-wide negative_control block.
    rows = [
        _det_row(stratum="A", gt_total=0, negative_control=True,
                 negative_control_floor_count=4, added_count=4),
        _det_row(stratum="A", gt_total=0, negative_control=True, box_free=True),
    ]
    agg = aggregate(rows)
    g = agg["detection"]["overall"]
    assert g["n_control_images"] == 1 and g["floor_boxes"] == 4
    assert g["floor_boxes_per_image"] == pytest.approx(4.0)
    assert g["n_box_free_excluded"] == 1
    # the run-wide block counts both — the divergence is real and must stay visible
    assert agg["negative_control"]["n_control_images"] == 2


def test_detection_view_refuses_to_pool_two_arms_of_one_engine():
    # detection_view() is a public rollup in its own right, so it must not be the one door
    # into this module through which two config arms can be averaged (rule #9, D-13.5).
    rows = [
        _det_row(stratum="A", config_id="stock", config_hash="aaa"),
        _det_row(stratum="A", config_id="tuned", config_hash="bbb"),
    ]
    with pytest.raises(ValueError, match="refusing to blend"):
        detection_view(rows)
    with pytest.raises(ValueError, match="carry no engine version"):
        detection_view([_det_row(stratum="A", version="")])


def test_na_reason_names_box_free_rows_instead_of_blaming_the_control_set():
    # A group whose only text-bearing row is box-free is unscoreable because the boxes are
    # missing — not because every image was a blank control. The reason must say which.
    rows = [
        _det_row(stratum="A", box_free=True, gt_total=6, found_count=4, omission_count=2),
        _det_row(stratum="A", gt_total=0, negative_control=True, added_count=2),
    ]
    g = aggregate(rows)["detection"]["per_stratum"]["A"]
    assert "1 row(s) excluded (no IoU matcher ran)" in g["recall_na_reason"]
    assert "1 row(s) excluded (no IoU matcher ran)" in g["precision_na_reason"]


def test_reading_arm_rows_are_excluded_from_detection_and_counted():
    # 13b's reading arm is HANDED the GT box and runs no matcher (`iou_thr=None`), so every
    # GT token is "found" by construction. Counting it would print a saturated 100%
    # detection recall — the exact failure this table exists to prevent.
    rows = [
        _det_row(stratum="A", iou_thr=None, gt_total=9, found_count=9),
        _det_row(stratum="A", gt_total=4, found_count=2, omission_count=2, added_count=2),
    ]
    g = aggregate(rows)["detection"]["per_stratum"]["A"]
    assert g["n_reading_arm_excluded"] == 1
    assert g["gt_total"] == 4 and g["found_count"] == 2  # the 9/9 handed boxes are gone
    assert g["detection_recall"] == pytest.approx(0.5)
    assert g["detection_precision"] == pytest.approx(0.5)


def test_a_pure_reading_arm_run_reports_na_naming_the_reading_arm():
    g = aggregate([_det_row(stratum="A", iou_thr=None, gt_total=5, found_count=5)])
    g = g["detection"]["overall"]
    assert g["detection_recall"] is None and g["detection_precision"] is None
    assert "reading arm" in g["recall_na_reason"]
    assert "nothing to detect" in g["recall_na_reason"]
