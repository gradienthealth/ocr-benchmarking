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

from harness.aggregate import aggregate


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
