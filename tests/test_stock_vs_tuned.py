"""Phase 13h tests — the six stock/tuned identities, the D-13.4 guard, the dev-slice draw.

SYNTHETIC ONLY. No render, no `gt.csv`, no `manifest.csv`, no dev slice is touched: the
manifest rows below are fabricated (`1.2.3.FAKE.…`, `FakeVendor…`), and the engines are
constructed but only ever asked for their declared config — never run on a real image.

What these pin, in the order the step-6 findings depend on them:
  1. the six arms carry six DISTINCT identities, and `aggregate()` refuses to blend any two;
  2. PP-OCRv6's correctness-required flags are IDENTICAL on both arms (so "stock thresholds,
     not PaddleOCR out of the box" is enforced, not just asserted in prose);
  3. the sweep cannot run against the frozen scored set (D-13.4);
  4. a config that regresses a stratum is disqualified even when the pooled number improves;
  5. the dev-slice draw excludes the scored set, spans vendors, and prints no identifier.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.sweep_stock_vs_tuned import (
    SweepError,
    freeze_winners,
    guard_not_the_scored_set,
    stratum_regressions,
)
from ground_truth.select_dev_slice import (
    BLANK_CONTROL_STRATA,
    DEFAULT_QUOTAS,
    SelectionError,
    build_summary,
    draw,
    read_excluded_uids,
    write_inputs,
)
from harness.aggregate import aggregate
from harness.runners.arms import ARM_NAMES, ArmNotFrozenError, build_arm, load_tuned_config

try:  # each engine is optional, exactly as in test_runner_contract.py
    from harness.runners.run_doctr import DoctrRunner
except ImportError:  # pragma: no cover
    DoctrRunner = None
try:
    from harness.runners.run_paddle_v6 import PaddleV6Runner
except ImportError:  # pragma: no cover
    PaddleV6Runner = None
try:
    from harness.runners.run_easyocr import (
        DETECTION_THRESHOLD,
        LIBRARY_LOW_TEXT,
        LIBRARY_TEXT_THRESHOLD,
        EasyOcrRunner,
    )
except ImportError:  # pragma: no cover
    EasyOcrRunner = None

requires_doctr = pytest.mark.skipif(DoctrRunner is None, reason="python-doctr not installed")
requires_paddle = pytest.mark.skipif(PaddleV6Runner is None, reason="paddleocr not installed")
requires_easyocr = pytest.mark.skipif(EasyOcrRunner is None, reason="easyocr not installed")


# --- 1. six arms, six identities --------------------------------------------------------


@requires_doctr
def test_doctr_stock_and_tuned_do_not_share_an_identity():
    stock, tuned = DoctrRunner(), DoctrRunner(bin_thresh=0.3, box_thresh=0.3)
    assert stock.config_id == "stock" and tuned.config_id == "tuned"
    assert stock.config_hash() != tuned.config_hash()


@requires_paddle
def test_paddle_stock_and_tuned_do_not_share_an_identity():
    stock = PaddleV6Runner()
    tuned = PaddleV6Runner(text_det_limit_side_len=1280)
    assert stock.config_id == "stock" and tuned.config_id == "tuned"
    assert stock.config_hash() != tuned.config_hash()


@requires_easyocr
def test_easyocr_stock_is_the_library_default_and_tuned_is_the_shipped_floor():
    """EasyOCR's pair runs BACKWARDS from the other two — the no-arg instance is tuned.

    Getting this the usual way round would answer the wrong question: the pair exists to
    separate "EasyOCR over-detects" from "our 0.2 threshold over-detects" (step 6).
    """
    shipped = EasyOcrRunner()
    library = EasyOcrRunner(text_threshold=LIBRARY_TEXT_THRESHOLD, low_text=LIBRARY_LOW_TEXT)
    assert shipped.config_id == "thr0.2"
    assert library.config_id == "stock"
    assert shipped.config_hash() != library.config_hash()
    assert shipped.config()["text_threshold"] == DETECTION_THRESHOLD


@requires_doctr
@requires_paddle
@requires_easyocr
def test_the_six_arms_produce_six_distinct_identities():
    """`aggregate()` keys on (model_name, version, config_id, config_hash, verifier_*).

    Six arms that collapse to fewer identities would average two different systems into one
    row with nothing to flag it — the failure D-13.5 exists to stop, at the scale step 6
    creates it.
    """
    runners = [
        DoctrRunner(),
        DoctrRunner(bin_thresh=0.3, box_thresh=0.3),
        PaddleV6Runner(),
        PaddleV6Runner(text_det_limit_side_len=1280),
        EasyOcrRunner(text_threshold=LIBRARY_TEXT_THRESHOLD, low_text=LIBRARY_LOW_TEXT),
        EasyOcrRunner(),
    ]
    identities = {(r.model_name, r.version, r.config_id, r.config_hash()) for r in runners}
    assert len(identities) == 6, "two step-6 arms share an identity and would aggregate as one"


def _row(**over) -> dict:
    row = dict(
        keep_total=1, false_redaction_count=0, keep_exact_match_count=1,
        found_count=1, omission_count=0, gt_total=1, added_count=0,
        elapsed=0.1, verifier_elapsed=None, cost=0.0,
        negative_control=False, negative_control_floor_count=0,
        stratum="synth_ct_axial", model_name="doctr", version="v1.0.1",
        config_id="stock", config_hash="aaaaaaaaaaaa",
        verifier_model_name=None, verifier_version=None,
    )
    row.update(over)
    return row


def test_aggregate_refuses_to_blend_a_stock_row_with_a_tuned_row():
    """Same engine, same version, different config — must fail, not average."""
    rows = [_row(), _row(config_id="tuned", config_hash="bbbbbbbbbbbb")]
    with pytest.raises(ValueError, match="config_id|tuple"):
        aggregate(rows)


def test_aggregate_accepts_one_arm_on_its_own():
    agg = aggregate([_row(), _row()])
    assert agg["config_id"] == "stock" and agg["n_images"] == 2


# --- 2. PP-OCR's correctness requirements are not tuning --------------------------------


@requires_paddle
def test_pp_ocr_correctness_flags_are_identical_on_both_arms():
    """`return_word_box`, `enable_mkldnn` and the three orientation stages are NOT knobs.

    Word-level output is contract-required (line boxes score IoU ~0.43 against a one-token
    GT box and fall under the 0.5 matcher bar), oneDNN raises on this stack, and the
    orientation stages move the pixels the boxes are expressed in. If a tuned arm ever
    differed on one of these, the step-6 delta would be measuring that instead of thresholds.
    """
    required = (
        "return_word_box", "enable_mkldnn",
        "use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation",
    )
    stock = PaddleV6Runner().config()
    tuned = PaddleV6Runner(
        text_det_thresh=0.2, text_det_box_thresh=0.3,
        text_det_unclip_ratio=2.0, text_det_limit_side_len=1280,
    ).config()
    for key in required:
        assert stock[key] == tuned[key], f"{key} differs between the stock and tuned arms"
    assert stock["return_word_box"] is True
    assert stock["enable_mkldnn"] is False


@requires_paddle
def test_paddle_stock_declares_its_thresholds_as_unset_not_as_copied_literals():
    """`None` means "PaddleOCR's default", which cannot go stale on a version bump."""
    stock = PaddleV6Runner().config()
    for key in ("text_det_thresh", "text_det_box_thresh",
                "text_det_unclip_ratio", "text_det_limit_side_len"):
        assert stock[key] is None


@requires_doctr
def test_config_id_is_derived_not_accepted_as_a_constructor_argument():
    """A label that can be passed in can be passed WRONG — a tuned run labelled "stock".

    It is also why `config_id` must not be an `__init__` parameter: the contract test
    requires every parameter to appear in `config()`, and a human label in the digest would
    split one configuration into two identities.
    """
    with pytest.raises(TypeError):
        DoctrRunner(config_id="stock", bin_thresh=0.3)  # type: ignore[call-arg]


# --- 3. the arms registry ---------------------------------------------------------------


def test_arm_names_are_the_six_of_step_6():
    assert len(ARM_NAMES) == 6
    for engine in ("doctr", "pp-ocrv6_medium", "easyocr"):
        assert f"{engine}:stock" in ARM_NAMES and f"{engine}:tuned" in ARM_NAMES


def test_build_arm_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm("surya:tuned")


def test_a_tuned_arm_raises_until_the_sweep_has_frozen_its_config(tmp_path):
    """Never fall back to stock values under a "tuned" label.

    A silent fallback would put a stock measurement into the report as the tuned row, making
    the step-6 delta read as "tuning gained nothing" when the truth is "the sweep has not
    run yet".
    """
    empty = tmp_path / "tuned_configs.json"
    empty.write_text(json.dumps({"engines": {}}), encoding="utf-8")
    with pytest.raises(ArmNotFrozenError, match="no frozen tuned config"):
        load_tuned_config("doctr", empty)


def test_a_missing_tuned_config_file_also_raises(tmp_path):
    with pytest.raises(ArmNotFrozenError, match="does not exist"):
        load_tuned_config("doctr", tmp_path / "nope.json")


def test_the_committed_tuned_configs_file_is_still_empty():
    """Guards against someone hand-writing a tuned config instead of measuring one.

    If this fails because the sweep really has run, update it to assert the frozen values
    — deliberately, not by deleting the test.
    """
    from harness.runners.arms import TUNED_CONFIGS

    doc = json.loads(TUNED_CONFIGS.read_text(encoding="utf-8"))
    assert doc["engines"] == {}, (
        "a tuned config appeared in tuned_configs.json — it must come from the dev-slice "
        "sweep (--freeze), never from a hand edit (D-13.4)"
    )


@requires_easyocr
def test_easyocr_arms_build_without_any_frozen_config():
    """Its pair is fixed by definition, so it needs no sweep and must not require one."""
    assert build_arm("easyocr:stock").config_id == "stock"
    assert build_arm("easyocr:tuned").config_id == "thr0.2"


# --- 4. the D-13.4 guard ----------------------------------------------------------------


def test_sweeping_the_frozen_scored_set_by_path_raises():
    from experiments.sweep_stock_vs_tuned import FROZEN_GT

    with pytest.raises(SweepError, match="FROZEN SCORED SET"):
        guard_not_the_scored_set(FROZEN_GT)


def test_sweeping_a_copy_of_the_scored_set_raises_on_content(tmp_path, monkeypatch):
    """The rename dodge: `cp gt.csv /tmp/dev.csv` must trip the hash check."""
    import experiments.sweep_stock_vs_tuned as sweep

    fake_gt = tmp_path / "totally_not_gt.csv"
    fake_gt.write_text("image_id,token_text\nffff0000,CMFN-0042\n", encoding="utf-8")
    hash_file = tmp_path / "gt.csv.sha256"
    hash_file.write_text(sweep.sha256_file(fake_gt) + "\n", encoding="utf-8")
    monkeypatch.setattr(sweep, "FROZEN_GT_HASH", hash_file)

    with pytest.raises(SweepError, match="same sha256"):
        sweep.guard_not_the_scored_set(fake_gt)


def test_a_genuine_dev_slice_passes_the_guard(tmp_path):
    dev = tmp_path / "dev_v1.csv"
    dev.write_text("image_id,token_text\nffff0000,CMFN-0042\n", encoding="utf-8")
    guard_not_the_scored_set(dev)  # must not raise


@pytest.mark.parametrize("flag", ["--force", "--allow-gt", "--no-guard"])
def test_the_guard_has_no_override_flag(flag, tmp_path):
    """A --force here would be a flag whose only use is voiding every reported number.

    Asserted against the real parser, not a grep of the source — the docstring explains why
    the flag does not exist, and a grep would be tripped by the explanation.
    """
    from experiments.sweep_stock_vs_tuned import _parse_args

    argv = ["--renders", str(tmp_path), "--gt", str(tmp_path / "dev.csv"),
            "--backmap", str(tmp_path / "b.csv"), "--manifest", str(tmp_path / "m.csv"),
            "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        _parse_args([*argv, flag])


# --- 5. per-stratum selection -----------------------------------------------------------


def _agg(overall: float, per_stratum: dict[str, float], metric: str) -> dict:
    return {
        "overall": {metric: overall},
        "per_stratum": {k: {metric: v} for k, v in per_stratum.items()},
    }


def test_a_config_that_destroys_one_stratum_is_disqualified_despite_a_better_pooled_number():
    """The `conf>=60, len>=2` failure, encoded: pooled improved, ultrasound was destroyed."""
    metric = "false_redaction_rate"
    stock = _agg(0.30, {"mg_2d": 0.40, "us_ge": 0.10}, metric)
    cand = _agg(0.20, {"mg_2d": 0.10, "us_ge": 0.45}, metric)  # pooled better, us_ge worse
    regressions = stratum_regressions(stock, cand, metric, tol=0.02)
    assert [r["stratum"] for r in regressions] == ["us_ge"]


def test_an_across_the_board_improvement_is_not_disqualified():
    metric = "false_redaction_rate"
    stock = _agg(0.30, {"mg_2d": 0.40, "us_ge": 0.10}, metric)
    cand = _agg(0.15, {"mg_2d": 0.20, "us_ge": 0.09}, metric)
    assert stratum_regressions(stock, cand, metric, tol=0.02) == []


def test_regression_direction_follows_the_metric():
    """keep_exact_match_rate is better HIGHER — a drop is the regression."""
    metric = "keep_exact_match_rate"
    stock = _agg(0.80, {"mg_2d": 0.90}, metric)
    cand = _agg(0.85, {"mg_2d": 0.50}, metric)
    assert [r["stratum"] for r in stratum_regressions(stock, cand, metric, 0.02)] == ["mg_2d"]


def test_a_stratum_missing_from_either_side_is_skipped_not_scored_as_equal():
    metric = "false_redaction_rate"
    stock = _agg(0.30, {"mg_2d": 0.40}, metric)
    cand = _agg(0.20, {"mg_2d": 0.30, "us_ge": 0.99}, metric)
    assert stratum_regressions(stock, cand, metric, 0.02) == []


# --- 6. freezing ------------------------------------------------------------------------


def _sweep_ctx(winner: dict | None, engine: str = "doctr") -> dict:
    return {
        "gt_name": "dev_v1.csv",
        "gt_hash": "0" * 64,
        "n_images": 26,
        "rank_metric": "false_redaction_rate",
        "engines": [{
            "engine": engine,
            "version": "v1.0.1",
            "sweep_size": 5,
            "max_stratum_regression": 0.02,
            "rank_metric": "false_redaction_rate",
            "winner": winner,
        }],
    }


def _trial(overall: float, per_stratum: dict[str, float], config: dict,
           config_id: str, metric: str = "false_redaction_rate") -> dict:
    return {
        "config": config,
        "config_id": config_id,
        "config_hash": "h" + config_id,
        "declared_config": config,
        "version": "v1.0.1",
        "overall": {metric: overall, "latency": {"mean": 1.0, "median": 1.0, "p95": 1.0}},
        "per_stratum": {k: {metric: v} for k, v in per_stratum.items()},
        "negative_control": {},
    }


def test_freezing_writes_the_winning_config(tmp_path):
    out = tmp_path / "tuned_configs.json"
    winner = {"config": {"bin_thresh": 0.3}, "config_hash": "abc123abc123"}
    assert freeze_winners(_sweep_ctx(winner), out) == ["doctr"]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["engines"]["doctr"]["config"] == {"bin_thresh": 0.3}
    assert doc["engines"]["doctr"]["sweep_size"] == 5


def test_an_engine_with_no_winner_freezes_nothing(tmp_path):
    """Every candidate regressed a stratum -> stock stands, `<engine>:tuned` stays unbuildable."""
    out = tmp_path / "tuned_configs.json"
    assert freeze_winners(_sweep_ctx(None), out) == []
    assert json.loads(out.read_text(encoding="utf-8"))["engines"] == {}


def test_easyocr_is_never_frozen(tmp_path):
    """Its pair is fixed by definition; a "frozen winner" would imply a choice not made."""
    out = tmp_path / "tuned_configs.json"
    winner = {"config": {"text_threshold": 0.2}, "config_hash": "abc123abc123"}
    assert freeze_winners(_sweep_ctx(winner, engine="easyocr"), out) == []


def test_a_re_sweep_with_no_winner_DELETES_the_stale_frozen_config(tmp_path):
    """Merging without deleting leaves `<engine>:tuned` resolving to the old sweep's answer.

    The run would print "froze nothing" while the file still hands out the previous config —
    now stamped with the NEW dev_set and rank_metric, which claim it came from this sweep.
    """
    out = tmp_path / "tuned_configs.json"
    winner = {"config": {"bin_thresh": 0.3}, "config_hash": "abc123abc123"}
    freeze_winners(_sweep_ctx(winner), out)
    assert json.loads(out.read_text(encoding="utf-8"))["engines"]["doctr"]

    assert freeze_winners(_sweep_ctx(None), out) == []
    assert "doctr" not in json.loads(out.read_text(encoding="utf-8"))["engines"]


def test_freezing_one_engine_does_not_unfreeze_another(tmp_path):
    """A `--engine easyocr` run must leave doctr's frozen config alone."""
    out = tmp_path / "tuned_configs.json"
    freeze_winners(_sweep_ctx({"config": {"bin_thresh": 0.3}, "config_hash": "a" * 12}), out)
    freeze_winners(_sweep_ctx(None, engine="pp-ocrv6_medium"), out)
    assert json.loads(out.read_text(encoding="utf-8"))["engines"]["doctr"]["config"] == {
        "bin_thresh": 0.3
    }


# --- 6b. the winner must actually beat stock --------------------------------------------


def _stub_engine(monkeypatch, trials_by_config_id: dict[str, dict]):
    """Run `sweep_engine`'s real selection logic over pre-computed numbers.

    Only `run_trial` is stubbed, so the grid, the stock lookup, the per-stratum guard, the
    ranking and the beat-stock rule are all the production ones. Running real engines here
    would measure the engines, not the selection rule under test.
    """
    import experiments.sweep_stock_vs_tuned as sweep

    def fake_run(runner, images, gt, allowlist):
        trial = trials_by_config_id[runner.config_id]
        return {
            "overall": trial["overall"],
            "per_stratum": trial["per_stratum"],
            "negative_control": {},
        }

    monkeypatch.setattr(sweep, "run_trial", fake_run)
    return sweep


@requires_doctr
def test_a_grid_where_nothing_beats_stock_declares_no_winner(monkeypatch):
    """Every candidate slightly WORSE than stock, but under the per-stratum tolerance.

    Nothing is disqualified, so the least-bad candidate would otherwise be crowned and
    frozen — reporting a tuning LOSS as the tuned arm of step 6.
    """
    sweep = _stub_engine(monkeypatch, {
        "stock": _trial(0.20, {"mg_2d": 0.20}, {}, "stock"),
        "tuned": _trial(0.21, {"mg_2d": 0.21}, {"bin_thresh": 0.3}, "tuned"),
    })
    res = sweep.sweep_engine("doctr", [], {}, set(), "false_redaction_rate", 0.02, 2)
    assert res["winner"] is None, "a config worse than stock was crowned as the tuned arm"
    assert res["sweep_size"] == 2


@requires_doctr
def test_a_genuine_improvement_still_wins(monkeypatch):
    sweep = _stub_engine(monkeypatch, {
        "stock": _trial(0.20, {"mg_2d": 0.20}, {}, "stock"),
        "tuned": _trial(0.05, {"mg_2d": 0.05}, {"bin_thresh": 0.3}, "tuned"),
    })
    res = sweep.sweep_engine("doctr", [], {}, set(), "false_redaction_rate", 0.02, 2)
    assert res["winner"] is not None
    # The grid's second entry, whatever it is — asserted against the grid, not a literal, so
    # tuning the grid does not silently turn this into a test of nothing.
    from experiments.sweep_stock_vs_tuned import doctr_grid

    assert res["winner"]["config"] == doctr_grid()[1]


def test_the_report_reads_the_latency_key_aggregate_actually_emits():
    """`aggregate._latency_stats` emits mean/median/p95 — a p50 lookup prints n/a forever."""
    from harness.aggregate import _latency_stats

    assert set(_latency_stats([0.1, 0.2, 0.3]) or {}) == {"mean", "median", "p95"}
    source = Path("experiments/sweep_stock_vs_tuned.py").read_text(encoding="utf-8")
    assert '.get("p50")' not in source


# --- 7. the dev-slice draw --------------------------------------------------------------


def _manifest_rows(spec: list[tuple[str, str, str, int]]) -> list[dict[str, str]]:
    """(stratum, vendor, uid_prefix, count) -> fabricated manifest rows."""
    rows = []
    for stratum, vendor, prefix, count in spec:
        for i in range(count):
            rows.append({
                "series_uid": f"1.2.3.FAKE.{prefix}.{i}",
                "strata": stratum,
                "manufacturer": vendor,
                "modality": "US" if stratum.startswith("us_") else "CT",
                "number_of_frames": "60",
            })
    return rows


ROWS = _manifest_rows([
    ("us_ge", "GE", "GE", 10),
    ("mg_2d", "HOLOGIC", "HOL", 10),
    ("mg_2d", "R2 Technology Inc", "R2", 10),
    ("ct_axial", "SIEMENS", "CTA", 10),
])
QUOTAS = {"us_ge": 3, "mg_2d": 8, "ct_axial": 4}


def test_the_draw_never_returns_an_excluded_series():
    """THE guard: a dev image that is also a gt_v1 image puts the tuning set inside the test set."""
    excluded = {r["series_uid"] for r in ROWS if r["strata"] == "us_ge"}
    chosen, report = draw(ROWS, excluded, QUOTAS, seed="t")
    assert not {r["series_uid"] for r in chosen} & excluded
    assert report["us_ge"]["available"] == 0 and report["us_ge"]["drawn"] == 0


def test_a_short_stratum_is_reported_not_silently_filled_from_elsewhere():
    chosen, report = draw(ROWS, set(), {"us_ge": 99}, seed="t")
    assert report["us_ge"] == {"requested": 99, "available": 10, "drawn": 10}
    assert len(chosen) == 10


def test_the_draw_is_deterministic_for_a_fixed_seed():
    a, _ = draw(ROWS, set(), QUOTAS, seed="dev_v1")
    b, _ = draw(ROWS, set(), QUOTAS, seed="dev_v1")
    assert [r["series_uid"] for r in a] == [r["series_uid"] for r in b]


def test_a_different_seed_draws_a_different_slice():
    a, _ = draw(ROWS, set(), QUOTAS, seed="dev_v1")
    b, _ = draw(ROWS, set(), QUOTAS, seed="dev_v2")
    assert [r["series_uid"] for r in a] != [r["series_uid"] for r in b]


def test_a_multi_vendor_stratum_is_split_across_vendors():
    """An 8-image mg_2d quota must not land on one manufacturer and tune to one scanner."""
    chosen, _ = draw(ROWS, set(), QUOTAS, seed="t")
    mg_vendors = [r["manufacturer"] for r in chosen if r["strata"] == "mg_2d"]
    assert len(set(mg_vendors)) == 2
    assert abs(mg_vendors.count("HOLOGIC") - mg_vendors.count("R2 Technology Inc")) <= 1


def test_the_default_quotas_include_blank_controls():
    """No blanks -> no invention floor -> tuning optimizes recall against noise (§8)."""
    assert sum(DEFAULT_QUOTAS.get(s, 0) for s in BLANK_CONTROL_STRATA) > 0


def test_the_blank_controls_are_not_drawn_from_ct_axial():
    """Measured 2026-08-10: gt_v1 uses all 66 of the manifest's 66 ct_axial series.

    Zero are left, so a ct_axial quota can only come up SHORT — and if any did exist,
    drawing one would tune against a frame inside the scored set and contaminate the
    headline hallucination floor, which is exactly what D-13.4 forbids.
    """
    assert "ct_axial" not in BLANK_CONTROL_STRATA
    assert "ct_axial" not in DEFAULT_QUOTAS


def test_the_blank_strata_are_the_ones_with_no_text_bearing_images():
    """`ct_scout` and `mg_tomo` returned ZERO text-bearing images in gt_v1.

    That makes them useless as tuning signal and exactly right as an invention floor. Two
    of them, not one, so a human finding text in one does not wipe out the floor.
    """
    assert set(BLANK_CONTROL_STRATA) == {"ct_scout", "mg_tomo"}
    assert len(BLANK_CONTROL_STRATA) >= 2


def test_the_summary_names_no_identifier(tmp_path):
    chosen, report = draw(ROWS, set(), QUOTAS, seed="t")
    summary = build_summary(report, chosen, len(ROWS), 0, "t", tmp_path / "inputs.csv")
    assert "1.2.3.FAKE" not in summary
    assert "us_ge" in summary and "HOLOGIC" in summary  # strata and vendors ARE PHI-free


def test_the_summary_warns_when_a_slice_has_no_blank_controls(tmp_path):
    chosen, report = draw(ROWS, set(), {"us_ge": 3}, seed="t")
    summary = build_summary(report, chosen, len(ROWS), 0, "t", tmp_path / "inputs.csv")
    assert "no blank frames in this slice" in summary


def test_inputs_csv_matches_render_pys_expected_header(tmp_path):
    from ground_truth.render import INPUT_COLUMNS

    out = tmp_path / "inputs.csv"
    chosen, _ = draw(ROWS, set(), QUOTAS, seed="t")
    write_inputs(chosen, out, "data/series")
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert set(reader.fieldnames or []) == set(INPUT_COLUMNS)
        rows = list(reader)
    # frame_idx empty => render.py uses n // 2 of the tar actually present (D-10.8).
    assert all(row["frame_idx"] == "" for row in rows)
    assert all(row["path"].endswith(".tar") for row in rows)


def test_an_exclusion_file_without_series_uid_is_rejected(tmp_path):
    """An exclusion file that excludes nothing would silently admit the scored set."""
    bad = tmp_path / "bad.csv"
    bad.write_text("image_id\nffff0000\n", encoding="utf-8")
    with pytest.raises(SelectionError, match="no 'series_uid' column"):
        read_excluded_uids([bad])


# --- 8. the sweep end to end, on synthetic pixels ---------------------------------------


@requires_doctr
def test_the_sweep_runs_end_to_end_on_synthetic_images_and_writes_a_phi_free_report(tmp_path):
    """One engine, two configs, two fabricated images — the whole path, no real data.

    Budgeted to two configs so the suite stays fast; the point is that the wiring holds
    (ImageRef assembly, GT load, allowlist, per-stratum rollup, report, freeze refusal), not
    that these particular thresholds are good.
    """
    from experiments.sweep_stock_vs_tuned import main
    from ground_truth.gt_schema import COLUMNS as GT_COLUMNS
    from tests.synthetic import make_blank_image, make_synthetic_image

    renders = tmp_path / "renders"
    renders.mkdir()
    text_id, blank_id = "aaaa0001", "bbbb0002"

    image, gt_tokens = make_synthetic_image(("CMFN-0042",))
    image.save(renders / f"{text_id}.png")
    blank, _ = make_blank_image()
    blank.save(renders / f"{blank_id}.png")
    w, h = image.size

    _write(renders / "render_manifest.csv",
           ("image_id", "frame_idx", "w", "h", "sha256", "fallback_used"),
           [[text_id, 0, w, h, "0" * 64, 0], [blank_id, 0, *blank.size, "1" * 64, 0]])

    _write(tmp_path / "backmap.csv", ("image_id", "series_uid"),
           [[text_id, "1.2.3.FAKE.A"], [blank_id, "1.2.3.FAKE.B"]])

    _write(tmp_path / "manifest.csv",
           ("series_uid", "modality", "strata", "manufacturer", "middle_frame_index"),
           [["1.2.3.FAKE.A", "MG", "mg_2d", "FakeVendorA", 0],
            ["1.2.3.FAKE.B", "CT", "ct_axial", "FakeVendorB", 0]])

    # Only the text-bearing image has rows; a confirmed-blank control is ABSENT from the GT
    # file, not present-and-empty — the same convention gt.csv uses.
    _write(tmp_path / "dev_v1.csv", GT_COLUMNS,
           [[text_id, "1.2.3.FAKE.A", "MG", "FakeVendorA", "mg_2d", 0, t.token_text,
             *t.bbox, t.label] for t in gt_tokens])

    out = tmp_path / "sweep_out"
    rc = main(["--renders", str(renders), "--gt", str(tmp_path / "dev_v1.csv"),
               "--backmap", str(tmp_path / "backmap.csv"),
               "--manifest", str(tmp_path / "manifest.csv"),
               "--out", str(out), "--engine", "doctr", "--budget", "2"])
    assert rc == 0

    report = (out / "sweep_report.txt").read_text(encoding="utf-8")
    assert "configurations tried: 2" in report
    assert "mg_2d" in report and "ct_axial" in report      # per-stratum, not pooled only
    assert "CMFN-0042" not in report                       # no token text, ever
    assert "1.2.3.FAKE" not in report                      # no identifier, ever

    doc = json.loads((out / "sweep_report.json").read_text(encoding="utf-8"))
    assert doc["engines"][0]["sweep_size"] == 2
    assert json.dumps(doc).find("CMFN-0042") == -1

    # --freeze was not passed, so the tuned arm is still unavailable.
    from harness.runners.arms import TUNED_CONFIGS

    assert json.loads(TUNED_CONFIGS.read_text(encoding="utf-8"))["engines"] == {}


def _write(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# --- 8. --from-report: choosing the ranking metric after the sweep -----------------------
#
# The sweep is ~2.5h of CPU and the project's two documents disagree about which metric
# leads (CLAUDE.md §1 vs the reading-quality framing). Before this path existed, picking the
# other metric meant running every engine again. These pin that re-selection reads the
# report and nothing else — and that the offline route is not a way around D-13.4.


def _dual_trial(config_id: str, config: dict, fr: float, kem: float) -> dict:
    """One trial carrying BOTH ranking metrics, which is what the real report carries."""
    return {
        "config": config,
        "config_id": config_id,
        "config_hash": "h" + config_id,
        "declared_config": config,
        "version": "v1.0.1",
        "overall": {
            "false_redaction_rate": fr,
            "keep_exact_match_rate": kem,
            "added_count": 0,
            "omission_count": 0,
            "latency": {"mean": 1.0, "median": 1.0, "p95": 1.0},
        },
        "per_stratum": {"mg_2d": {"false_redaction_rate": fr, "keep_exact_match_rate": kem}},
        "negative_control": {},
    }


# Deliberately crossed: `low_fr` is the best false-redaction config, `high_keep` is the best
# KEEP-exact config. Which one is "the tuned arm" is therefore decided entirely by --rank-by.
_TRIALS = [
    _dual_trial("stock", {}, fr=0.30, kem=0.60),
    _dual_trial("tuned", {"bin_thresh": 0.3}, fr=0.10, kem=0.65),
    _dual_trial("tuned2", {"bin_thresh": 0.1}, fr=0.20, kem=0.90),
]


def _report(dir_: Path, engine: str = "doctr", *, gt_hash: str = "a" * 64,
            gt_name: str = "dev_v1.csv", n_images: int = 22, trials=None) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    ctx = {
        "gt_name": gt_name,
        "gt_hash": gt_hash,
        "n_images": n_images,
        "rank_metric": "false_redaction_rate",
        "rank_direction": "lower",
        "max_stratum_regression": 0.02,
        "engines": [{
            "engine": engine,
            "version": "v1.0.1",
            "sweep_size": 3,
            "grid_truncated_by_budget": 0,
            "rank_metric": "false_redaction_rate",
            "rank_direction": "lower",
            "max_stratum_regression": 0.02,
            "stock": (trials or _TRIALS)[0],
            "trials": trials or _TRIALS,
            "disqualified": [],
            "winner": (trials or _TRIALS)[1],
        }],
    }
    (dir_ / "sweep_report.json").write_text(json.dumps(ctx, indent=2) + "\n", encoding="utf-8")
    return dir_


def test_the_ranking_metric_can_be_changed_without_rerunning_any_engine(tmp_path):
    """The whole point: same measured trials, other metric, different tuned arm."""
    import experiments.sweep_stock_vs_tuned as sweep

    shard = _report(tmp_path / "doctr")

    by_fr = sweep.main(["--from-report", str(shard), "--rank-by", "false_redaction_rate"])
    assert by_fr == 0

    ctx = sweep._reselect(sweep._parse_args(
        ["--from-report", str(shard), "--rank-by", "keep_exact_match_rate"]
    ))
    assert ctx["engines"][0]["winner"]["config_id"] == "tuned2"

    ctx_fr = sweep._reselect(sweep._parse_args(
        ["--from-report", str(shard), "--rank-by", "false_redaction_rate"]
    ))
    assert ctx_fr["engines"][0]["winner"]["config_id"] == "tuned"


def test_reselection_needs_no_gt_no_renders_and_builds_no_engine(tmp_path, monkeypatch):
    """No --gt, no --renders, and `build_runner` blows up if anything tries to construct one."""
    import experiments.sweep_stock_vs_tuned as sweep

    def explode(*a, **k):  # pragma: no cover - the point is that it is never called
        raise AssertionError("--from-report constructed an engine")

    monkeypatch.setattr(sweep, "build_runner", explode)
    monkeypatch.setattr(sweep, "load_dev_images", explode)
    assert sweep.main(["--from-report", str(_report(tmp_path / "doctr"))]) == 0


def test_per_engine_shards_are_merged(tmp_path):
    import experiments.sweep_stock_vs_tuned as sweep

    a = _report(tmp_path / "doctr", engine="doctr")
    b = _report(tmp_path / "paddle", engine="pp-ocrv6_medium")
    ctx = sweep._reselect(sweep._parse_args(
        ["--from-report", str(a), "--from-report", str(b)]
    ))
    assert [e["engine"] for e in ctx["engines"]] == ["doctr", "pp-ocrv6_medium"]


def test_shards_from_different_dev_sets_are_refused(tmp_path):
    """Two dev sets under one `dev_set` line in tuned_configs.json would be a false claim."""
    import experiments.sweep_stock_vs_tuned as sweep

    a = _report(tmp_path / "doctr", engine="doctr", gt_hash="a" * 64)
    b = _report(tmp_path / "paddle", engine="pp-ocrv6_medium", gt_hash="b" * 64)
    with pytest.raises(SweepError, match="different dev set"):
        sweep._reselect(sweep._parse_args(["--from-report", str(a), "--from-report", str(b)]))


def test_the_same_engine_in_two_shards_is_refused_not_silently_picked(tmp_path):
    import experiments.sweep_stock_vs_tuned as sweep

    a = _report(tmp_path / "one", engine="doctr")
    b = _report(tmp_path / "two", engine="doctr")
    with pytest.raises(SweepError, match="appears in both"):
        sweep._reselect(sweep._parse_args(["--from-report", str(a), "--from-report", str(b)]))


def test_a_report_of_the_scored_set_cannot_be_frozen_from(tmp_path):
    """D-13.4 has no override — including via a report that skips the --gt guard entirely."""
    import experiments.sweep_stock_vs_tuned as sweep

    frozen = sweep.FROZEN_GT_HASH.read_text(encoding="utf-8").strip()
    shard = _report(tmp_path / "doctr", gt_hash=frozen, gt_name="totally_not_gt.csv")
    with pytest.raises(SweepError, match="same sha256 as the frozen gt.csv"):
        sweep._reselect(sweep._parse_args(["--from-report", str(shard)]))


def test_a_report_named_gt_csv_is_refused_too(tmp_path):
    import experiments.sweep_stock_vs_tuned as sweep

    shard = _report(tmp_path / "doctr", gt_name="gt.csv")
    with pytest.raises(SweepError, match="FROZEN SCORED SET"):
        sweep._reselect(sweep._parse_args(["--from-report", str(shard)]))


def test_an_unfinished_shard_names_itself_instead_of_stack_tracing(tmp_path):
    import experiments.sweep_stock_vs_tuned as sweep

    (tmp_path / "running").mkdir()
    with pytest.raises(SweepError, match="has not finished"):
        sweep._reselect(sweep._parse_args(["--from-report", str(tmp_path / "running")]))


def test_sweeping_without_its_inputs_names_the_missing_flags(tmp_path):
    """The flags became optional so --from-report could omit them; sweeping still needs them."""
    import experiments.sweep_stock_vs_tuned as sweep

    with pytest.raises(SweepError, match=r"--renders.*--gt.*--backmap.*--manifest"):
        sweep.main(["--out", str(tmp_path / "out")])


def test_a_reselected_report_is_not_written_as_a_measured_one(tmp_path):
    """Otherwise it lands in a shard dir as sweep_report.json and reads back as a measurement."""
    import experiments.sweep_stock_vs_tuned as sweep

    shard = _report(tmp_path / "doctr")
    out = tmp_path / "out"
    sweep.main(["--from-report", str(shard), "--out", str(out)])
    assert (out / "reselected_report.json").is_file()
    assert not (out / "sweep_report.json").exists()


def test_freezing_offline_records_that_it_was_offline(tmp_path, monkeypatch):
    """`tuned_configs.json` must say the winner was re-ranked from shards, not measured now."""
    import experiments.sweep_stock_vs_tuned as sweep

    tuned = tmp_path / "tuned_configs.json"
    monkeypatch.setattr(sweep, "TUNED_CONFIGS", tuned)
    shard = _report(tmp_path / "doctr")
    sweep.main(["--from-report", str(shard), "--rank-by", "keep_exact_match_rate", "--freeze"])

    doc = json.loads(tuned.read_text(encoding="utf-8"))
    assert doc["rank_metric"] == "keep_exact_match_rate"
    assert doc["engines"]["doctr"]["config"] == {"bin_thresh": 0.1}   # the KEEP-exact winner
    assert doc["selected_offline_from"] == [str(shard / "sweep_report.json")]


def test_a_later_measured_freeze_drops_the_stale_offline_provenance(tmp_path):
    """Same trap as a stale winner: a leftover line describing a run that did not happen."""
    out = tmp_path / "tuned_configs.json"
    offline = _sweep_ctx({"config": {"bin_thresh": 0.3}, "config_hash": "a" * 12})
    offline["source_reports"] = ["experiments/sweep_dev_v1_doctr/sweep_report.json"]
    freeze_winners(offline, out)
    assert "selected_offline_from" in json.loads(out.read_text(encoding="utf-8"))

    freeze_winners(_sweep_ctx({"config": {"bin_thresh": 0.5}, "config_hash": "b" * 12}), out)
    assert "selected_offline_from" not in json.loads(out.read_text(encoding="utf-8"))
