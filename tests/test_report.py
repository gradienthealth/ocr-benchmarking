"""Phase 7 tests for harness/report.py — PHI-free, synthetic aggregate only.

The report consumes ONLY the aggregate object (counts/rates/strata/latency/cost/diagnostic),
so these tests build a real aggregate from synthetic score rows via the actual `aggregate()`
function — the shape is therefore authoritative, not hand-mocked. Fake PHI-shaped tokens are
declared as constants and deliberately NEVER placed into the aggregate; the PHI-guard test
proves none of them can appear in the rendered output.

The tests pin the CLAUDE.md invariants that live at the report layer: FR is the headline with
reading-quality shown alongside, Found and Added are separate columns (never averaged), all
strata break out, latency/cost/negative-control floor are present, CER/WER are quarantined in a
labeled Diagnostic appendix, and exactly three charts (no latency chart) are produced.
"""

from __future__ import annotations

import inspect
import pathlib
import tempfile

import pytest

from harness.aggregate import aggregate
from harness.report import write_report

# Fake, PHI-shaped tokens in the CLAUDE.md house style. These are NEVER inserted into the
# aggregate — the guard test asserts they cannot surface in the output.
FAKE_TOKENS = ["CMFN-00421", "GRDN-99887", "thryothor-11223"]

STRATA = ["us_header", "ct_scout", "secondary_capture", "mammo"]


def _diag(cer=None, wer=None, keep_chars_total=0, keep_chars_censored=0) -> dict:
    return dict(cer=cer, wer=wer, keep_chars_total=keep_chars_total,
                keep_chars_censored=keep_chars_censored)


def _mkrow(**over) -> dict:
    """A score row with every key aggregate() reads — numeric/string only, no token text."""
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
        stratum="us_header",
        model_name="synth-engine",
        # Non-empty by default: aggregate() rejects a blank version (D-8.4).
        version="0.0.0-synthetic",
        # Same, for config identity (D-13.5) — the report renders both in its header.
        config_id="synthetic",
        config_hash="0000synthetic",
        # Shared across every row by default (D-9.1 mirrors D-8.4): _synthetic_aggregate()
        # below has exactly one row that actually records a verifier_elapsed, but the
        # mixed-pair guard keys on identity, not elapsed, so every row in one batch must
        # agree on it regardless of which ones actually ran the verifier.
        verifier_model_name="synth-verifier",
        verifier_version="0.0.0-synthetic",
        diagnostic=_diag(),
    )
    row.update(over)
    return row


def _synthetic_aggregate() -> dict:
    """A 4-stratum aggregate with a KEEP mix, a hallucination, and a blank negative control."""
    rows = [
        _mkrow(stratum="us_header", keep_total=4, false_redaction_count=1,
               keep_exact_match_count=3, found_count=4, omission_count=0, gt_total=4,
               added_count=1, elapsed=0.12, verifier_elapsed=0.3, cost=0.001,
               diagnostic=_diag(cer=0.10, wer=0.20, keep_chars_total=20, keep_chars_censored=2)),
        _mkrow(stratum="ct_scout", keep_total=2, false_redaction_count=0,
               keep_exact_match_count=2, found_count=2, omission_count=1, gt_total=3,
               added_count=0, elapsed=0.20, cost=0.002,
               diagnostic=_diag(cer=0.05, wer=0.05, keep_chars_total=10, keep_chars_censored=0)),
        _mkrow(stratum="secondary_capture", keep_total=3, false_redaction_count=2,
               keep_exact_match_count=1, found_count=3, omission_count=2, gt_total=5,
               added_count=3, elapsed=0.35, cost=0.004,
               diagnostic=_diag(cer=0.40, wer=0.55, keep_chars_total=15, keep_chars_censored=8)),
        _mkrow(stratum="mammo", keep_total=1, false_redaction_count=0,
               keep_exact_match_count=1, found_count=1, omission_count=0, gt_total=1,
               added_count=0, elapsed=0.18, cost=0.001,
               diagnostic=_diag(cer=0.0, wer=0.0, keep_chars_total=6, keep_chars_censored=0)),
        # blank negative control (zero GT tokens) with pure hallucinations
        _mkrow(stratum="ct_scout", keep_total=0, gt_total=0, added_count=2,
               negative_control=True, negative_control_floor_count=2, elapsed=0.15, cost=0.001),
    ]
    return aggregate(rows)


@pytest.fixture()
def rendered(tmp_path):
    out = tmp_path / "report.md"
    md = write_report(
        _synthetic_aggregate(),
        out,
        run_metadata={"tier": "step-1", "version": "pp-ocrv6_medium-3.7.1",
                      "run_hash": "abc123", "set_hash": "def456",
                      "generated_at": "2026-07-23"},
    )
    return md, out, tmp_path


# --- headline: FR + reading quality --------------------------------------------


def test_fr_is_the_headline(rendered):
    md, _, _ = rendered
    assert "## Headline" in md
    assert "False-redaction rate" in md
    # headline value present (overall FR = 3 fr / 10 keep = 30.00%)
    assert "30.00%" in md


def test_reading_quality_metric_up_top(rendered):
    md, _, _ = rendered
    headline = md.split("## Headline", 1)[1].split("## Found vs Added", 1)[0]
    assert "KEEP exact-match rate" in headline
    assert "reading quality" in headline.lower()


def test_unreconciled_ranking_note_present(rendered):
    md, _, _ = rendered
    assert "Unreconciled ranking" in md
    assert "never averaged" in md.lower()


# --- Found vs Added: two columns, never blended --------------------------------


def test_found_and_added_are_separate_columns(rendered):
    md, _, _ = rendered
    fa = md.split("## Found vs Added", 1)[1].split("## Per-stratum", 1)[0]
    assert "Found" in fa and "Added" in fa and "Omission" in fa
    # never merged into a single blended score anywhere in the report
    for banned in ("accuracy", "combined score", "avg_error", "average error",
                   "blended", "overall error rate"):
        assert banned not in md.lower()


# --- per-stratum breakout: all four strata -------------------------------------


def test_all_four_strata_rows_present(rendered):
    md, _, _ = rendered
    breakout = md.split("## Per-stratum breakout", 1)[1].split("## Latency", 1)[0]
    for stratum in STRATA:
        assert stratum in breakout


# --- latency, cost, negative control -------------------------------------------


def test_latency_cost_and_negative_control_sections(rendered):
    md, _, _ = rendered
    assert "## Latency & cost" in md
    assert "Primary (model)" in md and "Verifier (re-read)" in md
    assert "## Negative-control hallucination floor" in md
    assert "Hallucinations / image" in md


# --- diagnostic appendix: CER/WER quarantined ----------------------------------


def test_diagnostic_appendix_present_and_labeled(rendered):
    md, _, _ = rendered
    assert "## Diagnostic appendix" in md
    appendix = md.split("## Diagnostic appendix", 1)[1]
    assert "DIAGNOSTIC ONLY" in appendix
    assert "CER" in appendix and "WER" in appendix
    # CER/WER must appear ONLY in the appendix, never in the headline/breakout.
    assert "CER" not in md.split("## Diagnostic appendix", 1)[0]


# --- PHI guard -----------------------------------------------------------------


def test_no_fake_token_text_appears_in_output(rendered):
    md, out, _ = rendered
    written = out.read_text(encoding="utf-8")
    for token in FAKE_TOKENS:
        assert token not in md
        assert token not in written


def test_write_report_signature_takes_no_token_or_row_argument():
    # Structural guarantee: the report cannot be fed raw tokens/rows — only the aggregate,
    # an out path, and PHI-free header metadata.
    params = list(inspect.signature(write_report).parameters)
    assert params == ["aggregate", "out_path", "run_metadata"]


# --- charts: exactly three, no latency chart -----------------------------------


def test_exactly_three_charts_no_latency_chart(rendered):
    _, out, tmp_path = rendered
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert pngs == sorted([
        "report_fr_by_stratum.png",
        "report_found_vs_added.png",
        "report_neg_control.png",
    ])
    assert not any("latency" in name for name in pngs)


def test_charts_referenced_in_markdown(rendered):
    md, _, _ = rendered
    for name in ("report_fr_by_stratum.png", "report_found_vs_added.png",
                 "report_neg_control.png"):
        assert f"({name})" in md


# --- missing header metadata is flagged, not fabricated ------------------------


def test_missing_run_metadata_is_flagged(tmp_path):
    md = write_report(_synthetic_aggregate(), tmp_path / "r.md")  # no run_metadata
    assert "(unspecified)" in md
    assert "not fully supplied" in md
    # model_name still comes from the aggregate
    assert "synth-engine" in md


def test_config_identity_rendered_in_header(tmp_path):
    """Two arms of one engine must be tellable apart by reading the report (D-13.5).

    Both come from the aggregate rather than `run_metadata`, so — unlike the header scalars
    — they cannot be forgotten or mistyped at report time: every scored row carries them.
    """
    md = write_report(_synthetic_aggregate(), tmp_path / "r.md")
    assert "| Config " in md
    assert "synthetic" in md
    assert "0000synthetic" in md  # the digest, not just the label


# --- detector-only view (Phase 13e) --------------------------------------------
# The third table: did the engine FIND the text, ignoring whether it read it. Two things
# are pinned here because getting them wrong invents signal — a stratum with no
# text-bearing images must render `n/a` with a reason rather than a saturated 100%, and a
# blank control's invented boxes must land in the hallucination floor, never in a rate.


def _detector_aggregate() -> dict:
    """Text-bearing + blank rows across three strata, one of them entirely blank."""
    rows = [
        # text-bearing: 3 of 4 tokens found, 1 invented box
        _mkrow(stratum="us_ge", gt_total=4, found_count=3, omission_count=1, added_count=1,
               box_free=False, iou_thr=0.5),
        # a stratum with NO text-bearing images at all (mg_tomo / ct_scout in the real set)
        _mkrow(stratum="mg_tomo", gt_total=0, negative_control=True,
               negative_control_floor_count=3, added_count=3, box_free=False, iou_thr=0.5),
        _mkrow(stratum="mg_tomo", gt_total=0, negative_control=True, box_free=False,
               iou_thr=0.5),
        # mixed stratum: one text-bearing image plus one blank control that invented a box
        _mkrow(stratum="ct_axial", gt_total=2, found_count=2, box_free=False, iou_thr=0.5),
        _mkrow(stratum="ct_axial", gt_total=0, negative_control=True,
               negative_control_floor_count=1, added_count=1, box_free=False, iou_thr=0.5),
    ]
    return aggregate(rows)


@pytest.fixture()
def detector_md(tmp_path):
    md = write_report(_detector_aggregate(), tmp_path / "det.md")
    return md.split("## Detector-only view", 1)[1].split("## Diagnostic appendix", 1)[0]


def _table_row(section: str, label: str) -> list[str]:
    for line in section.splitlines():
        if line.startswith(f"| {label} "):
            return [c.strip() for c in line.strip("|").split("|")]
    raise AssertionError(f"no table row for {label!r}")


def test_detector_section_is_its_own_table(detector_md):
    assert "Boxes only" in detector_md
    assert "never merged with the end-to-end or reader tables" in detector_md
    assert "Recall" in detector_md and "Precision" in detector_md
    # the matcher's bar is stated, so the table can't be read at the wrong threshold
    assert "IoU ≥ 0.5" in detector_md


def test_detection_rates_appear_only_in_the_detector_section(tmp_path):
    md = write_report(_detector_aggregate(), tmp_path / "det.md")
    before = md.split("## Detector-only view", 1)[0]
    assert "Detector-only" not in before
    # The end-to-end sections above never gain the detector table's columns. Keyed on the
    # column header itself, not on the bare word "precision" — banning a common word would
    # fire on unrelated prose edits upstream and read as a false alarm.
    assert "| Text imgs " not in before
    assert "detection_recall" not in before


def test_zero_text_stratum_renders_na_not_100_percent(detector_md):
    cells = _table_row(detector_md, "mg_tomo")
    stratum, images, blank = cells[0], cells[1], cells[2]
    recall, precision = cells[6], cells[9]
    assert stratum == "mg_tomo"
    assert images == "0" and blank == "2"
    assert recall == "n/a" and precision == "n/a"
    assert "100.00%" not in "".join(cells)


def test_na_cells_carry_their_reason(detector_md):
    assert "recall `n/a`" in detector_md
    assert "no boxed text-bearing images" in detector_md
    assert "no recall denominator" in detector_md


def test_blank_control_boxes_are_in_the_floor_and_not_in_any_rate(detector_md):
    # ct_axial: 1 text image (2/2 found, 0 added) + 1 blank control that invented a box.
    cells = _table_row(detector_md, "ct_axial")
    assert cells[1] == "1" and cells[2] == "1"  # 1 text-bearing, 1 blank
    assert cells[6] == "100.00%"  # recall over the text-bearing image only
    assert cells[8] == "0"  # Added: the control's box is NOT here
    assert cells[9] == "100.00%"
    # overall Added is the one invented box on a text-bearing image, not the 4 on blanks
    assert _table_row(detector_md, "Overall")[8] == "1"


def test_hallucination_floor_is_its_own_line(detector_md):
    assert "Hallucination floor" in detector_md
    # "boxed": box-free controls are excluded here, so this floor can differ from the
    # run-wide negative-control section, which counts every blank frame.
    assert "3 boxed confirmed-blank image(s) carried 4 box(es)" in detector_md
    assert "1.333 per image" in detector_md
    assert "held out of the precision denominator" in detector_md


def test_floor_breaks_out_by_stratum_not_just_pooled(detector_md):
    # Invention rates differ by orders of magnitude across strata (CLAUDE.md §8), so a
    # single pooled floor hides which stratum invents. Only blank-carrying strata appear.
    floor_tbl = detector_md.split("Hallucination floor (blank controls)", 1)[1]
    assert "| Blank images | Boxes | Boxes / image |" in floor_tbl
    assert _table_row(floor_tbl, "mg_tomo") == ["mg_tomo", "2", "3", "1.500"]
    assert _table_row(floor_tbl, "ct_axial") == ["ct_axial", "1", "1", "1.000"]
    # us_ge has no blank control images at all -> no floor row
    assert "| us_ge " not in floor_tbl


def test_detector_table_adds_no_fourth_chart(tmp_path):
    write_report(_detector_aggregate(), tmp_path / "det.md")
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert pngs == sorted(["det_fr_by_stratum.png", "det_found_vs_added.png",
                           "det_neg_control.png"])


def test_report_still_renders_without_a_detection_key(tmp_path):
    agg = _detector_aggregate()
    agg.pop("detection")
    md = write_report(agg, tmp_path / "old.md")
    # Fail loud, not silent: the section is still there, saying it could not be built.
    assert "## Detector-only view" in md
    assert "Not rendered" in md
    assert "| Text imgs " not in md
    assert "## Headline" in md


def _one_run(rows) -> dict:
    return aggregate(rows)


def test_floor_reads_na_with_a_reason_when_a_run_has_no_blank_controls():
    # "0 images carried 0 boxes" reads as a MEASURED floor of zero. A run with no blank
    # frames has no floor at all — the blind spot CLAUDE.md §8 says hid the ct_scout error.
    agg = _one_run([_mkrow(stratum="us_ge", gt_total=5, found_count=4, omission_count=1,
                           added_count=2, box_free=False, iou_thr=0.5)])
    md = write_report(agg, pathlib.Path(tempfile.mkdtemp()) / "r.md")
    sec = md.split("## Detector-only view", 1)[1].split("## Diagnostic appendix", 1)[0]
    assert "Hallucination floor (blank controls): `n/a`" in sec
    assert "no blank-control frames at all" in sec
    assert "**not** a floor of zero" in sec
    assert "carried 0 box(es)" not in sec
    assert "— — per image" not in sec  # the em-dash hole


def test_box_free_run_floor_names_the_reason_instead_of_reporting_zero_controls():
    rows = [_mkrow(stratum="us_ge", box_free=True, gt_total=5, found_count=4, omission_count=1,
                   iou_thr=0.5)]
    rows += [_mkrow(stratum="ct_axial", box_free=True, gt_total=0, negative_control=True,
                    iou_thr=0.5) for _ in range(3)]
    agg = _one_run(rows)
    md = write_report(agg, pathlib.Path(tempfile.mkdtemp()) / "r.md")
    sec = md.split("## Detector-only view", 1)[1].split("## Diagnostic appendix", 1)[0]
    assert "all 3 blank-control frame(s) here went through" in sec
    # the run-wide section still counts them — the divergence is stated, not hidden
    assert agg["negative_control"]["n_control_images"] == 3
    assert "including from the floor above" in sec
