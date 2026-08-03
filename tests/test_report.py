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
