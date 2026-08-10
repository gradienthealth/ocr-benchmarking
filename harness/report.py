"""Phase 7 — stratified markdown report + charts from an aggregate. 🟢 PHI-free.

`write_report(aggregate, out_path, *, run_metadata=None)` turns the Phase 6 aggregate object
(see `harness/aggregate.py`) into a human-readable markdown report plus three matplotlib
charts. It exists so engines can be compared on **reading quality** and **false redaction**
across strata.

PHI-safety is structural, not incidental. This module consumes ONLY the aggregate dict —
counts, rates, per-stratum metric tables, latency/cost scalars, the diagnostic rollup — plus
an optional PHI-free `run_metadata` dict of header scalars (tier / version / hashes / date).
It never accepts, imports, or touches raw tokens: no `rows`, no `GTToken`, no `token_text`,
no `raw_response`, no `gt.csv`, no pixels. There is no code path here that can read a
per-token string, so it is incapable of emitting one. A guard test (`tests/test_report.py`)
pins that: it renders on a synthetic aggregate and asserts no fake token string appears.

Two CLAUDE.md invariants are carried structurally:
- **Found vs Added are two columns, never one, never averaged.** Omission (the Found axis)
  and hallucination (the Added axis) stay in separate columns in every table and separate
  bars in the chart; no combined "accuracy"/"error" number is ever produced.
- **False-redaction rate is the headline** per plan.md, shown up top *alongside* the primary
  reading-quality metric (KEEP exact-match rate) because the current PM focus ranks by reading
  quality — a conflict the report flags rather than silently resolving. CER/WER are confined
  to a clearly-labeled Diagnostic appendix and are never a ranking signal.

- **Three questions, three tables, never merged.** The end-to-end tables answer "did the
  pipeline keep the token"; the detector-only table (Phase 13e) answers "did the engine
  *find* the text at all, ignoring the string." They are rendered as separate sections with
  separate denominators so a detection loss is never mistaken for a reading loss.

Engine-agnostic: it reports whatever strata/metrics the aggregate holds; there is no
engine-specific branch here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless, deterministic — no display server needed
import matplotlib.pyplot as plt  # noqa: E402

# Header metadata the report shows but the aggregate cannot supply (it carries only
# model_name). Passed in via run_metadata; missing keys render as "(unspecified)".
_META_KEYS = ("tier", "version", "run_hash", "set_hash", "generated_at")
_UNSPEC = "(unspecified)"
_NA = "—"


# --- scalar formatters ---------------------------------------------------------


def _fmt_rate(x: Optional[float], na: str = _NA) -> str:
    """A rate in [0, 1] as a percentage, or `na` when None (no data / 0 denominator).

    `na` exists for the detector table, which spells the empty case `n/a` rather than an
    em-dash: there, a missing rate means "this group had no denominator" and must read as
    an explicit non-result, never as a dash a skimmer can mistake for a small number — and
    emphatically never as a defaulted 100% (CLAUDE.md §8). One formatter, so the percent
    formatting cannot drift between the detector table and every other table.
    """
    return na if x is None else f"{x * 100:.2f}%"


def _fmt_int(x: Optional[int]) -> str:
    return _NA if x is None else f"{int(x)}"


def _fmt_float(x: Optional[float], places: int = 4) -> str:
    return _NA if x is None else f"{x:.{places}f}"


def _fmt_cost(x: Optional[float]) -> str:
    return _NA if x is None else f"${x:.6f}"


def _meta(run_metadata: Optional[dict], key: str) -> Optional[str]:
    if not run_metadata:
        return None
    v = run_metadata.get(key)
    return None if v is None else str(v)


# --- markdown table (column widths fit content — D-7.2) -------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavored markdown table with every column padded to its content width.

    Column width = max(header, widest cell). What this guarantees: no cell is ever
    truncated or wrapped at write time (we only ever pad, never clip), and the raw .md
    stays column-aligned in an editor / `git diff` / terminal.

    What it does NOT guarantee: anything about the *rendered* table. Renderers strip this
    whitespace, so display width comes from content length alone — a very long cell still
    overflows or scrolls in HTML. D-7.2 came from the PDF-report rule, where column widths
    are set in code and overflow is a real silent failure; markdown has no such mechanism.
    Rendered-width control would need a cell-length cap (truncate + ellipsis) or the D-7.1
    HTML path.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


# --- charts (matplotlib, exactly three — D-7.3) --------------------------------
# No latency-p95 chart: latency is already tabulated and a fourth chart adds little.


def _sorted_strata(per_stratum: dict) -> list[str]:
    return sorted(per_stratum, key=str)


def _chart_fr_by_stratum(per_stratum: dict, path: Path) -> None:
    """Per-stratum false-redaction rate (the headline), as a bar. None rates are skipped."""
    labels, values = [], []
    for s in _sorted_strata(per_stratum):
        rate = per_stratum[s]["false_redaction_rate"]
        if rate is not None:
            labels.append(s)
            values.append(rate * 100)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if labels:
        ax.bar(range(len(labels)), values, color="#c0392b")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "no KEEP tokens in any stratum", ha="center", va="center")
    ax.set_ylabel("False-redaction rate (%)")
    ax.set_title("False-redaction rate by stratum (headline)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _chart_found_vs_added(per_stratum: dict, path: Path) -> None:
    """Found (omission axis) vs Added (hallucination axis) as SIDE-BY-SIDE grouped bars.

    Never stacked, never summed — the two axes must read as distinct quantities.
    """
    labels = _sorted_strata(per_stratum)
    found = [per_stratum[s]["found_count"] for s in labels]
    added = [per_stratum[s]["added_count"] for s in labels]
    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if labels:
        ax.bar([i - width / 2 for i in x], found, width, label="Found (read)", color="#27ae60")
        ax.bar([i + width / 2 for i in x], added, width, label="Added (hallucination)",
               color="#e67e22")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no strata", ha="center", va="center")
    ax.set_ylabel("Word count")
    ax.set_title("Found vs Added by stratum (separate axes, never averaged)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _chart_negative_control(nc: dict, path: Path) -> None:
    """Negative-control hallucination floor: hallucinations per confirmed-blank image."""
    per_image = nc.get("hallucinations_per_image")
    height = 0.0 if per_image is None else per_image
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar([0], [height], width=0.5, color="#8e44ad")
    ax.set_xticks([0])
    ax.set_xticklabels(["negative control"])
    ax.set_ylabel("Hallucinations per image")
    ax.set_title("Hallucination floor (confirmed-blank frames)")
    ax.annotate(
        f"floor_count={nc.get('floor_count', 0)}, n={nc.get('n_control_images', 0)}",
        xy=(0, height), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- the report ----------------------------------------------------------------


def _header_block(agg: dict, run_metadata: Optional[dict]) -> str:
    model_name = agg.get("model_name") or _UNSPEC
    vals = {k: _meta(run_metadata, k) for k in _META_KEYS}
    lines = [
        "# OCR engine benchmark report",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["Engine (model_name)", model_name],
                # Config identity (D-13.5) comes from `agg`, not `run_metadata`: it is
                # carried on every scored row by the runner itself, so unlike the header
                # scalars below it cannot be forgotten or mistyped at report time.
                ["Config", agg.get("config_id") or _UNSPEC],
                ["Config hash", agg.get("config_hash") or _UNSPEC],
                ["Tier", vals["tier"] or _UNSPEC],
                ["Exact version", vals["version"] or _UNSPEC],
                ["Run hash", vals["run_hash"] or _UNSPEC],
                ["Set hash", vals["set_hash"] or _UNSPEC],
                ["Generated at", vals["generated_at"] or _UNSPEC],
                ["Images scored", _fmt_int(agg.get("n_images"))],
            ],
        ),
    ]
    missing = [k for k in _META_KEYS if vals[k] is None]
    if missing:
        lines += [
            "",
            f"> ⚠️ Header metadata not fully supplied: {', '.join(missing)}. Record engine "
            "tier + exact version string + run/set hash for every run — an engine version bump "
            "invalidates prior results the same way a `gt.csv` change does (CLAUDE.md §4, rule #9).",
        ]
    return "\n".join(lines)


def _headline_block(overall: dict) -> str:
    fr = _fmt_rate(overall["false_redaction_rate"])
    kem = _fmt_rate(overall["keep_exact_match_rate"])
    return "\n".join([
        "## Headline",
        "",
        _md_table(
            ["Metric", "Value", "Axis"],
            [
                ["False-redaction rate", fr, "headline (plan.md) — valid KEEP tokens wrongly redacted"],
                ["KEEP exact-match rate", kem, "primary reading quality — KEEP tokens read byte-exact"],
            ],
        ),
        "",
        "> **Unreconciled ranking:** plan.md designates **false-redaction rate** as THE headline; "
        "the current PM focus ranks engines by **OCR reading quality** (KEEP exact-match), with "
        "false-redaction demoted to secondary. The docs are **not yet reconciled**, so both are "
        "shown side by side here and are **never averaged** into a single score.",
    ])


def _found_vs_added_block(overall: dict, per_stratum: dict) -> str:
    def row(label: str, g: dict) -> list[str]:
        return [
            label,
            _fmt_int(g["found_count"]),
            _fmt_int(g["omission_count"]),
            _fmt_int(g["added_count"]),
            _fmt_int(g["gt_total"]),
        ]

    rows = [row(s, per_stratum[s]) for s in _sorted_strata(per_stratum)]
    rows.append(row("Overall", overall))
    return "\n".join([
        "## Found vs Added",
        "",
        "Two separate axes — **never averaged**. *Found*/*Omission* is the omission axis (real GT "
        "text the engine did / didn't read); *Added* is the hallucination axis (invented text — "
        "the dangerous one).",
        "",
        _md_table(["Stratum", "Found", "Omission", "Added", "GT total"], rows),
        "",
        "![Found vs Added by stratum](FOUND_VS_ADDED_CHART)",
    ])


def _per_stratum_block(overall: dict, per_stratum: dict) -> str:
    def row(label: str, g: dict) -> list[str]:
        return [
            label,
            _fmt_int(g["n_images"]),
            _fmt_rate(g["false_redaction_rate"]),
            _fmt_rate(g["keep_exact_match_rate"]),
            _fmt_int(g["found_count"]),
            _fmt_int(g["omission_count"]),
            _fmt_int(g["added_count"]),
        ]

    rows = [row(s, per_stratum[s]) for s in _sorted_strata(per_stratum)]
    rows.append(row("Overall", overall))
    headers = ["Stratum", "Images", "FR rate", "KEEP exact", "Found", "Omission", "Added"]
    return "\n".join([
        "## Per-stratum breakout",
        "",
        _md_table(headers, rows),
        "",
        "![False-redaction rate by stratum](FR_BY_STRATUM_CHART)",
    ])


def _latency_cost_block(overall: dict) -> str:
    def lat_row(label: str, stats: Optional[dict]) -> list[str]:
        if stats is None:
            return [label, _NA, _NA, _NA]
        return [label, _fmt_float(stats["mean"]), _fmt_float(stats["median"]),
                _fmt_float(stats["p95"])]

    lat = _md_table(
        ["Timer", "Mean (s)", "Median (s)", "p95 (s)"],
        [
            lat_row("Primary (model)", overall["latency"]),
            lat_row("Verifier (re-read)", overall["verifier_latency"]),
        ],
    )
    cost = overall["cost"]
    cost_tbl = _md_table(
        ["Cost", "Value"],
        [["Total", _fmt_cost(cost["sum"])], ["Mean / image", _fmt_cost(cost["mean"])]],
    )
    return "\n".join([
        "## Latency & cost",
        "",
        "Primary (model) and verifier (re-read) time are reported **separately** — never merged.",
        "",
        lat,
        "",
        cost_tbl,
    ])


def _negative_control_block(nc: dict) -> str:
    return "\n".join([
        "## Negative-control hallucination floor",
        "",
        "Every prediction on a confirmed-blank frame is a pure hallucination; this is the floor.",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Control images", _fmt_int(nc["n_control_images"])],
                ["Hallucinated words (floor count)", _fmt_int(nc["floor_count"])],
                ["Hallucinations / image", _fmt_float(nc["hallucinations_per_image"])],
            ],
        ),
        "",
        "![Hallucination floor](NEG_CONTROL_CHART)",
    ])


def _detector_block(det: dict) -> str:
    """Detector-only table — boxes vs GT at the matcher's IoU bar, strings ignored."""
    per_stratum = det["per_stratum"]
    overall = det["overall"]
    iou = det.get("iou_thr")
    thr = (
        f"IoU ≥ {iou}" if iou is not None
        else "an UNRECORDED or MIXED IoU threshold (see run metadata)"
    )

    def row(label: str, g: dict) -> list[str]:
        return [
            label,
            _fmt_int(g["n_text_images"]),
            _fmt_int(g["n_control_images"]),
            _fmt_int(g["gt_total"]),
            _fmt_int(g["found_count"]),
            _fmt_int(g["omission_count"]),
            _fmt_rate(g["detection_recall"], "n/a"),
            _fmt_int(g["predicted_count"]),
            _fmt_int(g["added_count"]),
            _fmt_rate(g["detection_precision"], "n/a"),
        ]

    labeled = [(s, per_stratum[s]) for s in _sorted_strata(per_stratum)]
    labeled.append(("Overall", overall))
    # "Text imgs", not "Images": the per-stratum breakout above already has an "Images"
    # column counting EVERY image, and two denominators under one header is how a detection
    # number gets read against an end-to-end one.
    headers = ["Stratum", "Text imgs", "Blank", "GT tokens", "Found", "Missed", "Recall",
               "Boxes", "Added", "Precision"]

    notes = []
    for label, g in labeled:
        if g["detection_recall"] is None:
            notes.append(f"- **{label}** — recall `n/a`: {g['recall_na_reason']}.")
        if g["detection_precision"] is None:
            notes.append(f"- **{label}** — precision `n/a`: {g['precision_na_reason']}.")

    lines = [
        "## Detector-only view",
        "",
        "Boxes only — **strings are never compared here**. A third, separate question: did the "
        f"engine *find* the text, regardless of whether it *read* it. Matched at {thr} by the "
        "same matcher as every other table, so a loss here is a detection loss, not a reading "
        "one. Reported on its own and never merged with the end-to-end or reader tables.",
        "",
        "*Text imgs* counts text-bearing images only — recall is denominated in their GT tokens, "
        "precision in the boxes predicted on them. *Blank* counts the zero-token control frames, "
        "which are held out of both rates and appear only in the floor line below.",
        "",
        _md_table(headers, [row(label, g) for label, g in labeled]),
    ]
    if notes:
        lines += [
            "",
            "Where a rate reads `n/a` the denominator is genuinely zero — **not** a 100% score:",
            "",
            *notes,
        ]
    # The floor gets the same n/a-with-reason treatment as the rates. Printing "0 images
    # carried 0 boxes" for a run with no controls reads as a MEASURED floor of zero — the
    # "no invention floor" blind spot CLAUDE.md §8 says hid the ct_scout error.
    n_ctrl = overall["n_control_images"]
    if n_ctrl:
        floor_line = (
            "**Hallucination floor (blank controls).** "
            f"{_fmt_int(n_ctrl)} boxed confirmed-blank image(s) carried "
            f"{_fmt_int(overall['floor_boxes'])} box(es) — "
            f"{_fmt_float(overall['floor_boxes_per_image'], 3)} per image. Detection precision "
            "on those frames is 0 by construction (every box there is a false positive), which "
            "is exactly why they are held out of the precision denominator above rather than "
            "averaged into it."
        )
    else:
        why = (
            f"all {overall['n_excluded_controls']} blank-control frame(s) here went through "
            "no IoU matcher (box-free, or the reading arm's given boxes) and structurally "
            "cannot report invented boxes"
            if overall["n_excluded_controls"]
            else "this run scored no blank-control frames at all"
        )
        floor_line = (
            "**Hallucination floor (blank controls): `n/a`** — "
            f"{why}, so no invention floor was measured. This is **not** a floor of zero: a "
            "run with no blank frames has nothing to bound its detection precision against, "
            "and reading the absence as \"invented nothing\" is exactly the blind spot that "
            "hid the ct_scout error (CLAUDE.md §8). The run-wide *Negative-control* section "
            "above counts every blank frame, box-free ones included."
        )
    lines += ["", floor_line]
    # Per-stratum floor, not just the pooled number: invention rates differ by two orders of
    # magnitude across strata, so one pooled floor hides which stratum is doing the inventing.
    floor_rows = [
        [s, _fmt_int(g["n_control_images"]), _fmt_int(g["floor_boxes"]),
         _fmt_float(g["floor_boxes_per_image"], 3)]
        for s, g in labeled[:-1] if g["n_control_images"]
    ]
    if floor_rows:
        lines += [
            "",
            _md_table(["Stratum", "Blank images", "Boxes", "Boxes / image"], floor_rows),
        ]
    n_bf = overall["n_box_free_excluded"]
    n_ra = overall["n_reading_arm_excluded"]
    if n_bf or n_ra:
        lines += [
            "",
            f"> ⚠️ {n_bf + n_ra} row(s) excluded from this table, **including from the floor "
            f"above** — {n_bf} box-free, {n_ra} reading-arm. The box-free path reports no "
            "hallucinations at all (no per-word structure to locate one with), and the reading "
            "arm is *handed the GT box* and runs no matcher, so every token is found by "
            "construction. Counting either would print a better detector than was measured — "
            "flawless precision, or a saturated 100% recall — and would dilute the floor "
            "toward zero. This is why the floor here can be smaller than the run-wide "
            "*Negative-control hallucination floor* section, which counts every blank frame.",
        ]
    return "\n".join(lines)


def _diagnostic_block(overall: dict, per_stratum: dict) -> str:
    """CER/WER appendix — clearly labeled DIAGNOSTIC ONLY; never a ranking signal."""
    d = overall.get("diagnostic", {})

    def row(label: str, g: dict) -> list[str]:
        gd = g.get("diagnostic", {})
        return [
            label,
            _fmt_float(gd.get("cer_mean")),
            _fmt_float(gd.get("wer_mean")),
            _fmt_int(gd.get("n_cer_images")),
            _fmt_rate(gd.get("char_censor_rate")),
        ]

    rows = [row(s, per_stratum[s]) for s in _sorted_strata(per_stratum)]
    rows.append(row("Overall", overall))
    return "\n".join([
        "## Diagnostic appendix",
        "",
        "> **DIAGNOSTIC ONLY — never a ranking signal.** CER/WER are for debugging read quality; "
        "engines are ranked by false-redaction / reading-quality above, not by these. "
        "`n` is the number of images that produced a CER (an image with no matched pairs has none). "
        "CER/WER are **macro means** over those images.",
        "",
        _md_table(
            ["Stratum", "CER (mean)", "WER (mean)", "n", "Char-censor rate"],
            rows,
        ),
    ])


def write_report(aggregate: dict, out_path, *, run_metadata: Optional[dict] = None) -> str:
    """Render a stratified markdown report + 3 charts from an aggregate; return the markdown.

    Args:
        aggregate: the Phase 6 aggregate dict (only source of metrics — no tokens ever).
        out_path: markdown file to write; chart PNGs are saved alongside it.
        run_metadata: optional PHI-free header scalars — tier / version / run_hash /
            set_hash / generated_at. Missing keys render as "(unspecified)" and are flagged;
            never fabricated. `model_name` always comes from the aggregate.

    Returns the rendered markdown string (also written to `out_path`).
    """
    out_path = Path(out_path)
    out_dir = out_path.parent
    stem = out_path.stem

    overall = aggregate["overall"]
    per_stratum = aggregate["per_stratum"]
    nc = aggregate["negative_control"]

    # Charts, saved next to the markdown; referenced by relative filename.
    fr_name = f"{stem}_fr_by_stratum.png"
    fa_name = f"{stem}_found_vs_added.png"
    ncc_name = f"{stem}_neg_control.png"
    _chart_fr_by_stratum(per_stratum, out_dir / fr_name)
    _chart_found_vs_added(per_stratum, out_dir / fa_name)
    _chart_negative_control(nc, out_dir / ncc_name)

    blocks = [
        _header_block(aggregate, run_metadata),
        _headline_block(overall),
        _found_vs_added_block(overall, per_stratum),
        _per_stratum_block(overall, per_stratum),
        _latency_cost_block(overall),
        _negative_control_block(nc),
    ]
    # Detector-only view: present on any aggregate built by `aggregate()`. Optional so a
    # hand-assembled aggregate from before Phase 13e still renders rather than crashing —
    # it is a second view, not a prerequisite of the report.
    detection = aggregate.get("detection")
    if detection:
        blocks.append(_detector_block(detection))
    else:
        # Say so rather than omitting silently: a missing section reads as "the engine had
        # no detection losses", which is the opposite of "nobody measured them."
        blocks.append(
            "## Detector-only view\n\n"
            "> ⚠️ **Not rendered** — this aggregate carries no `detection` key, so the "
            "detector-only table could not be built. Rebuild it with "
            "`harness.aggregate.aggregate()`; do not read the absence as a clean detection "
            "result."
        )
    blocks.append(_diagnostic_block(overall, per_stratum))
    md = "\n\n".join(blocks)
    # Wire chart filenames in last so the block helpers stay free of path detail.
    md = (
        md.replace("FR_BY_STRATUM_CHART", fr_name)
        .replace("FOUND_VS_ADDED_CHART", fa_name)
        .replace("NEG_CONTROL_CHART", ncc_name)
    ) + "\n"

    out_path.write_text(md, encoding="utf-8")
    return md
