"""Synthetic-only tests for `ground_truth/seed_tesseract.py` (Phase 10c). 🟢 PHI-FREE.

Every image here is fabricated in memory by `tests/synthetic.py` (blank frames and
`CMFN`-style fake tokens) and saved into `tmp_path`. Every OCR result is a fabricated
`image_to_data`-shaped DICT fed through an injected `ocr_fn`. Zero real renders, zero
`.dcm`, zero `gt.csv`, zero PHI.

Only one test needs the real `tesseract` binary: the synthetic positive control
(CLAUDE.md §8) — a "0 boxes" result is meaningless until the detector is shown to fire on
known text. It is guarded by `requires_tesseract`, whose skip is visible under `-ra` and
which can be forced into a hard failure with `SEED_REQUIRE_TESSERACT=1`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ground_truth import render, seed_tesseract
from tests.synthetic import FAKE_TOKENS, make_blank_image, make_synthetic_image

# ---------------------------------------------------------------------------
# helpers — all small, all synthetic
# ---------------------------------------------------------------------------

_SIZE = (128, 96)  # original render px; upscaled buffers are 2x this
_DATA_KEYS = ("level", "left", "top", "width", "height", "conf", "text")


class _Boom(Exception):
    """A per-image failure with a name the summary can be asserted against."""


def _provenance(**overrides: object) -> dict[str, object]:
    """A complete, obviously-synthetic D-10c.4 provenance block."""
    prov: dict[str, object] = {
        "tesseract_cmd_config": seed_tesseract.TESSERACT_CONFIG,
        "lang": seed_tesseract.LANG,
        "upscale_factor": seed_tesseract.UPSCALE_FACTOR,
        "upscale_filter": seed_tesseract.UPSCALE_FILTER_NAME,
        "coord_space": seed_tesseract.COORD_SPACE,
        "row_filter": seed_tesseract.ROW_FILTER,
        "conf_parse": seed_tesseract.CONF_PARSE,
        "env": dict(seed_tesseract.TESSERACT_ENV),
        "timeout_s": seed_tesseract.TIMEOUT_S,
        "tesseract_version": "tesseract 0.0.0-synthetic",
        "leptonica_version": "leptonica-0.0.0-synthetic",
        "pytesseract_version": "0.0.0-synthetic",
        "pillow_version": "0.0.0-synthetic",
        "tessdata_dir": "/synthetic/tessdata",
        "tessdata_variant": "best",
        "traineddata_sha256": "0" * 64,
    }
    assert set(prov) == set(seed_tesseract.PROVENANCE_FIELDS)
    prov.update(overrides)
    return prov


def _fake_data(rows: list[tuple[object, ...]]) -> dict[str, list[object]]:
    """Build an `image_to_data`-shaped DICT from (level,left,top,width,height,conf,text)."""
    return {key: [row[i] for row in rows] for i, key in enumerate(_DATA_KEYS)}


def _word_rows(texts, *, conf: float = 90.0, level: int = 5) -> list[tuple[object, ...]]:
    """One in-bounds word row per text, in UPSCALED space, stacked vertically."""
    return [(level, 2, 2 + i * 20, 40, 16, conf, text) for i, text in enumerate(texts)]


def _stub_ocr(rows: list[tuple[object, ...]], *, seen: list | None = None):
    """An OCRFn returning a fixed fabricated DICT, optionally recording each image size."""

    def ocr_fn(img):
        if seen is not None:
            seen.append(img.size)
        return _fake_data(rows)

    return ocr_fn


def _write_manifest(path: Path, rows: list[tuple[object, ...]]) -> Path:
    """Hand-write a render manifest with 10b's exact frozen header, LF-terminated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(render.MANIFEST_COLUMNS)]
    lines += [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_renders(tmp_path: Path, sizes: dict[str, tuple[int, int]]) -> tuple[Path, Path]:
    """Write one blank synthetic PNG per image_id plus a matching manifest."""
    renders_dir = tmp_path / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id, size in sizes.items():
        png = renders_dir / f"{image_id}.png"
        img, _ = make_blank_image(size)
        img.save(png, format="PNG")
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        rows.append((image_id, 0, size[0], size[1], sha, 0))
    manifest = _write_manifest(renders_dir / render.MANIFEST_NAME, rows)
    return renders_dir, manifest


def _out_dir(tmp_path: Path, name: str = "gt") -> Path:
    """The seed dir itself; its parent is where seed_summary.json belongs (D-10c.8)."""
    return tmp_path / name / seed_tesseract.SEED_DIR_NAME


def _run(
    renders_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    ocr_fn,
    *,
    on_existing: str = "overwrite",
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    return seed_tesseract.run_seed(
        renders_dir=renders_dir,
        manifest_path=manifest_path,
        out_dir=out_dir,
        provenance=_provenance() if provenance is None else provenance,
        on_existing=on_existing,
        ocr_fn=ocr_fn,
    )


def _seed(out_dir: Path, image_id: str) -> dict:
    return json.loads((out_dir / f"{image_id}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. bbox geometry (pure, no Tesseract)
# ---------------------------------------------------------------------------


def test_bbox_from_row_yields_ordered_in_bounds_corners() -> None:
    w, h = _SIZE
    x0, y0, x1, y1 = seed_tesseract.bbox_from_row(10, 20, 30, 40, 2)
    assert (x0, y0, x1, y1) == (5.0, 10.0, 20.0, 30.0)
    assert x0 < x1 and y0 < y1
    assert 0 <= x0 and 0 <= y0 and x1 <= w and y1 <= h


# ---------------------------------------------------------------------------
# 2. upscale round-trip (D-10c.4)
# ---------------------------------------------------------------------------


def test_upscaled_coords_divide_back_exactly_and_the_buffer_is_exactly_2x(
    tmp_path: Path,
) -> None:
    # Odd coordinates in upscaled space must come back as exact, UNROUNDED halves.
    data = _fake_data([(5, 21, 43, 5, 7, 88.0, "CMFN")])
    (token,) = seed_tesseract.tokens_from_data(data, upscale_factor=2, w=_SIZE[0], h=_SIZE[1])
    assert token.bbox == (10.5, 21.5, 13.0, 25.0)
    assert token.bbox[0] == 21 / 2 != 10 and token.bbox[0] != 11

    # And the image actually handed to Tesseract is exactly UPSCALE_FACTOR x the render.
    seen: list[tuple[int, int]] = []
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE})
    _run(renders, manifest, _out_dir(tmp_path), _stub_ocr(_word_rows(["CMFN"]), seen=seen))
    factor = seed_tesseract.UPSCALE_FACTOR
    assert seen == [(_SIZE[0] * factor, _SIZE[1] * factor)]


# ---------------------------------------------------------------------------
# 3. no filtering (D-10c.4)
# ---------------------------------------------------------------------------


def test_low_confidence_single_char_word_is_kept_while_non_word_rows_are_dropped() -> None:
    data = _fake_data(
        [
            (1, 0, 0, 128, 96, -1, "page"),
            (2, 0, 0, 128, 96, -1, "block"),
            (3, 0, 0, 128, 96, -1, "para"),
            (4, 2, 2, 40, 16, -1, "line"),
            (5, 2, 2, 6, 16, 8, "X"),  # conf 8, one char: KEPT, never filtered
            (5, 2, 22, 40, 16, 95, ""),  # empty text: parsing, not filtering
            (5, 2, 42, 40, 16, 95, "   "),  # whitespace only: same
        ]
    )
    tokens = seed_tesseract.tokens_from_data(data, upscale_factor=2, w=_SIZE[0], h=_SIZE[1])
    assert [t.text for t in tokens] == ["X"]
    assert tokens[0].confidence == 8.0


# ---------------------------------------------------------------------------
# 4. default label + no review state (D-10c.2)
# ---------------------------------------------------------------------------


def test_every_token_is_labelled_phi_and_no_field_implies_review_state(tmp_path: Path) -> None:
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE})
    out_dir = _out_dir(tmp_path)
    _run(renders, manifest, out_dir, _stub_ocr(_word_rows(["CMFN", "GRDN1234"])))

    seed = _seed(out_dir, "aaaa1111")
    assert [t["label"] for t in seed["tokens"]] == ["PHI", "PHI"]

    raw = (out_dir / "aaaa1111.json").read_text(encoding="utf-8").lower()
    for banned in ("reviewed", "accepted", "edited", "deferred", "status", "timestamp"):
        assert banned not in raw


# ---------------------------------------------------------------------------
# 5. raw text preserved (no normalize(), no strip-and-store, no casefold)
# ---------------------------------------------------------------------------


def test_raw_text_round_trips_byte_identically(tmp_path: Path) -> None:
    raw_text = "  CMFN‐042 "  # leading/trailing space + a UNICODE hyphen, not ASCII "-"
    data = _fake_data([(5, 2, 2, 40, 16, 91.5, raw_text)])
    (token,) = seed_tesseract.tokens_from_data(data, upscale_factor=2, w=_SIZE[0], h=_SIZE[1])
    assert token.text == raw_text  # the ROW_FILTER strips only to DECIDE, never to store

    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE})
    out_dir = _out_dir(tmp_path)
    _run(renders, manifest, out_dir, _stub_ocr([(5, 2, 2, 40, 16, 91.5, raw_text)]))
    assert _seed(out_dir, "aaaa1111")["tokens"][0]["text"] == raw_text


# ---------------------------------------------------------------------------
# 6. seed schema + complete provenance (D-10c.3 / D-10c.4)
# ---------------------------------------------------------------------------


def test_seed_schema_round_trips_and_incomplete_provenance_fails_loud(tmp_path: Path) -> None:
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE})
    out_dir = _out_dir(tmp_path)
    _run(renders, manifest, out_dir, _stub_ocr(_word_rows(["CMFN"])))

    seed = _seed(out_dir, "aaaa1111")
    assert set(seed) == {"image_id", "w", "h", "provenance", "tokens"}
    assert (seed["image_id"], seed["w"], seed["h"]) == ("aaaa1111", _SIZE[0], _SIZE[1])
    assert set(seed["provenance"]) == set(seed_tesseract.PROVENANCE_FIELDS)
    assert set(seed["tokens"][0]) == {"text", "bbox", "confidence", "label"}
    assert len(seed["tokens"][0]["bbox"]) == 4

    incomplete = _provenance()
    del incomplete["traineddata_sha256"]
    with pytest.raises(seed_tesseract.ProvenanceIncompleteError):
        seed_tesseract.check_provenance_complete(incomplete)

    out2 = _out_dir(tmp_path, "gt2")
    with pytest.raises(seed_tesseract.ProvenanceIncompleteError):
        _run(renders, manifest, out2, _stub_ocr(_word_rows(["CMFN"])), provenance=incomplete)
    assert not list(out2.glob("*.json"))


# ---------------------------------------------------------------------------
# 7. determinism
# ---------------------------------------------------------------------------


def test_two_runs_over_the_same_renders_are_byte_identical(tmp_path: Path) -> None:
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE, "bbbb2222": _SIZE})
    rows = _word_rows(["CMFN", "GRDN1234", "ACC-0001"])
    first = _out_dir(tmp_path, "run1")
    second = _out_dir(tmp_path, "run2")
    _run(renders, manifest, first, _stub_ocr(rows))
    _run(renders, manifest, second, _stub_ocr(rows))
    for image_id in ("aaaa1111", "bbbb2222"):
        a = (first / f"{image_id}.json").read_bytes()
        b = (second / f"{image_id}.json").read_bytes()
        assert a == b and a != b""


# ---------------------------------------------------------------------------
# 8. preflight runs before ANY write (D-10c.6)
# ---------------------------------------------------------------------------


def test_preflight_aborts_on_missing_png_extra_png_or_hash_mismatch(tmp_path: Path) -> None:
    ocr_fn = _stub_ocr(_word_rows(["CMFN"]))

    # (a) a manifest row with no PNG on disk
    renders_a, manifest_a = _make_renders(tmp_path / "a", {"aaaa1111": _SIZE})
    (renders_a / "aaaa1111.png").unlink()
    out_a = _out_dir(tmp_path / "a")
    with pytest.raises(seed_tesseract.PreflightError):
        _run(renders_a, manifest_a, out_a, ocr_fn)
    assert not out_a.exists() or not list(out_a.glob("*.json"))

    # (b) a PNG on disk that the manifest does not know about
    renders_b, manifest_b = _make_renders(tmp_path / "b", {"aaaa1111": _SIZE})
    extra, _ = make_blank_image(_SIZE)
    extra.save(renders_b / "cccc3333.png", format="PNG")
    out_b = _out_dir(tmp_path / "b")
    with pytest.raises(seed_tesseract.PreflightError):
        _run(renders_b, manifest_b, out_b, ocr_fn)
    assert not out_b.exists() or not list(out_b.glob("*.json"))

    # (c) a PNG whose bytes no longer match the manifest sha256
    renders_c, manifest_c = _make_renders(tmp_path / "c", {"aaaa1111": _SIZE, "bbbb2222": _SIZE})
    changed, _ = make_synthetic_image(("CMFN",), size=_SIZE)
    changed.save(renders_c / "bbbb2222.png", format="PNG")
    out_c = _out_dir(tmp_path / "c")
    with pytest.raises(seed_tesseract.PreflightError):
        _run(renders_c, manifest_c, out_c, ocr_fn)
    assert not out_c.exists() or not list(out_c.glob("*.json"))


# ---------------------------------------------------------------------------
# 9. manifest parsing (D-10c.6) — the regression pandas would introduce
# ---------------------------------------------------------------------------


def test_all_digit_image_id_stays_a_string_and_a_missing_manifest_fails_loud(
    tmp_path: Path,
) -> None:
    renders, manifest = _make_renders(tmp_path, {"12345678": _SIZE})
    out_dir = _out_dir(tmp_path)
    summary = _run(renders, manifest, out_dir, _stub_ocr(_word_rows(["CMFN"])))

    seed = _seed(out_dir, "12345678")  # the join to the PNG stem survived
    assert seed["image_id"] == "12345678"
    assert isinstance(seed["image_id"], str)
    assert "12345678" in summary["per_image_box_counts"]

    missing = _out_dir(tmp_path, "gt2")
    with pytest.raises(seed_tesseract.PreflightError):
        _run(renders, tmp_path / "nope.csv", missing, _stub_ocr(_word_rows(["CMFN"])))
    assert not missing.exists() or not list(missing.glob("*.json"))


# ---------------------------------------------------------------------------
# 10. re-run behaviour (D-10c.7)
# ---------------------------------------------------------------------------


def test_skip_preserves_overwrite_replaces_and_a_changed_provenance_aborts(tmp_path: Path) -> None:
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE})
    out_dir = _out_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = out_dir / "aaaa1111.json"
    ocr_fn = _stub_ocr(_word_rows(["CMFN"]))

    def _plant(prov: dict[str, object]) -> bytes:
        body = {"image_id": "aaaa1111", "w": _SIZE[0], "h": _SIZE[1], "provenance": prov,
                "tokens": []}
        raw = (json.dumps(body, sort_keys=True, indent=4) + "\n").encode("utf-8")
        seed_path.write_bytes(raw)
        return raw

    planted = _plant(_provenance())
    _run(renders, manifest, out_dir, ocr_fn, on_existing="skip")
    assert seed_path.read_bytes() == planted  # skip did not touch the file

    _run(renders, manifest, out_dir, ocr_fn, on_existing="overwrite")
    assert seed_path.read_bytes() != planted
    assert [t["text"] for t in _seed(out_dir, "aaaa1111")["tokens"]] == ["CMFN"]

    stale = _plant(_provenance(tessdata_variant="fast"))
    with pytest.raises(seed_tesseract.ProvenanceMismatchError):
        _run(renders, manifest, out_dir, ocr_fn, on_existing="skip")
    assert seed_path.read_bytes() == stale  # aborted, never silently skipped past

    with pytest.raises(seed_tesseract.AmbiguousRerunError):
        seed_tesseract.resolve_on_existing(out_dir, None, stdin_isatty=False)


# ---------------------------------------------------------------------------
# 11. per-image failure recording (D-10c.9)
# ---------------------------------------------------------------------------


def test_a_failed_image_writes_no_file_does_not_abort_and_is_retried_on_skip(
    tmp_path: Path,
) -> None:
    small, big = (64, 48), (128, 96)
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": small, "bbbb2222": big})
    out_dir = _out_dir(tmp_path)
    rows = _word_rows(["CMFN"])
    doomed = (big[0] * seed_tesseract.UPSCALE_FACTOR, big[1] * seed_tesseract.UPSCALE_FACTOR)

    def flaky(img):
        if img.size == doomed:
            raise _Boom("this message must never reach the summary")
        return _fake_data(rows)

    summary = _run(renders, manifest, out_dir, flaky)
    assert (out_dir / "aaaa1111.json").exists()
    assert not (out_dir / "bbbb2222.json").exists()
    assert summary["n_images_seeded"] == 1 and summary["n_images_failed"] == 1
    assert summary["failures"] == [{"image_id": "bbbb2222", "exception_type": "_Boom"}]
    assert set(summary["failures"][0]) == {"image_id", "exception_type"}

    # No placeholder file means the next `skip` run retries it automatically.
    retry = _run(renders, manifest, out_dir, _stub_ocr(rows), on_existing="skip")
    assert (out_dir / "bbbb2222.json").exists()
    assert retry["failures"] == []


# ---------------------------------------------------------------------------
# 12. seed_summary.json (D-10c.8)
# ---------------------------------------------------------------------------


def test_summary_sits_beside_the_seed_dir_and_holds_no_token_text(tmp_path: Path) -> None:
    seeded = [*FAKE_TOKENS, "XZQV-777"]
    renders, manifest = _make_renders(tmp_path, {"aaaa1111": _SIZE})
    out_dir = _out_dir(tmp_path)
    summary = _run(renders, manifest, out_dir, _stub_ocr(_word_rows(seeded)))

    summary_path = out_dir.parent / seed_tesseract.SUMMARY_NAME
    assert summary_path.exists()
    assert not (out_dir / seed_tesseract.SUMMARY_NAME).exists()

    raw = summary_path.read_text(encoding="utf-8")
    on_disk = json.loads(raw)
    assert on_disk == summary
    assert set(on_disk["provenance"]) == set(seed_tesseract.PROVENANCE_FIELDS)
    assert on_disk["n_images_manifest"] == on_disk["n_images_seeded"] == 1
    assert on_disk["n_tokens_total"] == len(seeded)
    assert on_disk["per_image_box_counts"]["aaaa1111"] == len(seeded)
    for token in seeded:
        assert token not in raw


# ---------------------------------------------------------------------------
# 13. synthetic positive control (CLAUDE.md §8) — the only test needing the binary
# ---------------------------------------------------------------------------

_NO_TESSERACT = (
    "tesseract binary not installed; install with: sudo apt-get install -y tesseract-ocr "
    "(set SEED_REQUIRE_TESSERACT=1 to make this a hard failure instead of a skip)"
)
# SEED_REQUIRE_TESSERACT=1 turns the guard off entirely so CI fails loud rather than
# letting the one detector-validating test vanish into a green run.
requires_tesseract = pytest.mark.skipif(
    not seed_tesseract.tesseract_binary_available()
    and os.environ.get("SEED_REQUIRE_TESSERACT") != "1",
    reason=_NO_TESSERACT,
)


def _tessdata_dir() -> Path:
    candidates = [Path(os.environ["TESSDATA_PREFIX"])] if os.environ.get("TESSDATA_PREFIX") else []
    candidates += [
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tessdata"),
    ]
    for cand in candidates:
        if (cand / f"{seed_tesseract.LANG}.traineddata").exists():
            return cand
    raise AssertionError(f"no {seed_tesseract.LANG}.traineddata found; set TESSDATA_PREFIX")


@requires_tesseract
def test_seeder_finds_fake_burned_in_text_and_nothing_on_a_blank_frame(tmp_path: Path) -> None:
    tessdata = _tessdata_dir()
    ocr_fn = seed_tesseract.default_ocr_fn(tessdata)
    prov = _provenance(tessdata_dir=str(tessdata))

    text_png = tmp_path / "text.png"
    img, _ = make_synthetic_image(("CMFN-00421", "GRDN5678"))
    img.save(text_png, format="PNG")
    found = seed_tesseract.seed_image(
        text_png, image_id="aaaa1111", w=img.width, h=img.height, provenance=prov, ocr_fn=ocr_fn
    )

    blank_png = tmp_path / "blank.png"
    blank, _ = make_blank_image()
    blank.save(blank_png, format="PNG")
    empty = seed_tesseract.seed_image(
        blank_png, image_id="bbbb2222", w=blank.width, h=blank.height,
        provenance=prov, ocr_fn=ocr_fn,
    )

    # The positive control is what makes the negative control mean anything.
    assert len(found["tokens"]) >= 1
    assert len(empty["tokens"]) <= 1
    assert len(empty["tokens"]) < len(found["tokens"])


# ---------------------------------------------------------------------------
# 14. the pinned configuration is FROZEN (D-10c.4 / CLAUDE.md rule #9)
# ---------------------------------------------------------------------------


def test_the_pinned_tesseract_configuration_is_frozen() -> None:
    """A deliberate change-detector: the literals live HERE, not read back from the module.

    Every value below is pinned by D-10c.4 (and D-10c.2b) and is recorded in the provenance
    block of every seed file. Changing any one of them changes what Tesseract reads, so it
    INVALIDATES EVERY SEED ALREADY WRITTEN — the same class of invalidation as a `gt.csv`
    edit or an engine bump (CLAUDE.md rule #9). Dropping `-c load_system_dawg=0`, for
    instance, switches dictionary correction back on and pulls ID-shaped strings toward
    English words; every other test in this file would still pass.

    So if this test fails: the correct response is to RE-RUN THE SEED under the new
    configuration (and re-review it), NOT to update the expected values here to match.
    Update these literals only in the same change that re-seeds, and say so loudly.
    """
    assert seed_tesseract.TESSERACT_CONFIG == (
        "--oem 1 --psm 11 -c load_system_dawg=0 -c load_freq_dawg=0 "
        "-c thresholding_method=0 -c invert_threshold=0.7 --dpi 300"
    )
    # Named individually so a failure points at the knob that moved, not just "a string".
    for flag in (
        "--oem 1",
        "--psm 11",
        "-c load_system_dawg=0",
        "-c load_freq_dawg=0",
        "-c thresholding_method=0",
        "-c invert_threshold=0.7",
        "--dpi 300",
    ):
        assert flag in seed_tesseract.TESSERACT_CONFIG, f"{flag} was removed: re-seed, don't re-pin"

    assert seed_tesseract.LANG == "eng"
    assert seed_tesseract.UPSCALE_FACTOR == 2
    assert seed_tesseract.UPSCALE_FILTER_NAME == "PIL.Image.Resampling.LANCZOS"
    assert seed_tesseract.TIMEOUT_S == 60
    assert seed_tesseract.WORD_LEVEL == 5
    assert seed_tesseract.CONF_FALLBACK == -1.0
    assert seed_tesseract.ROW_FILTER == "level == 5 AND text.strip() != ''"
    assert seed_tesseract.TESSERACT_ENV == {
        "LC_ALL": "C",
        "LC_NUMERIC": "C",
        "OMP_THREAD_LIMIT": "1",
    }
    assert seed_tesseract.TESSDATA_VARIANTS == ("best", "fast", "standard")
    assert seed_tesseract.PROVENANCE_FIELDS == (
        "tesseract_cmd_config",
        "lang",
        "upscale_factor",
        "upscale_filter",
        "coord_space",
        "row_filter",
        "conf_parse",
        "env",
        "timeout_s",
        "tesseract_version",
        "leptonica_version",
        "pytesseract_version",
        "pillow_version",
        "tessdata_dir",
        "tessdata_variant",
        "traineddata_sha256",
    )


# ---------------------------------------------------------------------------
# 15. main() / _parse_args — the PHI-critical error path (CLAUDE.md §0)
# ---------------------------------------------------------------------------

_MAIN_ARGS = (
    "--renders-dir",
    "--manifest",
    "--out-dir",
    "--tessdata-dir",
    "--tessdata-variant",
)


def _argv(tmp_path: Path, **overrides: str) -> list[str]:
    """A complete, syntactically valid CLI invocation over synthetic paths."""
    values = {
        "--renders-dir": str(tmp_path / "renders"),
        "--manifest": str(tmp_path / "renders" / render.MANIFEST_NAME),
        "--out-dir": str(_out_dir(tmp_path)),
        "--tessdata-dir": str(tmp_path / "tessdata"),
        "--tessdata-variant": "best",
        "--on-existing": "overwrite",
    }
    values.update(overrides)
    return [part for flag, value in values.items() for part in (flag, value)]


def test_main_fails_loud_on_a_bad_invocation_without_leaking_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonexistent --manifest must exit non-zero with a PHI-free one-liner, not a trace.

    The path itself is treated as untrusted text: it is planted with a marker so the test
    proves the message does not echo an operator-supplied filename back to the terminal.
    """
    marker = "marker-must-not-be-echoed"
    rc = seed_tesseract.main(_argv(tmp_path, **{"--manifest": str(tmp_path / f"{marker}.csv")}))
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert rc != 0
    assert "Traceback" not in output
    assert marker not in output
    assert output.startswith("FAILED: ")
    # Whichever gate fires first (no binary here, missing manifest where one is installed),
    # the whole report is one PHI-free FAILED line plus the static setup hint.
    assert output.count("FAILED: ") == 1


def test_main_suppresses_the_message_of_an_unexpected_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generic handler must print the exception TYPE only — never `{exc}`, never a trace.

    A crash on real data can carry OCR text or a filename in its message, so letting
    `{exc}` back into that handler is a PHI egress (CLAUDE.md §0). The planted message is
    shaped like a fake token so the assertion is unambiguous.
    """
    planted = "CMFN-00421-this-message-must-never-be-printed"

    def _explode() -> bool:
        raise _Boom(planted)

    monkeypatch.setattr(seed_tesseract, "tesseract_binary_available", _explode)

    rc = seed_tesseract.main(_argv(tmp_path))
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert rc != 0
    assert planted not in output
    assert "CMFN" not in output
    assert "Traceback" not in output
    assert "_Boom" in output  # the type IS reported, so the failure is still diagnosable


@pytest.mark.parametrize("omitted", _MAIN_ARGS)
def test_main_requires_every_flag_instead_of_defaulting(
    tmp_path: Path, omitted: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-10c.6/D-10c.2b: no silent defaults — a missing flag exits non-zero via argparse."""
    argv = _argv(tmp_path)
    index = argv.index(omitted)
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as excinfo:
        seed_tesseract.main(argv)
    assert excinfo.value.code not in (0, None)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err


# ---------------------------------------------------------------------------
# 16. tesseract_env() — the determinism env, applied and then restored
# ---------------------------------------------------------------------------


def _env_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in seed_tesseract.TESSERACT_ENV}


def test_tesseract_env_applies_the_pinned_env_and_restores_it_even_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One key previously SET to something else, two previously UNSET: both must come back.
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.delenv("LC_NUMERIC", raising=False)
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    before = _env_snapshot()

    with seed_tesseract.tesseract_env():
        assert _env_snapshot() == dict(seed_tesseract.TESSERACT_ENV)
    assert _env_snapshot() == before
    assert "LC_NUMERIC" not in os.environ  # previously unset ⇒ unset again, not ""
    assert "OMP_THREAD_LIMIT" not in os.environ

    with pytest.raises(_Boom), seed_tesseract.tesseract_env():
        assert os.environ["OMP_THREAD_LIMIT"] == "1"
        raise _Boom("a failing image must not leave the process env rewritten")
    assert _env_snapshot() == before
    assert "LC_NUMERIC" not in os.environ
    assert "OMP_THREAD_LIMIT" not in os.environ


# ---------------------------------------------------------------------------
# 17. build_provenance() rejects an unverifiable tessdata claim (D-10c.2b)
# ---------------------------------------------------------------------------


def test_build_provenance_rejects_an_unknown_variant_and_a_tessdata_dir_without_traineddata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both failures are arranged to land BEFORE the tesseract binary is invoked.

    The variant check is the first statement in `build_provenance`, so it needs nothing.
    The missing-traineddata check is reached before the version probe today, but the probe
    is stubbed here so the test does not silently depend on that ordering (or on a binary
    that is not installed in this environment).
    """
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()

    with pytest.raises(seed_tesseract.PreflightError):
        seed_tesseract.build_provenance(tessdata_dir=tessdata, tessdata_variant="bset")

    monkeypatch.setattr(
        seed_tesseract,
        "tesseract_version_block",
        lambda: {
            "tesseract_version": "tesseract 0.0.0-synthetic",
            "leptonica_version": "leptonica-0.0.0-synthetic",
        },
    )
    # An empty dir: the variant is a human CLAIM, and the hash that would pin it is absent.
    with pytest.raises(seed_tesseract.PreflightError):
        seed_tesseract.build_provenance(tessdata_dir=tessdata, tessdata_variant="best")
