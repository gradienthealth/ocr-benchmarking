"""Phase 10a tests for ground_truth/gt_schema.py — SYNTHETIC fake tokens only.

Every token, UID, and image id below is fabricated ("CMFN", "GRDN1234", "ACC-0001",
"9.9.9.synthetic.N"); no real patient IDs, accession numbers, or PHI appear here, and no
real path is ever read or written — every CSV is written under pytest's `tmp_path`. These
tests pin the FROZEN `gt.csv` column spec: that it stays derivable from `GTToken`, that the
validator collects rather than raises, that the loader round-trips `token_text` byte for
byte, and that no report can carry a field value back to the caller.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
from pathlib import Path

import pytest

from ground_truth import gt_schema
from ground_truth.gt_schema import GTValidationError, load_gt, validate_gt
from harness import metrics
from harness.contract import GTToken
from tests import synthetic

SIZE = (640, 480)

# One valid row, as a column -> value mapping. Individual tests override single cells.
_BASE: dict[str, str] = {
    "image_id": "synth-000",
    "series_uid": "9.9.9.synthetic.0",
    "modality": "CT",
    "vendor": "FakeVendorA",
    "stratum": "synth_ct_axial",
    "frame_idx": "0",
    "token_text": "CMFN",
    "x0": "24.0",
    "y0": "24.0",
    "x1": "88.0",
    "y1": "44.0",
    "label": "PHI",
}
_SIZES: dict[str, tuple[int, int]] = {"synth-000": SIZE}


def _row(**overrides: str) -> list[str]:
    cells = {**_BASE, **overrides}
    return [cells[c] for c in gt_schema.COLUMNS]


def _write(
    tmp_path: Path, rows: list[list[str]], header: list[str] | None = None
) -> Path:
    path = tmp_path / "gt.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(gt_schema.COLUMNS) if header is None else header)
        writer.writerows(rows)
    return path


def _codes(report: gt_schema.GTReport) -> set[str]:
    return {e.code for e in report.errors}


def _token_row(token: GTToken) -> list[str]:
    """Serialize a GTToken exactly as build_gt.py (10e) will have to."""
    return [
        token.image_id,
        token.series_uid,
        token.modality,
        token.vendor,
        token.stratum,
        str(token.frame_idx),
        token.token_text,
        *(repr(float(v)) for v in token.bbox),
        token.label,
    ]


# --- the frozen spec itself -------------------------------------------------


def test_columns_are_derivable_from_gttoken_fields() -> None:
    """Anti-drift: adding/renaming/reordering a GTToken field must fail HERE.

    The CSV spec is a hand-written literal on purpose (a frozen artifact's shape is
    explicit, not computed), so this is the test that stops it going stale.
    """
    expected: list[str] = []
    for field in dataclasses.fields(GTToken):
        if field.name == "bbox":
            expected.extend(gt_schema.BBOX_COLUMNS)
        else:
            expected.append(field.name)
    assert gt_schema.COLUMNS == tuple(expected)


def test_bbox_columns_are_corner_order() -> None:
    assert gt_schema.BBOX_COLUMNS == ("x0", "y0", "x1", "y1")


def test_allowed_labels_pinned_to_metrics_vocabulary() -> None:
    """Binary and case-sensitive. There is no OTHER."""
    assert gt_schema.ALLOWED_LABELS == {metrics.PHI, metrics.KEEP} == {"PHI", "KEEP"}


def test_module_never_prints_and_never_normalizes() -> None:
    """Structural guarantee, not a substring grep: no print() and no normalize() call.

    A `print()` would put PHI on stdout; `normalize()` here would normalize the ground
    truth twice (it is applied at match time, to both sides).
    """
    tree = ast.parse(Path(gt_schema.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "print" not in called
    assert "normalize" not in called
    assert "normalize" not in vars(gt_schema)


# --- round trip -------------------------------------------------------------


def test_round_trip_scene_gt(tmp_path: Path) -> None:
    """Write the whole synthetic scene's GT, validate clean, load back identical objects."""
    scene = synthetic.make_scene(seed=0)
    tokens: list[GTToken] = []
    for i, image in enumerate(scene.images):
        for j, token in enumerate(image.gt):
            # Mix both labels so the round trip covers the full vocabulary.
            label = metrics.KEEP if (i + j) % 2 else metrics.PHI
            tokens.append(dataclasses.replace(token, label=label))
    assert tokens, "scene must contribute GT rows"

    sizes = {image.image_id: SIZE for image in scene.images}
    path = _write(tmp_path, [_token_row(t) for t in tokens])

    report = validate_gt(path, sizes)
    assert report.ok, report.summary()
    assert report.n_rows == len(tokens)
    assert report.n_images == len({t.image_id for t in tokens})
    assert "PASS" in report.summary()

    loaded = load_gt(path, sizes)
    expected: dict[str, list[GTToken]] = {}
    for token in tokens:
        expected.setdefault(token.image_id, []).append(token)
    assert loaded == expected


def test_token_text_is_byte_identical(tmp_path: Path) -> None:
    """No strip, no case fold, no NFC — the loader returns exactly what the CSV holds."""
    raw = ("  CMFN  ", "acc-0001", "GRDN1234,CMFN", 'ACC "0099"', "CMFN-0042")
    rows = [_row(token_text=text) for text in raw]
    loaded = load_gt(_write(tmp_path, rows), _SIZES)
    assert [t.token_text for t in loaded["synth-000"]] == list(raw)


def test_blank_control_header_only_file_is_valid(tmp_path: Path) -> None:
    """A confirmed-blank frame contributes zero rows, so a header-only gt.csv is valid."""
    path = _write(tmp_path, [])
    report = validate_gt(path, _SIZES)
    assert report.ok, report.summary()
    assert (report.n_rows, report.n_images) == (0, 0)
    assert load_gt(path, _SIZES) == {}


def test_blank_control_image_is_absent_not_empty(tmp_path: Path) -> None:
    """The dict has no key for an image with no rows — callers must use .get(id, [])."""
    loaded = load_gt(_write(tmp_path, [_row()]), _SIZES)
    assert "synth-999" not in loaded
    assert loaded.get("synth-999", []) == []


# --- file-level failures ----------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        validate_gt(missing, _SIZES)
    with pytest.raises(FileNotFoundError):
        load_gt(missing, _SIZES)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    path.write_text("", encoding="utf-8")
    assert _codes(validate_gt(path, _SIZES)) == {gt_schema.FILE_EMPTY}


def test_bom_is_reported_but_does_not_cascade(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    path.write_text("\ufeff" + ",".join(gt_schema.COLUMNS) + "\r\n", encoding="utf-8")
    assert _codes(validate_gt(path, _SIZES)) == {gt_schema.HEADER_BOM}


# --- header failures --------------------------------------------------------


def test_missing_column(tmp_path: Path) -> None:
    header = [c for c in gt_schema.COLUMNS if c != "label"]
    path = _write(tmp_path, [_row()[:-1]], header=header)
    report = validate_gt(path, _SIZES)
    assert _codes(report) == {gt_schema.HEADER_MISSING_COLUMN}
    assert [e.column for e in report.errors] == ["label"]


def test_extra_column(tmp_path: Path) -> None:
    header = [*gt_schema.COLUMNS, "notes"]
    report = validate_gt(_write(tmp_path, [[*_row(), "hi"]], header=header), _SIZES)
    assert _codes(report) == {gt_schema.HEADER_EXTRA_COLUMN}


def test_duplicate_column(tmp_path: Path) -> None:
    header = [c if c != "x1" else "x0" for c in gt_schema.COLUMNS]
    report = validate_gt(_write(tmp_path, [_row()], header=header), _SIZES)
    assert gt_schema.HEADER_DUPLICATE_COLUMN in _codes(report)
    assert gt_schema.HEADER_MISSING_COLUMN in _codes(report)


def test_reordered_header_is_an_error_but_rows_still_validate(tmp_path: Path) -> None:
    """Same columns, wrong order: rows are read BY NAME, so row errors still surface."""
    header = list(gt_schema.COLUMNS)
    header[7], header[8] = header[8], header[7]  # swap x0 / y0
    cells = {**_BASE, "label": "OTHER"}
    row = [cells[c] for c in header]
    report = validate_gt(_write(tmp_path, [row], header=header), _SIZES)
    assert _codes(report) == {gt_schema.HEADER_ORDER, gt_schema.LABEL_INVALID}


# --- row failures -----------------------------------------------------------


def test_wrong_field_count(tmp_path: Path) -> None:
    report = validate_gt(_write(tmp_path, [_row()[:-1]]), _SIZES)
    assert _codes(report) == {gt_schema.ROW_FIELD_COUNT}


def test_blank_line_in_data_section(tmp_path: Path) -> None:
    report = validate_gt(_write(tmp_path, [_row(), []]), _SIZES)
    assert _codes(report) == {gt_schema.ROW_EMPTY}


@pytest.mark.parametrize("label", ["OTHER", "phi", "keep", "", "KEEP ", "Phi"])
def test_rejected_labels(tmp_path: Path, label: str) -> None:
    report = validate_gt(_write(tmp_path, [_row(label=label)]), _SIZES)
    assert _codes(report) == {gt_schema.LABEL_INVALID}


@pytest.mark.parametrize("column", ["image_id", "series_uid", "token_text"])
def test_required_fields_cannot_be_empty(tmp_path: Path, column: str) -> None:
    report = validate_gt(_write(tmp_path, [_row(**{column: ""})]), {"": SIZE, **_SIZES})
    assert _codes(report) == {gt_schema.FIELD_EMPTY}
    assert [e.column for e in report.errors] == [column]


@pytest.mark.parametrize("column", ["modality", "vendor", "stratum"])
def test_manifest_passthrough_fields_may_be_empty(tmp_path: Path, column: str) -> None:
    """run_harness stamps these authoritatively (rule #7), so a blank here is not a violation."""
    report = validate_gt(_write(tmp_path, [_row(**{column: ""})]), _SIZES)
    assert report.ok, report.summary()


def test_x_order(tmp_path: Path) -> None:
    for x1 in ("24.0", "10.0"):  # equal and inverted both violate x0 < x1
        report = validate_gt(_write(tmp_path, [_row(x1=x1)]), _SIZES)
        assert _codes(report) == {gt_schema.BOX_X_ORDER}


def test_y_order(tmp_path: Path) -> None:
    for y1 in ("24.0", "10.0"):
        report = validate_gt(_write(tmp_path, [_row(y1=y1)]), _SIZES)
        assert _codes(report) == {gt_schema.BOX_Y_ORDER}


@pytest.mark.parametrize("column", ["x0", "y0", "x1", "y1"])
def test_negative_coords(tmp_path: Path, column: str) -> None:
    report = validate_gt(_write(tmp_path, [_row(**{column: "-1.0"})]), _SIZES)
    assert _codes(report) == {gt_schema.COORD_NEGATIVE}


@pytest.mark.parametrize(("column", "value"), [("x1", "700.0"), ("y1", "500.0")])
def test_out_of_bounds(tmp_path: Path, column: str, value: str) -> None:
    report = validate_gt(_write(tmp_path, [_row(**{column: value})]), _SIZES)
    assert _codes(report) == {gt_schema.BOX_OUT_OF_BOUNDS}
    assert [e.column for e in report.errors] == [column]


def test_box_may_touch_the_far_edge(tmp_path: Path) -> None:
    report = validate_gt(_write(tmp_path, [_row(x1="640.0", y1="480.0")]), _SIZES)
    assert report.ok, report.summary()


def test_unknown_image_size_is_reported_never_skipped(tmp_path: Path) -> None:
    report = validate_gt(_write(tmp_path, [_row()]), {})
    assert _codes(report) == {gt_schema.IMAGE_SIZE_UNKNOWN}


@pytest.mark.parametrize("value", ["1.5", "abc", "", "03", " 3", "+3"])
def test_frame_idx_must_be_a_canonical_integer(tmp_path: Path, value: str) -> None:
    report = validate_gt(_write(tmp_path, [_row(frame_idx=value)]), _SIZES)
    assert _codes(report) == {gt_schema.FRAME_IDX_NOT_INT}


def test_frame_idx_negative(tmp_path: Path) -> None:
    report = validate_gt(_write(tmp_path, [_row(frame_idx="-1")]), _SIZES)
    assert _codes(report) == {gt_schema.FRAME_IDX_NEGATIVE}


@pytest.mark.parametrize("value", ["abc", "", "1,2"])
def test_coord_not_numeric(tmp_path: Path, value: str) -> None:
    report = validate_gt(_write(tmp_path, [_row(x0=value)]), _SIZES)
    assert _codes(report) == {gt_schema.COORD_NOT_NUMERIC}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_coord_not_finite(tmp_path: Path, value: str) -> None:
    """NaN/inf parse fine as floats and would silently poison IoU."""
    report = validate_gt(_write(tmp_path, [_row(x0=value)]), _SIZES)
    assert _codes(report) == {gt_schema.COORD_NOT_FINITE}


# --- collect-all + raising loader -------------------------------------------


def test_validator_collects_every_violation(tmp_path: Path) -> None:
    rows = [
        _row(),
        _row(label="OTHER"),
        _row(x1="10.0"),
        _row(frame_idx="1.5"),
    ]
    report = validate_gt(_write(tmp_path, rows), _SIZES)
    assert not report.ok
    assert report.counts_by_code() == {
        gt_schema.BOX_X_ORDER: 1,
        gt_schema.FRAME_IDX_NOT_INT: 1,
        gt_schema.LABEL_INVALID: 1,
    }
    assert [e.row for e in report.errors] == [2, 3, 4]  # 1-based data rows, header excluded
    assert report.n_rows == 4


def test_load_raises_on_any_violation(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row(label="OTHER")])
    with pytest.raises(GTValidationError) as excinfo:
        load_gt(path, _SIZES)
    assert excinfo.value.report.ok is False
    assert str(excinfo.value) == excinfo.value.report.summary()
    assert gt_schema.LABEL_INVALID in str(excinfo.value)


def test_error_detail_is_a_fixed_lookup() -> None:
    err = gt_schema.GTError(gt_schema.LABEL_INVALID, row=3, column="label")
    assert err.detail == gt_schema._CODE_DETAIL[gt_schema.LABEL_INVALID]
    assert not any(f.name == "detail" for f in dataclasses.fields(err))


# --- the PHI guarantee ------------------------------------------------------


def test_report_never_contains_a_field_value(tmp_path: Path) -> None:
    """A human pastes this report back to Claude, so it must carry no cell contents.

    Every fixture value here is fake, but the assertion is the real safeguard: if any
    value ever reaches a report, a real token_text would too.
    """
    scene = synthetic.make_scene(seed=0)
    tokens = [t for image in scene.images for t in image.gt]
    rows = [_token_row(t) for t in tokens]
    rows.append(_token_row(dataclasses.replace(tokens[0], label="OTHER")))
    rows.append(_token_row(dataclasses.replace(tokens[0], bbox=(9.0, 9.0, 1.0, 1.0))))
    rows.append(_token_row(dataclasses.replace(tokens[0], image_id="synth-unmapped")))

    sizes = {image.image_id: SIZE for image in scene.images}
    report = validate_gt(_write(tmp_path, rows), sizes)
    assert not report.ok

    secrets = (
        {t.token_text for t in tokens}
        | {t.series_uid for t in tokens}
        | {t.image_id for t in tokens}
        | {"synth-unmapped"}
    )
    haystacks = [report.summary(), repr(report.errors)]
    haystacks += [e.detail for e in report.errors]
    haystacks += [repr(e) for e in report.errors]
    with pytest.raises(GTValidationError) as excinfo:
        load_gt(_write(tmp_path, rows), sizes)
    haystacks.append(str(excinfo.value))

    for haystack in haystacks:
        for secret in secrets:
            assert secret not in haystack, f"report leaked a field value: {secret!r}"
