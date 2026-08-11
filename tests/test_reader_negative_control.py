"""Phase 13d — tests for the reader hallucination-floor procedure. 100% synthetic.

The script itself is PHI-touching (it opens real renders); everything asserted here is a
blank in-memory image from `tests/synthetic.py` and a scripted reader that returns canned
strings. Deliberately free of any model dependency, so these run in the shared benchmark venv
too — the control-box geometry and the counting rule are the parts that must be right no
matter which reader is plugged in.
"""

from __future__ import annotations

import csv

import pytest

from experiments.reader_negative_control import (
    BOX_ASPECT,
    BOX_MIN_HEIGHT,
    POSITIONS,
    control_boxes_for,
    guard_presence_matches_gt,
    read_blank_ids,
    read_gt_geometry,
)
from experiments.sweep_stock_vs_tuned import SweepError
from harness.reading import read_image
from tests import synthetic
from tests.test_reading import ScriptedReader, _image_ref, _saved

# --- the control-box grid ------------------------------------------------------------------


def test_boxes_are_the_requested_count_and_a_prefix_of_the_grid():
    """`--boxes-per-image N` must be a superset relationship, not a different sample.

    Ordered positions mean a 9-box run contains the 4-box run's boxes, so a follow-up at a
    higher N extends the earlier measurement instead of replacing it.
    """
    four = control_boxes_for(800, 600, 4)
    nine = control_boxes_for(800, 600, 9)
    assert len(four) == 4 and len(nine) == len(POSITIONS)
    assert nine[:4] == four


def test_boxes_stay_inside_the_frame():
    """A box off the edge still yields a crop (`crop_for_reading` clamps), but it would be a
    smaller, differently-shaped crop than the spec claims — so the spec places them properly."""
    w, h = 640, 480
    for x0, y0, x1, y1 in control_boxes_for(w, h, len(POSITIONS)):
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h


def test_box_shape_is_token_shaped_and_respects_the_minimum_height():
    """A VLM handed a square of noise is being asked a different question than the arm asks."""
    x0, y0, x1, y1 = control_boxes_for(1024, 1024, 1)[0]
    assert (y1 - y0) == pytest.approx(1024 * 0.035)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(BOX_ASPECT)

    tiny = control_boxes_for(200, 100, 1)[0]
    assert (tiny[3] - tiny[1]) == BOX_MIN_HEIGHT  # 100 * 0.035 = 3.5px would be meaningless


def test_the_grid_is_deterministic():
    """Two runs of the floor must measure the same boxes, or they are not the same floor."""
    assert control_boxes_for(700, 500, 6) == control_boxes_for(700, 500, 6)


# --- the control set -----------------------------------------------------------------------


def _presence(tmp_path, rows):
    path = tmp_path / "text_presence.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("image_id", "has_text"))
        writer.writerows(rows)
    return path


def test_anything_that_is_not_an_explicit_zero_stays_out_of_the_control_set(tmp_path):
    """Unknown means "not confirmed blank", and only a confirmed-blank frame is a control.

    Reading a blank-ish value as blank would put an unreviewed frame into the floor, where a
    correct read of real text becomes a fake invention — the number nobody can debug later.
    """
    path = _presence(tmp_path, [("aaa", "0"), ("bbb", ""), ("ccc", "false"), ("ddd", "1")])
    assert read_blank_ids(path) == {"aaa"}


def test_blank_ids_are_read_from_the_presence_file(tmp_path):
    path = _presence(tmp_path, [("aaa", "0"), ("bbb", "1"), ("ccc", "0")])
    assert read_blank_ids(path) == {"aaa", "ccc"}


def test_a_file_that_is_not_a_presence_csv_is_refused(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("image_id,token_text\naaa,SOMETHING\n", encoding="utf-8")
    with pytest.raises(SweepError, match="text-presence"):
        read_blank_ids(path)


# --- the gt.csv cross-check and the box scale ------------------------------------------------


def _fake_gt(tmp_path, rows):
    """A gt.csv-shaped file with SYNTHETIC token text — never a real one."""
    path = tmp_path / "fake_gt.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ("image_id", "series_uid", "modality", "vendor", "stratum", "frame_idx",
             "token_text", "x0", "y0", "x1", "y1", "label")
        )
        for image_id, stratum, y0, y1 in rows:
            writer.writerow(
                (image_id, "9.9.9", "CT", "FakeVendorA", stratum, 0, "CMFN", 10, y0, 90, y1, "KEEP")
            )
    return path


def test_a_frame_called_blank_that_has_tokens_stops_the_run(tmp_path):
    """The guard `read_image` cannot apply here, because this script passes no ground truth.

    Two of the 66 ct_axial controls really did turn out to carry text. Reading that text
    correctly and counting it as invention would fabricate the one number the whole procedure
    exists to produce.
    """
    with pytest.raises(SweepError, match="HAVE tokens"):
        guard_presence_matches_gt(
            {"aaa", "bbb"}, {"bbb", "ccc"}, tmp_path / "presence.csv", tmp_path / "gt.csv"
        )


def test_a_consistent_blank_set_passes_the_cross_check(tmp_path):
    guard_presence_matches_gt(
        {"aaa"}, {"bbb", "ccc"}, tmp_path / "presence.csv", tmp_path / "gt.csv"
    )


def test_gt_geometry_returns_ids_and_per_stratum_median_box_heights(tmp_path):
    path = _fake_gt(
        tmp_path,
        [("i1", "ct_axial", 0, 20), ("i2", "ct_axial", 0, 30), ("i3", "mg_2d", 0, 100)],
    )
    ids, by_stratum, overall = read_gt_geometry(path)
    assert ids == {"i1", "i2", "i3"}
    assert by_stratum == {"ct_axial": 25.0, "mg_2d": 100.0}
    assert overall == 30.0


def test_gt_geometry_refuses_a_file_without_the_box_columns(tmp_path):
    path = tmp_path / "not_gt.csv"
    path.write_text("image_id,has_text\naaa,0\n", encoding="utf-8")
    with pytest.raises(SweepError, match="not a gt.csv"):
        read_gt_geometry(path)


def test_a_measured_box_height_overrides_the_fraction_of_frame_fallback():
    """The floor must be measured on crops the same size as the scored set's, or it bounds
    a different input: 3.5% of a big mammo frame is ~90px, which `crop_for_reading`
    DOWNSCALES to 48, while every real token crop is an upscale."""
    measured = control_boxes_for(2560, 2560, 1, 22.0)[0]
    assert (measured[3] - measured[1]) == pytest.approx(22.0)

    fallback = control_boxes_for(2560, 2560, 1)[0]
    assert (fallback[3] - fallback[1]) == pytest.approx(2560 * 0.035)


def test_a_measured_height_below_the_floor_is_still_clamped():
    x0, y0, x1, y1 = control_boxes_for(800, 600, 1, 2.0)[0]
    assert (y1 - y0) == BOX_MIN_HEIGHT


# --- the counting rule -----------------------------------------------------------------------


def _floor(tmp_path, reply, n_boxes=4):
    """One blank frame, `n_boxes` control boxes, a reader that always answers `reply`."""
    image, _ = synthetic.make_blank_image((800, 600))
    path = _saved(tmp_path, image, "blank.png")
    return read_image(
        str(path),
        [],
        ScriptedReader([reply] * n_boxes),
        allowlist=set(),
        image_ref=_image_ref("synth-blank", path, image),
        control_boxes=control_boxes_for(800, 600, n_boxes),
    )


def test_every_non_empty_read_on_a_blank_frame_is_counted_as_invention(tmp_path):
    row = _floor(tmp_path, "ACC-0001")
    assert row["negative_control"] is True
    assert row["added_count"] == 4
    assert row["negative_control_floor_count"] == 4


def test_an_abstaining_reader_leaves_the_floor_at_zero(tmp_path):
    row = _floor(tmp_path, "")
    assert row["negative_control"] is True
    assert row["added_count"] == 0


def test_control_boxes_on_a_text_bearing_frame_are_refused(tmp_path):
    """The guard that makes this procedure safe: nobody has confirmed a text-bearing frame is
    empty anywhere, so a correct read there would be miscounted as invention."""
    image, gt = synthetic.make_synthetic_image(("CMFN",))
    path = _saved(tmp_path, image)
    with pytest.raises(ValueError, match="control box"):
        read_image(
            str(path),
            gt,
            ScriptedReader(["CMFN"]),
            allowlist=set(),
            image_ref=_image_ref("synth-000", path, image),
            control_boxes=control_boxes_for(image.width, image.height, 2),
        )
