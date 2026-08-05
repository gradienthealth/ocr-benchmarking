"""Synthetic-only tests for `ground_truth/render.py` (Phase 10b). 🟢 PHI-FREE.

Every DICOM here is fabricated in memory (`pydicom.Dataset` + `set_pixel_data` + a fake
`np.arange` pixel block) and written to `tmp_path`. Zero real `.dcm`, zero GCS, zero PHI —
UIDs are `9.9.9.synthetic.*` in the obviously-fake style of `tests/synthetic.py`.

Most tests inject `_identity_voi` as the grayscale-to-uint8 step so that the mechanics
around D-10.1 (frame selection, MONOCHROME1 inversion, forced RGB, determinism, the id
gate) are exercised without every expected pixel having to be pushed through a window.
The real D-10.1 policy has its own section below and is tested directly.
"""

from __future__ import annotations

import csv
import hashlib
import struct
import tarfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import FileMetaDataset
from pydicom.pixels import convert_color_space, set_pixel_data
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from ground_truth import render

FAKE_UID = "9.9.9.synthetic.{}"
FAKE_SERIES = "9.9.9.synthetic.series.{}"

# The fake UIDs above are deliberately non-conformant (letters in a VR UI) so no test value
# could ever be mistaken for a real identifier. Silence pydicom's write-time validation
# warning about that rather than making the fixtures look like real UIDs.
pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE


# --- synthetic DICOM fixtures --------------------------------------------------------


def _fake_pixels(shape: tuple[int, ...], salt: int = 0) -> np.ndarray:
    """A deterministic, obviously-fake pixel block in 0..250.

    `salt` makes each instance of a series pixel-wise DISTINCT. Without it every instance
    looked identical, and a test could not tell which instance's pixels were written — a
    renderer that always picked instance 0 would pass everything.
    """
    flat = (np.arange(int(np.prod(shape)), dtype=np.uint32) + salt * 37) % 251
    return flat.astype(np.uint8).reshape(shape)


def _make_dcm(
    dest: Path,
    *,
    sop_uid: str,
    instance_number: int | None = 1,
    photometric: str = "MONOCHROME2",
    n_frames: int = 1,
    rows: int = 4,
    cols: int = 6,
    pixels: np.ndarray | None = None,
    drop_photometric: bool = False,
    salt: int = 0,
    tags: dict[str, object] | None = None,
) -> np.ndarray:
    """Write one synthetic instance to `dest`; return the pixel block it contains."""
    if pixels is None:
        if photometric in ("RGB", "YBR_FULL"):
            shape: tuple[int, ...] = (rows, cols, 3)
        elif n_frames > 1:
            shape = (n_frames, rows, cols)
        else:
            shape = (rows, cols)
        pixels = _fake_pixels(shape, salt)
    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.PatientID = "SYNTH-0000"
    ds.StudyInstanceUID = FAKE_UID.format("study")
    ds.SeriesInstanceUID = FAKE_UID.format("series")
    # generate_instance_uid=False: the default would mint a fresh SOPInstanceUID and
    # scramble every expected image_id.
    set_pixel_data(ds, pixels, photometric, 8, generate_instance_uid=False)
    ds.SOPInstanceUID = sop_uid
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    if drop_photometric:
        del ds.PhotometricInterpretation
    for name, value in (tags or {}).items():
        setattr(ds, name, value)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(dest, enforce_file_format=True)
    return pixels


def _identity_voi(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Test-only stand-in for the D-10.1 policy: a straight, safe uint8 cast.

    Used by tests that assert on pixel IDENTITY (which instance/frame was picked, the
    MONOCHROME1 inversion, channel expansion). Injecting it keeps those expectations
    readable; the real policy is tested in its own section.
    """
    return render.clip_round_uint8(arr)


def _inputs_csv(path: Path, rows: list[tuple[str, str, object]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["series_uid", "path", "frame_idx"])
        for series_uid, src, frame in rows:
            writer.writerow([series_uid, src, "" if frame is None else frame])
    return path


def _run(
    tmp_path: Path,
    rows: list[tuple[str, str, object]],
    *,
    out_name: str = "out",
    backmap_name: str = "backmap.csv",
    id_len: int = render.DEFAULT_ID_LEN,
    voi_fn=_identity_voi,
) -> tuple[render.RenderSummary, dict[str, tuple[int, int, int, str, int]], Path]:
    inputs = render.read_input_list(_inputs_csv(tmp_path / "inputs.csv", rows))
    out_dir = tmp_path / out_name
    summary = render.render_set(
        inputs,
        out_dir,
        backmap_path=tmp_path / backmap_name,
        id_len=id_len,
        voi_fn=voi_fn,
    )
    return summary, render.read_manifest(summary.manifest_path), out_dir


def _one_series_dir(
    tmp_path: Path, name: str, specs: list[dict]
) -> tuple[Path, dict[str, np.ndarray]]:
    """Write several instances into one directory, each with distinct pixels.

    Filenames deliberately run backwards through the alphabet so a renderer that trusted
    directory order instead of (InstanceNumber, SOPInstanceUID) would pick the wrong slice.
    Returns the directory and {sop_uid: pixels} so a test can assert WHICH instance landed
    in the PNG, not merely which integer was recorded.
    """
    series_dir = tmp_path / name
    pixels: dict[str, np.ndarray] = {}
    for pos, spec in enumerate(specs):
        pixels[spec["sop_uid"]] = _make_dcm(
            series_dir / f"{chr(ord('z') - pos)}.dcm", salt=pos + 1, **spec
        )
    return series_dir, pixels


def _png_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        assert img.mode == "RGB"
        return np.array(img)


def _png_chunks(path: Path) -> list[str]:
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    chunks: list[str] = []
    pos = 8
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        chunks.append(raw[pos + 4 : pos + 8].decode("ascii"))
        pos += 12 + length
    return chunks


# --- image_id (D-10.2) ---------------------------------------------------------------


def test_image_id_uses_the_exact_settled_preimage() -> None:
    # sha256("9.9.9.synthetic.1|3") — UTF-8, literal "|", frame_idx as a bare decimal int.
    full = "a0fdebdc54349a732f51dc561d80673ce309226deeed1d647df813cb44351a65"
    assert render.compute_image_id(FAKE_UID.format(1), 3, 64) == full
    assert render.compute_image_id(FAKE_UID.format(1), 3) == full[:8]
    assert render.compute_image_id(FAKE_UID.format(7), 0, 64) == (
        "cb76e284e23fa7c332d52f1190eca7d52ec661d92dc7af5274323487ff5b3f6c"
    )


def test_pipe_separator_prevents_boundary_collision() -> None:
    # Without the "|", ("UID1", 23) and ("UID12", 3) would share a preimage.
    assert render.compute_image_id("UID1", 23, 64) != render.compute_image_id("UID12", 3, 64)


def test_image_id_is_stable_across_calls() -> None:
    a = render.compute_image_id(FAKE_UID.format(4), 2)
    assert a == render.compute_image_id(FAKE_UID.format(4), 2)


def test_id_len_changes_every_id(tmp_path: Path) -> None:
    specs = [{"sop_uid": FAKE_UID.format(i), "instance_number": i} for i in range(3)]
    series, _ = _one_series_dir(tmp_path, "s", specs)
    rows = [(FAKE_SERIES.format(0), str(series), None)]
    _, m8, _ = _run(tmp_path, rows, out_name="o8", backmap_name="b8.csv")
    _, m12, _ = _run(tmp_path, rows, out_name="o12", backmap_name="b12.csv", id_len=12)
    assert {len(k) for k in m8} == {8}
    assert {len(k) for k in m12} == {12}
    assert set(m8) != set(m12)
    # The longer id is a prefix-extension of the shorter one for the SAME image: same
    # digest, different truncation. (Asserted against the id we know, not recomputed
    # from m12 — that would just restate its own premise.)
    expected_full = render.compute_image_id(FAKE_UID.format(1), 1, 64)
    assert set(m8) == {expected_full[:8]} and set(m12) == {expected_full[:12]}


# --- input list (D-10.8) -------------------------------------------------------------


def test_input_list_rejects_wrong_header(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("series_uid,path\n1,2\n", encoding="utf-8")
    with pytest.raises(render.InputListError, match="header"):
        render.read_input_list(p)


def test_input_list_rejects_duplicate_series(tmp_path: Path) -> None:
    p = _inputs_csv(tmp_path / "dupe.csv", [("s1", "a", 1), ("s1", "b", 2)])
    with pytest.raises(render.InputListError, match="duplicate series_uid"):
        render.read_input_list(p)


def test_input_list_rejects_non_integer_frame_idx(tmp_path: Path) -> None:
    p = _inputs_csv(tmp_path / "frac.csv", [("s1", "a", "2.5")])
    with pytest.raises(render.InputListError, match="plain integer"):
        render.read_input_list(p)


def test_input_list_is_sorted_and_empty_frame_means_no_request(tmp_path: Path) -> None:
    p = _inputs_csv(tmp_path / "ord.csv", [("s2", "b", 4), ("s1", "a", None)])
    rows = render.read_input_list(p)
    assert [r.series_uid for r in rows] == ["s1", "s2"]
    assert rows[0].requested_frame_idx is None
    assert rows[1].requested_frame_idx == 4


def test_input_list_errors_never_echo_row_contents(tmp_path: Path) -> None:
    rows = [("secret-uid", "secret/path", 1), ("secret-uid", "x", 1)]
    p = _inputs_csv(tmp_path / "dupe2.csv", rows)
    with pytest.raises(render.InputListError) as exc:
        render.read_input_list(p)
    assert "secret-uid" not in str(exc.value)
    assert "secret/path" not in str(exc.value)


# --- frame selection -----------------------------------------------------------------


def test_multiframe_instance_picks_middle_frame(tmp_path: Path) -> None:
    series = tmp_path / "mf"
    frames = _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("mf"), n_frames=5)
    summary, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), None)])
    assert summary.n_rendered == 1
    image_id = render.compute_image_id(FAKE_UID.format("mf"), 2)
    assert manifest[image_id][0] == 2  # 5 // 2, never frame 0
    assert manifest[image_id][4] == 1  # no index requested -> fallback
    # The PIXELS must be frame 2's, not frame 0's. Asserting only the recorded integer let
    # a "always decode frame 0" mutation pass the whole suite — i.e. every render would be
    # the banner/title screen while the manifest claimed the middle frame.
    got = _png_pixels(out_dir / f"{image_id}.png")[:, :, 0]
    np.testing.assert_array_equal(got, frames[2])
    assert not np.array_equal(got, frames[0])


def test_multiframe_instance_honours_in_range_request(tmp_path: Path) -> None:
    series = tmp_path / "mf"
    frames = _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("mf"), n_frames=5)
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 1)])
    image_id = render.compute_image_id(FAKE_UID.format("mf"), 1)
    assert manifest[image_id][0] == 1
    assert manifest[image_id][4] == 0
    got = _png_pixels(out_dir / f"{image_id}.png")[:, :, 0]
    np.testing.assert_array_equal(got, frames[1])


def test_multi_instance_series_picks_middle_instance(tmp_path: Path) -> None:
    specs = [{"sop_uid": FAKE_UID.format(i), "instance_number": i} for i in range(5)]
    series, pixels = _one_series_dir(tmp_path, "mi", specs)
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), None)])
    image_id = render.compute_image_id(FAKE_UID.format(2), 2)
    assert manifest[image_id][0] == 2
    got = _png_pixels(out_dir / f"{image_id}.png")[:, :, 0]
    np.testing.assert_array_equal(got, pixels[FAKE_UID.format(2)])
    assert not np.array_equal(got, pixels[FAKE_UID.format(0)])


def test_single_frame_report_is_the_explicit_exception(tmp_path: Path) -> None:
    series = tmp_path / "report"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("rep"))
    _, manifest, _ = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    image_id = render.compute_image_id(FAKE_UID.format("rep"), 0)
    assert manifest[image_id][0] == 0
    assert manifest[image_id][4] == 0  # 0 was in range for a 1-frame image: not a fallback


def test_out_of_range_request_falls_back_and_records_what_was_rendered(tmp_path: Path) -> None:
    specs = [{"sop_uid": FAKE_UID.format(i), "instance_number": i} for i in range(4)]
    series, pixels = _one_series_dir(tmp_path, "oor", specs)
    summary, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 99)])
    requested_id = render.compute_image_id(FAKE_UID.format(2), 99)
    rendered_id = render.compute_image_id(FAKE_UID.format(2), 2)
    assert rendered_id in manifest and requested_id not in manifest
    assert manifest[rendered_id][0] == 2  # what was rendered, not what was asked for
    assert manifest[rendered_id][4] == 1
    assert summary.n_fallback == 1
    got = _png_pixels(out_dir / f"{rendered_id}.png")[:, :, 0]
    np.testing.assert_array_equal(got, pixels[FAKE_UID.format(2)])


def test_fallback_count_matches_the_per_image_flags(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _make_dcm(a / "a.dcm", sop_uid=FAKE_UID.format("a"), n_frames=3)
    _make_dcm(b / "b.dcm", sop_uid=FAKE_UID.format("b"), n_frames=3)
    summary, manifest, _ = _run(
        tmp_path,
        [(FAKE_SERIES.format(0), str(a), None), (FAKE_SERIES.format(1), str(b), 0)],
    )
    assert summary.n_fallback == sum(row[4] for row in manifest.values()) == 1


def test_instance_order_pinned_by_instance_number_then_sop_uid(tmp_path: Path) -> None:
    # Two instances share InstanceNumber 1, so only the UID tie-break decides the order;
    # filenames are written in the opposite order on purpose.
    specs = [
        {"sop_uid": FAKE_UID.format("b"), "instance_number": 1},
        {"sop_uid": FAKE_UID.format("a"), "instance_number": 1},
        {"sop_uid": FAKE_UID.format("c"), "instance_number": 2},
    ]
    series, pixels = _one_series_dir(tmp_path, "tie", specs)
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), None)])
    # sorted: (1,"...a"), (1,"...b"), (2,"...c") -> index 1 is "...b"
    image_id = render.compute_image_id(FAKE_UID.format("b"), 1)
    assert image_id in manifest
    got = _png_pixels(out_dir / f"{image_id}.png")[:, :, 0]
    np.testing.assert_array_equal(got, pixels[FAKE_UID.format("b")])


def test_missing_instance_number_sorts_last(tmp_path: Path) -> None:
    specs = [
        {"sop_uid": FAKE_UID.format("no"), "instance_number": None},
        {"sop_uid": FAKE_UID.format("yes"), "instance_number": 5},
    ]
    series, pixels = _one_series_dir(tmp_path, "miss", specs)
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), None)])
    image_id = render.compute_image_id(FAKE_UID.format("no"), 1)
    assert image_id in manifest
    got = _png_pixels(out_dir / f"{image_id}.png")[:, :, 0]
    np.testing.assert_array_equal(got, pixels[FAKE_UID.format("no")])


def test_mixed_multi_instance_multi_frame_raises_and_is_counted(tmp_path: Path) -> None:
    series = tmp_path / "mixed"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("m0"), instance_number=1)
    _make_dcm(series / "b.dcm", sop_uid=FAKE_UID.format("m1"), instance_number=2, n_frames=3)
    summary, manifest, _ = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), None)])
    assert summary.n_rendered == 0
    assert manifest == {}
    assert summary.errors == (("AmbiguousFrameAxisError", 1),)


# --- pixels: settled mechanics -------------------------------------------------------


def test_monochrome2_is_left_alone_and_becomes_three_identical_channels(tmp_path: Path) -> None:
    series = tmp_path / "m2"
    pixels = _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("m2"))
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    image_id = next(iter(manifest))
    got = _png_pixels(out_dir / f"{image_id}.png")
    assert got.shape == (*pixels.shape, 3)
    np.testing.assert_array_equal(got[:, :, 0], pixels)
    np.testing.assert_array_equal(got[:, :, 0], got[:, :, 1])
    np.testing.assert_array_equal(got[:, :, 1], got[:, :, 2])


def test_monochrome1_is_inverted_before_channel_expansion(tmp_path: Path) -> None:
    series = tmp_path / "m1"
    pixels = _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("m1"), photometric="MONOCHROME1")
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    got = _png_pixels(out_dir / f"{next(iter(manifest))}.png")
    np.testing.assert_array_equal(got[:, :, 0], 255 - pixels)
    np.testing.assert_array_equal(got[:, :, 0], got[:, :, 2])


def test_rgb_passes_through(tmp_path: Path) -> None:
    series = tmp_path / "rgb"
    pixels = _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("rgb"), photometric="RGB")
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    np.testing.assert_array_equal(_png_pixels(out_dir / f"{next(iter(manifest))}.png"), pixels)


def test_ybr_is_converted_to_rgb(tmp_path: Path) -> None:
    series = tmp_path / "ybr"
    pixels = _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("ybr"), photometric="YBR_FULL")
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    got = _png_pixels(out_dir / f"{next(iter(manifest))}.png")
    expected = convert_color_space(pixels, "YBR_FULL", "RGB")
    np.testing.assert_array_equal(got, expected)
    assert not np.array_equal(got, pixels)  # the conversion really happened


def test_missing_photometric_interpretation_fails_loud(tmp_path: Path) -> None:
    series = tmp_path / "nopi"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("nopi"), drop_photometric=True)
    summary, manifest, _ = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    assert manifest == {}
    assert summary.errors == (("UnsupportedPixelFormatError", 1),)


def test_every_output_is_rgb_8bit(tmp_path: Path) -> None:
    series = tmp_path / "mode"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("mode"), photometric="MONOCHROME1")
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    png = out_dir / f"{next(iter(manifest))}.png"
    with Image.open(png) as img:
        assert img.mode == "RGB"
    raw = png.read_bytes()
    bit_depth, color_type = raw[24], raw[25]  # IHDR payload starts at byte 16
    assert (bit_depth, color_type) == (8, 2)  # 8-bit truecolour RGB


def test_clip_round_uint8_never_wraps_or_nans() -> None:
    got = render.clip_round_uint8(np.array([-5.0, 0.4, 0.6, 255.6, 300.0, np.nan, np.inf]))
    assert got.dtype == np.uint8
    assert got.tolist() == [0, 0, 1, 255, 255, 0, 255]


def test_a_constant_frame_does_not_blow_up(tmp_path: Path) -> None:
    series = tmp_path / "flat"
    flat = np.zeros((4, 6), dtype=np.uint8)
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("flat"), pixels=flat)
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    got = _png_pixels(out_dir / f"{next(iter(manifest))}.png")
    assert got.max() == 0 and got.shape == (4, 6, 3)


# --- D-10.1: the windowing policy (resolved 2026-08-06) --------------------------------
# Expected values below are hand-computed from PS3.3 C.11.2.1.2 with y_min=0, y_max=255,
# never by re-running the implementation's own arithmetic.


def _ds(**tags: object) -> pydicom.Dataset:
    ds = pydicom.Dataset()
    for name, value in tags.items():
        setattr(ds, name, value)
    return ds


def test_voi_lut_sequence_raises(tmp_path: Path) -> None:
    """D-10.1a: 1 image of 384 sampled. Raise rather than carry an untestable branch."""
    lut_item = pydicom.Dataset()
    lut_item.LUTDescriptor = [4, 0, 8]
    lut_item.LUTData = [0, 85, 170, 255]
    ds = _ds(VOILUTSequence=[lut_item])
    with pytest.raises(render.UnsupportedVOIError, match="VOI LUT Sequence"):
        render.voi_to_uint8(np.zeros((2, 2)), ds)


def test_empty_voi_lut_sequence_is_not_a_lut() -> None:
    """An empty sequence is a present-but-unusable tag, not a reason to refuse."""
    got = render.voi_to_uint8(np.array([[0, 10]], dtype=np.uint16), _ds(VOILUTSequence=[]))
    assert got.tolist() == [[0, 255]]  # fell through to the min/max fallback


@pytest.mark.parametrize("func", ["SIGMOID", "LINEAR_EXACT", "sigmoid"])
def test_non_linear_voi_lut_function_raises(func: str) -> None:
    """D-10.1a: SIGMOID appears once in the census; honouring it would map that one
    image with a different function than the other ~964."""
    ds = _ds(VOILUTFunction=func, WindowCenter=2.5, WindowWidth=5)
    with pytest.raises(render.UnsupportedVOIError, match="VOILUTFunction"):
        render.voi_to_uint8(np.zeros((2, 2)), ds)


def test_explicit_linear_voi_lut_function_is_accepted() -> None:
    ds = _ds(VOILUTFunction="LINEAR", WindowCenter=2.5, WindowWidth=5)
    got = render.voi_to_uint8(np.array([[0, 2, 5]]), ds)
    assert got.tolist() == [[0, 128, 255]]


def test_linear_window_matches_the_dicom_formula() -> None:
    """center=2.5, width=5 -> lo=0.0, hi=4.0. Hand-computed, not re-derived:

    x<=0 -> 0 | x=1 -> 0.25*255=63.75 -> 64 | x=2 -> 0.5*255=127.5 -> 128
    x=3 -> 0.75*255=191.25 -> 191 | x=4 -> 255 | x>4 -> 255
    """
    ds = _ds(WindowCenter=2.5, WindowWidth=5)
    got = render.voi_to_uint8(np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int16), ds)
    assert got.dtype == np.uint8
    assert got.tolist() == [[0, 64, 128, 191, 255, 255]]


def test_multivalued_window_takes_index_zero() -> None:
    """D-10.1b-i: 51/384 sampled are multi-valued. Index 0, globally, so no per-image
    record of 'which window fired' is needed and the manifest keeps its 6 columns."""
    scalar = render.voi_to_uint8(
        np.array([[0, 2, 5]]), _ds(WindowCenter=2.5, WindowWidth=5)
    )
    multi = render.voi_to_uint8(
        np.array([[0, 2, 5]]), _ds(WindowCenter=[2.5, 900], WindowWidth=[5, 1200])
    )
    assert multi.tolist() == scalar.tolist() == [[0, 128, 255]]


def test_modality_lut_is_applied_before_windowing() -> None:
    """D-10.1b-iii. Slope 2 / intercept -10 maps stored 5,6,7 -> 0,2,4, which the
    center=2.5/width=5 window then maps to 0,128,255. Windowing the STORED values with
    the same window would clip all three to 255 (all are > hi=4), so this asserts the
    order, not merely that rescale ran.
    """
    ds = _ds(WindowCenter=2.5, WindowWidth=5, RescaleSlope=2, RescaleIntercept=-10)
    got = render.voi_to_uint8(np.array([[5, 6, 7]], dtype=np.int16), ds)
    assert got.tolist() == [[0, 128, 255]]


def test_window_width_one_is_a_hard_threshold() -> None:
    """width=1 makes lo==hi, so no pixel is 'inside' and the (width-1) division is never
    evaluated on a real element. Every output is 0 or 255, and nothing is NaN."""
    got = render.voi_to_uint8(np.array([[0, 1, 2, 3, 4]]), _ds(WindowCenter=2.5, WindowWidth=1))
    assert got.tolist() == [[0, 0, 0, 255, 255]]  # lo = hi = 2.0; x<=2 -> 0, x>2 -> 255


def test_window_width_below_one_raises() -> None:
    with pytest.raises(render.WindowTagError, match="WindowWidth < 1"):
        render.voi_to_uint8(np.zeros((2, 2)), _ds(WindowCenter=2.5, WindowWidth=0))


def test_half_a_window_pair_raises() -> None:
    """Absent is the legitimate fallback; half a pair is ambiguous and fails loud."""
    with pytest.raises(render.WindowTagError, match="only one of"):
        render.voi_to_uint8(np.zeros((2, 2)), _ds(WindowCenter=2.5))
    with pytest.raises(render.WindowTagError, match="only one of"):
        render.voi_to_uint8(np.zeros((2, 2)), _ds(WindowWidth=5))


def test_empty_window_element_raises() -> None:
    """A present-but-valueless pair is malformed, not the absent case."""
    with pytest.raises(render.WindowTagError, match="not numeric"):
        render.voi_to_uint8(np.zeros((2, 2)), _ds(WindowCenter=[], WindowWidth=[]))


def test_no_window_tags_falls_back_to_minmax() -> None:
    """D-10.1c. The endpoints land exactly on 0 and 255 and the interior is linear."""
    got = render.voi_to_uint8(np.array([[100, 140, 200]], dtype=np.uint16), _ds())
    assert got.tolist() == [[0, 102, 255]]  # 40/100 * 255 = 102


def test_minmax_is_bit_depth_independent() -> None:
    """The population this fallback actually serves is mg_tomo: 10/12-bit grayscale with
    no window tags. Pass-through would truncate it; min/max fills the 8-bit range."""
    values = np.array([[0, 256, 512, 768, 1023]], dtype=np.uint16)  # 10-bit
    got = render.voi_to_uint8(values, _ds())
    assert (got.min(), got.max()) == (0, 255)
    assert got.tolist() == [[0, 64, 128, 191, 255]]


def test_minmax_on_a_uniform_frame_returns_zeros_not_nan() -> None:
    """max == min has no range to stretch. Explicit guard, not a divide-by-zero."""
    got = render.voi_to_uint8(np.full((3, 4), 700, dtype=np.uint16), _ds())
    assert got.dtype == np.uint8
    assert got.shape == (3, 4)
    assert int(got.max()) == 0


def test_minmax_keeps_a_bright_outlier_at_255() -> None:
    """The accepted cost of min/max is that an outlier sets an endpoint. For this project
    the outlier IS the burned-in text, so it must land at 255, not be clipped away."""
    values = np.array([[10, 11, 12, 13, 1000]], dtype=np.uint16)
    got = render.voi_to_uint8(values, _ds())
    assert int(got[0, -1]) == 255


def test_the_real_policy_is_the_default_and_renders_end_to_end(tmp_path: Path) -> None:
    """No voi_fn injected: render_set falls through to voi_to_uint8 itself."""
    series = tmp_path / "policy"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("policy"))
    summary, manifest, out_dir = _run(
        tmp_path, [(FAKE_SERIES.format(0), str(series), 0)], voi_fn=None
    )
    assert summary.errors == () and summary.n_rendered == 1
    got = _png_pixels(out_dir / f"{next(iter(manifest))}.png")
    assert got.dtype == np.uint8 and got.shape == (4, 6, 3)


def test_policy_errors_are_counted_and_do_not_kill_the_run(tmp_path: Path) -> None:
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _make_dcm(good / "a.dcm", sop_uid=FAKE_UID.format("good"))
    _make_dcm(bad / "a.dcm", sop_uid=FAKE_UID.format("bad"), tags={"VOILUTFunction": "SIGMOID"})
    summary, manifest, _ = _run(
        tmp_path,
        [(FAKE_SERIES.format(0), str(good), 0), (FAKE_SERIES.format(1), str(bad), 0)],
        voi_fn=None,
    )
    assert summary.n_rendered == 1 and len(manifest) == 1
    assert summary.errors == (("UnsupportedVOIError", 1),)


def test_policy_error_messages_leak_no_pixel_or_patient_data(tmp_path: Path) -> None:
    """VOILUTFunction is closed-enumeration display metadata and may appear. A window
    VALUE is a measurement of the pixels, so it may not."""
    with pytest.raises(render.WindowTagError) as excinfo:
        render.voi_to_uint8(np.zeros((2, 2)), _ds(WindowCenter=1234.5, WindowWidth=0))
    assert "1234" not in str(excinfo.value)
    assert "SYNTH" not in str(excinfo.value) and "9.9.9" not in str(excinfo.value)


# --- determinism ---------------------------------------------------------------------


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    series = tmp_path / "det"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("det"), n_frames=3)
    rows = [(FAKE_SERIES.format(0), str(series), None)]
    _, m1, out1 = _run(tmp_path, rows, out_name="r1", backmap_name="b1.csv")
    _, m2, out2 = _run(tmp_path, rows, out_name="r2", backmap_name="b2.csv")
    assert m1 == m2
    image_id = next(iter(m1))
    assert (out1 / f"{image_id}.png").read_bytes() == (out2 / f"{image_id}.png").read_bytes()
    on_disk = hashlib.sha256((out1 / f"{image_id}.png").read_bytes()).hexdigest()
    assert m1[image_id][3] == on_disk


def test_png_carries_no_metadata_chunks(tmp_path: Path) -> None:
    series = tmp_path / "chunks"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("chunks"))
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    chunks = _png_chunks(out_dir / f"{next(iter(manifest))}.png")
    assert set(chunks).isdisjoint({"tIME", "pHYs", "tEXt", "iTXt", "zTXt", "iCCP"})
    assert chunks[0] == "IHDR" and chunks[-1] == "IEND"


def test_tar_and_directory_inputs_agree(tmp_path: Path) -> None:
    series = tmp_path / "fromdir"
    for i in range(3):
        _make_dcm(series / f"i{i}.dcm", sop_uid=FAKE_UID.format(i), instance_number=i)
    tar_path = tmp_path / "series.tar"
    with tarfile.open(tar_path, "w") as tar:
        for i in range(3):
            tar.add(series / f"i{i}.dcm", arcname=f"instances/{FAKE_UID.format(i)}.dcm")
    _, m_dir, out_dir = _run(
        tmp_path, [(FAKE_SERIES.format(0), str(series), None)], out_name="d", backmap_name="bd.csv"
    )
    _, m_tar, out_tar = _run(
        tmp_path,
        [(FAKE_SERIES.format(0), str(tar_path), None)],
        out_name="t",
        backmap_name="bt.csv",
    )
    assert m_dir == m_tar
    image_id = next(iter(m_dir))
    assert (out_dir / f"{image_id}.png").read_bytes() == (out_tar / f"{image_id}.png").read_bytes()


def test_manifest_is_sorted_lf_terminated_and_matches_the_files(tmp_path: Path) -> None:
    rows = []
    for i in range(4):
        series = tmp_path / f"s{i}"
        _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format(i))
        rows.append((FAKE_SERIES.format(3 - i), str(series), 0))
    summary, manifest, out_dir = _run(tmp_path, rows)
    raw = Path(summary.manifest_path).read_bytes()
    assert b"\r\n" not in raw
    lines = raw.decode("utf-8").strip().split("\n")
    assert lines[0] == ",".join(render.MANIFEST_COLUMNS)
    ids = [line.split(",")[0] for line in lines[1:]]
    assert ids == sorted(ids)
    for image_id, (_, w, h, sha, fallback) in manifest.items():
        png = out_dir / f"{image_id}.png"
        assert sha == hashlib.sha256(png.read_bytes()).hexdigest()
        assert (w, h) == (6, 4) and fallback in (0, 1)


def test_manifest_columns_are_frozen() -> None:
    # The frozen contract 10e reads to build harness.harness.ImageRef (id/w/h/frame_idx).
    assert render.MANIFEST_COLUMNS == (
        "image_id",
        "frame_idx",
        "w",
        "h",
        "sha256",
        "fallback_used",
    )


# --- the two-file PHI split ----------------------------------------------------------


def test_phi_free_manifest_holds_no_uid_while_the_backmap_does(tmp_path: Path) -> None:
    series = tmp_path / "split"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("split"))
    summary, _, _ = _run(tmp_path, [(FAKE_SERIES.format(7), str(series), 0)])
    manifest_text = Path(summary.manifest_path).read_text(encoding="utf-8")
    backmap_text = Path(summary.backmap_path).read_text(encoding="utf-8")
    assert Path(summary.manifest_path) != Path(summary.backmap_path)
    for uid in (FAKE_UID.format("split"), FAKE_SERIES.format(7)):
        assert uid not in manifest_text
        assert uid in backmap_text
    assert "SYNTH-0000" not in manifest_text  # no tag values either
    assert tuple(backmap_text.split("\n")[0].split(",")) == render.BACKMAP_COLUMNS


def test_backmap_accumulates_across_batches(tmp_path: Path) -> None:
    for i in (0, 1):
        series = tmp_path / f"batch{i}"
        _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format(i))
        _run(tmp_path, [(FAKE_SERIES.format(i), str(series), 0)])
    backmap = render.read_backmap(tmp_path / "backmap.csv")
    manifest = render.read_manifest(tmp_path / "out" / render.MANIFEST_NAME)
    assert len(backmap) == 2 and len(manifest) == 2


# --- the id gate (D-10.2) ------------------------------------------------------------


def test_collision_raises_before_anything_is_written(tmp_path: Path) -> None:
    rows = []
    for i in range(24):  # 24 ids into 16 buckets at --id-len 1: a collision is certain
        series = tmp_path / f"c{i}"
        _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format(f"col{i}"))
        rows.append((FAKE_SERIES.format(i), str(series), 0))
    with pytest.raises(render.IdCollisionError) as exc:
        _run(tmp_path, rows, id_len=1)
    message = str(exc.value)
    assert "--id-len 12" in message
    assert "Re-running unchanged cannot help" in message
    assert "gt.csv" in message
    # Both ids are named: the shared truncated id plus the two full digests behind it.
    full_ids = [tok.strip("'.") for tok in message.split() if len(tok.strip("'.")) == 64]
    assert len(full_ids) == 2 and full_ids[0] != full_ids[1]
    assert not list((tmp_path / "out").glob("*.png"))
    assert not (tmp_path / "backmap.csv").exists()


def test_collision_message_leaks_no_uid(tmp_path: Path) -> None:
    rows = []
    for i in range(24):
        series = tmp_path / f"c{i}"
        _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format(f"col{i}"))
        rows.append((FAKE_SERIES.format(i), str(series), 0))
    with pytest.raises(render.IdCollisionError) as exc:
        _run(tmp_path, rows, id_len=1)
    assert "9.9.9.synthetic" not in str(exc.value)


def test_rerunning_the_same_source_is_idempotent_not_a_collision(tmp_path: Path) -> None:
    series = tmp_path / "idem"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("idem"))
    rows = [(FAKE_SERIES.format(0), str(series), 0)]
    _, first, _ = _run(tmp_path, rows)
    _, second, _ = _run(tmp_path, rows)
    assert first == second


def test_changing_id_len_against_an_existing_backmap_raises(tmp_path: Path) -> None:
    series = tmp_path / "len"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("len"))
    rows = [(FAKE_SERIES.format(0), str(series), 0)]
    _run(tmp_path, rows)
    with pytest.raises(render.IdCollisionError, match="mixed id spaces"):
        _run(tmp_path, rows, out_name="out12", id_len=12)


def test_collision_against_an_already_rendered_image_raises(tmp_path: Path) -> None:
    """The case the whole D-10.2 gate exists for: a cross-run truncated-id collision."""
    series = tmp_path / "cross"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("cross"))
    doomed_id = render.compute_image_id(FAKE_UID.format("cross"), 0)
    # A back-map from an earlier run that already claims that id for a DIFFERENT instance.
    backmap = tmp_path / "prior.csv"
    with open(backmap, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(render.BACKMAP_COLUMNS)
        writer.writerow([doomed_id, FAKE_SERIES.format(9), FAKE_UID.format("other"), 0, 8])
    with pytest.raises(render.IdCollisionError, match="already-rendered image") as exc:
        _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)], backmap_name="prior.csv")
    assert "--id-len 12" in str(exc.value)
    assert "9.9.9.synthetic" not in str(exc.value)
    assert not list((tmp_path / "out").glob("*.png"))


def test_disjoint_batches_with_different_id_len_are_rejected(tmp_path: Path) -> None:
    """Two id spaces in one back-map: no id can key-collide, so the space itself is checked."""
    first = tmp_path / "spacea"
    second = tmp_path / "spaceb"
    _make_dcm(first / "a.dcm", sop_uid=FAKE_UID.format("spacea"))
    _make_dcm(second / "a.dcm", sop_uid=FAKE_UID.format("spaceb"))
    _run(tmp_path, [(FAKE_SERIES.format(0), str(first), 0)])
    with pytest.raises(render.IdCollisionError, match="mixed id spaces"):
        _run(tmp_path, [(FAKE_SERIES.format(1), str(second), 0)], out_name="out2", id_len=12)


def test_a_changed_hash_formula_is_caught(tmp_path: Path, monkeypatch) -> None:
    """Same instance, same --id-len, different id => the formula moved under us."""
    series = tmp_path / "drift"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("drift"))
    rows = [(FAKE_SERIES.format(0), str(series), 0)]
    _run(tmp_path, rows)
    real = render.compute_image_id
    monkeypatch.setattr(
        render,
        "compute_image_id",
        lambda uid, frame, id_len=render.DEFAULT_ID_LEN: real(uid, frame, 64)[-id_len:],
    )
    with pytest.raises(render.IdCollisionError, match="id formula or --id-len changed"):
        _run(tmp_path, rows, out_name="drift_out")


def test_manifest_and_backmap_must_be_different_files(tmp_path: Path) -> None:
    series = tmp_path / "same_path"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("same_path"))
    inputs = render.read_input_list(
        _inputs_csv(tmp_path / "in.csv", [(FAKE_SERIES.format(0), str(series), 0)])
    )
    shared = tmp_path / "shared.csv"
    with pytest.raises(render.InputListError, match="different files"):
        render.render_set(
            inputs,
            tmp_path / "out",
            manifest_path=shared,
            backmap_path=shared,
            voi_fn=_identity_voi,
        )


def test_single_dcm_file_input_works(tmp_path: Path) -> None:
    dcm = tmp_path / "loose" / "only.dcm"
    pixels = _make_dcm(dcm, sop_uid=FAKE_UID.format("loose"))
    _, manifest, out_dir = _run(tmp_path, [(FAKE_SERIES.format(0), str(dcm), 0)])
    image_id = render.compute_image_id(FAKE_UID.format("loose"), 0)
    assert image_id in manifest
    np.testing.assert_array_equal(_png_pixels(out_dir / f"{image_id}.png")[:, :, 0], pixels)


def test_uppercase_suffix_in_a_directory_is_found(tmp_path: Path) -> None:
    """A .DCM series must not silently render from a tar but vanish from a directory."""
    series = tmp_path / "shouty"
    _make_dcm(series / "A.DCM", sop_uid=FAKE_UID.format("shouty"))
    summary, manifest, _ = _run(tmp_path, [(FAKE_SERIES.format(0), str(series), 0)])
    assert summary.n_rendered == 1 and len(manifest) == 1


def test_two_input_rows_resolving_to_one_instance_raise(tmp_path: Path) -> None:
    series = tmp_path / "same"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("same"))
    rows = [(FAKE_SERIES.format(0), str(series), 0), (FAKE_SERIES.format(1), str(series), 0)]
    with pytest.raises(render.IdCollisionError, match="one row per series"):
        _run(tmp_path, rows)


# --- failure isolation + PHI-free stdout ---------------------------------------------


def test_one_bad_instance_does_not_kill_the_run(tmp_path: Path) -> None:
    good = tmp_path / "good"
    _make_dcm(good / "a.dcm", sop_uid=FAKE_UID.format("good"))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "a.dcm").write_bytes(b"this is not a dicom file")
    empty = tmp_path / "empty"
    empty.mkdir()
    summary, manifest, _ = _run(
        tmp_path,
        [
            (FAKE_SERIES.format(0), str(good), 0),
            (FAKE_SERIES.format(1), str(bad), 0),
            (FAKE_SERIES.format(2), str(empty), 0),
        ],
    )
    assert summary.n_rendered == 1 and len(manifest) == 1
    assert dict(summary.errors) == {"InvalidDicomError": 1, "NoInstancesError": 1}


def test_cli_stdout_is_counts_only(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(render, "voi_to_uint8", _identity_voi)
    series = tmp_path / "cli"
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("cli"))
    inputs = _inputs_csv(tmp_path / "in.csv", [(FAKE_SERIES.format(0), str(series), 0)])
    code = render.main(
        [
            "--inputs",
            str(inputs),
            "--out-dir",
            str(tmp_path / "cliout"),
            "--backmap",
            str(tmp_path / "clibackmap.csv"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "rendered 1/1" in out
    assert "9.9.9.synthetic" not in out
    assert "SYNTH-0000" not in out
    assert "Traceback" not in out


def test_cli_exits_nonzero_when_a_row_fails(tmp_path: Path, capsys) -> None:
    series = tmp_path / "cli2"
    # SIGMOID is the D-10.1a refusal, so this row fails under the real default policy.
    _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format("cli2"), tags={"VOILUTFunction": "SIGMOID"})
    inputs = _inputs_csv(tmp_path / "in2.csv", [(FAKE_SERIES.format(0), str(series), 0)])
    code = render.main(
        [
            "--inputs",
            str(inputs),
            "--out-dir",
            str(tmp_path / "out2"),
            "--backmap",
            str(tmp_path / "bm2.csv"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1  # a skipped row must never look like a complete run
    assert "UnsupportedVOIError" in out
    assert "rendered 0/1" in out


def test_cli_reports_a_fatal_collision_without_a_traceback(tmp_path: Path, capsys) -> None:
    rows = []
    for i in range(24):
        series = tmp_path / f"f{i}"
        _make_dcm(series / "a.dcm", sop_uid=FAKE_UID.format(f"fatal{i}"))
        rows.append((FAKE_SERIES.format(i), str(series), 0))
    inputs = _inputs_csv(tmp_path / "in3.csv", rows)
    code = render.main(
        [
            "--inputs",
            str(inputs),
            "--out-dir",
            str(tmp_path / "out3"),
            "--backmap",
            str(tmp_path / "bm3.csv"),
            "--id-len",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "IdCollisionError" in out
    assert "9.9.9.synthetic" not in out
    assert "Traceback" not in out
