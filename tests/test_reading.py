"""Phase 13b — tests for the reading arm. 100% synthetic: fake images, fake tokens.

Every image here is drawn by `tests/synthetic.py` from obviously-fake tokens ("CMFN",
"GRDN1234", "ACC-0001"); no render, no `gt.csv`, and no real identifier is touched. The
readers are stubs that return canned strings, so the whole scoring path is proven without
any model dependency — which is also what makes the fairness assertions meaningful: a stub
cannot accidentally "read well" and hide a scoring bug.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from harness import reading
from harness.aggregate import aggregate
from harness.contract import GTToken
from harness.cost import PRICES, priced
from harness.harness import ImageRef
from harness.reading import (
    CROP_HEIGHT,
    Reader,
    arm_config_hash,
    crop_for_reading,
    read_image,
    run_reading,
)
from tests import synthetic


class ScriptedReader(Reader):
    """Returns canned strings in call order. No model, no image understanding.

    `read()` ignores the crop entirely — the point is to drive `score()` through known
    inputs. It records the crops it was handed so the preprocessing assertions can inspect
    exactly what a real reader would have seen.
    """

    model_name = "stub-reader"
    version = synthetic.SYNTHETIC_VERSION
    version_source = None  # a test double, never sourced from an installed library
    config_id = synthetic.SYNTHETIC_CONFIG_ID

    def __init__(self, script: list[str], config_value: str = "a") -> None:
        self.script = list(script)
        self.calls = 0
        self.seen_crops: list[tuple[int, int]] = []
        self._config_value = config_value

    def read(self, crop: Image.Image) -> str:
        self.seen_crops.append(crop.size)
        text = self.script[self.calls] if self.calls < len(self.script) else ""
        self.calls += 1
        return text

    def config(self) -> dict[str, object]:
        return {"stub": self._config_value}


def _image_ref(image_id: str, path: Path, image: Image.Image, **kw) -> ImageRef:
    return ImageRef(
        id=image_id,
        path=str(path),
        w=image.width,
        h=image.height,
        stratum=kw.get("stratum", "synth_ct_axial"),
        modality=kw.get("modality", "CT"),
        vendor=kw.get("vendor", "FakeVendorA"),
        frame_idx=0,
    )


def _saved(tmp_path: Path, image: Image.Image, name: str = "img.png") -> Path:
    """Fixtures are in-memory (D-3.2); a test that needs a path saves it itself."""
    path = tmp_path / name
    image.save(path)
    return path


def _scene_image(tmp_path, tokens, *, label="KEEP", name="img.png"):
    image, gt = synthetic.make_synthetic_image(tokens, label=label)
    path = _saved(tmp_path, image, name)
    return image, gt, path


def _read(tmp_path, tokens, script, *, label="KEEP", control_boxes=(), allowlist=None):
    """Score one synthetic image with a scripted reader; returns the raw score row."""
    image, gt, path = _scene_image(tmp_path, tokens, label=label)
    reader = ScriptedReader(script)
    row = read_image(
        str(path),
        gt,
        reader,
        allowlist=set(allowlist if allowlist is not None else synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-000", path, image),
        control_boxes=control_boxes,
    )
    return row, gt, reader


# --- the scoring path -------------------------------------------------------------------


def test_exact_read_scores_no_false_redaction(tmp_path):
    """The happy path: every KEEP token read byte-exact and on the allowlist."""
    tokens = ("CMFN", "GRDN1234")
    row, _, _ = _read(tmp_path, tokens, list(tokens))

    assert row["keep_total"] == 2
    assert row["keep_exact_match_count"] == 2
    assert row["keep_exact_match_all"] is True
    assert row["false_redaction_count"] == 0
    assert row["found_count"] == 2
    assert row["omission_count"] == 0
    assert row["added_count"] == 0


def test_whitespace_is_normalized_but_case_is_not(tmp_path):
    """The frozen `normalize()` is reused, not re-implemented: strip yes, casefold no.

    `l` != `L` and `M` != `F` carry meaning in clinical tokens, so a case-insensitive
    reading arm would report a read as correct that the redaction pipeline would reject.
    """
    padded, _, _ = _read(tmp_path, ("CMFN",), ["  CMFN  "])
    assert padded["keep_exact_match_count"] == 1
    assert padded["false_redaction_count"] == 0

    lowered, _, _ = _read(tmp_path, ("CMFN",), ["cmfn"])
    assert lowered["found_count"] == 1  # it read *something* at that box
    assert lowered["keep_exact_match_count"] == 0
    assert lowered["false_redaction_count"] == 1  # ...and the good token gets blacked out


def test_empty_return_is_an_omission(tmp_path):
    """"" means "nothing readable here" — an omission, never a silent pass."""
    row, _, _ = _read(tmp_path, ("CMFN", "GRDN1234"), ["CMFN", ""])

    assert row["found_count"] == 1
    assert row["omission_count"] == 1
    assert row["gt_total"] == 2
    assert row["added_count"] == 0  # an omission is NEVER counted on the Added axis
    assert row["false_redaction_count"] == 1  # never read -> never confirmed -> redacted


def test_whitespace_only_return_is_also_an_omission(tmp_path):
    """A reader that returns "   " has read nothing; `normalize()` decides that, not len()."""
    row, _, _ = _read(tmp_path, ("CMFN",), ["   "])
    assert row["omission_count"] == 1
    assert row["found_count"] == 0


def test_control_boxes_are_refused_on_a_text_bearing_image(tmp_path):
    """Nobody has confirmed a region of a text-bearing frame is empty.

    The GT records where text IS, never where it is absent, so a control box there could
    land on unannotated glyphs and score a CORRECT read as a hallucination — an invented
    number on the Added axis. No matcher exists here to catch it, so the input is refused.
    """
    with pytest.raises(ValueError, match="confirmed-blank"):
        _read(
            tmp_path,
            ("CMFN",),
            ["CMFN", "ZZZZ-FAKE"],
            control_boxes=[(400.0, 300.0, 480.0, 324.0)],
        )


def test_empty_read_on_a_control_box_is_not_a_hallucination(tmp_path):
    """Silence on a blank box is the correct answer, not an error of either kind."""
    image, _ = synthetic.make_blank_image()
    path = _saved(tmp_path, image, "blank.png")

    row = read_image(
        str(path),
        [],
        ScriptedReader([""]),
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-blank", path, image),
        control_boxes=[(400.0, 300.0, 480.0, 324.0)],
    )
    assert row["added_count"] == 0
    assert row["omission_count"] == 0
    assert row["negative_control"] is True


def test_blank_frame_with_control_boxes_is_the_hallucination_floor(tmp_path):
    """Zero GT tokens + control boxes = the negative control for this arm.

    Handed only GT boxes a reader can never invent a location, so `added_count` would be
    structurally zero and the arm would have no floor at all. Control boxes on a
    confirmed-blank frame are the only way it can run one.
    """
    image, _ = synthetic.make_blank_image()
    path = _saved(tmp_path, image, "blank.png")
    reader = ScriptedReader(["INVENTED", ""])

    row = read_image(
        str(path),
        [],
        reader,
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-blank", path, image),
        control_boxes=[(100.0, 100.0, 200.0, 130.0), (300.0, 300.0, 400.0, 330.0)],
    )

    assert row["negative_control"] is True
    assert row["negative_control_floor_count"] == 1
    assert row["added_count"] == 1
    assert row["gt_total"] == 0
    assert row["keep_total"] == 0


def test_blank_image_without_control_boxes_contributes_zero_rows(tmp_path):
    """Nothing to read -> no row, rather than a row of zeros diluting every rate.

    Mirrors `load_gt()`, where a confirmed-blank frame is ABSENT from the ground truth
    rather than present-and-empty.
    """
    image, _ = synthetic.make_blank_image()
    path = _saved(tmp_path, image, "blank.png")
    reader = ScriptedReader([])

    row = read_image(
        str(path),
        [],
        reader,
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-blank", path, image),
    )

    assert row is None
    assert reader.calls == 0  # and the reader was never even invoked


def test_single_character_token_is_scored_normally(tmp_path):
    """No length floor anywhere. One `ct_secondary_capture` frame's entire content was
    `R` and `L`; a misread single character IS a false redaction, the headline metric."""
    row, _, reader = _read(
        tmp_path, ("L",), ["L"], allowlist=set(synthetic.FAKE_TOKENS) | {"L"}
    )

    assert reader.calls == 1
    assert row["gt_total"] == 1
    assert row["keep_total"] == 1
    assert row["keep_exact_match_count"] == 1
    assert row["false_redaction_count"] == 0


def test_misread_single_character_is_a_false_redaction(tmp_path):
    row, _, _ = _read(tmp_path, ("L",), ["1"])
    assert row["false_redaction_count"] == 1
    assert row["omission_count"] == 0  # it read something — wrongly. Not an omission.


def test_phi_tokens_do_not_enter_the_keep_denominator(tmp_path):
    """A misread PHI token is correct redaction, not false redaction — KEEP-only denominator."""
    row, _, _ = _read(tmp_path, ("CMFN",), ["WRONG"], label="PHI")
    assert row["keep_total"] == 0
    assert row["false_redaction_count"] == 0
    assert row["found_count"] == 1


# --- pinned crop preprocessing ----------------------------------------------------------


def test_crop_takes_no_preprocessing_arguments(tmp_path):
    """The structural fairness guarantee: there is no knob to turn.

    If a reader could pass padding/height/interpolation, the comparison could be rigged
    silently — the single easiest way to do so, per plan.md's fairness rules.
    """
    params = list(inspect.signature(crop_for_reading).parameters)
    assert params == ["image", "bbox"]


def test_reader_only_ever_sees_a_finished_crop():
    """`Reader.read` is handed a crop and nothing else — no path, no full image, no knobs."""
    params = list(inspect.signature(Reader.read).parameters)
    assert params == ["self", "crop"]


def test_every_crop_is_normalized_to_the_pinned_height(tmp_path):
    image, gt, path = _scene_image(tmp_path, ("CMFN", "GRDN1234", "ACC-0001"))
    reader = ScriptedReader(["CMFN", "GRDN1234", "ACC-0001"])
    read_image(
        str(path),
        gt,
        reader,
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-000", path, image),
    )

    assert len(reader.seen_crops) == 3
    assert {h for _, h in reader.seen_crops} == {CROP_HEIGHT}
    assert all(w >= 1 for w, _ in reader.seen_crops)


def test_padding_geometry_is_exact(tmp_path):
    """Pins the crop arithmetic on hand-computed numbers, not a tolerance.

    Box (100, 100)-(200, 120): height 20, so pad = 20 * 0.10 = 2px on every side. The padded
    box rounds OUTWARD to (98, 98)-(202, 122) = 104x24, and 104 * 48/24 = 208. A padding
    fraction or target height that drifts changes these integers immediately.
    """
    image, _ = synthetic.make_blank_image(size=(640, 480))
    assert crop_for_reading(image, (100.0, 100.0, 200.0, 120.0)).size == (208, 48)


def test_padding_widens_the_box_on_all_four_sides(tmp_path):
    """The padded crop strictly contains the tight GT box (checked pre-resize arithmetic)."""
    image, gt = synthetic.make_synthetic_image(("CMFN",))
    x0, y0, x1, y1 = gt[0].bbox
    pad = (y1 - y0) * reading.CROP_PAD_FRAC
    assert pad > 0

    # Reproduce the unresized padded box and confirm it covers strictly more than the GT box
    # in both axes, with room to spare inside the image (so no clamping is in play here).
    left, top = max(0, math.floor(x0 - pad)), max(0, math.floor(y0 - pad))
    right, bottom = math.ceil(x1 + pad), math.ceil(y1 + pad)
    assert left < x0 and top < y0 and right > x1 and bottom > y1
    assert right <= image.width and bottom <= image.height

    # ...and the delivered crop carries that wider aspect, not the tight box's.
    crop = crop_for_reading(image, gt[0].bbox)
    assert crop.size == (max(1, round((right - left) * CROP_HEIGHT / (bottom - top))), 48)


def test_crop_is_clamped_to_the_image_and_never_raises(tmp_path):
    """A box on (or over) the edge yields a valid crop — one bad GT row must not abort a run."""
    image, _ = synthetic.make_blank_image(size=(64, 48))

    for bbox in [
        (-20.0, -20.0, 10.0, 10.0),  # over the top-left corner
        (60.0, 44.0, 200.0, 200.0),  # off the bottom-right
        (500.0, 500.0, 520.0, 520.0),  # entirely outside
        (10.0, 10.0, 10.0, 10.0),  # degenerate: zero area
    ]:
        crop = crop_for_reading(image, bbox)
        assert crop.height == CROP_HEIGHT
        assert crop.width >= 1
        assert crop.mode == reading.CROP_MODE


def test_crop_mode_is_pinned_for_every_reader(tmp_path):
    image, gt = synthetic.make_synthetic_image(("CMFN",))
    assert image.mode == "L"  # the render is grayscale...
    assert crop_for_reading(image, gt[0].bbox).mode == "RGB"  # ...the crop never is


# --- detection metrics must not leak out of this arm ------------------------------------


def test_reading_module_does_not_import_the_matcher():
    """Enforced on the source, not by convention: the box is given, so nothing is matched."""
    tree = ast.parse(Path(reading.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not any(name.split(".")[-1] == "matching" for name in imported), imported


def test_rows_carry_no_iou_threshold(tmp_path):
    """`iou_thr=None` is a positive statement that no detection decision was made."""
    row, _, _ = _read(tmp_path, ("CMFN",), ["CMFN"])
    assert row["iou_thr"] is None
    assert row["box_free"] is False


def test_the_written_artifact_carries_its_own_caveats(tmp_path):
    """The artifact must be self-describing — the docstring's warning does not travel with it.

    `aggregate()` keeps neither `iou_thr` nor `box_free`, so without these keys the file
    would be byte-indistinguishable in shape from an end-to-end result: same `found_count`,
    same `false_redaction_rate`, no sign the boxes were free. Asserted on the file as
    written, not just the returned dict, because the file is what a report reads.
    """
    image, gt, path = _scene_image(tmp_path, ("CMFN",))
    out = tmp_path / "out" / "reading.json"
    run_reading(
        [_image_ref("synth-000", path, image)],
        {"synth-000": gt},
        ScriptedReader(["CMFN"]),
        allowlist=set(synthetic.FAKE_TOKENS),
        out_path=out,
    )
    written = json.loads(out.read_text())

    assert written["arm"] == "reader"
    assert written["oracle_detector"] is True
    assert written["iou_thr"] is None  # positive statement: no matcher ran
    assert "undefined" in written["detection_metrics"]
    assert written["crop_preprocessing"]["height"] == CROP_HEIGHT


def test_no_box_level_metric_is_computed_anywhere_in_the_arm(tmp_path):
    """No function in the module computes or exposes a detection number.

    (The banned-substring scan that used to stand here proved nothing: `aggregate()` emits
    no such key on ANY arm, so it passed identically on end-to-end results.)
    """
    public = [n for n in vars(reading) if not n.startswith("_")]
    banned = ("precision", "recall", "iou", "detect")
    assert [n for n in public if any(b in n.lower() for b in banned)] == []

    source = Path(reading.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "iou" not in called


# --- run_reading: stamping, output file, identity ---------------------------------------


def test_run_reading_writes_its_result_and_stamps_the_manifest_values(tmp_path):
    """PHI rule: results go to a file. Rule #7: stratum/vendor come from the ImageRef."""
    image, gt, path = _scene_image(tmp_path, ("CMFN",))
    out = tmp_path / "nested" / "reading.json"

    result = run_reading(
        [_image_ref("synth-000", path, image, stratum="synth_xr_chest", vendor="FakeVendorB")],
        {"synth-000": gt},
        ScriptedReader(["CMFN"]),
        allowlist=set(synthetic.FAKE_TOKENS),
        out_path=out,
    )

    assert out.exists()
    assert "synth_xr_chest" in result["per_stratum"]
    assert result["n_images"] == 1
    assert result["model_name"] == "stub-reader"
    assert result["config_id"] == synthetic.SYNTHETIC_CONFIG_ID


def test_run_reading_skips_images_with_nothing_to_read(tmp_path):
    """`n_images` counts images the reader actually saw, not images that were offered."""
    image, gt, path = _scene_image(tmp_path, ("CMFN",))
    blank, _ = synthetic.make_blank_image()
    blank_path = _saved(tmp_path, blank, "blank.png")

    result = run_reading(
        [
            _image_ref("synth-000", path, image),
            _image_ref("synth-blank", blank_path, blank),
        ],
        {"synth-000": gt},
        ScriptedReader(["CMFN"]),
        allowlist=set(synthetic.FAKE_TOKENS),
        out_path=tmp_path / "reading.json",
    )

    assert result["n_images"] == 1


def test_two_reader_configs_get_distinct_identities_and_cannot_be_blended(tmp_path):
    """D-13.5 one level up: every reader is its own arm.

    Two configs of one reader at one version must not average into a single row — the same
    failure that made docTR-stock and docTR-tuned indistinguishable.
    """
    image, gt, path = _scene_image(tmp_path, ("CMFN",))
    img_ref = _image_ref("synth-000", path, image)

    def rows_for(reader):
        row = read_image(
            str(path),
            gt,
            reader,
            allowlist=set(synthetic.FAKE_TOKENS),
            image_ref=img_ref,
        )
        row.update(
            stratum=img_ref.stratum,
            model_name=reader.model_name,
            version=reader.version,
            config_id=reader.config_id,
            config_hash=arm_config_hash(reader),
            verifier_model_name=None,
            verifier_version=None,
            verifier_elapsed=None,
        )
        return [row]

    a = ScriptedReader(["CMFN"], config_value="a")
    b = ScriptedReader(["CMFN"], config_value="b")
    assert a.config_hash() != b.config_hash()

    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate(rows_for(a) + rows_for(b))


def test_reading_is_deterministic(tmp_path):
    """Same reader, same image, same row — timing excluded, which is the only free variable."""
    image, gt, path = _scene_image(tmp_path, ("CMFN", "GRDN1234"))
    img_ref = _image_ref("synth-000", path, image)

    def once():
        row = read_image(
            str(path),
            gt,
            ScriptedReader(["CMFN", "GRDN1234"]),
            allowlist=set(synthetic.FAKE_TOKENS),
            image_ref=img_ref,
        )
        row.pop("elapsed")
        return row

    assert once() == once()


def test_changing_the_crop_pipeline_changes_the_arm_identity(monkeypatch):
    """A rebound crop constant cannot merge with the old numbers.

    Python cannot make a module global immutable, so the guarantee is that a change is not
    SILENT: the live crop settings are hashed into the arm identity, and `aggregate()`'s
    D-13.5 guard then refuses to blend 48px-crop rows with 32px-crop rows. Without this, a
    model's numbers could be averaged across two different inputs with nothing to flag it.
    """
    reader = ScriptedReader(["CMFN"])
    at_48 = arm_config_hash(reader)

    monkeypatch.setattr(reading, "CROP_HEIGHT", 32)
    at_32 = arm_config_hash(reader)

    assert at_48 != at_32
    assert reader.config_hash() == reader.config_hash()  # the reader's own config is unchanged


def test_cost_falls_back_to_self_hosted_for_an_untagged_reader(tmp_path):
    """Reuses `estimate_cost`; a local reader is $0 without any pricing logic here."""
    row, _, _ = _read(tmp_path, ("CMFN",), ["CMFN"])
    assert row["cost"] == 0.0


class CloudReader(ScriptedReader):
    """A reader billed per request, to pin that cost is counted per CROP, not per image."""

    model_name = "stub-cloud-reader"

    @priced("cloud")
    def read(self, crop: Image.Image) -> str:
        return super().read(crop)


def test_cost_is_billed_per_crop_not_per_image(tmp_path):
    """Three boxes on one frame are three reader calls, so three billable requests.

    Pricing it once per image would understate a flat-per-request cloud reader by the token
    count of the frame — and cost per config is a reported number, not a footnote.
    """
    tokens = ("CMFN", "GRDN1234", "ACC-0001")
    image, gt, path = _scene_image(tmp_path, tokens)
    reader = CloudReader(list(tokens))

    row = read_image(
        str(path),
        gt,
        reader,
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-000", path, image),
    )

    assert reader.calls == 3
    assert row["cost"] == pytest.approx(3 * PRICES["cloud"])


# --- the contract move ------------------------------------------------------------------


def test_match_types_are_still_importable_from_the_matcher():
    """`Match`/`MatchResult` moved to contract.py; `matching.py` re-exports them."""
    from harness import contract, matching

    assert matching.MatchResult is contract.MatchResult
    assert matching.Match is contract.Match


def test_gt_token_shape_is_unchanged_by_this_arm():
    """Sanity: the reading arm consumes the frozen GT schema, it does not extend it."""
    assert [f for f in GTToken.__dataclass_fields__] == [
        "image_id",
        "series_uid",
        "modality",
        "vendor",
        "stratum",
        "frame_idx",
        "token_text",
        "bbox",
        "label",
    ]
