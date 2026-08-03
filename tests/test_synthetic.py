"""Phase 3 acceptance tests for the synthetic fixture factory (plan.md Phase 3)."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tests import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS = ("CMFN", "ACC-0001")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Acceptance 1: determinism ------------------------------------------------


def test_clean_image_deterministic():
    img_a, gt_a = synthetic.make_synthetic_image(TOKENS, seed=7)
    img_b, gt_b = synthetic.make_synthetic_image(TOKENS, seed=7)
    assert png_bytes(img_a) == png_bytes(img_b)
    assert gt_a == gt_b


def test_hard_image_deterministic():
    img_a, gt_a = synthetic.make_synthetic_image(TOKENS, seed=7, hard=True)
    img_b, gt_b = synthetic.make_synthetic_image(TOKENS, seed=7, hard=True)
    assert png_bytes(img_a) == png_bytes(img_b)
    assert gt_a == gt_b


def test_different_seed_changes_hard_output():
    img_a, _ = synthetic.make_synthetic_image(TOKENS, seed=1, hard=True)
    img_b, _ = synthetic.make_synthetic_image(TOKENS, seed=2, hard=True)
    assert png_bytes(img_a) != png_bytes(img_b)


def test_hard_differs_from_clean():
    clean, _ = synthetic.make_synthetic_image(TOKENS, seed=7, hard=False)
    hard, _ = synthetic.make_synthetic_image(TOKENS, seed=7, hard=True)
    assert png_bytes(clean) != png_bytes(hard)


def test_scene_deterministic():
    scene_a = synthetic.make_scene(seed=0)
    scene_b = synthetic.make_scene(seed=0)
    for a, b in zip(scene_a.images, scene_b.images, strict=True):
        assert png_bytes(a.image) == png_bytes(b.image)
        assert a.gt == b.gt


# --- Acceptance 2: blank control ---------------------------------------------


def test_blank_image_zero_gt_rows(blank_image):
    img, gt = blank_image
    assert gt == []
    assert img.getbbox() is None  # uniformly black — no drawn content at all


# --- Acceptance 3: scene coverage ---------------------------------------------


def test_scene_spans_strata_vendors_and_blank(scene):
    assert len({s.stratum for s in scene.images}) >= 2
    assert len({s.vendor for s in scene.images}) >= 2
    assert len(scene.blanks) >= 1
    assert all(not b.gt for b in scene.blanks)


def test_scene_hard_fraction_about_30_percent(scene):
    frac = sum(s.hard for s in scene.images) / len(scene.images)
    assert 0.2 <= frac <= 0.4


def test_scene_gt_boxes_match_metadata(scene):
    for s in scene.images:
        for t in s.gt:
            assert t.image_id == s.image_id
            assert t.stratum == s.stratum
            assert t.vendor == s.vendor
            # modality must agree with the stratum, e.g. synth_xr_chest -> XR
            assert s.stratum.startswith(f"synth_{t.modality.lower()}")
            x0, y0, x1, y1 = t.bbox
            assert 0 <= x0 < x1 <= s.image.width
            assert 0 <= y0 < y1 <= s.image.height


def test_allowlist_known_and_one_token_off_it(scene):
    assert scene.allowlist == frozenset(synthetic.FAKE_TOKENS)
    drawn = {t.token_text for s in scene.images for t in s.gt}
    assert synthetic.NOT_ALLOWLISTED in drawn
    assert synthetic.NOT_ALLOWLISTED not in scene.allowlist


# --- Acceptance 4: standalone import (no pytest required by the module) -------


def test_synthetic_module_imports_without_pytest():
    code = (
        "import sys; "
        "import tests.synthetic; "
        "assert 'pytest' not in sys.modules, 'tests.synthetic pulled in pytest'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


# --- Fabricators ----------------------------------------------------------------


def _gt():
    _, gt = synthetic.make_synthetic_image(TOKENS, seed=3)
    return gt


def test_perfect_output_matches_gt():
    gt = _gt()
    out = synthetic.perfect_output(gt)
    assert [w.text for w in out.words] == [t.token_text for t in gt]
    assert [w.bbox for w in out.words] == [t.bbox for t in gt]
    assert not out.box_free


def test_misread_output_one_char_off():
    gt = _gt()
    out = synthetic.misread_output(gt, index=0)
    assert out.words[0].text != gt[0].token_text
    assert len(out.words[0].text) == len(gt[0].token_text)
    diffs = sum(a != b for a, b in zip(out.words[0].text, gt[0].token_text, strict=True))
    assert diffs == 1
    assert [w.text for w in out.words[1:]] == [t.token_text for t in gt[1:]]


def test_mutate_token_spec_example():
    assert synthetic.mutate_token("CMFN") == "CMEN"


def test_omission_output_drops_one_box():
    gt = _gt()
    out = synthetic.omission_output(gt, index=0)
    assert len(out.words) == len(gt) - 1
    assert [w.text for w in out.words] == [t.token_text for t in gt[1:]]


def test_hallucination_on_blank():
    out = synthetic.hallucination_output(gt=[])
    assert len(out.words) == 1
    assert out.words[0].bbox is not None
    assert out.words[0].text not in synthetic.FAKE_TOKENS


def test_box_free_output():
    gt = _gt()
    out = synthetic.box_free_output(gt)
    assert out.box_free
    assert len(out.words) == 1
    assert out.words[0].bbox is None and out.words[0].confidence is None
    for t in gt:
        assert t.token_text in out.words[0].text


def test_fabricated_raw_response_is_marked_synthetic():
    gt = _gt()
    for out in (
        synthetic.perfect_output(gt),
        synthetic.misread_output(gt),
        synthetic.omission_output(gt),
        synthetic.hallucination_output(gt),
        synthetic.box_free_output(gt),
    ):
        assert out.raw_response == {"synthetic": True}


# --- D-3.2: in-memory factory, tmp_path when a file is needed --------------------


def test_image_roundtrips_through_tmp_path(tmp_path):
    img, gt = synthetic.make_synthetic_image(TOKENS, seed=5)
    p = tmp_path / "synth.png"
    img.save(p)
    assert png_bytes(Image.open(p).convert("L")) == png_bytes(img)
    assert gt  # the GT still comes from the factory, not the file
