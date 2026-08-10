"""Synthetic fixture factory — 100% fake images, tokens, and engine outputs.

This module is the safety foundation for Phases 4-7: everything the matcher/scorer
tests consume is generated here, in memory, from obviously-fake tokens ("CMFN",
"GRDN1234", "ACC-0001" style). It touches ZERO PHI by construction: no manifest,
no gt.csv, no GCS, no real-ID shapes.

Design rules (plan.md Phase 3, D-3.1/D-3.2):
- Deterministic: every random choice (hard-set font size, rotation, noise) is driven
  by an explicit seed. Same seed -> byte-identical PNG bytes and identical GT boxes.
  No time, no unseeded RNG, no dict-ordering dependence.
- In-memory only (D-3.2): nothing here writes a file. Tests that need a real path
  save the returned PIL.Image into pytest's tmp_path themselves.
- Hard subset (D-3.1): `hard=True` adds *slight* font-size/rotation/noise variation,
  nothing adversarial. Clean images ignore the seed's jitter entirely.
- Plain importable module: no pytest dependency, so runner tests can reuse it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harness.contract import GTToken, OCROutput, OCRWord

# Obviously-synthetic token pool (no real-ID shapes). These six are the scene
# allowlist; NOT_ALLOWLISTED is drawn in the scene but deliberately kept off it.
FAKE_TOKENS = ("CMFN", "GRDN1234", "ACC-0001", "CMFN-0042", "GRDN5678", "ACC-0099")
NOT_ALLOWLISTED = "XZQV-777"

_MARGIN = 24
_LINE_GAP = 18
_BASE_FONT_SIZE = 20
_HARD_FONT_SIZES = (16, 18, 22, 24)  # slight variation only (D-3.1)
_HARD_MAX_ROT_DEG = 5.0
_HARD_NOISE_FRACTION = 0.005

# Placeholder for OCROutput.raw_response in fabricated outputs. The real field is
# PHI-bearing; this constant marks fabricated ones as safe.
SYNTHETIC_RAW_RESPONSE = {"synthetic": True}

# OCROutput.version is required and must be non-empty (D-8.4), so fabricated outputs need
# a version too. A deliberately unreal marker: it can never collide with a real engine's
# version string, so a fixture row can never be mistaken for a measured one.
SYNTHETIC_VERSION = "0.0.0-synthetic"


def _font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled default font (pinned Pillow==11.1.0) — no system-font
    # dependence, so rendering is identical across machines.
    return ImageFont.load_default(size=size)


def _draw_token(
    img: Image.Image, token: str, x: int, y: int, font: ImageFont.FreeTypeFont, angle: float
) -> tuple[float, float, float, float]:
    """Draw one token (optionally slightly rotated) and return its exact pixel bbox.

    The token is drawn on its own transparent tile so rotation is per-token; the GT
    bbox is the tile's non-zero content bounds offset by the paste position — exact
    and deterministic, including the anti-aliased fringe of a rotated tile.
    """
    pad = 4
    tb = font.getbbox(token)
    tile = Image.new("LA", (tb[2] - tb[0] + 2 * pad, tb[3] - tb[1] + 2 * pad), (0, 0))
    ImageDraw.Draw(tile).text((pad - tb[0], pad - tb[1]), token, font=font, fill=(255, 255))
    if angle:
        tile = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    content = tile.getbbox()
    luma, alpha = tile.split()
    img.paste(luma, (x, y), alpha)
    return (x + content[0], y + content[1], x + content[2], y + content[3])


def make_blank_image(size: tuple[int, int] = (640, 480)) -> tuple[Image.Image, list[GTToken]]:
    """Negative control: a uniform black frame with exactly zero GT rows."""
    return Image.new("L", size, 0), []


def make_synthetic_image(
    tokens: tuple[str, ...] | list[str],
    size: tuple[int, int] = (640, 480),
    seed: int = 0,
    hard: bool = False,
    *,
    image_id: str = "synth-000",
    series_uid: str = "9.9.9.synthetic.0",
    modality: str = "CT",
    vendor: str = "FakeVendorA",
    stratum: str = "synth_ct_axial",
    frame_idx: int = 0,
    label: str = "PHI",
) -> tuple[Image.Image, list[GTToken]]:
    """Draw fake tokens at known pixel boxes; return the image and exact GT rows.

    Clean (`hard=False`): fixed font size, no rotation, no noise — `seed` has no
    effect on the output (kept in the signature so call sites are uniform).
    Hard (`hard=True`): per-token font size and small rotation plus sparse pixel
    noise, all drawn from `random.Random(seed)` / a numpy RNG derived from it.
    """
    rng = random.Random(seed)
    img = Image.new("L", size, 0)
    gt: list[GTToken] = []
    y = _MARGIN
    for token in tokens:
        font_size = rng.choice(_HARD_FONT_SIZES) if hard else _BASE_FONT_SIZE
        angle = rng.uniform(-_HARD_MAX_ROT_DEG, _HARD_MAX_ROT_DEG) if hard else 0.0
        bbox = _draw_token(img, token, _MARGIN, y, _font(font_size), angle)
        if bbox[2] > size[0] or bbox[3] > size[1]:
            raise ValueError(f"token {token!r} does not fit in image of size {size}")
        gt.append(
            GTToken(
                image_id=image_id,
                series_uid=series_uid,
                modality=modality,
                vendor=vendor,
                stratum=stratum,
                frame_idx=frame_idx,
                token_text=token,
                bbox=bbox,
                label=label,
            )
        )
        y = int(bbox[3]) + _LINE_GAP
    if hard:
        arr = np.array(img)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        mask = np_rng.random(arr.shape) < _HARD_NOISE_FRACTION
        arr[mask] = np_rng.integers(0, 256, size=arr.shape, dtype=np.uint8)[mask]
        img = Image.fromarray(arr, mode="L")
    return img, gt


# ---------------------------------------------------------------------------
# Scene: a fixed small set of images spanning >=2 strata and >=2 vendors,
# with >=1 blank control and ~30% hard images (2 of 7), plus a known allowlist.
# ---------------------------------------------------------------------------


@dataclass
class SceneImage:
    image_id: str
    image: Image.Image
    gt: list[GTToken]
    stratum: str
    vendor: str
    hard: bool


@dataclass
class Scene:
    images: list[SceneImage]
    allowlist: frozenset[str]

    @property
    def blanks(self) -> list[SceneImage]:
        return [s for s in self.images if not s.gt]


# (tokens, modality, stratum, vendor, hard); tokens=() is the blank control.
_SCENE_SPEC: tuple[tuple[tuple[str, ...], str, str, str, bool], ...] = (
    (("CMFN", "ACC-0001"), "CT", "synth_ct_axial", "FakeVendorA", False),
    (("GRDN1234",), "CT", "synth_ct_axial", "FakeVendorB", False),
    (("CMFN-0042", "GRDN5678"), "XR", "synth_xr_chest", "FakeVendorA", False),
    (("ACC-0099",), "XR", "synth_xr_chest", "FakeVendorB", True),
    (("CMFN", "GRDN1234"), "XR", "synth_xr_chest", "FakeVendorA", True),
    (("ACC-0001", NOT_ALLOWLISTED), "CT", "synth_ct_axial", "FakeVendorB", False),
    ((), "CT", "synth_ct_axial", "FakeVendorA", False),  # blank negative control
)


def make_scene(seed: int = 0, size: tuple[int, int] = (640, 480)) -> Scene:
    """Build the fixed scene. Deterministic: same seed -> identical images and GT."""
    images: list[SceneImage] = []
    for i, (tokens, modality, stratum, vendor, hard) in enumerate(_SCENE_SPEC):
        image_id = f"synth-{i:03d}"
        if tokens:
            img, gt = make_synthetic_image(
                tokens,
                size=size,
                seed=seed * 1000 + i,
                hard=hard,
                image_id=image_id,
                series_uid=f"9.9.9.synthetic.{i}",
                modality=modality,
                vendor=vendor,
                stratum=stratum,
            )
        else:
            img, gt = make_blank_image(size)
        images.append(SceneImage(image_id, img, gt, stratum, vendor, hard))
    return Scene(images=images, allowlist=frozenset(FAKE_TOKENS))


# ---------------------------------------------------------------------------
# OCROutput fabricators: simulate engine behaviours against a known GT.
# ---------------------------------------------------------------------------


def mutate_token(text: str) -> str:
    """Deterministic one-character misread: 'CMFN' -> 'CMEN'."""
    i = len(text) // 2
    repl = chr(ord(text[i]) - 1)
    if not repl.isalnum():
        repl = chr(ord(text[i]) + 1)
    return text[:i] + repl + text[i + 1 :]


def perfect_output(
    gt: list[GTToken],
    model_name: str = "synthetic-perfect",
    version: str = SYNTHETIC_VERSION,
) -> OCROutput:
    """Every GT token read exactly, at its exact box."""
    words = [OCRWord(text=t.token_text, bbox=t.bbox, confidence=0.99) for t in gt]
    return OCROutput(
        words=words,
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name=model_name,
        version=version,
    )


def misread_output(
    gt: list[GTToken],
    index: int = 0,
    model_name: str = "synthetic-misread",
    version: str = SYNTHETIC_VERSION,
) -> OCROutput:
    """Perfect read except the token at `index` has one character wrong."""
    words = [
        OCRWord(
            text=mutate_token(t.token_text) if i == index else t.token_text,
            bbox=t.bbox,
            confidence=0.99 if i != index else 0.61,
        )
        for i, t in enumerate(gt)
    ]
    return OCROutput(
        words=words,
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name=model_name,
        version=version,
    )


def omission_output(
    gt: list[GTToken],
    index: int = 0,
    model_name: str = "synthetic-omission",
    version: str = SYNTHETIC_VERSION,
) -> OCROutput:
    """Perfect read except the token at `index` is not detected at all."""
    words = [
        OCRWord(text=t.token_text, bbox=t.bbox, confidence=0.99)
        for i, t in enumerate(gt)
        if i != index
    ]
    return OCROutput(
        words=words,
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name=model_name,
        version=version,
    )


def hallucination_output(
    gt: list[GTToken] | None = None,
    model_name: str = "synthetic-hallucination",
    version: str = SYNTHETIC_VERSION,
) -> OCROutput:
    """All GT tokens read correctly PLUS one invented box. With gt=[] (a blank
    image) the output is a single hallucinated word — the negative-control case."""
    words = [OCRWord(text=t.token_text, bbox=t.bbox, confidence=0.99) for t in gt or []]
    words.append(OCRWord(text="ZZZZ-FAKE", bbox=(500.0, 400.0, 590.0, 424.0), confidence=0.42))
    return OCROutput(
        words=words,
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name=model_name,
        version=version,
    )


def box_free_output(
    gt: list[GTToken],
    model_name: str = "synthetic-boxfree",
    version: str = SYNTHETIC_VERSION,
) -> OCROutput:
    """A VLM-style blob: one text response, no boxes, no confidence."""
    blob = " ".join(t.token_text for t in gt)
    word = OCRWord(text=blob, bbox=None, confidence=None)
    return OCROutput(
        words=[word],
        raw_response=SYNTHETIC_RAW_RESPONSE,
        model_name=model_name,
        version=version,
        box_free=True,
    )
