"""Phase 13b — the READING arm: crop a GT box, hand it to a reader, score the string.

Every engine in the end-to-end path does two jobs at once: **find** the text (emit a box)
and **read** it (emit a string). One number cannot separate them, and the two failures have
opposite consequences — text never found is never redacted (PHI leaks); text found and
misread is wrongly redacted (good data destroyed). This module measures the reading half
alone: `gt_v1`'s human-drawn boxes are handed to the model, so detection error is held at
zero and a recognizer with no detector of its own becomes testable.

**Scoring is the existing frozen path, not a second one.** Each crop's returned string is
judged by `metrics.score()` with the frozen `normalize()`, exactly like an end-to-end
engine's string, and rolled up by the existing `aggregate()`. Nothing here re-implements a
comparison.

**No matcher.** The box is GIVEN, so there is nothing to match: this module does not import
`matching.py`, never computes an IoU, and stamps `iou_thr=None` on every row.

**Detection metrics are UNDEFINED for this arm** and are never emitted. Note that the score
row's `found_count`/`omission_count` keys mean something different here than on the
end-to-end path: they are *the reader returned text* vs *the reader returned empty*, NOT
*the detector found the box* — the box was free. That is why every result this module
produces is tagged `arm="reader"`: readers run under an **oracle-detector** condition and
their numbers must be reported in their own table, never blended into the end-to-end
ranking (plan.md, "Step 7 in detail", fairness rules).

**Identical preprocessing for every reader.** Padding, height normalization, interpolation
and channel mode are module-level constants applied by `crop_for_reading()`, which takes no
preprocessing arguments at all, and a `Reader` only ever sees the finished crop — no path,
no full image, no knob. Python cannot stop a determined caller from rebinding a module
global, so the second half of the guarantee is that it cannot be done *silently*: the live
crop settings are hashed into every row's `config_hash` (see `arm_config_hash`) and written
into the result, so a changed crop pipeline is a different arm that `aggregate()` refuses to
blend with the old one — the D-13.5 rule applied at the preprocessing layer.

PHI: this module reads renders and ground truth, so it is PHI-handling. It writes its
result to a file and returns only PHI-free aggregates (counts, rates, hashes, ids). No
token text, no crop, and no pixel value is returned, printed, or logged anywhere in it.
"""

from __future__ import annotations

import dataclasses
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter

from PIL import Image

from harness.aggregate import aggregate
from harness.contract import (
    GTToken,
    Match,
    MatchResult,
    OCRWord,
    config_digest,
    normalize,
)
from harness.cost import estimate_cost
from harness.harness import ImageRef
from harness.metrics import score

BBox = tuple[float, float, float, float]

# --- PINNED CROP PREPROCESSING (identical for every reader; see module docstring) -------
# These are constants, not parameters. `crop_for_reading()` exposes no way to override them
# and readers never see anything but its output. Changing a value here changes what every
# reader is fed, so it invalidates cross-run comparability the same way an engine version
# bump does (CLAUDE.md rule #9) — change it loudly or not at all.

CROP_PAD_FRAC = 0.10
# Padding on all four sides, as a fraction of the GT box HEIGHT (height, not width, so a
# long token and a short one get geometrically similar margins). GT boxes are drawn tight
# around the glyphs, and text recognizers are trained on crops that carry a little
# background; a zero-pad crop clips anti-aliased edges and reads worse for reasons that
# have nothing to do with reading quality.

CROP_HEIGHT = 48
# Target height in pixels, aspect ratio preserved. 48 is >= both common recognizer input
# heights (CRNN / PARSeq 32, PP-OCR / SVTR 48), so a 32-height reader downsamples from real
# pixels instead of upsampling a crop that was already shrunk for it. Burned-in overlay
# text is small, so this is an upscale in the large majority of cases either way.

CROP_RESAMPLE = Image.Resampling.BICUBIC
# Bicubic is the standard choice for upscaling small text: LANCZOS rings on the hard,
# high-contrast glyph edges typical of burned-in overlays, and BILINEAR is softer.

CROP_MODE = "RGB"
# Three channels: what every candidate recognizer expects. Renders are grayscale, so this
# is a widening, never a color decision.


def crop_spec() -> dict[str, object]:
    """The crop pipeline's current settings, read LIVE — the arm's preprocessing identity.

    Read at call time, not captured at import, precisely so that a rebound module constant
    shows up here and therefore in `arm_config_hash()`. Two runs at different padding or
    target height are then different arms and `aggregate()` refuses to merge them, instead
    of averaging a model's numbers across two different inputs with nothing to flag it.
    """
    return {
        "pad_frac": CROP_PAD_FRAC,
        "height": CROP_HEIGHT,
        "resample": CROP_RESAMPLE.name,
        "mode": CROP_MODE,
    }


def arm_config_hash(reader: Reader) -> str:
    """Identity of one READING arm: the reader's own config PLUS the crop pipeline.

    Deliberately not the same value as `reader.config_hash()`. What this arm measures is a
    (reader, preprocessing) pair — the same reader fed 32px crops instead of 48px is a
    different measurement, and `aggregate()`'s guard only protects what is in the hash.
    """
    return config_digest({"reader": reader.config(), "crop": crop_spec()})


class Reader(ABC):
    """One recognizer scored on given boxes: crop in, string out.

    Deliberately NOT a `Runner` subclass and not a `Runner` at all. A `Runner` takes an
    `ImageRef` and returns a full `OCROutput` (its own boxes, confidences, raw response); a
    reader is handed one crop and returns one string. Forcing them into one interface would
    mean either a reader that fabricates boxes or a runner whose `run()` cannot see the
    image — so they sit beside each other instead.

    The identity fields mirror `Runner`'s for one reason: `aggregate()` keys on
    `(model_name, version, config_id, config_hash, verifier_*)` and refuses to blend rows
    that differ. Every reader is its own arm, and a dozen readers without distinct
    identities would silently average into one meaningless number (D-13.5).
    """

    model_name: str

    version: str
    # EXACT version string, sourced from the real library (rule #9) — never hand-typed.

    config_id: str
    # Short human label for this arm's config ("stock", "parseq", "svtrv2") — for the
    # report, where a bare hash is unreadable.

    version_source: str | None = None
    # Dotted module path whose `__version__` IS `version`, cross-checked by the reader's
    # own test. Same contract as `Runner.version_source`; None is reserved for test doubles.

    @abstractmethod
    def read(self, crop: Image.Image) -> str:
        """Read one preprocessed crop and return the string. That is the whole interface.

        The crop arrives already padded, height-normalized and converted (see the pinned
        constants). An implementation must NOT re-crop, re-scale, or otherwise undo that —
        it is what makes two readers' numbers comparable. Return "" for "nothing readable
        here"; that is scored as an omission, which is a real answer, not a failure.
        """
        ...

    @abstractmethod
    def config(self) -> dict[str, object]:
        """Every knob that changes what this reader outputs, as a flat dict (D-13.5).

        Abstract for the same reason `Runner.config()` is: a default `{}` would let a
        reader that declares nothing hash identically to every other config of the same
        model — the exact collision the config identity exists to stop.
        """
        ...

    def config_hash(self) -> str:
        """Stable digest of `config()`. Do NOT override — see `contract.config_digest`."""
        return config_digest(self.config())


def crop_for_reading(image: Image.Image, bbox: BBox) -> Image.Image:
    """Cut `bbox` out of `image` and apply THE pinned preprocessing. No overrides.

    The signature is `(image, bbox)` and nothing else on purpose: there is no argument by
    which a caller — or a reader — could give one model a gentler crop than another. Every
    reader in the arm sees pixels produced by exactly this function.

    Steps, in order: pad by `CROP_PAD_FRAC` of the box height on all four sides; round the
    padded box OUTWARD (floor the mins, ceil the maxes) so padding is never silently eaten
    by truncation; clamp to the image bounds; crop; convert to `CROP_MODE`; resize to
    `CROP_HEIGHT` with the aspect ratio preserved, width floored at 1px so a degenerate box
    still yields a valid image rather than raising.

    A box that lies entirely outside the image, or is degenerate, still returns a valid
    1px-wide crop; deciding whether such a box should exist is the ground truth's job, not
    this function's, and raising here would abort a whole run over one bad row.
    """
    x0, y0, x1, y1 = bbox
    pad = (y1 - y0) * CROP_PAD_FRAC

    left = max(0, math.floor(x0 - pad))
    top = max(0, math.floor(y0 - pad))
    right = min(image.width, math.ceil(x1 + pad))
    bottom = min(image.height, math.ceil(y1 + pad))

    # A box off the edge (or inverted) can collapse to zero area after clamping; keep at
    # least one pixel so `resize` has something to work with.
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)

    crop = image.crop((left, top, right, bottom)).convert(CROP_MODE)
    width = max(1, round(crop.width * CROP_HEIGHT / crop.height))
    return crop.resize((width, CROP_HEIGHT), CROP_RESAMPLE)


def _crop_cost(image_ref: ImageRef, crop: Image.Image, reader: Reader) -> float:
    """Cost of ONE `reader.read` call, priced on the crop it was actually handed.

    Reuses the existing `estimate_cost` — no pricing logic lives here — but bills per CROP,
    not per image, because that is what this arm sends: a 30-token frame is thirty calls to
    a cloud reader, and each carries 48px-tall crop pixels rather than the whole frame.
    Pricing it once per image would understate a flat-per-request rule by ~30x and overstate
    a per-megapixel one by more.
    """
    return estimate_cost(
        dataclasses.replace(image_ref, w=crop.width, h=crop.height), reader.read
    )


def read_image(
    image_path: str,
    gt_tokens: Sequence[GTToken],
    reader: Reader,
    *,
    allowlist: set[str],
    image_ref: ImageRef,
    control_boxes: Iterable[BBox] = (),
) -> dict | None:
    """Read every GT box (and every control box) on one image and score the strings.

    Mapping onto the two axes, which are never collapsed into one accuracy number:
      - reader returns text on a GT box  -> a MATCH; whether it is *right* is `score()`'s
        call, and a wrong read on a KEEP token is a false redaction (the headline metric).
      - reader returns empty on a GT box -> an OMISSION (the Found axis).
      - reader returns text on a CONTROL box -> a HALLUCINATION (the Added axis).

    `control_boxes` are boxes known to contain no text. They exist because a reader handed
    only GT boxes can never invent a *location*, so `added_count` would be structurally zero
    and the arm would have no hallucination floor at all. plan.md requires every generative
    arm to run its negative control before its accuracy numbers are reported; for this arm,
    control boxes are the only way to run one.

    They are accepted ONLY on an image with zero GT tokens — a confirmed-blank frame — and
    passing them alongside GT tokens raises. On a text-bearing frame, "this region has no
    text" is not something anyone has confirmed: the GT records where text IS, never where
    it is absent, so a control box could land on unannotated glyphs and a CORRECT read would
    be scored as a hallucination — an invented number on the Added axis, the dangerous one.
    The end-to-end path cannot make that mistake because IoU clears it; here there is no
    matcher to catch it, so the input is refused instead. The cost of the strict rule is
    real and worth stating in any write-up: the blank control set is all `ct_axial`, so the
    reader hallucination floor is measured on that stratum only.

    Raises:
        ValueError: `control_boxes` given for an image that also has GT tokens.

    Returns None when the image yields zero crops (no GT tokens and no control boxes): a
    blank frame with nothing to read contributes zero rows rather than a row of zeros,
    which would otherwise dilute every rate with images the reader never saw. This mirrors
    `load_gt()`, where a blank image is ABSENT from the ground truth, not present-and-empty.

    Timing: `elapsed` sums ONLY the `reader.read` calls — cropping, scoring and file I/O
    are excluded, the same rule `run_harness` applies to `run_func`. Cost comes from the
    existing `estimate_cost`, so a cloud reader tagged `@priced(...)` is costed with no new
    pricing logic here.
    """
    boxes = list(control_boxes)
    if boxes and gt_tokens:
        raise ValueError(
            f"read_image(): {len(boxes)} control box(es) passed for an image that has "
            f"{len(gt_tokens)} GT tokens. Control boxes are only valid on a confirmed-blank "
            "frame: on a text-bearing image nobody has confirmed that a given region is "
            "empty, so a correct read there would be scored as a hallucination."
        )
    if not gt_tokens and not boxes:
        return None

    matches: list[Match] = []
    omissions: list[GTToken] = []
    hallucinations: list[OCRWord] = []
    elapsed = 0.0
    cost = 0.0

    with Image.open(image_path) as image:
        image.load()

        for token in gt_tokens:
            crop = crop_for_reading(image, token.bbox)
            t0 = perf_counter()
            text = reader.read(crop)  # the ONLY model call, and the ONLY thing timed
            elapsed += perf_counter() - t0
            cost += _crop_cost(image_ref, crop, reader)
            if normalize(text):
                # The reader's raw string, un-normalized — `normalize()` is applied at
                # scoring time, exactly as `matching.py` leaves an engine's text alone.
                # `iou=None`: the box was given, so there is no overlap to report.
                word = OCRWord(text=text, bbox=token.bbox, confidence=None)
                matches.append(Match(gt=token, word=word, iou=None))
            else:
                omissions.append(token)

        for box in boxes:
            crop = crop_for_reading(image, box)
            t0 = perf_counter()
            text = reader.read(crop)
            elapsed += perf_counter() - t0
            cost += _crop_cost(image_ref, crop, reader)
            if normalize(text):
                hallucinations.append(OCRWord(text=text, bbox=box, confidence=None))

    result = MatchResult(
        matches=matches,
        omissions=omissions,
        hallucinations=hallucinations,
        box_free=False,  # boxes exist on every word here — they are the GT's
        iou_thr=None,  # no matcher ran; detection metrics are undefined for this arm
    )
    return score(result, allowlist, elapsed=elapsed, cost=cost)


def run_reading(
    images: Sequence[ImageRef],
    ground_truth: Mapping[str, list[GTToken]],
    reader: Reader,
    *,
    allowlist: set[str],
    out_path: str | Path,
    control_boxes: Mapping[str, Sequence[BBox]] | None = None,
) -> dict:
    """Run the reading arm over `images`, write the aggregate to `out_path`, return it.

    The mirror of `run_harness` for readers: same authoritative stamping (rule #7 —
    `stratum`/`modality`/`vendor` come from the `ImageRef`, never re-derived), same
    `aggregate()`, same identity guard. `images` is the authoritative set; an image absent
    from `ground_truth` has zero GT tokens, which is the confirmed-blank control.

    `control_boxes` maps `image_id` to boxes known to contain no text — valid only for an
    image with zero GT tokens, i.e. a confirmed-blank frame; anything else raises (see
    `read_image`). Images that end up with nothing to read at all are skipped, so `n_images`
    counts images the reader actually saw.

    The return value is PHI-free — counts, rates, latencies, identities — and is what gets
    written to `out_path`. It is tagged `arm="reader"` because these numbers come from an
    oracle-detector condition and must be reported in their own table, never blended into
    the end-to-end ranking.

    Raises:
        ValueError: propagated from `aggregate()` if rows carry more than one identity —
            i.e. someone fed two different reader configs to one call (D-13.5).
    """
    controls = control_boxes or {}
    rows = []
    for img in images:
        row = read_image(
            img.path,
            ground_truth.get(img.id, []),
            reader,
            allowlist=allowlist,
            image_ref=img,
            control_boxes=controls.get(img.id, ()),
        )
        if row is None:
            continue  # nothing to read on this image; see read_image's docstring

        # Authoritative stamp (rule #7), identical to run_harness's.
        row["stratum"] = img.stratum
        row["modality"] = img.modality
        row["vendor"] = img.vendor
        row["image_id"] = img.id
        row["model_name"] = reader.model_name
        row["version"] = reader.version
        row["config_id"] = reader.config_id
        # The ARM's identity, not the reader's alone: it covers the crop pipeline too, so
        # the same reader fed different crops cannot merge with itself (see arm_config_hash).
        row["config_hash"] = arm_config_hash(reader)
        # No verifier on this arm: the reader IS the read. Present and None so the row
        # shape matches the end-to-end path's and `aggregate()`'s identity tuple is filled.
        row["verifier_elapsed"] = None
        row["verifier_model_name"] = None
        row["verifier_version"] = None
        rows.append(row)

    # The caveats travel WITH the numbers. `aggregate()` keeps neither `iou_thr` nor
    # `box_free` (they are per-image row fields), so without these keys the written artifact
    # would be indistinguishable from an end-to-end result: same `found_count`,
    # `omission_count`, `false_redaction_rate` key names, no sign that the boxes were free.
    # Anything rendering this file can see, in the file itself, that detection was an oracle.
    result = {
        "arm": "reader",
        "oracle_detector": True,
        "detection_metrics": "undefined — GT boxes were supplied; no detector ran",
        "iou_thr": None,
        "crop_preprocessing": crop_spec(),
        **aggregate(rows),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
