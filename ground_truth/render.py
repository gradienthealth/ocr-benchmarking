#!/usr/bin/env python3
"""Deterministic DICOM -> PNG renderer for the middle frame (Phase 10b). 🔴 PHI-TOUCHING.

>>> RUN THIS YOURSELF. Never let Claude read the PNGs or the back-map. <<<
The PNGs contain burned-in patient identifiers. View them in a STANDALONE image viewer
(OS file browser, Preview, eog, feh) — NOT in the Claude-connected IDE, or the pixels leak
back into an agent that is not covered by a BAA (CLAUDE.md §0). This data is pseudonymized
PHI, not de-identified.

WHAT THIS WRITES — three artifacts, exactly one of which is safe to show Claude
------------------------------------------------------------------------------------
  1. <out-dir>/<image_id>.png          PHI (burned-in text). Gitignored, Read-denied.
  2. <out-dir>/render_manifest.csv     PHI-FREE: image_id,frame_idx,w,h,sha256,fallback_used.
                                       Hashes and integers only — no UID, no tag value, no
                                       token text. THIS is the one you may paste to Claude.
  3. --backmap CSV                     image_id -> series_uid + SOPInstanceUID (+ the id
                                       inputs). PHI-adjacent (linkable identifiers):
                                       gitignored, hook-denied, never shown to Claude.
                                       10e needs it to join vendor/stratum/modality.

HOW TO RUN IT
------------------------------------------------------------------------------------
Prereqs: the series are ALREADY on local disk (this script never downloads and never
deletes an input). Pinned deps are already installed: pydicom==3.0.1, Pillow==11.1.0,
numpy==2.2.1, plus the pylibjpeg/gdcm decoder plugins for compressed transfer syntaxes.

You generate the input list yourself (D-10.8: this script never reads manifest.csv).
Keep it under ground_truth/ so it is gitignored + hook-denied like the back-map, e.g.
ground_truth/render_inputs.csv — header exactly `series_uid,path,frame_idx`:

    series_uid,path,frame_idx
    1.2.840.<uid-a>,data/series/1.2.840.<uid-a>.tar,42
    1.2.840.<uid-b>,data/series/1.2.840.<uid-b>.tar,
    1.2.840.<uid-c>,data/loose/<uid-c>/,7

`path` may be a local .tar (members `instances/*.dcm`), a directory of .dcm files, or a
single .dcm file. An empty `frame_idx` means "no request — use n // 2".

    .venv/bin/python -m ground_truth.render --inputs ground_truth/render_inputs.csv

Optional: --out-dir renders/gt (default) · --manifest-out <out-dir>/render_manifest.csv ·
--backmap ground_truth/render_backmap.csv · --id-len 8.

stdout is PHI-free by construction: counts, progress, output DIRECTORY paths, and
exception TYPE names only. Never a UID, a tag value, a tar member name, or a traceback.
Re-running is safe and idempotent: same inputs -> byte-identical PNGs and identical
sha256s (same machine/zlib), and the back-map recognizes an already-rendered image
instead of treating it as a collision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import os
import tarfile
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, pixel_array

# --- Output shapes -----------------------------------------------------------------

# The PHI-free manifest. Exactly these columns, in this order (plan.md 10b / Phase 10
# decisions block). `frame_idx` here is the frame ACTUALLY rendered, not the one that was
# requested — see _select_frame(). 10e and harness.harness.ImageRef MUST take `frame_idx`
# from this file and NEVER re-join manifest.csv for it: there is exactly one PNG per
# image_id, and this row describes that PNG.
#
# READ THIS BEFORE USING `frame_idx` FOR ANYTHING BUT PROVENANCE: it is an index into the
# pinned SELECTION AXIS, which is instances for a multi-instance series and frames for a
# single multi-frame instance (see _select_frame). So `frame_idx = 2` on a multi-instance
# series means "the 3rd instance in (InstanceNumber, SOPInstanceUID) order, whose only
# frame", NOT "frame 2 of some instance". Re-deriving pixels from (SOPInstanceUID,
# frame_idx) therefore needs this same axis rule plus the back-map — which is exactly why
# nothing downstream should re-derive pixels at all: score the PNG this row describes.
MANIFEST_COLUMNS: tuple[str, ...] = ("image_id", "frame_idx", "w", "h", "sha256", "fallback_used")

# The back-map. PHI-adjacent: it is the only link from a PHI-free image_id back to the
# identifiers. `sop_instance_uid` + `src_frame_idx` are the exact image_id preimage inputs
# and `id_len` the truncation length, which is what lets a rerun tell "already rendered"
# apart from "the hash formula changed under me" (see _check_ids).
BACKMAP_COLUMNS: tuple[str, ...] = (
    "image_id",
    "series_uid",
    "sop_instance_uid",
    "src_frame_idx",
    "id_len",
)

INPUT_COLUMNS: frozenset[str] = frozenset({"series_uid", "path", "frame_idx"})

DEFAULT_OUT_DIR = "renders/gt"
DEFAULT_BACKMAP = "ground_truth/render_backmap.csv"
MANIFEST_NAME = "render_manifest.csv"

# D-10.2 (resolved 2026-08-05): 8 hex chars of sha256 over `f"{SOPInstanceUID}|{frame_idx}"`.
# Short enough for readable filenames, so collisions are unlikely but POSSIBLE — hence
# _check_ids() raises instead of silently merging two images' ground truth.
DEFAULT_ID_LEN = 8
FULL_ID_LEN = 64  # the untruncated sha256, carried only so a collision can name both ids

# Determinism (plan.md §0.5): Pillow save parameters are pinned so the PNG bytes are a
# pure function of the pixels. No `dpi` (would emit pHYs), no pnginfo/text chunks, no
# timestamps, fixed compression. Byte-identity is guaranteed for a fixed environment —
# a different zlib build can compress the same pixels differently.
PNG_SAVE_KWARGS: dict[str, object] = {"format": "PNG", "optimize": False, "compress_level": 6}

_PROGRESS_EVERY = 10


# --- Errors ------------------------------------------------------------------------
# Every message below is PHI-free: ids are hashes, counts are counts, and no branch
# interpolates a UID, a path, a date, or a pixel value (CLAUDE.md §7 — PHI never to
# stdout). Three format tags (PhotometricInterpretation, SamplesPerPixel, VOILUTFunction)
# DO appear: all three are closed-enumeration display metadata, not identifiers, and
# without them an unsupported image is undebuggable. Nothing else from the dataset may be
# interpolated here — in particular no window value, which is a measurement of the pixels.


class RenderError(Exception):
    """Base class for this module's failures."""


class InputListError(RenderError):
    """The --inputs CSV is malformed. Fatal: nothing is rendered."""


class IdCollisionError(RenderError):
    """Two distinct images hashed to the same image_id, or the id formula changed."""


class NoInstancesError(RenderError):
    """An input path contained no readable .dcm instance."""


class AmbiguousFrameAxisError(RenderError):
    """A multi-instance series whose selected instance is itself multi-frame.

    Resolved 2026-08-05: raise. The requested index is a manifest index over the series'
    instances, so with two candidate axes its meaning is genuinely undefined; guessing
    would bury a provenance lie. The row is skipped and counted, the run continues.
    """


class UnsupportedPixelFormatError(RenderError):
    """Missing PhotometricInterpretation or an unexpected sample count. Fail loud."""


class UnsupportedVOIError(RenderError):
    """D-10.1a: a VOI LUT Sequence, or a VOILUTFunction this policy refuses to guess at.

    Resolved 2026-08-06 from `census_out/tag_census.md` (384 series sampled): a VOI LUT
    Sequence appears on 1 image and a non-LINEAR VOILUTFunction (SIGMOID) on 1. Rather
    than carry two extra mapping functions that no test population can exercise, both
    raise, the row is skipped and counted, and the run continues. Cost is at most ~2
    images of 966; the benefit is that ONE mapping produced every scored pixel.
    """


class WindowTagError(RenderError):
    """Window Center/Width present but unusable (half a pair, non-numeric, width < 1).

    Not the same as "absent": absent is the legitimate D-10.1c fallback below. A broken
    pair is ambiguous, so it fails loud rather than silently falling through to min/max.
    """


# --- D-10.1: the windowing policy (RESOLVED 2026-08-06) ------------------------------
# Settled from `census_out/tag_census.md` (40 series/stratum, 384 sampled). The census
# numbers that drove each branch are recorded beside it, because the policy is only
# defensible with them: an engine or dataset change re-opens the decision.
#
# Scope: this function only ever sees MONOCHROME1/2 frames. Colour (SamplesPerPixel = 3,
# 140/384 sampled) never reaches it — to_rgb8() routes colour straight to the 8-bit RGB
# path, which is correct: VOI LUT / windowing are defined for grayscale only.


def voi_to_uint8(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Map one raw grayscale frame to uint8. D-10.1, resolved 2026-08-06.

    Order is fixed and matches DICOM's grayscale pipeline (PS3.3 C.11):

      1. D-10.1a  raise on a VOI LUT Sequence or a non-LINEAR VOILUTFunction.
      2. D-10.1b-iii  modality LUT / rescale FIRST. Window Center/Width are defined in
         the modality LUT's OUTPUT space, so windowing before rescale is simply wrong.
      3. D-10.1b  Window Center/Width present -> the PS3.3 C.11.2.1.2 LINEAR function,
         evaluated straight to 0-255, taking index 0 when multi-valued.
      4. D-10.1c  neither present -> per-image min/max stretch.

    Deterministic: every branch is a pure function of (pixels, tags), so a rerun on the
    same instance reproduces the same PNG bytes.
    """
    # D-10.1a. `.get` rather than hasattr: an empty sequence is not a usable LUT.
    if ds.get("VOILUTSequence"):
        raise UnsupportedVOIError("VOI LUT Sequence present: D-10.1a refuses to render it")
    voi_func = str(getattr(ds, "VOILUTFunction", "") or "").strip().upper()
    if voi_func not in ("", "LINEAR"):
        # SIGMOID / LINEAR_EXACT would map one or two images with a different function
        # than the other ~964, which is exactly the cross-engine comparability problem.
        raise UnsupportedVOIError(f"VOILUTFunction={voi_func!r} is not handled (D-10.1a)")

    # D-10.1b-iii. A no-op when the dataset has neither ModalityLUTSequence nor
    # RescaleSlope/Intercept (pydicom returns `arr` unchanged), so it is safe to call
    # unconditionally. It is NOT a no-op for CT: the census found rescale on 37/37
    # ct_axial and 36/36 ct_scout, plus 27 signed (PixelRepresentation = 1) images.
    values = np.asarray(apply_modality_lut(arr, ds), dtype=np.float64)

    window = _window_center_width(ds)
    if window is None:
        return _minmax_uint8(values)
    return _linear_window_uint8(values, *window)


def _window_center_width(ds: pydicom.Dataset) -> tuple[float, float] | None:
    """Return (center, width) at index 0, or None when neither tag is present.

    D-10.1b-i (multi-valued windows, 51/384 sampled — 23/36 ct_scout, 15/37 ct_axial):
    take index 0. It is pydicom's own default, DICOM treats the values as alternative
    presentations with the first as the conventional default, and — the reason that
    matters here — it is a GLOBAL rule, so no per-image record of "which window fired"
    is needed and the frozen manifest keeps its 6 columns. A named lookup through
    WindowCenterWidthExplanation would be data-dependent and the tag is vendor free text
    that is often absent even when the window is multi-valued.
    """
    has_center = "WindowCenter" in ds
    has_width = "WindowWidth" in ds
    if not has_center and not has_width:
        return None
    if has_center != has_width:
        raise WindowTagError("only one of WindowCenter/WindowWidth is present")
    try:
        center = float(_at_index_0(ds["WindowCenter"]))
        width = float(_at_index_0(ds["WindowWidth"]))
    except (TypeError, ValueError, IndexError) as exc:
        # Type name only: a malformed DS value could be anything, including tag content.
        raise WindowTagError(f"WindowCenter/Width is not numeric ({type(exc).__name__})") from None
    if width < 1.0:
        # PS3.3 C.11.2.1.2 requires WindowWidth >= 1 for LINEAR; below that the mapping
        # is undefined, so this is a fail-loud case rather than a clamp.
        raise WindowTagError("WindowWidth < 1 is undefined for a LINEAR window")
    return center, width


def _at_index_0(elem: pydicom.DataElement) -> object:
    """Index 0 of a multi-valued element, or the value itself when VM <= 1.

    Branching on `elem.VM` rather than `isinstance(value, list)` because pydicom wraps
    multi-valued DS elements in `MultiValue`, which is a MutableSequence and NOT a list
    subclass — an isinstance check silently misses every multi-valued window and then
    fails on `float(MultiValue)`. This mirrors pydicom's own `apply_windowing`.
    """
    return elem.value[0] if elem.VM > 1 else elem.value


def _linear_window_uint8(values: np.ndarray, center: float, width: float) -> np.ndarray:
    """The PS3.3 C.11.2.1.2 LINEAR function evaluated with y_min = 0, y_max = 255.

    D-10.1b-ii: written out here rather than calling `pydicom.pixels.apply_voi_lut`,
    because that function's output range is NOT 0-255 — it is y_min..y_max derived from
    BitsStored/PixelRepresentation and then shifted by RescaleSlope/Intercept
    (pydicom/pixels/processing.py). For a typical CT (BitsStored 16, unsigned,
    RescaleIntercept -1024) that is -1024..64511, so clipping it into 8 bits yields a
    near-black frame with the burned-in text gone — the failure
    `scripts/diagnose_black_render.py` exists to diagnose. Evaluating the standard
    formula directly into 0-255 removes that trap and stays hand-checkable in a test.

    Clipping is the right behaviour for this project rather than a compromise: burned-in
    overlay text sits at a pixel extreme, so it saturates to 255 against a background
    that saturates to 0 — maximum contrast for OCR.
    """
    lo = center - 0.5 - (width - 1.0) / 2.0
    hi = center - 0.5 + (width - 1.0) / 2.0
    out = np.empty(values.shape, dtype=np.float64)
    below = values <= lo
    above = values > hi
    inside = ~(below | above)
    out[below] = 0.0
    out[above] = 255.0
    # width == 1 makes `inside` empty (lo == hi), so the (width - 1) division is never
    # evaluated on a real element. Guarded by test_window_width_one_is_a_hard_threshold.
    if inside.any():
        out[inside] = ((values[inside] - (center - 0.5)) / (width - 1.0) + 0.5) * 255.0
    return clip_round_uint8(out)


def _minmax_uint8(values: np.ndarray) -> np.ndarray:
    """D-10.1c: per-image min/max stretch, for frames carrying no windowing tags.

    Chosen 2026-08-06. The census puts 135/384 in the neither-tag bucket, but colour
    never reaches this function, so the population that actually lands here is led by
    mg_tomo: 38/40 neither, all SamplesPerPixel = 1, BitsStored 10 or 12. Pass-through
    would truncate those to 8 bits and full-range scaling by 2**BitsStored - 1 renders
    dark when the data occupies part of its nominal range; min/max reliably produces a
    visible frame at any bit depth.

    Known cost, accepted: one hot or dead pixel (or a PixelPaddingValue) sets an endpoint
    and compresses real content. That does not threaten legibility here — burned-in text
    IS the bright outlier, so it lands at 255 either way. A percentile stretch would trade
    that for a tunable baked into a frozen artifact plus a much wider degenerate case
    (p_lo == p_hi whenever one value holds most of the frame), so it is deliberately not
    used unless a rendered histogram shows it is needed.

    max == min (a uniform frame) has no range to stretch, so it returns all zeros rather
    than dividing by zero. Black is the honest render of a frame with no contrast.
    """
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return clip_round_uint8((values - lo) * (255.0 / (hi - lo)))


# `voi_to_uint8` is injected rather than called directly so the policy stays swappable and
# unit-testable in isolation, and so the surrounding pixel mechanics (MONOCHROME1
# inversion, YBR->RGB, forced RGB) can be tested without going through a window.
VOIFunc = Callable[[np.ndarray, pydicom.Dataset], np.ndarray]


def clip_round_uint8(arr: np.ndarray) -> np.ndarray:
    """Cast a float array to uint8 without wrapping. Any D-10.1 mapping ends here.

    `(arr * 255).astype(np.uint8)` — what the provisional scripts do — wraps 256.0 to 0
    and -1.0 to 255, which turns a bright burned-in token into a black one. NaN/inf are
    zeroed first, then clipped, then rounded with np.rint (round-half-to-even, so it is
    deterministic rather than platform-dependent).
    """
    return np.rint(np.clip(np.nan_to_num(np.asarray(arr, dtype=np.float64)), 0.0, 255.0)).astype(
        np.uint8
    )


# --- Input list --------------------------------------------------------------------


@dataclass(frozen=True)
class InputRow:
    """One requested render: a series, where its instances are, and the wanted index."""

    series_uid: str
    path: str
    requested_frame_idx: int | None


def read_input_list(path: str | Path) -> list[InputRow]:
    """Parse the human-authored input list (D-10.8: manifest.csv is never read here).

    D-10.8 (resolved 2026-08-05): render.py takes an explicit list of local paths + frame
    indices. It does not read manifest.csv, does not widen harness/manifest.py's
    `_COLUMN_MAP`, and needs no vendor/stratum at all — the series_uid -> vendor/stratum/
    modality join belongs to 10e, where it is actually used.

    Errors name the 1-based row number only, never the row's contents.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise InputListError("input list is empty (no header row)")
        header = {(name or "").strip() for name in reader.fieldnames}
        if header != INPUT_COLUMNS:
            raise InputListError(
                f"input list header must be exactly {sorted(INPUT_COLUMNS)}, "
                f"got {sorted(header)}"
            )
        rows: list[InputRow] = []
        seen: set[str] = set()
        for lineno, raw in enumerate(reader, start=2):
            series_uid = (raw.get("series_uid") or "").strip()
            src = (raw.get("path") or "").strip()
            if not series_uid or not src:
                raise InputListError(f"row {lineno}: series_uid and path must be non-empty")
            if series_uid in seen:
                raise InputListError(
                    f"row {lineno}: duplicate series_uid (one row per series; "
                    "two rows for one series would race for the same PNG)"
                )
            seen.add(series_uid)
            frame_raw = (raw.get("frame_idx") or "").strip()
            requested: int | None = None
            if frame_raw:
                try:
                    requested = int(frame_raw)
                except ValueError as exc:
                    raise InputListError(
                        f"row {lineno}: frame_idx must be a plain integer or empty"
                    ) from exc
            rows.append(InputRow(series_uid, src, requested))
    # Determinism: fix the work order before anything runs. Ordering does not change any
    # metric (D-1.1) but it does make progress output and error counts reproducible.
    rows.sort(key=lambda r: (r.series_uid, r.path))
    return rows


# --- Instance ordering + frame selection -------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One instance found under an input path, with its pinned sort key."""

    sort_key: tuple[int, int, str]
    sop_uid: str
    n_frames: int
    tar_member: str | None  # set iff the source is a .tar
    dcm_path: str | None  # set iff the source is a loose .dcm


_DCM_SUFFIXES = (".dcm", ".dicom")


def _is_dcm_name(name: str) -> bool:
    return name.lower().endswith(_DCM_SUFFIXES)


def _iter_sources(path: Path) -> Iterable[tuple[str | None, str | None, bytes | None]]:
    """Yield (tar_member, dcm_path, dicom_bytes) for every .dcm under `path`, sorted.

    `bytes` is None for an on-disk file so the header pass can stop reading at the pixel
    data instead of pulling a whole multi-hundred-MB series into memory; a tar member has
    to be read out to be parsed at all.

    `Path.glob` returns readdir order, which is not stable across machines, so every
    listing is sorted before use.
    """
    if path.is_dir():
        # Case-INSENSITIVE suffix match: `rglob("*.dcm")` is case-sensitive on Linux, so an
        # `A.DCM` series would render from a tar but come back "no instances" from a
        # directory. Same rule on both paths, or packaging silently changes the result.
        found = (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _DCM_SUFFIXES)
        for dcm in sorted(str(p) for p in found):
            yield None, dcm, None
        return
    if path.suffix.lower() == ".tar":
        with tarfile.open(path) as tar:
            members = sorted(
                (m for m in tar.getmembers() if m.isfile() and _is_dcm_name(m.name)),
                key=lambda m: m.name,
            )
            for member in members:
                handle = tar.extractfile(member)
                if handle is None:  # pragma: no cover - tarfile guards this via isfile()
                    continue
                yield member.name, None, handle.read()
        return
    yield None, str(path), None


def _silence_pydicom() -> None:
    """Stop pydicom from printing tag VALUES to the terminal. Process-global on purpose.

    MEASURED 2026-08-05 while running scripts/tag_census.py: this dataset contains UIDs
    longer than the 64 characters VR UI allows, and pydicom's validator warns with the
    offending value inline ("Invalid value for VR UI: '1.3.12...'"). That is a raw
    identifier on the terminal, one copy/paste from an agent's context — a disclosure
    outside the BAA (CLAUDE.md §0). The library's warning path is a PHI egress path.

    This lives in render_set(), not main(), so the guarantee travels with the function:
    anything that imports and calls the renderer gets it, not just the CLI.
    """
    warnings.simplefilter("ignore")
    logging.getLogger("pydicom").setLevel(logging.CRITICAL)
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE


def _read_header(blob: bytes) -> pydicom.Dataset:
    return pydicom.dcmread(io.BytesIO(blob), stop_before_pixels=True)


def _header_of(dcm_path: str | None, blob: bytes | None) -> pydicom.Dataset:
    if blob is not None:
        return _read_header(blob)
    return pydicom.dcmread(str(dcm_path), stop_before_pixels=True)


def _sort_key(hdr: pydicom.Dataset, sop_uid: str) -> tuple[int, int, str]:
    """Pin the instance order: InstanceNumber, tie-broken by SOPInstanceUID.

    "Index i of the series" is undefined without a fixed order, and an unpinned sort can
    select a different slice on a later run. Instances with no InstanceNumber sort LAST
    (flag 1) rather than as 0, so a missing tag cannot masquerade as the first slice.
    """
    raw = getattr(hdr, "InstanceNumber", None)
    if raw is None or str(raw).strip() == "":
        return (1, 0, sop_uid)
    return (0, int(raw), sop_uid)


def _collect_candidates(path: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for member, dcm_path, blob in _iter_sources(path):
        hdr = _header_of(dcm_path, blob)
        sop_uid = str(getattr(hdr, "SOPInstanceUID", "") or "")
        if not sop_uid:
            raise UnsupportedPixelFormatError("instance has no SOPInstanceUID")
        n_frames = int(getattr(hdr, "NumberOfFrames", 1) or 1)
        candidates.append(_Candidate(_sort_key(hdr, sop_uid), sop_uid, n_frames, member, dcm_path))
    if not candidates:
        raise NoInstancesError("no readable .dcm instance under this input path")
    candidates.sort(key=lambda c: c.sort_key)
    return candidates


@dataclass(frozen=True)
class _Plan:
    """A fully resolved render, decided from headers only — no pixels touched yet."""

    row: InputRow
    candidate: _Candidate
    frame_idx: int
    fallback_used: bool
    image_id: str
    full_id: str  # untruncated digest, so a collision message can name BOTH ids


def _select_frame(candidates: Sequence[_Candidate], requested: int | None) -> tuple[int, bool, str]:
    """Return (frame_idx, fallback_used, axis) for the middle-frame rule.

    Frame selection (resolved 2026-08-05): use the requested index when it is in range,
    else `n // 2`. NEVER frame 0 by default — frame 0 is typically a banner/title screen,
    so scoring it measures the wrong thing (CLAUDE.md §8). Three explicit shapes:

      * one instance with NumberOfFrames > 1  -> axis is frames,    n = NumberOfFrames
      * many single-frame instances           -> axis is instances, n = n_instances
      * one instance with one frame           -> the legitimate single-frame exception
        (dose/protocol reports): frame_idx = 0 because there IS only one frame.

    The manifest records what was actually rendered plus `fallback_used`, so substituting
    the middle for an out-of-range request is visible rather than a silent provenance lie.
    `fallback_used = 1` covers BOTH "the request was out of range" and "no index was
    requested" — it means "the requested index is not what you are looking at", and the
    frozen 6-column manifest has no room to distinguish the two. Over-flagging is the safe
    direction; a per-reason breakdown would need a 7th column and a re-render.
    """
    if len(candidates) == 1:
        n = candidates[0].n_frames
        axis = "frame" if n > 1 else "single"
    else:
        n = len(candidates)
        axis = "instance"
    if requested is not None and 0 <= requested < n:
        return requested, False, axis
    return n // 2, True, axis


def _plan_row(row: InputRow, id_len: int) -> _Plan:
    candidates = _collect_candidates(Path(row.path))
    frame_idx, fallback_used, axis = _select_frame(candidates, row.requested_frame_idx)
    chosen = candidates[frame_idx] if axis == "instance" else candidates[0]
    if axis == "instance" and chosen.n_frames > 1:
        raise AmbiguousFrameAxisError(
            "multi-instance series whose selected instance is multi-frame: the requested "
            "index could mean either axis (see AmbiguousFrameAxisError)"
        )
    return _Plan(
        row,
        chosen,
        frame_idx,
        fallback_used,
        compute_image_id(chosen.sop_uid, frame_idx, id_len),
        compute_image_id(chosen.sop_uid, frame_idx, FULL_ID_LEN),
    )


# --- image_id -----------------------------------------------------------------------


def compute_image_id(sop_instance_uid: str, frame_idx: int, id_len: int = DEFAULT_ID_LEN) -> str:
    """D-10.2 (resolved 2026-08-05): sha256 of `f"{SOPInstanceUID}|{frame_idx}"`, truncated.

    The preimage is exact and must never drift: UTF-8, a literal `|` separator so that
    ("UID1", 23) cannot collide with ("UID12", 3), and `frame_idx` as a plain decimal int
    with no padding. `frame_idx` is the value RECORDED in the manifest (whichever axis it
    came from), so the id and the manifest row always describe the same PNG.
    """
    digest = hashlib.sha256(f"{sop_instance_uid}|{frame_idx}".encode()).hexdigest()
    return digest[:id_len]


def _check_ids(
    plans: Sequence[_Plan],
    backmap: dict[str, tuple[str, str, int, int]],
    id_len: int,
) -> None:
    """Raise on any id ambiguity BEFORE a single byte is written.

    D-10.2 recovery note, repeated in the error text because it is counter-intuitive:
    re-running unchanged CANNOT clear a collision — sha256 is deterministic and
    reproduces it exactly. The only fix is a longer id (`--id-len 12`), which changes
    EVERY image_id, so it is safe only before `gt.csv` exists; afterwards it is a
    CLAUDE.md rule #8 invalidation of every number already reported.
    """
    # Guard the whole id SPACE first. A per-id lookup can never catch a mixed --id-len,
    # because an 8-char key cannot collide with a 12-char one: two disjoint batches would
    # otherwise pile two incompatible id spaces into one back-map, undetected.
    known_lens = {entry[3] for entry in backmap.values()}
    if known_lens and known_lens != {id_len}:
        raise IdCollisionError(
            f"back-map already holds id lengths {sorted(known_lens)} but this run uses "
            f"{id_len}: mixed id spaces are not comparable. Re-render the whole set with a "
            "single --id-len, or start a fresh back-map."
        )
    fix = (
        "Re-running unchanged cannot help (sha256 is deterministic and reproduces the "
        "same ids); rerun with --id-len 12. WARNING: changing --id-len changes EVERY "
        "image_id, so it is only safe before gt.csv exists — afterwards it invalidates "
        "every result already reported (CLAUDE.md rule #8)."
    )
    # 1. Collisions inside this render set.
    by_id: dict[str, tuple[str, int]] = {}
    full_by_id: dict[str, str] = {}
    for plan in plans:
        source = (plan.candidate.sop_uid, plan.frame_idx)
        prior = by_id.get(plan.image_id)
        if prior is None:
            by_id[plan.image_id] = source
            full_by_id[plan.image_id] = plan.full_id
        elif prior != source:
            # Name BOTH ids: the shared truncated id and the two full digests it came from
            # (all three are hashes, so all three are PHI-free).
            raise IdCollisionError(
                f"image_id collision within this render set: two different instances both "
                f"hash to '{plan.image_id}' at --id-len {id_len}. Full digests: "
                f"'{full_by_id[plan.image_id]}' and '{plan.full_id}'. {fix}"
            )
        else:
            raise IdCollisionError(
                f"two input rows resolve to the same instance and frame (image_id "
                f"'{plan.image_id}'). Fix the input list — one row per series."
            )
    # 2. Collisions against images rendered by an earlier run.
    for image_id, source in by_id.items():
        known = backmap.get(image_id)
        if known is None:
            continue
        _, known_sop, known_frame, _ = known
        if (known_sop, known_frame) != source:
            raise IdCollisionError(
                f"image_id collision against an already-rendered image: '{image_id}' is "
                f"already mapped to a different instance/frame. {fix}"
            )
    # 3. The same instance+frame recorded under a different id (formula drift).
    reverse = {source: image_id for image_id, source in by_id.items()}
    for image_id, (_, known_sop, known_frame, _) in backmap.items():
        current = reverse.get((known_sop, known_frame))
        if current is not None and current != image_id:
            raise IdCollisionError(
                f"the same instance+frame is recorded as '{image_id}' in the back-map but "
                f"hashes to '{current}' now — the id formula or --id-len changed. {fix}"
            )


# --- Pixels -------------------------------------------------------------------------


def _decode_frame(blob: bytes, frame_idx: int) -> np.ndarray:
    """Decode exactly one frame.

    `raw=False` is pinned explicitly (it is also the default) because that is what performs
    the YBR/YCbCr -> RGB conversion, including the case where a JPEG stream's component ids
    say RGB while the tag still says YBR — heuristics we do NOT want to reimplement. So the
    array reaching to_rgb8() is already RGB for colour input; converting again would corrupt
    every colour frame (caught by test_ybr_is_converted_to_rgb).

    Decoding one frame instead of the whole stack is also the minimum necessary
    (CLAUDE.md §7): only the frame we score is ever materialized.
    """
    return pixel_array(io.BytesIO(blob), index=frame_idx, raw=False)


def validate_pixel_tags(ds: pydicom.Dataset) -> tuple[str, int]:
    """Return (PhotometricInterpretation, SamplesPerPixel) or fail loud.

    Called BEFORE decoding as well as inside to_rgb8(): a missing
    PhotometricInterpretation must surface as this error rather than as whatever the
    decoder happens to raise three layers down.
    """
    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").strip().upper()
    if not photometric:
        raise UnsupportedPixelFormatError("no PhotometricInterpretation — cannot render safely")
    if photometric == "PALETTE COLOR":
        # Needs apply_color_lut and a palette policy of its own. Fail loud rather than
        # silently rendering the index values as if they were grayscale.
        raise UnsupportedPixelFormatError("PALETTE COLOR is not handled")
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if photometric.startswith("MONOCHROME"):
        if samples != 1:
            raise UnsupportedPixelFormatError(
                f"{photometric} with SamplesPerPixel={samples} is not a shape we handle"
            )
    elif samples != 3:
        raise UnsupportedPixelFormatError(
            f"{photometric} with SamplesPerPixel={samples} is not a shape we handle"
        )
    elif not (photometric == "RGB" or photometric.startswith("YBR")):
        raise UnsupportedPixelFormatError(f"unhandled PhotometricInterpretation: {photometric}")
    return photometric, samples


def to_rgb8(arr: np.ndarray, ds: pydicom.Dataset, voi_fn: VOIFunc) -> np.ndarray:
    """One decoded frame -> an (h, w, 3) uint8 RGB array.

    PNG mode (resolved 2026-08-05): force RGB for EVERY image, monochrome included. One
    uniform input type for every engine, runner and test fixture means nothing downstream
    branches on channel count. The MONOCHROME1 inversion happens BEFORE the channel
    expansion, and the expansion is the last step.
    """
    photometric, samples = validate_pixel_tags(ds)

    if photometric.startswith("MONOCHROME"):
        gray = voi_fn(arr, ds)  # D-10.1 lives entirely inside voi_fn
        if gray.dtype != np.uint8:
            raise UnsupportedPixelFormatError("the D-10.1 policy must return uint8")
        if photometric == "MONOCHROME1":
            # MONOCHROME1 means "0 = white": invert so text reads dark-on-light like every
            # other frame. Done on the 8-bit result, i.e. before channel expansion.
            gray = 255 - gray
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise UnsupportedPixelFormatError("colour frame did not decode to 3 channels")
    if arr.dtype != np.uint8:
        arr = clip_round_uint8(arr)
    return np.ascontiguousarray(arr)


def _save_png(rgb: np.ndarray, dest: Path) -> str:
    """Write the PNG atomically with pinned encoder settings; return its sha256.

    The hash is over the FILE bytes as written, which is what "byte-identical rerun"
    actually means. The temp-file + os.replace dance means a crash mid-encode cannot leave
    a half-written PNG that a later run would treat as done.
    """
    tmp = dest.with_name(dest.name + ".part")
    image = Image.fromarray(rgb, mode="RGB")  # fromarray -> empty .info, so no stray chunks
    try:
        with open(tmp, "wb") as fh:
            image.save(fh, **PNG_SAVE_KWARGS)
        os.replace(tmp, dest)
    finally:
        # A failed encode must not leave partial pixel data (which is PHI) lying around.
        tmp.unlink(missing_ok=True)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


# --- Sidecar files ------------------------------------------------------------------


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Rewrite a sidecar atomically.

    Both sidecars are rewritten whole on every run, so a crash mid-write would truncate
    them. That is survivable for the manifest (re-derivable from the PNGs) but NOT for the
    back-map: it is the only image_id -> SOPInstanceUID link in existence, and losing it
    orphans every PNG already rendered. Write to `.part`, then os.replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            # csv defaults to CRLF; pin LF so the file bytes are stable across platforms.
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_manifest(path: str | Path) -> dict[str, tuple[int, int, int, str, int]]:
    """Load an existing PHI-free manifest as image_id -> (frame_idx, w, h, sha256, fallback)."""
    out: dict[str, tuple[int, int, int, str, int]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != MANIFEST_COLUMNS:
            raise InputListError(f"existing manifest header is not {list(MANIFEST_COLUMNS)}")
        for raw in reader:
            out[raw["image_id"]] = (
                int(raw["frame_idx"]),
                int(raw["w"]),
                int(raw["h"]),
                raw["sha256"],
                int(raw["fallback_used"]),
            )
    return out


def read_backmap(path: str | Path) -> dict[str, tuple[str, str, int, int]]:
    """Load the back-map as image_id -> (series_uid, sop_instance_uid, frame_idx, id_len).

    Claude never reads the FILE; this loader exists so a rerun can tell "already rendered"
    apart from a genuine collision (see _check_ids).
    """
    out: dict[str, tuple[str, str, int, int]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != BACKMAP_COLUMNS:
            raise InputListError(f"existing back-map header is not {list(BACKMAP_COLUMNS)}")
        for raw in reader:
            out[raw["image_id"]] = (
                raw["series_uid"],
                raw["sop_instance_uid"],
                int(raw["src_frame_idx"]),
                int(raw["id_len"]),
            )
    return out


# --- Driver -------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderSummary:
    """PHI-free result of a run: counts and hashes only."""

    n_inputs: int
    n_rendered: int
    n_fallback: int
    errors: tuple[tuple[str, int], ...]
    manifest_path: str
    backmap_path: str
    out_dir: str


def render_set(
    inputs: Sequence[InputRow],
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    manifest_path: str | Path | None = None,
    backmap_path: str | Path = DEFAULT_BACKMAP,
    id_len: int = DEFAULT_ID_LEN,
    voi_fn: VOIFunc | None = None,
    progress: bool = False,
) -> RenderSummary:
    """Render every input's middle frame. Two passes, so the id check is a real gate.

    Pass 1 reads HEADERS ONLY (stop_before_pixels) and resolves the exact instance, frame
    and image_id for every row. The collision gate then runs over the whole set and
    against the back-map. Only then does pass 2 decode pixels — and it re-uses the tar
    member / file path recorded in pass 1, so a re-listing can never select a different
    slice than the one the id was computed from.
    """
    # Resolved at call time (not as a default argument) so the D-10.1 policy stays a single
    # swappable seam: tests inject one, and nothing silently renders without a policy.
    _silence_pydicom()  # before the first dcmread: a warning can carry a raw UID
    policy: VOIFunc = voi_fn if voi_fn is not None else voi_to_uint8
    out = Path(out_dir)
    manifest = Path(manifest_path) if manifest_path is not None else out / MANIFEST_NAME
    backmap_file = Path(backmap_path)
    if manifest.resolve() == backmap_file.resolve():
        # The whole PHI split rests on these being two files. Pointed at one path, the
        # back-map write would land UIDs at the path a human is told is safe to paste.
        raise InputListError(
            "--manifest-out and --backmap must be different files: the manifest is the "
            "PHI-free artifact and the back-map holds the identifiers"
        )
    errors: Counter[str] = Counter()

    plans: list[_Plan] = []
    for row in inputs:
        try:
            plans.append(_plan_row(row, id_len))
        except Exception as exc:  # noqa: BLE001 - type name only; the message may hold PHI
            errors[type(exc).__name__] += 1

    backmap = read_backmap(backmap_file)
    _check_ids(plans, backmap, id_len)  # fatal: raises before anything is written

    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)  # merge, so rendering in batches accumulates
    n_rendered = 0
    n_fallback = 0
    for done, plan in enumerate(plans, start=1):
        try:
            blob = _load_blob(plan)
            # Headers carry every tag the D-10.1 policy needs (window/VOI LUT/rescale), so
            # the tag pass stays pixel-free and only the one scored frame is decoded.
            hdr = _read_header(blob)
            validate_pixel_tags(hdr)  # fail loud on the tags before decoding any pixels
            frame_to_decode = plan.frame_idx if plan.candidate.n_frames > 1 else 0
            rgb = to_rgb8(_decode_frame(blob, frame_to_decode), hdr, policy)
            sha = _save_png(rgb, out / f"{plan.image_id}.png")
            rows[plan.image_id] = (
                plan.frame_idx,
                int(rgb.shape[1]),
                int(rgb.shape[0]),
                sha,
                int(plan.fallback_used),
            )
            backmap[plan.image_id] = (
                plan.row.series_uid,
                plan.candidate.sop_uid,
                plan.frame_idx,
                id_len,
            )
            n_rendered += 1
            n_fallback += int(plan.fallback_used)
        except Exception as exc:  # noqa: BLE001 - type name only; the message may hold PHI
            errors[type(exc).__name__] += 1
        if progress and done % _PROGRESS_EVERY == 0:
            print(f"  {done}/{len(plans)} done")

    # Rewrite both sidecars whole and sorted by image_id: append order depends on how the
    # work was batched, and the files should not.
    _write_csv(
        manifest,
        MANIFEST_COLUMNS,
        [(image_id, *rows[image_id]) for image_id in sorted(rows)],
    )
    _write_csv(
        backmap_file,
        BACKMAP_COLUMNS,
        [(image_id, *backmap[image_id]) for image_id in sorted(backmap)],
    )
    return RenderSummary(
        n_inputs=len(inputs),
        n_rendered=n_rendered,
        n_fallback=n_fallback,
        errors=tuple(sorted(errors.items())),
        manifest_path=str(manifest),
        backmap_path=str(backmap_file),
        out_dir=str(out),
    )


def _load_blob(plan: _Plan) -> bytes:
    """Re-read exactly the instance pass 1 chose — never a fresh directory listing."""
    if plan.candidate.dcm_path is not None:
        return Path(plan.candidate.dcm_path).read_bytes()
    with tarfile.open(plan.row.path) as tar:
        handle = tar.extractfile(plan.candidate.tar_member or "")
        if handle is None:
            raise NoInstancesError("tar member vanished between the header and pixel pass")
        return handle.read()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the middle frame of local DICOM series to deterministic PNGs.",
    )
    # Plain string paths, never argparse.FileType: FileType's own error messages echo the
    # path, and a series path contains a UID.
    parser.add_argument("--inputs", required=True, help="CSV: series_uid,path,frame_idx")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest-out", default=None, help=f"default: <out-dir>/{MANIFEST_NAME}")
    parser.add_argument("--backmap", default=DEFAULT_BACKMAP)
    parser.add_argument(
        "--id-len",
        type=int,
        default=DEFAULT_ID_LEN,
        help="hex chars of the image_id (D-10.2). Raising it changes EVERY image_id.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # pydicom logs offending VR values and warnings can quote tag content: silence both so
    # a malformed instance cannot print PHI to the terminal.
    logging.getLogger("pydicom").setLevel(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = read_input_list(args.inputs)
            print(f"inputs: {len(inputs)} series")
            summary = render_set(
                inputs,
                args.out_dir,
                manifest_path=args.manifest_out,
                backmap_path=args.backmap,
                id_len=args.id_len,
                progress=True,
            )
    except RenderError as exc:
        # RenderError messages are PHI-free by construction (see the Errors section).
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - never let a traceback render tag data
        print(f"FAILED: {type(exc).__name__} (message suppressed: may contain PHI)")
        return 1

    print(f"rendered {summary.n_rendered}/{summary.n_inputs}")
    print(f"middle-frame fallback used: {summary.n_fallback}")
    print(f"failed {summary.n_inputs - summary.n_rendered} {list(summary.errors)}")
    print(f"PNGs (PHI, standalone viewer only) -> {summary.out_dir}/")
    print(f"PHI-free manifest (safe to share)  -> {summary.manifest_path}")
    print(f"back-map (PHI-adjacent, never share, never let Claude read) -> {summary.backmap_path}")
    # Fail loud (plan.md §0.8): any row that did not render makes the exit code non-zero,
    # so a partial run can never be mistaken for a complete one by a wrapper script.
    return 0 if summary.n_rendered == summary.n_inputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
