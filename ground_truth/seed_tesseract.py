#!/usr/bin/env python3
"""Local-Tesseract SEED for the Phase-10d ground-truth review (Phase 10c). 🔴 PHI-TOUCHING.

>>> RUN THIS YOURSELF. Never let Claude read its output, the PNGs, or the text. <<<
This reads the Phase-10b renders, whose pixels carry burned-in patient identifiers, and
writes back the strings Tesseract read out of them. Every seed file is therefore PHI
(CLAUDE.md §0/§3). Open them in 10d's review UI, never in the Claude-connected IDE, or
the tokens leak into an agent that is not covered by a BAA. This data is pseudonymized
PHI, not de-identified.

WHAT THIS DOES
------------------------------------------------------------------------------------
Pre-fills boxes + strings so the human corrects a seed instead of drawing every box from
scratch. Tesseract only: it is LOCAL (no egress, no BAA needed), it is NEUTRAL — never a
candidate engine, so seeding cannot bias the bake-off — and its errors are obvious
garbage rather than plausible near-misses. There is no HTTP call anywhere in this file
and there must never be one; that would be PHI egress.

WHAT THIS WRITES — two artifacts, exactly one of which is safe to show Claude
------------------------------------------------------------------------------------
  1. <out-dir>/<image_id>.json   PHI (read-back tokens). Gitignored, Read-denied. D-10c.1.
  2. <out-dir>/../seed_summary.json
                                 PHI-FREE: provenance, counts, per-image box counts keyed
                                 by image_id (a hash), image dimensions, the confidence
                                 histogram, preflight results and {image_id,
                                 exception_type} failures. No token text, no filenames.
                                 THIS is the one you may paste to Claude. D-10c.8.

stdout is PHI-free by construction: counts, distributions, version strings, the pinned
config, output DIRECTORY paths, and exception TYPE names only — never a token, never a
message, never a traceback.

Setup (once — YOUR step: the sandbox cannot install system packages)
------------------------------------------------------------------------------------
    sudo apt-get install -y tesseract-ocr
    .venv/bin/pip install -e ".[tesseract]"        # the WRAPPER only (D-10c.5)

    # `apt` gives you the `standard` traineddata. `best` and `fast` are separate
    # downloads and produce DIFFERENT seeds, so the variant is a recorded human claim,
    # never a guess (--tessdata-variant is required):
    mkdir -p tessdata_best
    curl -L -o tessdata_best/eng.traineddata \\
      https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

Run
------------------------------------------------------------------------------------
    .venv/bin/python -m ground_truth.seed_tesseract \\
        --renders-dir renders/gt \\
        --manifest renders/gt/render_manifest.csv \\
        --out-dir ground_truth/seed \\
        --tessdata-dir tessdata_best \\
        --tessdata-variant best \\
        --on-existing skip

Re-running is safe: a preflight re-hashes every PNG before a single byte is written, and
`--on-existing skip` aborts rather than mixing two provenances into one seed directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import PIL
from PIL import Image

from ground_truth.render import MANIFEST_COLUMNS, RenderError, read_manifest

# --- Determinism (CLAUDE.md §7, rule #9) ---------------------------------------------
#
# Same renders -> byte-identical seed directory. What that rests on:
#   * `sorted(manifest)` iteration — the manifest, not a directory listing, is the
#     authority on which images get seeded (D-10c.6).
#   * TESSERACT_ENV applied to the Tesseract subprocess. OMP_THREAD_LIMIT=1 is a
#     DETERMINISM requirement, not a perf tweak (thread count changes the result);
#     LC_ALL/LC_NUMERIC=C keep confidence parsing locale-independent.
#   * json.dump(..., sort_keys=True) and atomic .part + os.replace writes.
#   * bbox = coord / UPSCALE_FACTOR with UPSCALE_FACTOR a power of two, so the division
#     is exact in binary and the float repr is stable.
#
# A bump to ANY of Tesseract, Leptonica, eng.traineddata, pytesseract or Pillow CHANGES
# THE SEED — the same class of invalidation as an OCR engine bump (CLAUDE.md rule #9).
# That is why each is version-recorded and the traineddata is hashed, in EVERY seed file.
#
# The one thing that can break byte-identity is a TIMEOUT: it is load-dependent, so the
# same renders can seed on one run and time out on the next. Recording failures (D-10c.9)
# is what makes that visible instead of silent.

# --- Pinned configuration (D-10c.4) ---------------------------------------------------
#
# D-10c.4: `--oem 1` LSTM only; `--psm 11` sparse text (burned-in overlays are scattered,
# not a page). `-c load_system_dawg=0 -c load_freq_dawg=0` kill dictionary correction so
# ID-shaped strings are NOT pulled toward English words — the whole point of this project.
# `thresholding_method=0` pins Otsu (Tesseract 5 also ships Sauvola and LeptonicaOtsu, and
# the default is not guaranteed stable across builds). `invert_threshold=0.7` pins the
# default explicitly: it governs light-on-dark inversion, which is the COMMON case for
# burned-in overlays, so it is the FIRST knob to revisit if the seed under-detects — and
# changing it RE-INVALIDATES EVERY SEED ALREADY WRITTEN. `--dpi 300` is a pinned CLAIM,
# not a measurement; it stays pinned even though the 2x upscale changes true effective
# DPI, and is never computed per image.
TESSERACT_CONFIG = (
    "--oem 1 --psm 11 -c load_system_dawg=0 -c load_freq_dawg=0 "
    "-c thresholding_method=0 -c invert_threshold=0.7 --dpi 300"
)
LANG = "eng"

# D-10c.4: the upscale is the ONE pixel operation 10c may perform, and only because a pure
# resize is EXACTLY INVERTIBLE — boxes come back as coord / UPSCALE_FACTOR, unrounded.
# NO other preprocessing (no threshold, no denoise, no contrast): 10b owns rendering
# (CLAUDE.md §7), and any non-invertible op would put the seed's boxes in a coordinate
# space that no longer describes the PNG the human reviews.
UPSCALE_FACTOR = 2
UPSCALE_FILTER_NAME = "PIL.Image.Resampling.LANCZOS"
COORD_SPACE = "original render pixels (x / upscale_factor, floats, unrounded)"

# D-10c.4: `level == 5` is image_to_data's WORD level; levels 1-4 (page/block/para/line)
# are structure, not tokens. Dropping empty-text rows is PARSING, not filtering.
ROW_FILTER = "level == 5 AND text.strip() != ''"
CONF_PARSE = "float(conf), fallback -1.0, kept as-is (never dropped)"
CONF_FALLBACK = -1.0

TESSERACT_ENV = {"LC_ALL": "C", "LC_NUMERIC": "C", "OMP_THREAD_LIMIT": "1"}
TIMEOUT_S = 60
WORD_LEVEL = 5

SEED_DIR_NAME = "seed"
SUMMARY_NAME = "seed_summary.json"

# D-10c.2b — RESOLVED: the settled variant is `best` (tessdata_best, LSTM float); accuracy
# over speed, since this is a one-time seeding run. The script still never GUESSES it: the
# three variants produce DIFFERENT seeds, `apt-get install tesseract-ocr` gives you
# `standard` rather than `best`, and the variant is a claim this code cannot verify. So it
# stays a required flag and a recorded human claim, cross-checked by the traineddata hash
# that goes into every seed file. Switching variants later re-invalidates every seed
# already written (CLAUDE.md rule #9).
TESSDATA_VARIANTS = ("best", "fast", "standard")

# The D-10c.4 provenance block, recorded IN EVERY SEED FILE. Every field is mandatory: a
# seed written with unknown provenance cannot be told apart later from one written under a
# different engine build, which is precisely the rule #9 invalidation this exists to catch.
PROVENANCE_FIELDS = (
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

# An image_id is a hash, so it is PHI-free and printable. A seed-dir stem that is NOT
# hash-shaped is an unknown string that could carry PHI, so it is only ever counted.
_HASH_STEM_RE = re.compile(r"^[0-9a-f]{8,64}$")

_HIST_BUCKETS = (
    "-1",
    "0-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90-100",
)


# --- Errors ---------------------------------------------------------------------------
# Every message raised below is PHI-free by construction: image_ids are hashes, counts are
# counts, and no branch interpolates a token string, a render filename, a pixel value or a
# DICOM tag. Nothing else from the dataset may ever be interpolated here.


class SeedError(Exception):
    """Base class for this module's failures."""


class TesseractMissingError(SeedError):
    """The `tesseract` binary is not on PATH. Actionable setup message, never a traceback."""


class PreflightError(SeedError):
    """The inputs are not in a state where seeding could be trusted. Nothing is written."""


class ProvenanceMismatchError(SeedError):
    """An existing seed was written under different provenance. ABORT — never skip past it."""


class AmbiguousRerunError(SeedError):
    """Existing seeds, no --on-existing, and no TTY to ask. Fail loud rather than guess."""


class BboxOutOfBoundsError(SeedError):
    """A converted bbox falls outside the render. The coordinate space is wrong somewhere."""


class ProvenanceIncompleteError(SeedError):
    """A provenance field is missing/None/empty. Refuse to write a seed of unknown origin."""


# --- The seed token -------------------------------------------------------------------

OCRFn = Callable[[Image.Image], Mapping[str, Sequence[object]]]


@dataclass(frozen=True)
class SeedToken:
    """One Tesseract word, in the harness's shared bbox vocabulary."""

    text: str
    # RAW. `harness.contract.normalize()` is FROZEN and applied only at MATCH time, so the
    # seed stores Tesseract's exact string: no normalize(), no strip-and-store, no
    # casefold. Cleaning here would hide a real read error behind a tidy-looking seed.

    bbox: tuple[float, float, float, float]
    # (x0, y0, x1, y1), axis-aligned, TOP-LEFT origin, in pixels of the ORIGINAL render —
    # identical to OCRWord/GTToken (harness/contract.py). Unrounded floats: both contract
    # bboxes are tuple[float, float, float, float], so unrounded is conformant (D-10c.4).

    confidence: float
    label: str
    # D-10c.2: ALWAYS "PHI". Tesseract cannot know a token's label, so the default takes
    # the conservative direction (redact rather than expose). Safe only because 10d shows
    # reviewed-state explicitly — which is why NOTHING here implies review happened.

    def to_json(self) -> dict[str, object]:
        x0, y0, x1, y1 = self.bbox
        return {
            "text": self.text,
            "bbox": [x0, y0, x1, y1],
            "confidence": self.confidence,
            "label": self.label,
        }


# --- Geometry + row parsing -----------------------------------------------------------


def bbox_from_row(
    left: int, top: int, width: int, height: int, upscale_factor: int
) -> tuple[float, float, float, float]:
    """image_to_data's (left, top, width, height) -> (x0, y0, x1, y1) in ORIGINAL px.

    D-10c.4: plain division, NEVER rounded. `upscale_factor` is a power of two so the
    quotient is exact in binary and the float repr is byte-stable across runs.
    """
    f = float(upscale_factor)
    return (left / f, top / f, (left + width) / f, (top + height) / f)


def tokens_from_data(
    data: Mapping[str, Sequence[object]], *, upscale_factor: int, w: int, h: int
) -> list[SeedToken]:
    """Convert one image_to_data DICT into seed tokens.

    D-10c.4: KEEP EVERY TOKEN — no confidence threshold, no minimum length, no alnum
    filter. The CONF=50/MINLEN=3 in `scripts/` existed because those scripts COUNTED
    detections; here the seed is what the human sees, so a dropped faint token means no
    box appears in 10d and that token silently never reaches gt.csv. Clutter is a DISPLAY
    problem (10d dims/toggles low-confidence boxes), never a reason to drop a row here.
    """
    tokens: list[SeedToken] = []
    levels = data["level"]
    texts = data["text"]
    confs = data["conf"]
    lefts, tops, widths, heights = data["left"], data["top"], data["width"], data["height"]
    for i in range(len(levels)):
        # ROW_FILTER. Level 5 is the word level; empty text is a structural row with no
        # string in it. Dropping those is parsing, not filtering.
        if int(levels[i]) != WORD_LEVEL:
            continue
        text = str(texts[i])
        if text.strip() == "":
            continue
        try:
            confidence = float(confs[i])
        except (TypeError, ValueError):
            # CONF_PARSE: a -1.0 fallback is RECORDED, never used to drop the row.
            confidence = CONF_FALLBACK
        bbox = bbox_from_row(
            int(lefts[i]), int(tops[i]), int(widths[i]), int(heights[i]), upscale_factor
        )
        x0, y0, x1, y1 = bbox
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            # D-10c.3: this is what w/h are FOR. An out-of-bounds box means the coordinate
            # space is wrong (wrong upscale factor, wrong image), and a silently clamped
            # box would put a wrong region into frozen gt.csv.
            raise BboxOutOfBoundsError(
                f"bbox falls outside the {w}x{h} render after dividing by "
                f"upscale_factor={upscale_factor}"
            )
        # Zero-width / zero-height boxes are KEPT (counted by count_degenerate, never
        # dropped) — same D-10c.4 reasoning: a degenerate box is still a token the human
        # must see, and dropping it would erase it from gt.csv without a trace.
        tokens.append(SeedToken(text=text, bbox=bbox, confidence=confidence, label="PHI"))
    return tokens


def count_degenerate(tokens: Sequence[SeedToken | Mapping[str, object]]) -> int:
    """How many boxes have zero width or zero height. Reported, never used to filter.

    Takes SeedTokens or their `to_json()` dicts: `run_seed` only ever holds the serialized
    form, and one definition of "degenerate" beats the same test written twice.
    """
    n = 0
    for token in tokens:
        bbox = token.bbox if isinstance(token, SeedToken) else token["bbox"]
        x0, y0, x1, y1 = bbox  # type: ignore[misc]
        n += int(x0 == x1 or y0 == y1)
    return n


# --- Files ----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks (renders are large; never slurp them)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, obj: object) -> None:
    """Write JSON to `.part`, then os.replace.

    A killed run must leave only complete, valid files (D-10c.1): 10d opens these one at a
    time and a truncated seed would either crash it or, worse, present a partial token list
    as if it were the whole image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, sort_keys=True, ensure_ascii=True, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        # A half-written seed still contains PHI; do not leave one lying around.
        tmp.unlink(missing_ok=True)


# --- Tesseract ------------------------------------------------------------------------

_SETUP_HINT = (
    "install it first:\n"
    "    sudo apt-get install -y tesseract-ocr\n"
    "    .venv/bin/pip install -e \".[tesseract]\"\n"
    "then fetch the traineddata variant you intend to record (apt gives you 'standard'):\n"
    "    mkdir -p tessdata_best\n"
    "    curl -L -o tessdata_best/eng.traineddata \\\n"
    "      https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata\n"
    "and pass --tessdata-dir tessdata_best --tessdata-variant best"
)


def tesseract_binary_available() -> bool:
    """True iff the `tesseract` BINARY is on PATH (pip only ever carries the wrapper)."""
    return shutil.which("tesseract") is not None


def tesseract_version_block() -> dict[str, str]:
    """Capture Tesseract + Leptonica versions once per run, from `tesseract --version`.

    `pytesseract.get_tesseract_version()` returns the number only; the full stdout also
    carries the Leptonica line, and that output contains no PHI, so capturing it is fine.
    Once per RUN, not per image.
    """
    try:
        proc = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TesseractMissingError(
            f"the `tesseract` binary is not on PATH — {_SETUP_HINT}"
        ) from exc
    lines = [ln.strip() for ln in ((proc.stdout or "") + (proc.stderr or "")).splitlines()]
    lines = [ln for ln in lines if ln]
    leptonica = next((ln for ln in lines if "leptonica-" in ln), "")
    return {
        "tesseract_version": lines[0] if lines else "",
        "leptonica_version": leptonica,
    }


def traineddata_sha256(tessdata_dir: Path, lang: str = LANG) -> str:
    """Hash the eng.traineddata ACTUALLY loaded — the variant flag is only a claim.

    The traineddata is one of the five things whose bump changes every seed (rule #9), and
    it is the only one with no version string of its own, so it is hashed instead.
    """
    path = Path(tessdata_dir) / f"{lang}.traineddata"
    if not path.exists():
        raise PreflightError(
            f"no {lang}.traineddata under --tessdata-dir — {_SETUP_HINT}"
        )
    return sha256_file(path)


def check_provenance_complete(prov: Mapping[str, object]) -> None:
    """Fail loud rather than write a seed whose origin is partly unknown."""
    bad = [
        field
        for field in PROVENANCE_FIELDS
        if field not in prov
        or prov[field] is None
        or (isinstance(prov[field], str) and prov[field].strip() == "")
    ]
    if bad:
        raise ProvenanceIncompleteError(
            "provenance block is incomplete (missing/None/empty): "
            f"{sorted(bad)} — refusing to write a seed of unknown origin (rule #9)"
        )


def build_provenance(*, tessdata_dir: Path, tessdata_variant: str) -> dict[str, object]:
    """Assemble the D-10c.4 provenance block that goes into EVERY seed file."""
    if tessdata_variant not in TESSDATA_VARIANTS:
        raise PreflightError(
            f"--tessdata-variant must be one of {list(TESSDATA_VARIANTS)} (D-10c.2b: the "
            "three produce different seeds, so it is recorded, never guessed)"
        )
    try:
        pytesseract_version = importlib.metadata.version("pytesseract")
    except importlib.metadata.PackageNotFoundError as exc:
        # D-10c.5: `pytesseract` is a declared OPTIONAL extra (`.[tesseract]`), never a
        # core dependency. pip carries the WRAPPER only — the binary, Leptonica and the
        # traineddata are system-level, installed separately, and THOSE determine the seed.
        raise PreflightError(
            'pytesseract is not installed — run `.venv/bin/pip install -e ".[tesseract]"` '
            "(that installs the wrapper only; the tesseract binary is a separate step)"
        ) from exc
    prov: dict[str, object] = {
        # The PINNED base string. `--tessdata-dir` is appended at call time and recorded
        # separately, so this field stays comparable across machines with different paths.
        "tesseract_cmd_config": TESSERACT_CONFIG,
        "lang": LANG,
        "upscale_factor": UPSCALE_FACTOR,
        "upscale_filter": UPSCALE_FILTER_NAME,
        "coord_space": COORD_SPACE,
        "row_filter": ROW_FILTER,
        "conf_parse": CONF_PARSE,
        "env": dict(TESSERACT_ENV),
        "timeout_s": TIMEOUT_S,
        "pytesseract_version": pytesseract_version,
        "pillow_version": PIL.__version__,
        "tessdata_dir": str(tessdata_dir),
        "tessdata_variant": tessdata_variant,
        "traineddata_sha256": traineddata_sha256(Path(tessdata_dir)),
        **tesseract_version_block(),
    }
    check_provenance_complete(prov)
    return prov


def default_ocr_fn(tessdata_dir: Path) -> OCRFn:
    """The real Tesseract call. Imported LAZILY so this module imports without pytesseract.

    D-10c.5: `pytesseract` is a pinned optional extra (`tesseract = ["pytesseract==0.3.13"]`),
    never a core dependency, and pip installs the wrapper ONLY.
    """
    try:
        import pytesseract  # noqa: PLC0415 - lazy on purpose: see the docstring
    except ImportError as exc:
        raise PreflightError(
            'pytesseract is not installed — run `.venv/bin/pip install -e ".[tesseract]"`'
        ) from exc

    config = f"{TESSERACT_CONFIG} --tessdata-dir {shlex.quote(str(tessdata_dir))}"

    def _run(img: Image.Image) -> Mapping[str, Sequence[object]]:
        return pytesseract.image_to_data(
            img,
            lang=LANG,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=TIMEOUT_S,
        )

    return _run


@contextmanager
def tesseract_env() -> Iterator[None]:
    """Apply TESSERACT_ENV to the Tesseract subprocess, then restore os.environ.

    pytesseract shells out and the child inherits os.environ, so this is how the pinned
    env actually reaches Tesseract. OMP_THREAD_LIMIT=1 is a determinism requirement.
    Lives here, not only in main(), so the guarantee travels with any caller of run_seed().
    """
    saved = {key: os.environ.get(key) for key in TESSERACT_ENV}
    os.environ.update(TESSERACT_ENV)
    try:
        yield
    finally:
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


# --- Seeding one image ----------------------------------------------------------------


def seed_image(
    png_path: Path,
    *,
    image_id: str,
    w: int,
    h: int,
    provenance: Mapping[str, object],
    ocr_fn: OCRFn,
) -> dict[str, object]:
    """Read one render and return its seed record. Writes NOTHING (the caller writes).

    D-10c.3: `w`/`h` are kept in the record and are not convenience — they are what asserts
    every bbox is in-bounds, what verifies the upscaled buffer is exactly UPSCALE_FACTOR x
    the render, and what 10d uses to size its canvas.
    """
    with Image.open(png_path) as img:
        if (img.width, img.height) != (w, h):
            raise PreflightError(
                f"render dimensions {img.width}x{img.height} do not match the manifest's "
                f"{w}x{h} for image_id {image_id} — the manifest and the PNG disagree"
            )
        upscaled = img.resize((w * UPSCALE_FACTOR, h * UPSCALE_FACTOR), Image.Resampling.LANCZOS)
    if (upscaled.width, upscaled.height) != (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR):
        raise PreflightError(
            f"upscaled buffer is not exactly {UPSCALE_FACTOR}x the render for image_id "
            f"{image_id} — the coordinate round-trip would be wrong"
        )
    data = ocr_fn(upscaled)
    tokens = tokens_from_data(data, upscale_factor=UPSCALE_FACTOR, w=w, h=h)
    # D-10c.1: one JSON per image, and NOTHING that implies review state — no `reviewed`,
    # `accepted`, `status`, `edited`, `deferred`, no timestamp. A defaulted PHI label must
    # never be mistakable for one the human confirmed; 10d owns reviewed-state entirely.
    return {
        "image_id": image_id,
        "w": w,
        "h": h,
        "provenance": dict(provenance),
        "tokens": [t.to_json() for t in tokens],
    }


# --- Preflight (D-10c.6 / D-10c.7) ----------------------------------------------------


def preflight(
    *,
    renders_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    on_existing: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Gate the whole run BEFORE any byte is written. Every failure raises.

    D-10c.6: manifest-driven, plus a BIDIRECTIONAL disk sweep, plus a re-hash of every PNG.
    Running this over the whole set first means a stale or truncated render aborts the run
    instead of leaving a half-seeded directory whose files silently came from other pixels.
    """
    # D-10c.4: the provenance block is what makes a seed reproducible and what rule #9's
    # invalidation check compares against. A seed file written with an incomplete block has
    # UNKNOWN provenance, which is worse than no seed at all — so fail here, before any write,
    # rather than per-image partway through. `build_provenance` already guarantees this for the
    # CLI path; the check is repeated for callers that construct the block themselves.
    check_provenance_complete(provenance)

    # `read_manifest` returns {} for a MISSING file rather than raising, so check existence
    # ourselves — a missing manifest must never read as "zero images, nothing to do".
    if not manifest_path.exists():
        raise PreflightError(
            f"manifest not found: pass --manifest pointing at 10b's {list(MANIFEST_COLUMNS)} "
            "CSV (renders/gt/render_manifest.csv by default)"
        )
    try:
        # Reuse 10b's parser: it validates the header and keeps image_id/sha256 as str.
        # NEVER pandas here — it type-infers, and an all-digit 8-hex image_id like
        # "12345678" would be coerced to int64 and silently fail to join to the PNG stem.
        manifest = read_manifest(manifest_path)
    except RenderError as exc:
        # render.py's messages are PHI-free by construction (see its Errors section).
        raise PreflightError(f"manifest is unreadable: {exc}") from exc
    if not manifest:
        raise PreflightError("manifest has zero rows — nothing to seed; fix the manifest")
    if not renders_dir.exists():
        raise PreflightError("--renders-dir does not exist")

    manifest_ids = set(manifest)
    stems = {p.stem for p in renders_dir.glob("*.png")}
    missing = sorted(manifest_ids - stems)
    extra = stems - manifest_ids
    if missing or extra:
        parts = []
        if missing:
            # image_ids are hashes, so printing them is PHI-free and actionable.
            parts.append(f"{len(missing)} manifest rows have no PNG: {missing}")
        if extra:
            # COUNT ONLY: an unknown stem is not necessarily a hash and could carry PHI.
            parts.append(
                f"{len(extra)} PNGs on disk are absent from the manifest (ids withheld: an "
                "unknown stem could carry PHI) — inspect --renders-dir yourself"
            )
        raise PreflightError("renders and manifest disagree; " + "; ".join(parts))

    mismatches = sorted(
        image_id
        for image_id in manifest_ids
        if sha256_file(renders_dir / f"{image_id}.png") != manifest[image_id][3]
    )
    if mismatches:
        raise PreflightError(
            f"{len(mismatches)} PNGs do not match the manifest sha256: {mismatches} — "
            "the renders are stale or truncated; re-render before seeding"
        )

    # D-10c.7 (extended by review 2026-08-06): the THIRD sweep — seed dir vs manifest.
    # The two sweeps above only compare renders to the manifest, and `n_existing_seeds`
    # below only counts ids that ARE in the manifest, so a seed whose image_id has since
    # left the manifest is invisible: seed 100 images, re-render against a 90-row manifest,
    # re-run with `--on-existing overwrite`, and the 10 dropped seeds stay on disk carrying
    # stale provenance that the provenance guard never inspects (it, too, iterates the
    # manifest). 10d/10e read the DIRECTORY, so those orphans would be pulled into the
    # frozen gt.csv — the exact provenance mixing D-10c.7 exists to prevent, on the one axis
    # its guard does not cover. Abort before any byte is written.
    # `.part` files are in-flight atomic writes, not seeds; `*.json` already excludes them.
    orphan_stems = sorted(p.stem for p in out_dir.glob("*.json")) if out_dir.exists() else []
    orphan_stems = [stem for stem in orphan_stems if stem not in manifest_ids]
    if orphan_stems:
        hash_shaped = [stem for stem in orphan_stems if _HASH_STEM_RE.match(stem)]
        parts = [f"{len(orphan_stems)} seed file(s) in --out-dir are absent from the manifest"]
        if hash_shaped:
            parts.append(f"image_ids: {hash_shaped}")
        n_withheld = len(orphan_stems) - len(hash_shaped)
        if n_withheld:
            parts.append(
                f"{n_withheld} further stem(s) withheld (not hash-shaped, so they could carry "
                "PHI) — inspect --out-dir yourself"
            )
        raise PreflightError(
            "; ".join(parts) + " — these are ORPHANS with stale provenance that 10d/10e would "
            "read straight into gt.csv (rule #9). Delete them, or start a fresh --out-dir."
        )

    n_existing = sum(1 for image_id in manifest_ids if (out_dir / f"{image_id}.json").exists())
    if on_existing == "skip":
        # D-10c.7: the guard that makes `skip` safe. Every seed carries the full provenance
        # block, so a config or version change would otherwise leave a seed directory
        # silently MIXING two provenances — exactly the rule #9 invalidation. Abort, never
        # skip past it.
        differing: list[str] = []
        for image_id in sorted(manifest_ids):
            path = out_dir / f"{image_id}.json"
            if not path.exists():
                continue
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                prior = existing["provenance"]
            except Exception:  # noqa: BLE001 - an unreadable seed is treated as a mismatch
                differing.append(image_id)
                continue
            if prior != dict(provenance):
                differing.append(image_id)
        if differing:
            raise ProvenanceMismatchError(
                f"{len(differing)} existing seed(s) were written under DIFFERENT provenance: "
                f"{differing} — skipping them would mix two engine builds in one seed dir "
                "(rule #9). Re-run with --on-existing overwrite, or start a fresh --out-dir."
            )

    return {
        "n_manifest_rows": len(manifest),
        "n_missing_pngs": len(missing),
        "missing_pngs": missing,
        "n_extra_pngs": len(extra),
        "n_hash_mismatches": len(mismatches),
        "hash_mismatches": mismatches,
        "n_existing_seeds": n_existing,
        # Always 0 on a successful preflight (a non-zero count raises above) — it is in the
        # summary so the PHI-free artifact RECORDS that the orphan sweep ran.
        "n_orphan_seeds": len(orphan_stems),
    }


# --- The run --------------------------------------------------------------------------


def _bucket(conf: float) -> str:
    if conf < 0:
        return "-1"
    if conf >= 90:
        return "90-100"
    low = int(conf // 10) * 10
    return f"{low}-{low + 9}"


def _confidence_block(values: Sequence[float]) -> dict[str, object]:
    histogram = dict.fromkeys(_HIST_BUCKETS, 0)
    for value in values:
        histogram[_bucket(value)] += 1
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None, "histogram": histogram}
    return {
        "n": len(values),
        "min": min(values),
        "median": float(statistics.median(values)),
        "max": max(values),
        "histogram": histogram,
    }


def run_seed(
    *,
    renders_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    provenance: Mapping[str, object],
    on_existing: str,
    ocr_fn: OCRFn,
) -> dict[str, object]:
    """Preflight, seed every manifest image in sorted order, write the files + summary."""
    pf = preflight(
        renders_dir=renders_dir,
        manifest_path=manifest_path,
        out_dir=out_dir,
        on_existing=on_existing,
        provenance=provenance,
    )
    manifest = read_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeded = 0
    n_skipped = 0
    n_tokens = 0
    n_degenerate = 0
    per_image_box_counts: dict[str, int] = {}
    image_dims: dict[str, list[int]] = {}
    confidences: list[float] = []
    failures: list[dict[str, str]] = []

    with tesseract_env():
        # Determinism: the manifest is the authority and sorted() fixes the work order.
        for image_id in sorted(manifest):
            _, w, h, _, _ = manifest[image_id]
            seed_path = out_dir / f"{image_id}.json"
            if on_existing == "skip" and seed_path.exists():
                # Provenance already verified identical in preflight, so this is a true
                # resume. The summary describes THIS run, so a skipped image contributes no
                # counts — its numbers live in the seed file that was already written.
                n_skipped += 1
                continue
            try:
                record = seed_image(
                    renders_dir / f"{image_id}.png",
                    image_id=image_id,
                    w=w,
                    h=h,
                    provenance=provenance,
                    ocr_fn=ocr_fn,
                )
            except Exception as exc:  # noqa: BLE001 - TYPE ONLY; the message may hold PHI
                # D-10c.9: a failed/timed-out image gets NO FILE — not even a
                # `status: "failed"` placeholder, whose existence would make a later
                # `--on-existing skip` stop retrying and turn a transient timeout into a
                # permanent hole in gt.csv. No file = the next skip run retries it. The
                # run continues; the failure is recorded as image_id + exception TYPE, never
                # a message, traceback or filename (CLAUDE.md §0).
                failures.append({"image_id": image_id, "exception_type": type(exc).__name__})
                continue
            write_json_atomic(seed_path, record)
            tokens = record["tokens"]
            assert isinstance(tokens, list)
            n_seeded += 1
            n_tokens += len(tokens)
            per_image_box_counts[image_id] = len(tokens)
            image_dims[image_id] = [w, h]
            # D-10c.4: degenerate boxes are KEPT and counted — by count_degenerate, the one
            # place that defines "degenerate", never re-derived inline here.
            n_degenerate += count_degenerate(tokens)
            for token in tokens:
                confidences.append(float(token["confidence"]))

    # D-10c.8: PHI-free BY CONSTRUCTION. No token text, no strings read from any image, no
    # filenames — the keys below are image_ids, which are hashes. This file lives NEXT TO
    # the seed dir, not inside it, because the seed dir is gitignored + hook-denied and
    # this is the one artifact the human (and Claude) may read.
    summary: dict[str, object] = {
        "provenance": dict(provenance),
        "renders_dir": str(renders_dir),
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "on_existing": on_existing,
        "n_images_manifest": len(manifest),
        "n_images_seeded": n_seeded,
        "n_images_skipped": n_skipped,
        "n_images_failed": len(failures),
        "n_tokens_total": n_tokens,
        "n_degenerate_boxes": n_degenerate,
        "per_image_box_counts": per_image_box_counts,
        "image_dims": image_dims,
        "confidence": _confidence_block(confidences),
        "preflight": pf,
        "failures": failures,
    }
    write_json_atomic(out_dir.parent / SUMMARY_NAME, summary)
    return summary


def resolve_on_existing(
    out_dir: Path, flag: str | None, *, stdin_isatty: bool | None = None
) -> str:
    """D-10c.7: support BOTH skip and overwrite, and ASK when it is genuinely ambiguous.

    A batch run must stay explicit, so a non-TTY with existing seeds and no flag fails loud
    rather than picking a default that could either destroy corrected work or silently
    resume from seeds written under a different config.
    """
    if flag is not None:
        return flag
    existing = sorted(out_dir.glob("*.json")) if out_dir.exists() else []
    if not existing:
        return "overwrite"  # nothing to clash with
    isatty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
    if not isatty:
        raise AmbiguousRerunError(
            f"--out-dir already holds {len(existing)} seed file(s) and --on-existing was not "
            "passed, and stdin is not a TTY so there is nobody to ask. Pass "
            "--on-existing skip or --on-existing overwrite explicitly."
        )
    while True:
        print(f"--out-dir already holds {len(existing)} seed file(s).")
        answer = input("skip / overwrite / abort? ").strip().lower()
        if answer in ("skip", "overwrite"):
            return answer
        if answer == "abort":
            raise AmbiguousRerunError("aborted at the re-run prompt; nothing was written")


# --- CLI ------------------------------------------------------------------------------


def format_summary_lines(summary: Mapping[str, object]) -> list[str]:
    """PHI-free stdout: counts and distributions only — never a token, never a filename."""
    prov = summary["provenance"]
    assert isinstance(prov, Mapping)
    conf = summary["confidence"]
    assert isinstance(conf, Mapping)
    failures = summary["failures"]
    assert isinstance(failures, list)
    by_type: dict[str, int] = {}
    for failure in failures:
        by_type[failure["exception_type"]] = by_type.get(failure["exception_type"], 0) + 1
    return [
        f"tesseract:   {prov['tesseract_version']}",
        f"leptonica:   {prov['leptonica_version']}",
        f"pytesseract: {prov['pytesseract_version']}  pillow: {prov['pillow_version']}",
        f"tessdata:    variant={prov['tessdata_variant']} "
        f"sha256={prov['traineddata_sha256']}",
        f"config:      {prov['tesseract_cmd_config']}",
        f"upscale:     {prov['upscale_factor']}x {prov['upscale_filter']}",
        f"on-existing: {summary['on_existing']}",
        f"images:      {summary['n_images_seeded']} seeded, "
        f"{summary['n_images_skipped']} skipped, {summary['n_images_failed']} failed "
        f"of {summary['n_images_manifest']} in the manifest",
        f"tokens:      {summary['n_tokens_total']} total, "
        f"{summary['n_degenerate_boxes']} degenerate box(es)",
        f"confidence:  n={conf['n']} min={conf['min']} median={conf['median']} "
        f"max={conf['max']}",
        f"  histogram: {conf['histogram']}",
        f"failures by exception type: {sorted(by_type.items())}",
        f"seeds (PHI — review UI only, never Claude) -> {summary['out_dir']}/",
        f"PHI-free summary (safe to share)           -> "
        f"{Path(str(summary['out_dir'])).parent / SUMMARY_NAME}",
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed Phase-10d ground-truth review with LOCAL Tesseract (Phase 10c).",
    )
    # Plain string paths, never argparse.FileType: FileType's own error messages echo the
    # path back to the terminal. No silent defaults either (D-10c.6) — 10b's defaults are
    # what the human passes, not what this script may assume.
    parser.add_argument("--renders-dir", required=True, help="dir of 10b's <image_id>.png")
    parser.add_argument("--manifest", required=True, help="10b's render_manifest.csv")
    parser.add_argument("--out-dir", required=True, help=f"the {SEED_DIR_NAME} dir itself")
    parser.add_argument("--tessdata-dir", required=True, help="dir holding eng.traineddata")
    parser.add_argument(
        "--tessdata-variant",
        required=True,
        choices=TESSDATA_VARIANTS,
        help="D-10c.2b: recorded, never guessed — apt installs 'standard', not 'best'",
    )
    parser.add_argument(
        "--on-existing",
        choices=("skip", "overwrite"),
        default=None,
        help="D-10c.7: omit to be prompted (TTY) or to fail loud (non-TTY)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    try:
        if not tesseract_binary_available():
            raise TesseractMissingError(f"the `tesseract` binary is not on PATH — {_SETUP_HINT}")
        # env applied before the version probe too, so LC_* cannot alter what is recorded.
        with tesseract_env():
            provenance = build_provenance(
                tessdata_dir=Path(args.tessdata_dir),
                tessdata_variant=args.tessdata_variant,
            )
            on_existing = resolve_on_existing(out_dir, args.on_existing)
            summary = run_seed(
                renders_dir=Path(args.renders_dir),
                manifest_path=Path(args.manifest),
                out_dir=out_dir,
                provenance=provenance,
                on_existing=on_existing,
                ocr_fn=default_ocr_fn(Path(args.tessdata_dir)),
            )
    except SeedError as exc:
        # Every SeedError message is PHI-free by construction (see the Errors section).
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - never let a traceback render token text
        print(f"FAILED: {type(exc).__name__} (message suppressed: may contain PHI)")
        return 1

    for line in format_summary_lines(summary):
        print(line)
    # Fail loud (CLAUDE.md §0.8 / plan §0): a partial run must never look complete.
    complete = (
        int(summary["n_images_seeded"]) + int(summary["n_images_skipped"])
        == int(summary["n_images_manifest"])
        and not summary["failures"]
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
