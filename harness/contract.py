"""The frozen data contract — the shared spine every OCR runner converts into.

This module defines the small set of dataclasses and the single string normalizer
that the whole harness agrees on. Every OCR engine (docTR, PaddleOCR, EasyOCR, a
self-hosted VLM, cloud OCR) has its OWN native output shape; each engine's runner is
responsible for converting that native output into `OCRWord` / `OCROutput` HERE, in
the runner — NEVER in the harness. The matcher, scorer, and reporter only ever see
these types, so the comparison across engines stays fair (CLAUDE.md engine-agnostic
rule): keep engine-specific code in `runners/`, keep this contract fixed.

`GTToken` (the ground-truth schema) also lives in this file — deliberately NOT in a
separate `gt_schema.py` inside `ground_truth/`. `matching.py` and `metrics.py` need
the GT shape, and routing it through this PHI-free module lets them import it without
taking a dependency on the PHI-touching `ground_truth/` package.

This file touches ZERO PHI: it is pure dataclasses plus a string normalizer. It has
no engine-specific imports and depends only on the standard library.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class OCRWord:
    """One word as read by an engine, in the harness's shared vocabulary.

    bbox convention (shared with `GTToken`): `(x0, y0, x1, y1)`, axis-aligned, with a
    TOP-LEFT origin, measured in PIXELS of the exact image fed to the engine. This is
    the PIL / OpenCV / numpy convention where y increases DOWNWARD: `(x0, y0)` is the
    top-left corner and `(x1, y1)` is the bottom-right corner. It is NOT bottom-left
    Cartesian. Any polygon-to-box conversion (quads, rotated boxes) happens inside the
    engine's runner, never in this file.
    """

    text: str
    # Exact engine read — zero cleanup at this layer. No stripping, casing, or
    # normalization here; `normalize()` is applied later, only at match time.

    bbox: tuple[float, float, float, float] | None
    # (x0, y0, x1, y1) in PIXELS of the fed image, top-left origin (see class doc).
    # None when the engine gives no box (e.g. a box-free VLM path).

    confidence: float | None
    # None = the engine provides no per-word confidence score.


@dataclass
class OCROutput:
    """One engine's full output for one image: its words plus provenance."""

    words: list[OCRWord]

    raw_response: object
    # PHI-BEARING. The engine's opaque native response — it contains the read-back
    # PHI tokens (patient IDs, accession numbers). NEVER print it, NEVER log it to
    # stdout, NEVER commit it. If it must ever be persisted, hash or redact it first.
    # See CLAUDE.md §8 (cloud raw_response leakage). Typed `object` on purpose: the
    # harness treats it as opaque and never inspects it.

    model_name: str

    version: str
    # EXACT engine version string (rule #9), sourced by the runner from the real engine
    # (e.g. the library's `__version__`) — NEVER hand-typed in the harness. REQUIRED and
    # validated non-empty in `__post_init__` (D-8.4): there is no "not set" state, because
    # a batch in which every row is blank looks like one consistent `(model_name, version)`
    # pair to `aggregate()` and would be averaged across engine builds silently.
    # `aggregate()` refuses to blend rows whose `(model_name, version)` differ, so a
    # version bump can never merge with prior results (D-8.3).

    config_id: str
    # SHORT HUMAN LABEL for this arm's configuration — "stock", "tuned", "parseq",
    # "thr0.2". Exists for the report: `config_hash` below is what actually protects the
    # aggregation, but a bare hash is unreadable in a findings table (D-13.5).
    # REQUIRED and validated non-empty for the same reason as `version`.

    config_hash: str
    # AUTO-COMPUTED digest of every knob that changes what the engine outputs, produced by
    # `Runner.config_hash()` from the runner's declared `config()` dict — never hand-typed
    # and never assembled here (D-13.5).
    #
    # Why this exists in addition to `config_id`: `aggregate()`'s guard keyed only on
    # `(model_name, version, verifier_*)`, so docTR-stock, docTR-tuned and docTR+parseq all
    # presented the SAME identity — same library, same `__version__` — and averaged into one
    # meaningless row. A human label alone fails open: it prevents that collision only if
    # someone remembers to change the label when they change a knob, and forgetting is
    # silent. A hash over the declared config cannot be forgotten — change a threshold and
    # the identity changes with it.
    #
    # REQUIRED and validated non-empty (see `__post_init__`): a blank default would let a
    # runner that declares nothing sail through, which is precisely the bug.

    box_free: bool = False
    # True for the VLM "blob" path (a single text response, no per-word boxes). Set by
    # the runner. Box-free outputs keep the primary engine's box and swap only the string.

    def __post_init__(self) -> None:
        """Reject a blank version or config identity at construction — the earliest point.

        Making the fields required stops an OMITTED value (TypeError from the dataclass
        constructor); this stops an EXPLICITLY BLANK one. Both matter, and neither is a
        runner-only concern: a hand-built `OCROutput` in a script or notebook is exactly
        how un-versioned or un-configured rows would otherwise reach `aggregate()`.
        Fixtures must pass explicit markers (e.g. `"0.0.0-synthetic"`), not "".
        """
        if not self.version.strip():
            raise ValueError(
                "OCROutput.version must be a non-empty exact engine version string "
                "(rule #9) — source it from the engine itself (e.g. the library's "
                "__version__), never hand-type it. Synthetic fixtures should pass an "
                "explicit marker such as '0.0.0-synthetic'."
            )
        if not self.config_id.strip():
            raise ValueError(
                "OCROutput.config_id must be a non-empty label for this arm's config "
                "(e.g. 'stock', 'tuned', 'parseq') — it is how a reader tells two arms of "
                "the same engine apart in the report (D-13.5). Synthetic fixtures should "
                "pass an explicit marker such as 'synthetic'."
            )
        if not self.config_hash.strip():
            raise ValueError(
                "OCROutput.config_hash must be a non-empty digest of the runner's declared "
                "config — produce it with Runner.config_hash(), never hand-type it "
                "(D-13.5). Without it, two arms of the same engine at the same version "
                "carry identical identities and aggregate() averages them silently. "
                "Synthetic fixtures should pass an explicit marker such as '0000synthetic'."
            )


@dataclass(frozen=True)
class GTToken:
    """One ground-truth token for one image — the frozen GT schema.

    bbox convention (identical to `OCRWord`): `(x0, y0, x1, y1)`, axis-aligned,
    TOP-LEFT origin, in PIXELS of the input image — the PIL / OpenCV / numpy
    convention where y increases DOWNWARD, `(x0, y0)` is the top-left corner and
    `(x1, y1)` is the bottom-right corner. NOT bottom-left Cartesian. Polygon-to-box
    conversion, if any, happens where the GT is built, never in this file.

    `token_text` is PHI (the actual patient ID / accession number). This dataclass is
    the shape; its instances are constructed from `gt.csv`, which Claude never reads.
    """

    image_id: str
    series_uid: str
    modality: str
    vendor: str
    stratum: str
    frame_idx: int
    token_text: str
    bbox: tuple[float, float, float, float]
    label: str  # "PHI" | "KEEP"


def normalize(s: str) -> str:
    """Canonicalize a string for exact matching. FROZEN.

    Applies, in order:
      1. Unicode NFC normalization (`unicodedata.normalize("NFC", s)`)
      2. stripping of surrounding whitespace (`.strip()`)

    It is CASE-SENSITIVE and does NOTHING else: no lowercasing, no punctuation
    stripping, no fuzzy matching. In clinical tokens these distinctions carry meaning
    — `l` != `L`, `M` != `F` — so collapsing them would corrupt the comparison.

    Idempotent: `normalize(normalize(s)) == normalize(s)`. NFC is idempotent, and NFC
    never introduces surrounding whitespace, so a single strip after it suffices.

    CHANGING THIS FUNCTION INVALIDATES ALL REPORTED RESULTS. Both the ground truth and
    every engine's output pass through it before matching; altering it silently shifts
    every score. It is frozen — treat any edit the way you would an edit to `gt.csv`.
    """
    return unicodedata.normalize("NFC", s).strip()
