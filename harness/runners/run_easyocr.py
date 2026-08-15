"""Phase 9 — the EasyOCR runner. 🟢 PHI-free by construction.

EasyOCR is **self-hosted and local**: weights are fetched once from the project's public
release assets at first use and inference runs in-environment, so no image — and therefore no
PHI — ever leaves the box. No BAA is needed (CLAUDE.md §4). The only PHI-bearing object this
module handles is `OCROutput.raw_response` (the read-back tokens), stored opaquely and never
printed or logged, per `contract.py`'s warning.

**This file holds ALL EasyOCR-specific knowledge.** The quad→pixel-box conversion below is the
crux of the runner and lives here, never in `harness.py`/`matching.py`/`aggregate.py` — that is
what keeps the cross-engine comparison fair (CLAUDE.md engine-agnostic rule).

Detection threshold: 0.2
------------------------
`DETECTION_THRESHOLD` is applied to **both** `text_threshold` and `low_text` (library defaults
0.7 and 0.4). Both are required, because they gate a detection at two different points of
CRAFT's postprocess — verified against the installed `easyocr/craft_utils.py`:

    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)   # line 27: region binarization
    ...
    if np.max(textmap[labels==k]) < text_threshold: continue   # line 41: component peak

`low_text` binarizes the score map into candidate regions; `text_threshold` then discards any
resulting connected component whose *peak* score is below it. A region must clear both, so the
effective detection threshold is `max(low_text, text_threshold)` — leaving either at its
default would pin the effective floor at that default rather than at 0.2.

`link_threshold` stays at its library default value (0.4): it controls character→region linking,
not detection sensitivity, so it is not part of "the detection threshold." Since Phase 13h it is
passed EXPLICITLY at that value rather than omitted. Behaviour is unchanged, but `config()`
declares it, and a declared knob whose value is assumed rather than passed is a digest that can
quietly stop matching what the engine did — an easyocr release that moved the default would
change the run without changing the identity (rule #9).

⚠️ **Latency caveat (D-9.1):** CPU-only (`easyocr.Reader(gpu=False)`, torch 2.13.0+cpu). Every
latency number this runner produces is CPU-only and is **NOT** representative of GPU-served
production numbers. Flag that wherever Phase 9 latency is reported.

⚠️ **Confidence is RECOGNITION confidence, per detected region.** `readtext` returns the
recognizer's (CTC decoder) score, not a detection score. Same *domain* as the docTR and
PP-OCRv6 runners' confidence, so a `conf_threshold` is broadly comparable across the three
local arms. It is NOT comparable to a generative arm's score (a mean per-token decode
probability), so never carry a `conf_threshold` across those two families (CLAUDE.md §4).

⚠️ **Two measured geometry behaviors (2026-07-31), recorded here, NOT tuned around:**
- *Boxes are vertically inflated.* On synthetic fixtures EasyOCR returns regions ~1.9x taller
  than the glyph content they bound (26px vs a 14px tight GT box), which lands IoU against a
  tight ground-truth box at ~0.41-0.50 — at or below the matcher's 0.5 threshold. The
  conversion below is not the cause: the returned box correctly *contains* the true box, it is
  simply looser. `add_margin` (library default 0.1) accounts for only part of it — measured
  IoU rises to just 0.493 at `add_margin=0.0`, still under 0.5 — so it is left at its default
  rather than tuned for a knob that does not fix the problem. How much this matters against
  *real* ground truth depends on how tightly Phase 10's boxes are drawn.
- *Output is line-level, not word-level.* A line holding two tokens comes back as one region.
  EasyOCR has no word-box mode, and `width_ths` does not split it (measured at 0.5/0.1/0.0) —
  the detector emits the whole line as a single region. Unlike PaddleOCR, where
  `return_word_box=True` fixed this, there is no in-library remedy.

Cost: deliberately **untagged** — no `@priced(...)`. `harness.cost.estimate_cost` falls back to
the `self_hosted` rule ($0) for an untagged `run_func`, which is exactly right for a local
engine. No pricing logic belongs here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import easyocr
import numpy as np

from harness.contract import OCROutput, OCRWord
from harness.runners.base import Runner

if TYPE_CHECKING:  # only the type hint needs it; avoids a hard import at runtime
    from harness.harness import ImageRef

BBox = tuple[float, float, float, float]

# Applied to BOTH `text_threshold` and `low_text` — see the module docstring for why both are
# needed to make 0.2 the *effective* threshold. Module-level so a test can assert the runner
# really passes it, rather than trusting a literal buried in a call.
DETECTION_THRESHOLD = 0.2

# One language, matching the burned-in-overlay use case (Latin alphanumeric IDs).
LANGUAGES = ["en"]

# EasyOCR's own default for `link_threshold`, restated here so `config()` can declare it
# (D-13.5). This runner does NOT pass it — it governs character→region linking rather than
# detection sensitivity (module docstring) — but a knob left at its default is still a
# choice, and a tuned arm that moves it must produce a different config hash than this one.
# If EasyOCR ever changes this default, update the constant: the value in the digest must be
# the value the engine actually used, or the identity lies.
LINK_THRESHOLD = 0.4

# EasyOCR's LIBRARY defaults for the two detection knobs. Phase 13 step 6's stock arm is
# built from these, and it is the only "stock" in the lineup that is not simply "no
# argument": this runner has shipped a non-default config since Phase 9, so its stock/tuned
# pair runs the other way round from docTR's and PP-OCR's. See `EasyOcrRunner.config_id`.
LIBRARY_TEXT_THRESHOLD = 0.7
LIBRARY_LOW_TEXT = 0.4


def _to_pixel_bbox(region: Any, page_w: int, page_h: int) -> BBox:
    """Convert one EasyOCR 4-point region to the contract's pixel box. THE CRUX.

    `readtext` emits `[[x0,y0], [x1,y0], [x1,y1], [x0,y1]]` — a 4-point quad already in PIXELS
    of the fed image, top-left origin (x right, y DOWN). That is the same axis convention the
    contract wants (contract.py:30-36), so this is never a y-flip and never a rescale. The
    points come back as `numpy.int32`, hence the explicit float cast.

    Reduces all four points to min/max rather than trusting their winding order, which keeps it
    correct for a rotated quad (EasyOCR's `rotation_info` path can emit one) — the enclosing
    axis-aligned box is what the IoU matcher expects.

    Coordinates are clamped to the page: EasyOCR pads each region by `add_margin` and can push a
    box outside the image it was read from.
    """
    pts = np.asarray(region, dtype=float).reshape(-1, 2)
    xs = np.clip(pts[:, 0], 0.0, float(page_w))
    ys = np.clip(pts[:, 1], 0.0, float(page_h))
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class EasyOcrRunner(Runner):
    """EasyOCR wired into the fixed harness loop. `run` satisfies `RunFunc` as-is."""

    model_name = "easyocr"
    version_source = "easyocr"  # `version` must equal easyocr.__version__ (contract test)

    def __init__(
        self,
        *,
        text_threshold: float = DETECTION_THRESHOLD,
        low_text: float = DETECTION_THRESHOLD,
        link_threshold: float = LINK_THRESHOLD,
    ) -> None:
        """Defaults to the 0.2/0.2 floor config Phase 9 shipped — which is the TUNED arm.

        **This runner's stock/tuned pair runs backwards from the other two.** docTR and
        PP-OCRv6 are stock by default and tuned by argument; EasyOCR has shipped a
        non-default config since Phase 9 (0.2 on both knobs vs library defaults 0.7/0.4), so
        the no-argument instance is *our* configuration and the library-default instance is
        the one that has to be constructed explicitly. Getting this backwards would answer
        the wrong question: the pair exists to separate "EasyOCR over-detects" from "our
        threshold choice over-detects" (Phase 13 step 6).

        `config_id` is therefore derived to three values, not two — see `_derive_config_id`.
        """
        # Read from the installed library, never hand-typed (base.py's contract + rule #9).
        self.version: str = easyocr.__version__
        self.languages: list[str] = list(LANGUAGES)
        self.text_threshold: float = text_threshold
        self.low_text: float = low_text
        self.link_threshold: float = link_threshold
        self.config_id: str = self._derive_config_id()
        self._reader: easyocr.Reader | None = None

    def _derive_config_id(self) -> str:
        """"thr0.2" for the shipped floor config, "stock" for library defaults, else "tuned".

        Derived rather than passed in (Phase 13h) so an arm cannot run one configuration
        under another's name. "thr0.2" is kept verbatim for the shipped config: it names the
        thing that is actually distinctive about the floor arm, where "tuned" would not.
        """
        knobs = (self.text_threshold, self.low_text, self.link_threshold)
        if knobs == (DETECTION_THRESHOLD, DETECTION_THRESHOLD, LINK_THRESHOLD):
            return "thr0.2"
        if knobs == (LIBRARY_TEXT_THRESHOLD, LIBRARY_LOW_TEXT, LINK_THRESHOLD):
            return "stock"
        return "tuned"

    def config(self) -> dict[str, object]:
        """The knobs that decide what EasyOCR detects and reads (D-13.5).

        `text_threshold` and `low_text` are separate knobs, not one `detection_threshold`
        applied twice: the effective floor is `max(low_text, text_threshold)` (see the module
        docstring), so an arm that moves only one of them is a genuinely different
        configuration and must hash differently. The shipped floor config sets both to 0.2.

        `link_threshold` is declared because an arm may move it — a knob left at a default is
        still a choice, and its value belongs in the digest. `languages` is declared because
        a language change swaps the recognition model.
        """
        return {
            "languages": self.languages,
            "text_threshold": self.text_threshold,
            "low_text": self.low_text,
            "link_threshold": self.link_threshold,
        }

    @property
    def reader(self) -> easyocr.Reader:
        """The EasyOCR reader, built once on first use and reused after.

        Lazy so constructing an `EasyOcrRunner` (e.g. to read `version` for metadata, or to
        register it in a test list) costs nothing and needs no weights. Cached so that
        `run_harness`'s timer — which wraps only `run_func` — measures inference, not reader
        construction and a weight download.
        """
        if self._reader is None:
            # gpu=False is D-9.1. verbose=False keeps the reader from writing progress lines to
            # stdout, which the harness treats as a channel that must stay clean.
            self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        return self._reader

    def run(self, image_ref: ImageRef) -> OCROutput:
        """Read one image with EasyOCR and convert its output to the frozen contract.

        Pixels are loaded here, from `image_ref.path`, by EasyOCR's own reader so the decode and
        channel conventions match what the detector expects. Nothing is printed or logged.
        """
        results = self.reader.readtext(
            image_ref.path,
            text_threshold=self.text_threshold,
            low_text=self.low_text,
            link_threshold=self.link_threshold,
        )

        words: list[OCRWord] = []
        for region, text, confidence in results:
            if not text.strip():
                # A glyph-free region carries nothing the engine actually *read*, so emitting it
                # would add an unmatchable prediction that the scorer counts as a HALLUCINATION,
                # inflating the Added axis. Structural filter, not text cleanup: punctuation
                # survives and no surviving word's text is altered (contract.py:38-40 —
                # normalize() runs later, at match time). Mirrors the PP-OCRv6 runner.
                continue
            words.append(
                OCRWord(
                    text=text,  # raw engine read, zero cleanup
                    bbox=_to_pixel_bbox(region, image_ref.w, image_ref.h),
                    # numpy.float64 -> float; the contract wants a plain float.
                    confidence=float(confidence),
                )
            )

        return OCROutput(
            words=words,
            # PHI-bearing and opaque: the harness never inspects it and this runner never
            # prints it (contract.py:56-61).
            raw_response=results,
            model_name=self.model_name,
            version=self.version,
            config_id=self.config_id,
            config_hash=self.config_hash(),
            box_free=False,  # EasyOCR gives per-region boxes
        )
