"""Phase 9 — the PP-OCRv6_medium runner (PaddleOCR). 🟢 PHI-free by construction.

PaddleOCR is **self-hosted and local**: weights are fetched once from a public model hub at
first use and inference runs in-environment, so no image — and therefore no PHI — ever leaves
the box. No BAA is needed (CLAUDE.md §4). The only PHI-bearing object this module handles is
`OCROutput.raw_response` (the read-back tokens), stored opaquely and never printed or logged,
per `contract.py`'s warning.

**This file holds ALL PaddleOCR-specific knowledge.** The polygon→pixel-box conversion below is
the crux of the runner and lives here, never in `harness.py`/`matching.py`/`aggregate.py` — that
is what keeps the cross-engine comparison fair (CLAUDE.md engine-agnostic rule).

Engine selection rationale (D-9.2; record in run metadata/report)
----------------------------------------------------------------
PP-OCRv6_medium is **+4.6% detection / +5.1% recognition over PP-OCRv5_server** (86.2/83.2 vs
81.6/78.1), and v5 was itself a large jump over v4 (**53.0 → 80.1** weighted recognition
accuracy, mobile-tier benchmark). Gains span **screen, industrial, rotated, and ancient-text**
categories. Deliberately NOT phrased as "biggest gains on screen/dot-matrix text" — that
superlative is not supported by the source benchmark (rotated +13.8pp and ancient +12.0pp both
exceed industrial/dot-matrix's +9.0pp). Apache-2.0, self-hosted, $0 marginal cost.

Runtime (D-9.2): **native Paddle Inference**, via `paddleocr` → `paddlex` → `paddlepaddle`, not
the ONNX or safetensors variants — so PaddleOCR's own DB postprocess and CRNN decode are reused
rather than reimplemented against a different runtime.

⚠️ **oneDNN must stay OFF (`enable_mkldnn=False`).** PaddleOCR enables the oneDNN/MKL-DNN CPU
backend by default. With the in-env stack (paddlepaddle 3.3.1 + PP-OCRv6_medium) that path
raises `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]` from the PIR executor and **no inference runs at
all**. Disabling it is a correctness requirement here, not a tuning knob — but note it also
removes a CPU accelerator, so it compounds the D-9.1 latency caveat below. Re-test before
turning it back on after any paddlepaddle bump.

⚠️ **Latency caveat (D-9.1):** CPU-only build (`paddlepaddle`, not `paddlepaddle-gpu`;
`is_compiled_with_cuda() == False`), with oneDNN disabled per above. Every latency number this
runner produces is CPU-only and is **NOT** representative of GPU-served production numbers.
Flag that wherever Phase 9 latency is reported.

⚠️ **Confidence is RECOGNITION confidence, and it is per-LINE, inherited by each word.**
PaddleOCR's OCR *pipeline* surfaces `rec_scores` only. Detection scores (`dt_scores`) exist in
paddlex's standalone text-detection predictor but the OCR pipeline drops them, so a per-word
detection confidence is genuinely unavailable without bypassing the pipeline (and thus its DB
postprocess → crop → CRNN chain). PP-OCR also scores a whole text line, not a word, so every
word this runner emits carries **its parent line's** recognition score. Consequences:
- This is the same *domain* as the docTR runner's confidence (also recognition), so a
  `conf_threshold` is broadly comparable between docTR and PP-OCRv6. It is NOT comparable to a
  generative arm's score (a mean per-token decode probability) — never carry a threshold across
  those two families (CLAUDE.md §4).
- It is a per-line score on a per-word field: words on a line are not independently scored, so
  a single bad word in an otherwise clean line will not stand out by confidence alone. Keep that
  in mind when tuning the verifier's abstain threshold (D-11.2).

Cost: deliberately **untagged** — no `@priced(...)`. `harness.cost.estimate_cost` falls back to
the `self_hosted` rule ($0) for an untagged `run_func`, which is exactly right for a local
engine. No pricing logic belongs here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import paddleocr
from paddleocr import PaddleOCR

from harness.contract import OCROutput, OCRWord
from harness.runners.base import Runner

if TYPE_CHECKING:  # only the type hint needs it; avoids a hard import at runtime
    from harness.harness import ImageRef

BBox = tuple[float, float, float, float]

# D-9.2: name the models explicitly rather than leaning on PaddleOCR's default resolution.
# `PaddleOCR()` with no `lang`/`ocr_version` happens to resolve to this same pair today, but
# that default is version-dependent — pinning the names here means a paddleocr bump cannot
# silently swap the model tier out from under a run (rule #9).
DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"


def _to_pixel_bbox(region: Any, page_w: int, page_h: int) -> BBox:
    """Convert one PP-OCR 4-point word polygon to the contract's pixel box. THE CRUX.

    PaddleOCR emits `((x0,y0), (x1,y0), (x1,y1), (x0,y1))` — a 4-point quad already in PIXELS
    of the fed image, top-left origin (x right, y DOWN). That is the same axis convention the
    contract wants (contract.py:30-36), so this is never a y-flip and never a rescale.

    Reduces all four points to min/max rather than trusting their winding order, which keeps it
    correct for a rotated quad (PP-OCR's detector can emit one) — the enclosing axis-aligned box
    is what the IoU matcher expects.

    Coordinates are clamped to the page: PP-OCR's unclip step expands each detected region by
    `text_det_unclip_ratio` and can push a box a hair outside the image it was read from.
    """
    pts = np.asarray(region, dtype=float).reshape(-1, 2)
    xs = np.clip(pts[:, 0], 0.0, float(page_w))
    ys = np.clip(pts[:, 1], 0.0, float(page_h))
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class PaddleV6Runner(Runner):
    """PP-OCRv6_medium wired into the fixed harness loop. `run` satisfies `RunFunc` as-is."""

    model_name = "pp-ocrv6_medium"
    version_source = "paddleocr"  # `version` must equal paddleocr.__version__ (contract test)
    config_id = "stock"
    # "stock" here means STOCK THRESHOLDS, not "PaddleOCR out of the box" — the settings in
    # `config()` below are correctness requirements present on every arm. Phase 13 step 6
    # says to state that explicitly in the report or the finding is mislabelled.

    def __init__(self) -> None:
        # Read from the installed library, never hand-typed (base.py's contract + rule #9).
        self.version: str = paddleocr.__version__
        self.det_model: str = DET_MODEL
        self.rec_model: str = REC_MODEL
        self._ocr: PaddleOCR | None = None

    def config(self) -> dict[str, object]:
        """Everything passed to `PaddleOCR(...)` that can change a box or a string (D-13.5).

        The model names are declared because a tier swap (medium -> server) is a different
        engine configuration entirely. The three correctness-required flags are declared even
        though they are constant today: if a future arm ever flips one, the hash must change
        rather than quietly merge with these results.

        Detection thresholds are NOT declared here because this runner does not set them —
        it takes PaddleOCR's defaults. The tuned arm (Phase 13 step 6) sets
        `text_det_thresh` / `text_det_box_thresh` / `text_det_unclip_ratio` /
        `text_det_limit_side_len` and must add them to its own `config()`, which is exactly
        what gives it a distinct hash.
        """
        return {
            "det_model": self.det_model,
            "rec_model": self.rec_model,
            "return_word_box": True,
            "enable_mkldnn": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

    @property
    def ocr(self) -> PaddleOCR:
        """The PP-OCRv6_medium pipeline, built once on first use and reused after.

        Lazy so constructing a `PaddleV6Runner` (e.g. to read `version` for metadata, or to
        register it in a test list) costs nothing and needs no weights. Cached so that
        `run_harness`'s timer — which wraps only `run_func` — measures inference, not pipeline
        construction and a weight download.
        """
        if self._ocr is None:
            self._ocr = PaddleOCR(
                text_detection_model_name=DET_MODEL,
                text_recognition_model_name=REC_MODEL,
                # Word-level boxes. REQUIRED by the contract: without this PP-OCR returns one
                # box per text LINE, so an overlay line like "ID: CMFN-0042 ACC-0099" becomes a
                # single ~236px box that scores IoU ~0.43 against the ~102px GT box for one
                # token — below the matcher's 0.5 threshold. Line-level would silently break
                # matching on exactly the burned-in shape this project scores.
                return_word_box=True,
                # See the module docstring: the oneDNN path raises NotImplementedError on this
                # stack, so this is a correctness requirement, not a performance knob.
                enable_mkldnn=False,
                # Preprocessing stages OFF, for determinism and minimum-necessary work: each
                # can rotate or warp the image, which would move the pixels the returned boxes
                # are expressed in and break the "pixels of the fed image" contract.
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    def run(self, image_ref: ImageRef) -> OCROutput:
        """Read one image with PP-OCRv6_medium and convert its output to the frozen contract.

        Pixels are loaded here, from `image_ref.path`, by PaddleOCR's own reader so the decode
        and channel conventions match what the pipeline expects. Nothing is printed or logged.
        """
        results = self.ocr.predict(image_ref.path)

        words: list[OCRWord] = []
        for res in results:
            # Per detected text line: its words, their quads, and the line's recognition score.
            # `text_word`/`text_word_region` are present only because `return_word_box=True`;
            # index i of each list refers to the same line.
            line_words = res.get("text_word") or []
            line_regions = res.get("text_word_region") or []
            line_scores = res.get("rec_scores") or []

            # strict=True on both zips: a length mismatch between texts and their regions means
            # PP-OCR's output shape changed underneath us. Plain zip() would silently truncate,
            # dropping words that the scorer would then count as OMISSIONS — a corrupted
            # measurement that still renders as a real result. Fail loudly instead.
            for i, (texts, regions) in enumerate(zip(line_words, line_regions, strict=True)):
                # Per-line recognition score, inherited by every word on that line — PP-OCR
                # exposes no per-word score (see the module docstring).
                score = float(line_scores[i]) if i < len(line_scores) else None
                for text, region in zip(texts, regions, strict=True):
                    if not text.strip():
                        # A glyph-free separator ("  ") that PP-OCR's character grouping emits
                        # between words. It is not something the engine *read*, so emitting it
                        # would add an unmatchable prediction that the scorer counts as a
                        # HALLUCINATION — inflating the Added axis, the metric that matters
                        # most here. Dropping it is a structural filter, not text cleanup:
                        # punctuation like ":" survives, and no surviving word's text is
                        # altered (contract.py:38-40 — normalize() runs later, at match time).
                        continue
                    words.append(
                        OCRWord(
                            text=text,  # raw engine read, zero cleanup
                            bbox=_to_pixel_bbox(region, image_ref.w, image_ref.h),
                            confidence=score,
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
            box_free=False,  # PP-OCR gives per-word boxes
        )
