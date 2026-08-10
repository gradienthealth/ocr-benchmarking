"""Phase 9 — the docTR runner (Step-1 incumbent). 🟢 PHI-free by construction.

docTR is **self-hosted and local**: weights are fetched once from Mindee's public CDN at
first use and inference runs in-environment, so no image — and therefore no PHI — ever
leaves the box. No BAA is needed (CLAUDE.md §4). The only PHI-bearing object this module
handles is `OCROutput.raw_response` (the read-back tokens), which is stored opaquely and
never printed or logged, per `contract.py`'s warning.

**This file holds ALL docTR-specific knowledge.** The coordinate conversion below is the
crux of the runner and lives here, never in `harness.py`/`matching.py`/`aggregate.py` —
that is what keeps the cross-engine comparison fair (CLAUDE.md engine-agnostic rule).

Configuration (D-9.2): pretrained defaults, no architecture override —
`ocr_predictor(pretrained=True)`.

⚠️ **Discrepancy worth knowing (surfaced 2026-07-31, NOT silently averaged):** plan.md D-9.2
describes "the pretrained defaults" as `db_resnet50` + `crnn_vgg16_bn`. In the installed
docTR **v1.0.1** the default detection architecture is actually **`fast_base`** (a FAST
model), not `db_resnet50`; recognition is `crnn_vgg16_bn` as documented. This module follows
the *literal* decision — pretrained defaults, no override — so the resolved pair is
`fast_base` + `crnn_vgg16_bn`. Both resolved names are read off the library at runtime and
exposed as `det_arch`/`reco_arch` so a run's metadata records the model tier that actually
ran (rule #9), rather than the one the plan text assumed. If D-9.2 meant `db_resnet50`
specifically, that is an architecture override and a deliberate change — raise it, don't
patch it in quietly, since it makes results non-comparable with anything run before.

⚠️ **Latency caveat (D-9.1):** installed as a CPU-only build (`torch==2.13.0+cpu`), so every
latency number this runner produces is CPU-only and is NOT representative of GPU-served
production numbers. Flag that wherever Phase 9 latency is reported.

Cost: deliberately **untagged** — no `@priced(...)`. `harness.cost.estimate_cost` falls back
to the `self_hosted` rule ($0) for an untagged `run_func`, which is exactly right for a local
engine. No pricing logic belongs here.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import doctr
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from harness.contract import OCROutput, OCRWord
from harness.runners.base import Runner

if TYPE_CHECKING:  # only the type hint needs it; avoids a hard import at runtime
    from harness.harness import ImageRef

# docTR's geometry is normalized to the page; the contract wants pixels (contract.py:42-43).
BBox = tuple[float, float, float, float]


def _resolved_arch(param: str) -> str:
    """The architecture `ocr_predictor()` selects by default, read from the library itself.

    Sourced from the live signature rather than hand-typed for the same reason `version`
    is (rule #9): the plan's assumed default and the installed library's actual default
    disagree (see the module docstring), and the metadata must record what really ran.
    """
    return str(inspect.signature(ocr_predictor).parameters[param].default)


def _to_pixel_bbox(geometry: Any, page_w: int, page_h: int) -> BBox:
    """Convert one docTR word geometry to the contract's pixel box. THE CRUX.

    docTR emits geometry **normalized to the page** in one of two shapes:
      - `((xmin, ymin), (xmax, ymax))` — a straight box, the default since
        `ocr_predictor(pretrained=True)` runs with `assume_straight_pages=True`;
      - a 4-point `(4, 2)` polygon — if straight-page mode is ever turned off.
    Both are handled by reducing every point to its min/max, so this stays correct if that
    flag changes.

    Returns `(x0, y0, x1, y1)` in PIXELS with a TOP-LEFT origin — docTR's normalized axes
    already run x-right / y-DOWN, so this is a pure scale, never a y-flip. Scale factors
    come from `page.dimensions` (docTR's own view of the page it read), not from
    `ImageRef.w/h`, so a mismatch between the manifest's dimensions and the actual file
    cannot silently skew the boxes.

    Coordinates are clamped to the page: docTR's postprocessing can push a box a hair
    outside [0, 1], which would otherwise emit a box outside the image it was read from.
    """
    pts = np.asarray(geometry, dtype=float).reshape(-1, 2)
    xs = np.clip(pts[:, 0] * page_w, 0.0, float(page_w))
    ys = np.clip(pts[:, 1] * page_h, 0.0, float(page_h))
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class DoctrRunner(Runner):
    """docTR wired into the fixed harness loop. `run` satisfies `harness.RunFunc` as-is."""

    model_name = "doctr"
    version_source = "doctr"  # `version` must equal doctr.__version__ (contract test)

    def __init__(self) -> None:
        # Read from the installed library, never hand-typed (base.py's contract + rule #9).
        # Resolves to the literal string "v1.0.1" — note docTR includes the "v" prefix.
        self.version: str = doctr.__version__
        self.det_arch: str = _resolved_arch("det_arch")
        self.reco_arch: str = _resolved_arch("reco_arch")
        self._predictor = None

    @property
    def predictor(self):
        """The pretrained predictor, built once on first use and reused after.

        Lazy so that constructing a `DoctrRunner` (e.g. to read `version` for metadata, or
        to register it in a test list) costs nothing and needs no weights. Cached so that
        `run_harness`'s timer — which wraps only `run_func` — measures inference, not model
        construction and a weight download.
        """
        if self._predictor is None:
            self._predictor = ocr_predictor(pretrained=True)  # D-9.2: no arch override
        return self._predictor

    def run(self, image_ref: ImageRef) -> OCROutput:
        """Read one image with docTR and convert its native output to the frozen contract.

        Pixels are loaded here, from `image_ref.path`, via docTR's own `DocumentFile` loader
        so the decode and channel conventions match what the predictor expects (it requires
        3-channel `(H, W, 3)` arrays). Nothing about the result is printed or logged.
        """
        pages = DocumentFile.from_images(image_ref.path)
        doc = self.predictor(pages)

        words: list[OCRWord] = []
        for page in doc.pages:
            page_h, page_w = page.dimensions  # docTR order is (height, width)
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        words.append(
                            OCRWord(
                                # Raw engine read, zero cleanup — `normalize()` is applied
                                # later, only at match time (contract.py:38-40).
                                text=word.value,
                                bbox=_to_pixel_bbox(word.geometry, page_w, page_h),
                                # docTR's RECOGNITION confidence, carried through unchanged.
                                # NOT the detection score (that is `word.objectness_score`).
                                # The two are different numbers — don't conflate them when
                                # wiring verifier gating, and don't assume a `conf_threshold`
                                # tuned here transfers to another engine.
                                confidence=float(word.confidence),
                            )
                        )

        return OCROutput(
            words=words,
            # PHI-bearing and opaque: the harness never inspects it and this runner never
            # prints it (contract.py:56-61).
            raw_response=doc,
            model_name=self.model_name,
            version=self.version,
            box_free=False,  # docTR gives per-word boxes
        )
