"""Phase 8 — the abstract `Runner` base class. 🟢 PHI-free.

Formalizes "the one model-specific line" (harness.py's `run_func`): every OCR engine
converts its own native output into the frozen `OCRWord`/`OCROutput` contract INSIDE
its own runner's `run()`, never in `harness.py`/`matching.py`/`aggregate.py`. Keeping
that conversion here, per engine, is what keeps the cross-engine comparison fair
(CLAUDE.md engine-agnostic rule).

A bound `Runner().run` method satisfies `harness.harness.RunFunc` unchanged, so a
`Runner` subclass drops straight into `run_harness` with no other wiring.

`box_free` is deliberately NOT a class attribute here (D-8.2, resolved): it lives only
on `OCROutput`, set by `run()` itself, so there is exactly one source of truth for it.

This file touches ZERO PHI: it is a pure abstract interface plus a cost default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from harness.contract import OCROutput
from harness.cost import estimate_cost

if TYPE_CHECKING:  # avoid an import cycle with harness.py; only used for the type hint
    from harness.harness import ImageRef


class Runner(ABC):
    """One OCR engine, wired into the fixed harness loop.

    Subclasses set `model_name`/`version` (class attributes or in their own `__init__`)
    and implement `run()`. `version` must be sourced from the real engine (e.g. its
    library's `__version__`) — never hand-typed — since `aggregate()` keys results by
    `(model_name, version)` and refuses to blend a version bump into prior results.
    """

    model_name: str
    version: str

    @abstractmethod
    def run(self, image_ref: "ImageRef") -> OCROutput:
        """Native engine output -> OCRWord/OCROutput. Boxes in pixels of the fed image."""
        ...

    def cost(self, image_ref: "ImageRef") -> float:
        """Estimated dollar cost of running this engine on `image_ref`.

        Delegates to `harness.cost.estimate_cost`, which reads the pricing rule off
        `self.run.pricing` (set via `@priced(...)` on the subclass's `run`); an
        untagged runner falls back to `self_hosted` ($0). No pricing logic lives here.
        """
        return estimate_cost(image_ref, self.run)
