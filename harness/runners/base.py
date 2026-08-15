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

from harness.contract import OCROutput, config_digest
from harness.cost import estimate_cost

if TYPE_CHECKING:  # avoid an import cycle with harness.py; only used for the type hint
    from harness.harness import ImageRef


class Runner(ABC):
    """One OCR engine, wired into the fixed harness loop.

    Subclasses set `model_name`/`version`/`config_id` (class attributes or in their own
    `__init__`) and implement `run()` and `config()`. `version` must be sourced from the
    real engine (e.g. its library's `__version__`) — never hand-typed — since `aggregate()`
    keys results by `(model_name, version, config_id, config_hash, verifier_*)` and refuses
    to blend a version OR config change into prior results.
    """

    model_name: str
    version: str

    config_id: str
    # Short human label for this arm's configuration — "stock", "tuned", "parseq".
    # Rendered in the report header so two arms of one engine are tellable apart by eye.

    version_source: str | None = None
    # Dotted module path whose `__version__` IS the value of `version` — e.g. "doctr".
    # The shared contract test imports this module and asserts the two agree, so the
    # "never hand-type a version" rule (rule #9) is enforced for EVERY registered runner
    # instead of only the ones someone wrote a per-engine test for.
    #
    # `None` means "not sourced from an installed library" and is reserved for non-engine
    # test doubles. It is not a quiet escape hatch: the contract test requires any runner
    # declaring None to also appear in an explicit exemption list in the test file, so
    # opting out is a visible, reviewable edit rather than a default.
    #
    # NOT a second source of truth in the D-8.2 sense: nothing in the harness ever reads
    # `version_source` to make a decision — only the test reads it, to cross-check the
    # value the runner already set. Two values that must agree, verified once, not two
    # values that different code paths might each believe.

    @abstractmethod
    def run(self, image_ref: "ImageRef") -> OCROutput:
        """Native engine output -> OCRWord/OCROutput. Boxes in pixels of the fed image."""
        ...

    @abstractmethod
    def config(self) -> dict[str, object]:
        """Every knob that changes what this engine outputs, as a flat dict (D-13.5).

        Feeds `config_hash()`, which joins `aggregate()`'s identity guard. Declare a knob
        here if changing it could change a single box or string: architecture choices,
        detection/recognition thresholds, resize limits, orientation stages. Do NOT declare
        things that cannot change the output (device, batch size, log level) — they would
        split one arm into two identities for no reason.

        **Abstract on purpose.** A default `{}` would let a runner that declares nothing
        hash identically to every other config of the same engine — the exact collision
        D-13.5 exists to stop. An incomplete dict is the one remaining way to get a false
        identity, so `tests/test_runner_contract.py` asserts each runner's declared keys
        cover its `__init__` signature and its known knobs.
        """
        ...

    def config_hash(self) -> str:
        """Stable 12-hex-char digest of `config()`. Do NOT override.

        Delegates to `contract.config_digest`, the single implementation, so no runner can
        hash differently — a per-engine digest would make cross-engine identities
        incomparable and reintroduce the collision. It lives in `contract.py` rather than
        here because `reading.py`'s `Reader` needs the identical algorithm and must not
        import the runners package to get it (Phase 13b).

        Like `version_source`, this is NOT a second source of truth (D-8.2): the runner
        declares `config()` and everything downstream derives from it.
        """
        return config_digest(self.config())

    def cost(self, image_ref: "ImageRef") -> float:
        """Estimated dollar cost of running this engine on `image_ref`.

        Delegates to `harness.cost.estimate_cost`, which reads the pricing rule off
        `self.run.pricing` (set via `@priced(...)` on the subclass's `run`); an
        untagged runner falls back to `self_hosted` ($0). No pricing logic lives here.
        """
        return estimate_cost(image_ref, self.run)
