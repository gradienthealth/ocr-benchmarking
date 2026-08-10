"""Phase 6 cost estimation — table-driven per-engine pricing. 🟢 PHI-free.

Every engine's marginal cost is derived from the image's pixel dimensions (megapixels),
so cost is computable without ever loading pixels — only `ImageRef.w`/`.h`. Two things are
kept deliberately separate:

- **Structural constants** (how a rule turns `(w, h)` into a resource count) live in the
  per-rule functions below. These follow from each engine's documented tokenization/tiling
  and are stable (Phase 6 decision D-6.4).
- **Dollar rates** (`$` per resource unit) live in the `PRICES` table. These are volatile —
  when a vendor changes prices, edit `PRICES`, not the functions. Rates verified Jul 2026.

`estimate_cost` returns **dollars** so a single comparable float flows through `score()` and
`aggregate()`. Engines are tagged, not sniffed: a `run_func` carries a `.pricing` attribute
(set via the `@priced(...)` decorator in its runner); an untagged callable falls back to
`self_hosted` → `$0`, never a guess. This keeps engine-specific pricing knowledge in the
runner, out of the fixed harness (CLAUDE.md engine-agnostic rule).

This module touches ZERO PHI: pure arithmetic over image dimensions and a price table.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # avoid an import cycle with harness.py; only .w/.h are used at runtime
    from harness.harness import ImageRef

# $ per resource unit. Structural constants that produce the unit count are in the
# functions below; only the volatile rates live here (verified Jul 2026).
PRICES: dict[str, float] = {
    "claude": 5.00 / 1_000_000,   # $/input-token — Claude Opus 4.8 input ($5.00 / 1M)
    "gpt4o": 2.50 / 1_000_000,    # $/input-token — GPT-4o input ($2.50 / 1M)
    "cloud": 0.0015,              # $/image — AWS Textract / Google Vision ($1.50 / 1000)
    "self_hosted": 0.0,           # $/image — docTR / PaddleOCR / EasyOCR / Qwen VLM
}

_TILE = 512          # GPT-4o high-detail tile edge, in pixels
_TILE_TOKENS = 170   # tokens per 512x512 tile
_BASE_TOKENS = 85    # GPT-4o fixed per-image base
_CLAUDE_DIVISOR = 750  # Anthropic image-token approximation: tokens ~= (w*h) / 750


def _claude_tokens(w: int, h: int) -> int:
    """Anthropic image tokens: ceil((w*h) / 750) (D-6.4)."""
    return math.ceil((w * h) / _CLAUDE_DIVISOR)


def _gpt4o_tokens(w: int, h: int) -> int:
    """GPT-4o high-detail image tokens: 85 base + 170 per 512x512 tile (D-6.4).

    Tiling is applied to the fed image's pixel dimensions directly. GPT-4o's own
    pre-tiling resize (shortest side to 768, longest <= 2048) is intentionally NOT
    modeled here: the harness scores the exact image it feeds, and D-6.4 fixes the
    tile/token constants — keeping the estimate deterministic and comparable.
    """
    tiles = math.ceil(w / _TILE) * math.ceil(h / _TILE)
    return _BASE_TOKENS + _TILE_TOKENS * tiles


# rule -> (resource_fn(w, h) -> units, price_key). One function per pricing rule (D-6.4).
_RULES: dict[str, tuple[Callable[[int, int], float], str]] = {
    "claude": (_claude_tokens, "claude"),
    "gpt4o": (_gpt4o_tokens, "gpt4o"),
    "cloud": (lambda w, h: 1, "cloud"),          # flat per image
    "self_hosted": (lambda w, h: 0, "self_hosted"),  # no marginal cost
}


def estimate_cost(img: "ImageRef", run_func: Callable) -> float:
    """Estimated dollar cost of running `run_func` on one image, from its dimensions.

    The pricing rule is read from `run_func.pricing` (set by `@priced(...)` in the
    runner). An untagged callable falls back to `self_hosted` ($0) — a fail-safe
    default, never a heuristic guess about the engine.
    """
    rule = getattr(run_func, "pricing", "self_hosted")
    resource_fn, price_key = _RULES[rule]
    return resource_fn(img.w, img.h) * PRICES[price_key]


def priced(rule: str) -> Callable[[Callable], Callable]:
    """Tag a `run_func` with its pricing rule: `@priced("claude")`.

    Attaches `.pricing = rule` so `estimate_cost` can dispatch without the harness
    knowing anything engine-specific. Validates the rule name up front so a typo
    fails at decoration time, not silently at cost time.
    """
    if rule not in _RULES:
        raise ValueError(f"unknown pricing rule {rule!r}; expected one of {sorted(_RULES)}")

    def deco(fn: Callable) -> Callable:
        fn.pricing = rule
        return fn

    return deco
