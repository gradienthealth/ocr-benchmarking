"""Phase 13h — the named stock/tuned arms of step 6. 🟢 PHI-free.

Step 6 measures every end-to-end engine TWICE — once at stock config, once at tuned — and
reports both as separate rows. A single number per engine cannot distinguish "this engine
reads burned-in tokens well" from "our config for this engine happened to work," and the
findings section will be read as the former (plan.md, "Step 6 in detail").

This module is the one place the six arms are named. It builds runners; it does not score,
rank, or decide anything — `harness.py`/`matching.py`/`aggregate.py` stay untouched so the
cross-engine comparison stays fair.

    from harness.runners.arms import ARM_NAMES, build_arm
    runner = build_arm("pp-ocrv6_medium:tuned")

WHY THE TUNED ARMS ARE NOT LITERALS HERE
---------------------------------------------------------------------------------
A tuned arm's thresholds are an OUTPUT of the dev-slice sweep, not a constant someone picks.
They are read from `experiments/tuned_configs.json`, which the sweep writes and a human
freezes. Until that file has an entry for an engine, `build_arm("<engine>:tuned")` raises
instead of guessing — an arm that silently fell back to stock values while calling itself
"tuned" would put a mislabelled row straight into the report.

D-13.4 — WHY THE SWEEP RUNS ON A DEV SLICE
---------------------------------------------------------------------------------
Resolved 2026-08-10: tuning happens on `dev_v1`, a separately annotated slice drawn from
OUTSIDE the frozen `gt_v1` scored set. Sweeping thresholds against `gt.csv` and reporting the
best result is fitting the test set — the tuned number becomes an optimistic bound rather
than a measurement, and the stock-vs-tuned delta gets inflated by exactly the amount of
overfitting (plan.md "Trap 1"). Holding a slice out of `gt_v1` instead was rejected because
four strata carry 3 or fewer text-bearing images and cannot be split at all.

The consequence for this module: once a tuned config is frozen here, it is FROZEN. Re-tuning
after seeing `gt_v1` numbers reintroduces precisely the bias the dev slice was annotated to
avoid.

EASYOCR RUNS BACKWARDS — READ BEFORE USING IT
---------------------------------------------------------------------------------
docTR and PP-OCRv6 are stock with no argument and tuned by argument. EasyOCR has shipped a
NON-default config since Phase 9 (0.2 on both detection knobs vs library defaults 0.7/0.4),
so its no-argument instance is *our* configuration — the tuned side — and its stock arm is
the one that must be constructed explicitly, from the library defaults. Its pair therefore
needs no sweep: both ends are already known, which is why `easyocr:tuned` is buildable today
while the other two tuned arms are not. It answers the question someone will definitely ask —
how much of EasyOCR's over-redaction is EasyOCR, and how much is our threshold choice?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # importing a runner pulls in its engine; keep that out of type-check time
    from harness.runners.base import Runner

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNED_CONFIGS = REPO_ROOT / "experiments" / "tuned_configs.json"

# The six arms of step 6. Ordered engine-major so a report lists each engine's pair together.
ARM_NAMES = (
    "doctr:stock",
    "doctr:tuned",
    "pp-ocrv6_medium:stock",
    "pp-ocrv6_medium:tuned",
    "easyocr:stock",
    "easyocr:tuned",
)


class ArmNotFrozenError(RuntimeError):
    """A tuned arm was requested before the dev-slice sweep froze its thresholds."""


def load_tuned_config(engine: str, path: Path | None = None) -> dict[str, object]:
    """The frozen tuned knobs for `engine`, or raise. PHI-free file: knob values only.

    Raises `ArmNotFrozenError` rather than returning `{}` — an empty dict would build a
    runner identical to the stock arm, which would then report a stock measurement under the
    "tuned" label and make the step-6 delta read as zero improvement rather than as work not
    yet done.
    """
    path = path or TUNED_CONFIGS
    if not path.is_file():
        raise ArmNotFrozenError(
            f"{path} does not exist — no tuned config has been frozen yet. Run the dev-slice "
            "sweep (experiments/sweep_stock_vs_tuned.py) first; D-13.4 forbids tuning "
            "against gt_v1."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    entry = doc.get("engines", {}).get(engine)
    if not entry or not entry.get("config"):
        raise ArmNotFrozenError(
            f"no frozen tuned config for {engine!r} in {path}. The sweep writes it; until "
            "then this arm must not run, because a tuned arm built from stock values would "
            "report a stock measurement under a tuned label (CLAUDE.md rule #9)."
        )
    return dict(entry["config"])


def build_arm(name: str, tuned_configs: Path | None = None) -> Runner:
    """Construct one named arm. Engine imports are local so one missing engine is not fatal.

    A missing library raises that library's own ImportError, which names the extra to
    install — the same behaviour `tests/test_runner_contract.py` already relies on.
    """
    if name not in ARM_NAMES:
        raise ValueError(f"unknown arm {name!r}; the six step-6 arms are {list(ARM_NAMES)}")
    engine, _, variant = name.partition(":")

    if engine == "doctr":
        from harness.runners.run_doctr import DoctrRunner

        if variant == "stock":
            return DoctrRunner()  # D-9.2: pretrained defaults, no override
        return DoctrRunner(**load_tuned_config(engine, tuned_configs))

    if engine == "pp-ocrv6_medium":
        from harness.runners.run_paddle_v6 import PaddleV6Runner

        if variant == "stock":
            # Stock THRESHOLDS. return_word_box / enable_mkldnn / the three orientation
            # stages are correctness requirements and are set identically on both arms by
            # the runner itself — see PaddleV6Runner.config().
            return PaddleV6Runner()
        return PaddleV6Runner(**load_tuned_config(engine, tuned_configs))

    from harness.runners.run_easyocr import (
        LIBRARY_LOW_TEXT,
        LIBRARY_TEXT_THRESHOLD,
        EasyOcrRunner,
    )

    if variant == "stock":
        return EasyOcrRunner(
            text_threshold=LIBRARY_TEXT_THRESHOLD,
            low_text=LIBRARY_LOW_TEXT,
        )
    # The shipped 0.2/0.2 floor config — no sweep needed, see the module docstring.
    return EasyOcrRunner()
