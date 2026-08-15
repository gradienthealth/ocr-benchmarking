"""Phase 13g — exact-match bench for reading arms, on SYNTHETIC fixtures only. 🟢 PHI-free.

Runs one or more `Reader`s over an identical, deterministic set of fake `CMFN`-style crops
and reports exact-match rate per arm. Every pixel is drawn in-process by `tests/synthetic.py`
from obviously-fake tokens; no render, no `gt.csv`, no manifest is touched, so this bench is
safe to run anywhere and needs no BAA clearance for any arm.

It exists because the Gemini arm (`13g`) is BAA-gated and cannot be measured on real frames
until a human opens the gate — but the question "does the arm work at all, and roughly how
well does it read short IDs" can be answered without a single real pixel. The crops are
built by the same pinned `crop_for_reading()` every reading arm uses, so the numbers are
comparable across arms in exactly the way the reading arm intends.

Run (one command, no line continuations):

    python experiments/bench_reader_synthetic.py --reader gemini --reader doctr-crnn --out results/reader_synth.json

WHAT THE NUMBER IS AND IS NOT
---------------------------------------------------------------------------------
It IS: exact match after the frozen `normalize()`, on fixture glyphs rendered with Pillow's
bundled font at 16-24px — the same comparison the real arm makes.
It is NOT a substitute for the real measurement. Fixture text is clean, evenly spaced and
noise-free; burned-in overlay text is not. Read this table as "the arm is wired up and this
is its floor on easy text," never as a projected score on `gt_v1`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.contract import normalize  # noqa: E402
from harness.reading import Reader, crop_for_reading  # noqa: E402
from tests import synthetic  # noqa: E402

# Fixed fixture set: fake, ID-shaped, and deliberately including 8-character tokens, the
# length the Gemini arm's ~59% projection is quoted against. Frozen — changing it changes
# every number this bench has ever produced, so change it loudly or not at all.
SCENES: tuple[tuple[tuple[str, ...], bool], ...] = (
    (("CMFN", "GRDN1234", "ACC-0001"), False),
    (("CMFN-0042", "GRDN5678", "ACC-0099"), False),
    (("GRDN0001", "GRDN9999", "ACC-1234"), False),
    (("CMFN", "GRDN1234", "ACC-0001"), True),
    (("CMFN-0042", "GRDN5678", "ACC-0099"), True),
    (("GRDN0001", "GRDN9999", "ACC-1234"), True),
)

EIGHT_CHAR = 8


@dataclass(frozen=True)
class Fixture:
    """One crop and the string it is known to contain. Both are fabricated."""

    expected: str
    crop: object  # PIL.Image.Image; typed loosely so this module imports without PIL stubs


def build_fixtures() -> list[Fixture]:
    """The identical crop set every arm is scored on. Deterministic across machines."""
    fixtures: list[Fixture] = []
    for index, (tokens, hard) in enumerate(SCENES):
        image, gt = synthetic.make_synthetic_image(
            tokens, seed=index, hard=hard, label="KEEP", image_id=f"synth-{index:03d}"
        )
        fixtures.extend(Fixture(t.token_text, crop_for_reading(image, t.bbox)) for t in gt)
    return fixtures


def _may_show_predictions(reader: Reader) -> bool:
    """True unless the reader could have read something other than this bench's fixtures.

    `--show-predictions` exists because an exact-match rate is undiagnosable without seeing
    what came back, and every crop in this bench is drawn by `build_fixtures()` from fake
    `CMFN`-style tokens — so a prediction here is a read of pixels this file just generated,
    never a patient identifier.

    The one arm that needs an explicit check is Gemini, because it is the only reader whose
    `source` can be `REAL`. A `REAL` reader gets counts only, whatever the flag says: it has
    no business in this bench at all, and if one ever appears here the flag must not be the
    thing that decides whether its output is printed.
    """
    from harness.readers.read_gemini import CropSource, GeminiReader

    if isinstance(reader, GeminiReader):
        return reader.source is CropSource.SYNTHETIC
    return True


def bench(
    reader: Reader, fixtures: list[Fixture], *, show_predictions: bool = False
) -> dict[str, object]:
    """Score one arm. Counts and rates by default — never a predicted string.

    `show_predictions` is honoured only for a reader that cannot have seen a real crop
    (see `_may_show_predictions`); for every other arm the flag is ignored rather than
    obeyed, because a bench that prints model output is one edit from printing it on renders.
    """
    exact = wrong = empty = 0
    exact_8 = total_8 = 0
    elapsed = 0.0
    predictions: list[dict[str, str]] = []
    show = show_predictions and _may_show_predictions(reader)

    for fixture in fixtures:
        start = perf_counter()
        text = reader.read(fixture.crop)
        elapsed += perf_counter() - start

        hit = normalize(text) == normalize(fixture.expected)
        exact += hit
        empty += not normalize(text)
        wrong += bool(normalize(text)) and not hit
        if len(fixture.expected) == EIGHT_CHAR:
            total_8 += 1
            exact_8 += hit
        if show and not hit:
            predictions.append({"expected": fixture.expected, "got": text})

    total = len(fixtures)
    return {
        "model_name": reader.model_name,
        "version": reader.version,
        "config_id": reader.config_id,
        "config_hash": reader.config_hash(),
        "crops": total,
        "exact_match": exact,
        "exact_match_rate": exact / total if total else None,
        "misread": wrong,
        "empty": empty,
        "exact_match_rate_8char": exact_8 / total_8 if total_8 else None,
        "crops_8char": total_8,
        "seconds_per_crop": elapsed / total if total else None,
        "usage": reader.usage_summary() if hasattr(reader, "usage_summary") else None,
        "misses": predictions or None,  # synthetic arms only; see `_may_show_predictions`
    }


def build_gemini(fixtures: list[Fixture]) -> Reader:
    """The 13g arm on the synthetic path: registered fixture pixels, no BAA flag needed."""
    from harness.readers.read_gemini import CropSource, GeminiReader, crop_digest

    return GeminiReader(
        source=CropSource.SYNTHETIC,
        synthetic_digests=frozenset(crop_digest(f.crop) for f in fixtures),
    )


def build_doctr_crnn(fixtures: list[Fixture]) -> Reader:
    """docTR's recognizer alone, as a local reference point for the same crops.

    Not a registered arm and not a `Runner`: docTR's harness role is end-to-end. This wraps
    its recognition head only, which is what a reading arm compares against — the reader
    question is "given the box, what does it say," and a detector must not be in the answer.
    """
    import doctr
    import numpy as np
    from doctr.models import recognition_predictor

    predictor = recognition_predictor(pretrained=True)

    class DoctrCrnnReader(Reader):
        model_name = "doctr-recognition"
        version = doctr.__version__
        version_source = "doctr"
        config_id = "crnn_vgg16_bn"

        def read(self, crop) -> str:
            out = predictor([np.asarray(crop.convert("RGB"))])
            return out[0][0] if out else ""

        def config(self) -> dict[str, object]:
            return {"arch": "crnn_vgg16_bn", "pretrained": True}

    del fixtures  # a local reader needs no registration: it never leaves the environment
    return DoctrCrnnReader()


BUILDERS = {"gemini": build_gemini, "doctr-crnn": build_doctr_crnn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", action="append", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--show-predictions",
        action="store_true",
        help="print expected/got for each miss. Honoured ONLY for readers that can have "
        "read nothing but registered synthetic fixtures; silently ignored otherwise.",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()
    rows: list[dict[str, object]] = []
    for name in args.reader:
        try:
            reader = BUILDERS[name](fixtures)
            rows.append(bench(reader, fixtures, show_predictions=args.show_predictions))
        except Exception as exc:  # an unavailable arm is reported, never silently dropped
            rows.append(
                {
                    "model_name": name,
                    "status": "NOT MEASURED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {"fixtures": len(fixtures), "scenes": len(SCENES), "arms": rows}
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
