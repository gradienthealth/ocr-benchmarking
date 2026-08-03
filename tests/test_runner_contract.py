"""Phase 8/9 tests for harness/runners/*. SYNTHETIC fixtures only, zero PHI.

Parametrized over `REGISTERED_RUNNERS`: the contract test itself makes no assumption
about any specific engine, so a future `Runner` subclass is covered by adding it to that
list — a one-line registration, not a new test or a change to the test's logic (plan.md
Phase 8 acceptance criteria). `EchoRunner` is a trivial test double, not a real engine —
no pixel access, no GT dependency — defined here rather than in base.py since it's
test-only scaffolding.

Phase 9 adds REAL engines (docTR, then PP-OCRv6_medium, then EasyOCR). Two consequences for
this file:
- A real runner loads pixels from `ImageRef.path`, so the shared image fixture now writes
  an actual PNG (into pytest's tmp dir, never the repo) instead of using a dummy path.
  Every image here comes from `tests.synthetic` — fake tokens, no PHI, ever.
- Each engine lives behind its own optional extra, so each is registered conditionally. An
  absent engine is NOT hidden: `test_<engine>_runner_is_registered` skips with an explicit
  reason, which `-ra` (see pyproject) prints — a missing engine must never read as green.
"""

from __future__ import annotations

import pytest

from harness.contract import GTToken, OCROutput, OCRWord
from harness.harness import ImageRef
from harness.matching import iou
from harness.runners import Runner
from tests.synthetic import make_synthetic_image

try:  # optional extra: pip install -e ".[doctr]"
    from harness.runners.run_doctr import DoctrRunner
except ImportError:  # pragma: no cover - exercised only in a core-only install
    DoctrRunner = None

try:  # optional extra: pip install -e ".[paddle]"
    from harness.runners.run_paddle_v6 import PaddleV6Runner
except ImportError:  # pragma: no cover - exercised only in a core-only install
    PaddleV6Runner = None

try:  # optional extra: pip install -e ".[easyocr]"
    from harness.runners.run_easyocr import DETECTION_THRESHOLD, EasyOcrRunner
except ImportError:  # pragma: no cover - exercised only in a core-only install
    EasyOcrRunner = None
    DETECTION_THRESHOLD = None


class EchoRunner(Runner):
    """Emits one fixed word inside the image bounds. Not a real engine."""

    model_name = "echo-runner"
    version = "echo-0.0.1"

    def run(self, image_ref: ImageRef) -> OCROutput:
        w = min(50.0, float(image_ref.w))
        h = min(20.0, float(image_ref.h))
        word = OCRWord(text="ECHO", bbox=(0.0, 0.0, w, h), confidence=0.99)
        return OCROutput(
            words=[word],
            raw_response={"synthetic": True},
            model_name=self.model_name,
            version=self.version,
        )


# One shared instance per engine: a real runner caches a built predictor, so reusing the
# instance keeps the whole file to a single model construction / weight load.
_DOCTR_RUNNER = DoctrRunner() if DoctrRunner is not None else None
_PADDLE_RUNNER = PaddleV6Runner() if PaddleV6Runner is not None else None
_EASYOCR_RUNNER = EasyOcrRunner() if EasyOcrRunner is not None else None

REGISTERED_RUNNERS = (
    [EchoRunner()]
    + ([_DOCTR_RUNNER] if _DOCTR_RUNNER is not None else [])
    + ([_PADDLE_RUNNER] if _PADDLE_RUNNER is not None else [])
    + ([_EASYOCR_RUNNER] if _EASYOCR_RUNNER is not None else [])
)

_NO_DOCTR = "python-doctr not installed; install with: pip install -e '.[doctr]'"
requires_doctr = pytest.mark.skipif(_DOCTR_RUNNER is None, reason=_NO_DOCTR)

_NO_PADDLE = "paddleocr not installed; install with: pip install -e '.[paddle]'"
requires_paddle = pytest.mark.skipif(_PADDLE_RUNNER is None, reason=_NO_PADDLE)

_NO_EASYOCR = "easyocr not installed; install with: pip install -e '.[easyocr]'"
requires_easyocr = pytest.mark.skipif(_EASYOCR_RUNNER is None, reason=_NO_EASYOCR)

# A single fake token whose exact pixel box `make_synthetic_image` returns as ground truth.
_TOKEN = "CMFN-0042"


@pytest.fixture(scope="session")
def synthetic_image(tmp_path_factory) -> tuple[ImageRef, list[GTToken]]:
    """One synthetic image on disk plus its exact GT boxes.

    Session-scoped so a real engine reads the same file for every test in this module.
    The PNG is written to pytest's tmp dir — outside the repo, and synthetic regardless.
    `w`/`h` are taken from the saved image so `ImageRef` cannot disagree with the pixels.
    """
    image, gt = make_synthetic_image((_TOKEN,))
    path = tmp_path_factory.mktemp("synthetic") / "synth-000.png"
    image.save(path)
    w, h = image.size
    ref = ImageRef(
        id="synth-000",
        path=str(path),
        w=w,
        h=h,
        stratum="synth_ct_axial",
        modality="CT",
        vendor="FakeVendorA",
        frame_idx=0,
    )
    return ref, gt


@pytest.mark.parametrize("runner", REGISTERED_RUNNERS, ids=lambda r: r.model_name)
def test_runner_contract(runner: Runner, synthetic_image):
    img, _ = synthetic_image
    out = runner.run(img)

    assert isinstance(out, OCROutput)

    for word in out.words:
        if word.bbox is None:
            assert out.box_free, "a non-box-free output must not have a None bbox"
        else:
            x0, y0, x1, y1 = word.bbox
            assert 0 <= x0 <= x1 <= img.w
            assert 0 <= y0 <= y1 <= img.h
        assert word.confidence is None or isinstance(word.confidence, float)

    assert isinstance(out.model_name, str) and out.model_name
    assert isinstance(out.version, str) and out.version


def test_doctr_runner_is_registered():
    """Fail loud, not silent: an absent engine must be visible, not just missing."""
    if _DOCTR_RUNNER is None:
        pytest.skip(_NO_DOCTR)
    assert any(r.model_name == "doctr" for r in REGISTERED_RUNNERS)


@requires_doctr
def test_doctr_version_comes_from_the_library():
    """`version` must be the installed library's own string, never hand-typed (rule #9)."""
    import doctr

    assert _DOCTR_RUNNER.version == doctr.__version__
    assert _DOCTR_RUNNER.version  # non-empty: aggregate() keys on (model_name, version)


@requires_doctr
def test_doctr_coordinate_conversion(synthetic_image):
    """The crux: docTR's normalized geometry -> pixel box, checked against the KNOWN box.

    The expectation is not fabricated — `make_synthetic_image` drew `_TOKEN` and returned
    the exact pixel bbox it drew it at, and the comparison reuses the harness's own frozen
    `iou` (harness/matching.py) rather than a reimplementation. IoU >= 0.5 is the Phase 9
    acceptance threshold and matches the matcher's default.
    """
    img, gt = synthetic_image
    truth = gt[0].bbox

    out = _DOCTR_RUNNER.run(img)

    assert out.words, "docTR found no text on a synthetic image that has one token"
    best = max(iou(w.bbox, truth) for w in out.words)
    assert best >= 0.5, f"best IoU {best:.4f} vs known box {truth} is below the 0.5 threshold"


@requires_doctr
def test_doctr_is_deterministic(synthetic_image):
    """Same image twice -> identical output. Determinism is a correctness property here."""
    img, _ = synthetic_image

    first = _DOCTR_RUNNER.run(img)
    second = _DOCTR_RUNNER.run(img)

    def fingerprint(out: OCROutput) -> list[tuple]:
        return [(w.text, w.bbox, w.confidence) for w in out.words]

    assert fingerprint(first) == fingerprint(second)


# --- PP-OCRv6_medium (PaddleOCR) ----------------------------------------------


def test_paddle_runner_is_registered():
    """Fail loud, not silent: an absent engine must be visible, not just missing."""
    if _PADDLE_RUNNER is None:
        pytest.skip(_NO_PADDLE)
    assert any(r.model_name == "pp-ocrv6_medium" for r in REGISTERED_RUNNERS)


@requires_paddle
def test_paddle_version_comes_from_the_library():
    """`version` must be the installed library's own string, never hand-typed (rule #9)."""
    import paddleocr

    assert _PADDLE_RUNNER.version == paddleocr.__version__
    assert _PADDLE_RUNNER.version  # non-empty: aggregate() keys on (model_name, version)


@requires_paddle
def test_paddle_coordinate_conversion(synthetic_image):
    """The crux: PP-OCR's 4-point quad -> pixel min/max box, checked against the KNOWN box.

    The expectation is not fabricated — `make_synthetic_image` drew `_TOKEN` and returned the
    exact pixel bbox it drew it at, and the comparison reuses the harness's own frozen `iou`
    (harness/matching.py) rather than a reimplementation. IoU >= 0.5 is the Phase 9 acceptance
    threshold and matches the matcher's default.
    """
    img, gt = synthetic_image
    truth = gt[0].bbox

    out = _PADDLE_RUNNER.run(img)

    assert out.words, "PP-OCRv6 found no text on a synthetic image that has one token"
    best = max(iou(w.bbox, truth) for w in out.words)
    assert best >= 0.5, f"best IoU {best:.4f} vs known box {truth} is below the 0.5 threshold"


@requires_paddle
def test_paddle_returns_word_level_not_line_level_boxes(tmp_path):
    """Guards `return_word_box=True` — the setting the whole contract hinges on.

    PP-OCR detects text LINES. Without word-level output an overlay line like
    "ID: CMFN-0042 ACC-0099" comes back as ONE box spanning the whole line, which scores below
    the matcher's 0.5 IoU against the box for a single token — silently breaking matching on
    exactly the burned-in shape this project scores. The single-token fixture used by the
    coordinate test above CANNOT catch that (there, line == word), so this test draws two fake
    tokens on ONE line and asserts they come back as separate words.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (640, 200), 0)
    ImageDraw.Draw(image).text(
        (30, 30), f"ID: {_TOKEN} ACC-0099", font=ImageFont.load_default(size=20), fill=255
    )
    path = tmp_path / "two-tokens-one-line.png"
    image.save(path)
    w, h = image.size
    ref = ImageRef(
        id="synth-line", path=str(path), w=w, h=h,
        stratum="synth_ct_axial", modality="CT", vendor="FakeVendorA", frame_idx=0,
    )

    texts = [word.text for word in _PADDLE_RUNNER.run(ref).words]

    # Line-level output would yield a single "ID: CMFN-0042 ACC-0099" entry and neither of these.
    assert _TOKEN in texts, f"{_TOKEN!r} not returned as its own word; got {texts}"
    assert "ACC-0099" in texts, f"'ACC-0099' not returned as its own word; got {texts}"


@requires_paddle
def test_paddle_blank_image_returns_no_words(blank_image, tmp_path):
    """The negative-control path: a blank frame must traverse the runner without crashing.

    Exercises the empty-detection branch — PP-OCR returns a result whose word keys are absent
    or empty, which is where a missing-key access would blow up. Zero words is also the correct
    hallucination-floor answer for a truly blank frame, so it is asserted rather than just
    checked for absence of an exception.
    """
    image, gt = blank_image
    assert gt == []  # fixture sanity: the blank control has no GT tokens by construction
    path = tmp_path / "blank.png"
    image.save(path)
    w, h = image.size
    ref = ImageRef(
        id="synth-blank", path=str(path), w=w, h=h,
        stratum="synth_ct_axial", modality="CT", vendor="FakeVendorA", frame_idx=0,
    )

    out = _PADDLE_RUNNER.run(ref)

    assert isinstance(out.words, list)
    assert out.words == [], f"hallucinated {len(out.words)} word(s) on a blank frame"


@requires_paddle
def test_paddle_is_deterministic(synthetic_image):
    """Same image twice -> identical output. Determinism is a correctness property here."""
    img, _ = synthetic_image

    first = _PADDLE_RUNNER.run(img)
    second = _PADDLE_RUNNER.run(img)

    def fingerprint(out: OCROutput) -> list[tuple]:
        return [(w.text, w.bbox, w.confidence) for w in out.words]

    assert fingerprint(first) == fingerprint(second)


# --- EasyOCR (floor/baseline arm) ---------------------------------------------


def _one_line_two_tokens(tmp_path, name: str) -> ImageRef:
    """A synthetic image with two fake tokens on ONE line — the burned-in-overlay shape."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (640, 200), 0)
    ImageDraw.Draw(image).text(
        (30, 30), f"ID: {_TOKEN} ACC-0099", font=ImageFont.load_default(size=20), fill=255
    )
    path = tmp_path / name
    image.save(path)
    w, h = image.size
    return ImageRef(
        id="synth-line", path=str(path), w=w, h=h,
        stratum="synth_ct_axial", modality="CT", vendor="FakeVendorA", frame_idx=0,
    )


def test_easyocr_runner_is_registered():
    """Fail loud, not silent: an absent engine must be visible, not just missing."""
    if _EASYOCR_RUNNER is None:
        pytest.skip(_NO_EASYOCR)
    assert any(r.model_name == "easyocr" for r in REGISTERED_RUNNERS)


@requires_easyocr
def test_easyocr_version_comes_from_the_library():
    """`version` must be the installed library's own string, never hand-typed (rule #9)."""
    import easyocr

    assert _EASYOCR_RUNNER.version == easyocr.__version__
    assert _EASYOCR_RUNNER.version  # non-empty: aggregate() keys on (model_name, version)


@requires_easyocr
def test_easyocr_passes_the_0_2_detection_threshold_to_the_detector(synthetic_image, monkeypatch):
    """The floor config is 0.2 on BOTH knobs — asserted at the call, not at the constant.

    EasyOCR gates a detection twice: `low_text` binarizes CRAFT's score map into candidate
    regions, then `text_threshold` discards any component whose peak score is below it. A
    region must clear both, so the effective threshold is `max(low_text, text_threshold)` and
    leaving either at its library default (0.4 / 0.7) would pin the effective floor there
    instead of at 0.2. Spying on the real `readtext` call is what makes this test a guard: a
    constant can be right while the call site quietly drifts back toward the defaults.
    """
    img, _ = synthetic_image
    captured: dict = {}
    real_readtext = _EASYOCR_RUNNER.reader.readtext

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_readtext(*args, **kwargs)

    monkeypatch.setattr(_EASYOCR_RUNNER.reader, "readtext", spy)
    _EASYOCR_RUNNER.run(img)

    assert DETECTION_THRESHOLD == 0.2
    assert captured["text_threshold"] == 0.2, "text_threshold drifted off the 0.2 floor config"
    assert captured["low_text"] == 0.2, "low_text drifted off the 0.2 floor config"


@requires_easyocr
def test_easyocr_box_conversion_lands_on_the_known_gt_box(synthetic_image):
    """Proves the coordinate conversion itself: right place, right axes, right units.

    `make_synthetic_image` drew `_TOKEN` and returned the exact pixel bbox it drew it at, so
    this expectation is measured, not fabricated. Containment (plus a width sanity check) is
    what isolates *conversion* correctness from EasyOCR's box tightness: a y-flip, a
    normalized-coordinate leak, or a transposed quad would all put the box somewhere else
    entirely. How tightly the box hugs the text is a separate property — see the IoU test below.
    """
    img, gt = synthetic_image
    x0, y0, x1, y1 = gt[0].bbox

    out = _EASYOCR_RUNNER.run(img)

    assert out.words, "EasyOCR found no text on a synthetic image that has one token"
    best = max(out.words, key=lambda w: iou(w.bbox, gt[0].bbox)).bbox
    bx0, by0, bx1, by1 = best
    assert bx0 <= x0 and by0 <= y0 and bx1 >= x1 and by1 >= y1, (
        f"predicted box {best} does not contain the known box {gt[0].bbox} — the conversion "
        "is misplacing the box, not merely loosening it"
    )
    assert (bx1 - bx0) < 2 * (x1 - x0), f"predicted box {best} is not the width of the token"


@requires_easyocr
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED 2026-07-31, not a flake: EasyOCR's regions are ~1.9x taller than the glyph "
        "content they bound (26px vs a 14px tight GT box), so IoU against a tight ground-truth "
        "box lands at ~0.41-0.50 — at or below the matcher's 0.5 threshold. The conversion is "
        "correct (see the containment test above); the engine's boxes are simply loose. "
        "add_margin=0.0 only reaches 0.493, so it is not tuned around. strict=True: if this "
        "ever passes, the geometry changed and the Phase 9 write-up must be updated."
    ),
)
def test_easyocr_coordinate_conversion_meets_the_0_5_iou_bar(synthetic_image):
    """Phase 9's acceptance criterion (plan.md): IoU >= 0.5 vs the known box. EasyOCR misses it.

    Kept as a strict xfail rather than deleted or weakened so the bar stays visible in the test
    run (`-ra` prints it) and the failure is attributable. Consequence for the bake-off: at the
    harness's 0.5 IoU matcher threshold a correct EasyOCR read can still fail to match, scoring
    as an omission AND a hallucination — a geometry failure, not a reading failure.
    """
    img, gt = synthetic_image

    out = _EASYOCR_RUNNER.run(img)

    assert out.words, "EasyOCR found no text on a synthetic image that has one token"
    best = max(iou(w.bbox, gt[0].bbox) for w in out.words)
    assert best >= 0.5, f"best IoU {best:.4f} vs known box {gt[0].bbox} is below the 0.5 threshold"


@requires_easyocr
def test_easyocr_returns_line_level_not_word_level_boxes(tmp_path):
    """Pins MEASURED granularity: EasyOCR returns one region per LINE, not per word.

    PaddleOCR had the same default and `return_word_box=True` fixed it. EasyOCR has no
    equivalent, and `width_ths` does not split the line (measured at 0.5 / 0.1 / 0.0) — the
    detector emits the whole line as a single region. This test asserts the limitation rather
    than hiding it, so a future EasyOCR release that changes the behavior fails here loudly
    instead of silently altering what the bake-off measures.
    """
    ref = _one_line_two_tokens(tmp_path, "easyocr-two-tokens-one-line.png")

    texts = [word.text for word in _EASYOCR_RUNNER.run(ref).words]

    assert _TOKEN not in texts, (
        f"EasyOCR now returns word-level boxes ({texts}) — granularity changed; update the "
        "Phase 9 findings and the runner docstring before relying on this."
    )
    assert any(_TOKEN in t for t in texts), f"{_TOKEN!r} not read at all; got {texts}"


@requires_easyocr
def test_easyocr_blank_image_returns_no_words(blank_image, tmp_path):
    """The negative-control path: a blank frame must traverse the runner without crashing.

    Exercises the empty-detection branch, where `readtext` returns an empty list. Zero words is
    also the correct hallucination-floor answer for a truly blank frame — and a real question
    for this arm specifically, since a 0.2 detection threshold is exactly the config that would
    make an engine fire on noise. Measured 2026-07-31: zero words at 0.2.
    """
    image, gt = blank_image
    assert gt == []  # fixture sanity: the blank control has no GT tokens by construction
    path = tmp_path / "blank.png"
    image.save(path)
    w, h = image.size
    ref = ImageRef(
        id="synth-blank", path=str(path), w=w, h=h,
        stratum="synth_ct_axial", modality="CT", vendor="FakeVendorA", frame_idx=0,
    )

    out = _EASYOCR_RUNNER.run(ref)

    assert isinstance(out.words, list)
    assert out.words == [], f"hallucinated {len(out.words)} word(s) on a blank frame"


@requires_easyocr
def test_easyocr_is_deterministic(synthetic_image):
    """Same image twice -> identical output. Determinism is a correctness property here."""
    img, _ = synthetic_image

    first = _EASYOCR_RUNNER.run(img)
    second = _EASYOCR_RUNNER.run(img)

    def fingerprint(out: OCROutput) -> list[tuple]:
        return [(w.text, w.bbox, w.confidence) for w in out.words]

    assert fingerprint(first) == fingerprint(second)
