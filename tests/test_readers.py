"""Phase 13c — the shared reader test scaffold. 100% synthetic: fake images, fake tokens.

Every crop in this file is drawn by `tests/synthetic.py` from an obviously-fake token
(`CMFN-00421`); no render, no `gt.csv`, and no real identifier is touched, so the models
here only ever see PHI-free pixels (CLAUDE.md §0).

Parametrized over `REGISTERED_READERS` so each new reader inherits the same four checks
rather than getting its own bespoke ones — the point of the arm is that readers are
comparable, and a per-reader test file is how that quietly stops being true. What the shared
checks establish:

- **contract** — `read()` returns a plain `str` and the identity fields are populated, so
  `aggregate()` can key on them (D-13.5).
- **version sourcing** — `version` equals the installed library's `__version__` (rule #9).
- **charset** — the decoder can represent `A-Z0-9-`. Several recognizers ship Chinese-first
  dictionaries; one that cannot spell a patient ID would score badly for a reason that has
  nothing to do with reading quality.
- **determinism** — the same crop twice gives the identical string. A reader that drifts
  makes every number in the arm unreproducible (plan.md §2.1).
- **known token** — a synthetic `CMFN-00421` crop reads back exactly, which is what proves
  the preprocessing/channel-order plumbing is right rather than merely not crashing.

Reader-specific constraint tests (OpenOCR's `drop_score` and its stray result file) live at
the bottom, next to the shared ones, because they are the same kind of claim: something the
library does by default that would corrupt a measurement if it happened here.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import string
from pathlib import Path

import pytest

from harness.reading import Reader, crop_for_reading
from tests.synthetic import make_synthetic_image

try:
    from harness.readers.read_svtrv2 import SVTRv2Reader
except ImportError:  # pragma: no cover - exercised only where openocr is absent
    SVTRv2Reader = None

try:
    from harness.readers.read_doctr_parseq import DoctrParseqReader
except ImportError:  # pragma: no cover - exercised only where doctr is absent
    DoctrParseqReader = None

REPO_ROOT = Path(__file__).resolve().parents[1]

# The fake token every reader is asked to read back. Deliberately ID-SHAPED — letters, a
# hyphen and digits — because that is the character class this project scores, and a gate or
# a charset that handles vendor overlay text can still fail on it (CLAUDE.md §8).
TOKEN = "CMFN-00421"

# The characters a patient ID / accession number can be made of. A reader whose decoder
# cannot emit these cannot be scored on reading quality at all.
REQUIRED_CHARS = frozenset(string.ascii_uppercase + string.digits + "-")

# One shared instance per reader: a real reader caches a built model, so reusing the
# instance keeps this whole file to a single weight load per reader.
_SVTRV2 = SVTRv2Reader() if SVTRv2Reader is not None else None
_PARSEQ = DoctrParseqReader() if DoctrParseqReader is not None else None

REGISTERED_READERS = [r for r in (_SVTRV2, _PARSEQ) if r is not None]

# Identity, not library name: `model_name` alone collides for two arms of one library, which
# is the whole point of D-13.5 — and a test id that collides is a test that silently doesn't
# run twice.
def _reader_id(reader: Reader) -> str:
    return f"{reader.model_name}:{reader.config_id}"


_NO_OPENOCR = "openocr-python not installed; install with: pip install -e '.[openocr]'"
requires_openocr = pytest.mark.skipif(_SVTRV2 is None, reason=_NO_OPENOCR)

_NO_DOCTR = "python-doctr not installed; install with: pip install -e '.[doctr]'"
requires_doctr = pytest.mark.skipif(_PARSEQ is None, reason=_NO_DOCTR)


def test_every_reader_is_registered():
    """An import failure must fail, not silently delete the parametrized tests.

    `REGISTERED_READERS` drops a reader whose module will not import. That is right for a
    machine without the engine installed, but it means a broken import here empties the list
    and every shared test below vanishes — passing, with no skip and no failure, which is
    exactly the "completed while silently skipped" outcome CLAUDE.md forbids. The skipif
    marks cover the two reader-specific tests; this covers the shared ones.
    """
    assert SVTRv2Reader is not None, _NO_OPENOCR
    assert DoctrParseqReader is not None, _NO_DOCTR
    assert len(REGISTERED_READERS) == 2


@pytest.fixture(scope="module")
def token_crop():
    """One preprocessed crop of a synthetic `CMFN-00421`, built exactly as the arm builds it.

    Goes through `crop_for_reading` rather than `Image.crop` on purpose: these tests then
    exercise the same pixels a scored run would hand the model, so a reader that silently
    disagrees with the pinned preprocessing (channel order, height, padding) fails here
    instead of on real data.
    """
    image, gt = make_synthetic_image((TOKEN,))
    return crop_for_reading(image, gt[0].bbox)


@pytest.mark.parametrize("reader", REGISTERED_READERS, ids=_reader_id)
def test_reader_contract(reader: Reader, token_crop):
    text = reader.read(token_crop)

    assert isinstance(text, str)
    assert isinstance(reader.model_name, str) and reader.model_name
    assert isinstance(reader.version, str) and reader.version
    assert isinstance(reader.config_id, str) and reader.config_id
    assert isinstance(reader.config(), dict) and reader.config()
    assert len(reader.config_hash()) == 12


@pytest.mark.parametrize("reader", REGISTERED_READERS, ids=_reader_id)
def test_reader_version_comes_from_its_library(reader: Reader):
    """`version` must be read from the installed library, never hand-typed (rule #9).

    An engine version bump invalidates prior results the same way a `gt.csv` change does, so
    a hand-typed string that goes stale would blend two different models' numbers under one
    identity. `version_source` names the module the value came from; this test imports it and
    checks the two still agree.
    """
    assert reader.version_source is not None, (
        f"{type(reader).__name__} declares version_source=None, which is reserved for test "
        "doubles with no library behind them"
    )
    module = importlib.import_module(reader.version_source)
    library_version = getattr(module, "__version__", None)
    assert library_version, f"{reader.version_source} exposes no usable __version__"
    assert reader.version == library_version


@pytest.mark.parametrize("reader", REGISTERED_READERS, ids=_reader_id)
def test_reader_charset_covers_id_characters(reader: Reader):
    missing = REQUIRED_CHARS - reader.charset()
    assert not missing, (
        f"{reader.model_name}'s decoder cannot emit {sorted(missing)} — it cannot spell a "
        "patient ID, so its score would measure its dictionary, not its reading"
    )


@pytest.mark.parametrize("reader", REGISTERED_READERS, ids=_reader_id)
def test_reader_is_deterministic(reader: Reader, token_crop):
    """Same crop twice -> identical string. Determinism is a correctness property here."""
    assert reader.read(token_crop) == reader.read(token_crop)


@pytest.mark.parametrize("reader", REGISTERED_READERS, ids=_reader_id)
def test_reader_reads_a_known_token(reader: Reader, token_crop):
    """The synthetic `CMFN-00421` crop reads back exactly — plumbing proof, not a benchmark.

    This is not a quality measurement: one clean synthetic crop says nothing about burned-in
    overlay text. It says the crop reached the model in the form the model expects. A wrong
    channel order or an undone resize typically still returns *something*, so only an exact
    match distinguishes "wired correctly" from "wired plausibly".
    """
    assert reader.read(token_crop) == TOKEN


# ---------------------------------------------------------------------------
# OpenOCR-specific constraints (see harness/readers/read_svtrv2.py's docstring)
# ---------------------------------------------------------------------------


@requires_openocr
def test_openocr_rec_task_applies_no_score_filter():
    """The recognition path must not drop low-confidence reads (CLAUDE.md §8, D-10c.4).

    `SVTRv2Reader` passes `drop_score=0.0`, but that argument is INERT on this path in
    openocr-python 0.1.5 — `OpenOCR.__init__` forwards it to the `'ocr'` task only. So the
    reader's own setting cannot be what this test checks; it asserts the library property the
    reader actually depends on. If a release starts filtering rec results, a read that the
    model produced would vanish before the scorer sees it, which silently shrinks the
    denominator of every rate in the arm and flatters the engine. That must fail loudly here.
    """
    from openocr import OpenOCR
    from tools.infer_rec import OpenRecognizer

    assert "drop_score" not in inspect.signature(OpenRecognizer.__init__).parameters, (
        "OpenRecognizer gained a drop_score parameter — check whether it now filters, and "
        "make sure SVTRv2Reader's 0.0 actually reaches it"
    )

    # The forwarding is a call inside __init__, not a signature, so it is read from the
    # source: this asserts the wiring, which is the thing that can change under us.
    tree = ast.parse(inspect.getsource(OpenOCR.__init__).lstrip())
    rec_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_init_rec_task"
    ]
    assert len(rec_calls) == 1
    forwarded = {kw.arg for kw in rec_calls[0].keywords}
    assert "drop_score" not in forwarded, (
        "OpenOCR now forwards drop_score to the rec task — SVTRv2Reader passes 0.0, so this "
        "should be safe, but confirm the value survives before deleting this assertion"
    )


@requires_openocr
def test_openocr_reader_writes_no_stray_file(token_crop, tmp_path, monkeypatch):
    """A read must not create a file the caller did not ask for.

    OpenOCR's end-to-end task writes `./e2e_results/system_results.txt` into the working
    directory. On real data that file is engine-read token text — PHI — in an unmanaged
    location, so "we don't call that task" has to be enforced rather than assumed.

    Builds its OWN reader instead of reusing the shared `_SVTRV2`, and does so inside the
    redirected working directory: the shared instance has already been constructed by an
    earlier test, so reusing it would cover `read()` only and leave the model-construction
    path — the one that resolves configs and materializes a checkpoint — untested. That also
    makes the test independent of the order the file happens to run in.

    The repo-root check afterwards is a backstop, not a claim about this read: with the CWD
    redirected, a relative write lands in `tmp_path` and is caught above. It catches a write
    that used an absolute or repo-relative path instead.
    """
    monkeypatch.chdir(tmp_path)
    reader = SVTRv2Reader()
    reader.read(token_crop)

    assert os.listdir(tmp_path) == []
    for leaked in ("e2e_results", "rec_results"):
        assert not (REPO_ROOT / leaked).exists(), (
            f"{leaked}/ appeared in the repo — a reader wrote engine output to disk"
        )


# ---------------------------------------------------------------------------
# Config identity (D-13.5)
# ---------------------------------------------------------------------------


def test_registered_readers_have_distinct_identities():
    """No two readers may share `aggregate()`'s identity key, or their rows would merge.

    A dozen readers without distinct identities average into one meaningless number — the
    failure Phase 13a exists to prevent. This is the cheap, always-on guard against the next
    reader being added with a copy-pasted `config_id`.
    """
    identities = [
        (r.model_name, r.version, r.config_id, r.config_hash()) for r in REGISTERED_READERS
    ]
    assert len(set(identities)) == len(identities)


@requires_doctr
def test_parseq_reader_does_not_collide_with_the_doctr_runner():
    """docTR-parseq vs docTR-stock: same library, same version, different measurement.

    This is the concrete case D-13.5 was written for. The two arms deliberately keep the same
    `model_name` so a report shows two configurations of one library, so `config_id` and
    `config_hash` are the only things standing between them and a silent average.
    """
    from harness.runners.run_doctr import DoctrRunner

    runner = DoctrRunner()

    assert (runner.model_name, runner.version) == (_PARSEQ.model_name, _PARSEQ.version)
    assert runner.config_id != _PARSEQ.config_id
    assert runner.config_hash() != _PARSEQ.config_hash()
