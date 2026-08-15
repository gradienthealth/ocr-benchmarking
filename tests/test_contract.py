"""Phase 1 tests for harness/contract.py — SYNTHETIC fake tokens only.

Every string below is fabricated (obviously-fake IDs like "CMFN", "GRDN1234",
"ACC-0001"); no real patient IDs / accession numbers / PHI appear here. These
tests pin the FROZEN data contract: the OCRWord / OCROutput / GTToken shapes and
the frozen `normalize()` behaviour that the matcher and scorer depend on.
"""

from __future__ import annotations

import dataclasses
import unicodedata

import pytest

from harness.contract import GTToken, OCROutput, OCRWord, normalize

# --- normalize(): NFC + strip, case-sensitive, no fuzzing -------------------

# "e" + U+0301 COMBINING ACUTE ACCENT (decomposed) vs U+00E9 "é" (composed).
_E_DECOMPOSED = "é"
_E_COMPOSED = "é"


def test_nfc_composes_combining_form():
    # Decomposed input must collapse to the single composed code point.
    assert _E_DECOMPOSED != _E_COMPOSED  # sanity: they are distinct strings
    assert normalize(_E_DECOMPOSED) == _E_COMPOSED
    assert normalize(_E_DECOMPOSED) == unicodedata.normalize("NFC", _E_DECOMPOSED)


def test_whitespace_stripped_outer_not_inner():
    assert normalize("  CMFN  ") == "CMFN"
    assert normalize("\tGRDN1234\n") == "GRDN1234"
    assert normalize("\n\r ACC-0001 \t") == "ACC-0001"
    # Interior whitespace is preserved verbatim.
    assert normalize("  ACC 0001  ") == "ACC 0001"
    assert normalize("A\tB") == "A\tB"


def test_case_is_preserved():
    assert normalize("CMFN") == "CMFN"
    assert normalize("cmfn") != normalize("CMFN")
    # l vs L must not collapse.
    assert normalize("labL") != normalize("LABl")
    assert normalize("lL") == "lL"


def test_no_punctuation_stripping():
    assert normalize("ACC-0001.") == "ACC-0001."
    assert normalize("GRDN_1234!") == "GRDN_1234!"
    assert normalize("  (CMFN)  ") == "(CMFN)"


@pytest.mark.parametrize(
    "raw",
    [
        "CMFN",
        "cmfn",
        "GRDN1234",
        "ACC-0001.",
        "  padded  ",
        "\tG R D N\n",
        _E_DECOMPOSED,
        _E_COMPOSED,
        "café",
        "Straße",
        "",
        "   ",
    ],
)
def test_normalize_is_idempotent(raw):
    once = normalize(raw)
    assert normalize(once) == once


def test_empty_and_whitespace_only():
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize("\t\n\r ") == ""


@pytest.mark.parametrize(
    "raw",
    [
        "café",   # already-composed accented token
        "Straße",  # eszett (no case folding under NFC)
        "naïve",
    ],
)
def test_non_ascii_round_trips_through_nfc(raw):
    result = normalize(raw)
    assert result == unicodedata.normalize("NFC", raw)
    # NFC leaves an already-composed, unpadded token unchanged.
    assert result == raw


# --- dataclass construction -------------------------------------------------


def test_ocrword_construction_with_bbox():
    word = OCRWord(text="CMFN", bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.97)
    assert word.text == "CMFN"
    assert word.bbox == (1.0, 2.0, 3.0, 4.0)
    assert word.confidence == 0.97


def test_ocrword_construction_with_none_fields():
    word = OCRWord(text="GRDN1234", bbox=None, confidence=None)
    assert word.text == "GRDN1234"
    assert word.bbox is None
    assert word.confidence is None


def test_ocroutput_defaults_box_free_false():
    words = [
        OCRWord(text="ACC-0001", bbox=(0.0, 0.0, 5.0, 5.0), confidence=0.5),
        OCRWord(text="CMFN", bbox=None, confidence=None),
    ]
    out = OCROutput(
        words=words,
        raw_response={"fake": "payload"},
        model_name="fake-ocr-v0",
        version="fake-0.0.1",
        config_id="fake-stock",
        config_hash="0000fakehash",
    )
    assert out.words == words
    assert out.raw_response == {"fake": "payload"}
    assert out.model_name == "fake-ocr-v0"
    assert out.box_free is False


def test_ocroutput_requires_a_version_argument():
    """Omitting `version` is a constructor error, not a silent "" (D-8.4).

    This is the hand-built path: a script or notebook that assembles an OCROutput
    directly is exactly how un-versioned rows would otherwise reach aggregate().
    """
    with pytest.raises(TypeError):
        OCROutput(  # type: ignore[call-arg]
            words=[OCRWord(text="CMFN", bbox=None, confidence=None)],
            raw_response="fake-raw",
            model_name="fake-ocr-v0",
        )


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
def test_ocroutput_rejects_a_blank_version(blank):
    """Passing `version=""` explicitly is rejected too — required is not enough.

    A required field only proves the argument was supplied. Whitespace counts as blank
    because " " carries no more provenance than "" while looking set.
    """
    with pytest.raises(ValueError, match="non-empty exact engine version"):
        OCROutput(
            words=[OCRWord(text="CMFN", bbox=None, confidence=None)],
            raw_response="fake-raw",
            model_name="fake-ocr-v0",
            version=blank,
            config_id="fake-stock",
            config_hash="0000fakehash",
        )


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
def test_ocroutput_rejects_a_blank_config_id(blank):
    """Same argument as the version check, for the arm's human label (D-13.5)."""
    with pytest.raises(ValueError, match="config_id must be a non-empty"):
        OCROutput(
            words=[OCRWord(text="CMFN", bbox=None, confidence=None)],
            raw_response="fake-raw",
            model_name="fake-ocr-v0",
            version="fake-0.0.1",
            config_id=blank,
            config_hash="0000fakehash",
        )


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
def test_ocroutput_rejects_a_blank_config_hash(blank):
    """The digest is what stops two arms of one engine version merging (D-13.5).

    Caught here, at construction, rather than only in `aggregate()`: a hand-built
    `OCROutput` in a script or notebook is exactly how an un-configured row would
    otherwise reach the aggregation.
    """
    with pytest.raises(ValueError, match="config_hash must be a non-empty"):
        OCROutput(
            words=[OCRWord(text="CMFN", bbox=None, confidence=None)],
            raw_response="fake-raw",
            model_name="fake-ocr-v0",
            version="fake-0.0.1",
            config_id="fake-stock",
            config_hash=blank,
        )


def test_ocroutput_box_free_explicit_true():
    out = OCROutput(
        words=[OCRWord(text="CMFN", bbox=None, confidence=None)],
        raw_response="fake-raw",
        model_name="fake-vlm-v0",
        version="fake-vlm-0.0.1",
        config_id="fake-stock",
        config_hash="0000fakehash",
        box_free=True,
    )
    assert out.box_free is True
    assert out.words[0].text == "CMFN"


def test_gttoken_construction():
    tok = GTToken(
        image_id="img-0001",
        series_uid="1.2.3.FAKE.SERIES.1",
        modality="CT",
        vendor="ACME_SCANNER",
        stratum="ct_axial",
        frame_idx=10,
        token_text="GRDN1234",
        bbox=(1.0, 2.0, 3.0, 4.0),
        label="KEEP",
    )
    assert tok.image_id == "img-0001"
    assert tok.series_uid == "1.2.3.FAKE.SERIES.1"
    assert tok.modality == "CT"
    assert tok.vendor == "ACME_SCANNER"
    assert tok.stratum == "ct_axial"
    assert tok.frame_idx == 10
    assert tok.token_text == "GRDN1234"
    assert tok.bbox == (1.0, 2.0, 3.0, 4.0)
    assert tok.label == "KEEP"


@pytest.mark.parametrize("label", ["PHI", "KEEP"])
def test_gttoken_accepts_each_valid_label(label):
    tok = GTToken(
        image_id="img-0002",
        series_uid="1.2.3.FAKE.SERIES.2",
        modality="CT",
        vendor="BETACORP",
        stratum="ct_axial",
        frame_idx=12,
        token_text="ACC-0001",
        bbox=(0.0, 0.0, 1.0, 1.0),
        label=label,
    )
    assert tok.label == label


# --- (im)mutability ---------------------------------------------------------


def test_ocrword_is_frozen():
    word = OCRWord(text="CMFN", bbox=None, confidence=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        word.text = "GRDN1234"  # type: ignore[misc]


def test_gttoken_is_frozen():
    tok = GTToken(
        image_id="img-0003",
        series_uid="1.2.3.FAKE.SERIES.3",
        modality="CT",
        vendor="ACME_SCANNER",
        stratum="ct_axial",
        frame_idx=14,
        token_text="CMFN",
        bbox=(1.0, 2.0, 3.0, 4.0),
        label="PHI",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tok.token_text = "GRDN1234"  # type: ignore[misc]


def test_ocroutput_is_mutable():
    out = OCROutput(
        words=[OCRWord(text="CMFN", bbox=None, confidence=None)],
        raw_response="fake-raw",
        model_name="fake-ocr-v0",
        version="fake-0.0.1",
        config_id="fake-stock",
        config_hash="0000fakehash",
    )
    # OCROutput is NOT frozen: reassignment and list mutation both succeed.
    out.box_free = True
    assert out.box_free is True
    out.model_name = "fake-ocr-v1"
    assert out.model_name == "fake-ocr-v1"
    out.words.append(OCRWord(text="ACC-0001", bbox=None, confidence=None))
    assert len(out.words) == 2
