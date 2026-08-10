"""Phase 1 tests for harness/manifest.py — SYNTHETIC fake manifest only.

Every value below is fabricated (fake UIDs/dates/institutions); the real
`manifest.csv` is never read here. The PHI-shaped columns carry obvious sentinel
values so the negative test can prove they are unreachable through the public API.
"""

from __future__ import annotations

import pandas as pd
import pytest

from harness.manifest import _EXPOSED_COLUMNS, BLANK_CONTROL_STRATUM, load_manifest

# Sentinels for the columns that MUST be dropped. If any of these strings ever shows
# up in an accessor's output, PHI/quasi-identifiers are leaking.
_PHI_SENTINELS = [
    "1.2.3.FAKE.STUDY",          # study_uid
    "1.2.3.FAKE.SERIES",         # series_uid
    "1.2.840.FAKE.SOPCLASS",     # sop_class_uid
    "FAKE_GENERAL_HOSPITAL",     # institution_name
    "1999-12-31",                # study_date (full date = PHI)
]

# The 16 real manifest columns (from build_manifest.py), all with fake values.
_FAKE_ROWS = [
    # series_uid, modality, strata, manufacturer, model, body_part, series_desc,
    # n_frames, middle_frame_index, photometric, sop_class, study_uid,
    # institution, study_date, source_tar, dest_tar
    ("1.2.3.FAKE.SERIES.1", "CT", "ct_axial", "ACME_SCANNER", "M1", "CHEST", "ax",
     "20", "10", "MONOCHROME2", "1.2.840.FAKE.SOPCLASS", "1.2.3.FAKE.STUDY",
     "FAKE_GENERAL_HOSPITAL", "1999-12-31", "gs://x/src1.tar", "gs://y/dst1.tar"),
    ("1.2.3.FAKE.SERIES.2", "CT", "ct_axial", "ACME_SCANNER", "M1", "CHEST", "ax",
     "24", "12", "MONOCHROME2", "1.2.840.FAKE.SOPCLASS", "1.2.3.FAKE.STUDY",
     "FAKE_GENERAL_HOSPITAL", "1999-12-31", "gs://x/src2.tar", "gs://y/dst2.tar"),
    ("1.2.3.FAKE.SERIES.3", "CT", "ct_axial", "BETACORP", "Z9", "CHEST", "ax",
     "30", "14", "MONOCHROME2", "1.2.840.FAKE.SOPCLASS", "1.2.3.FAKE.STUDY",
     "FAKE_GENERAL_HOSPITAL", "1999-12-31", "gs://x/src3.tar", "gs://y/dst3.tar"),
    ("1.2.3.FAKE.SERIES.4", "CT", "ct_scout", "BETACORP", "Z9", "CHEST", "scout",
     "1", "0", "MONOCHROME2", "1.2.840.FAKE.SOPCLASS", "1.2.3.FAKE.STUDY",
     "FAKE_GENERAL_HOSPITAL", "1999-12-31", "gs://x/src4.tar", "gs://y/dst4.tar"),
    ("1.2.3.FAKE.SERIES.5", "CT", "ct_scout", "BETACORP", "Z9", "CHEST", "scout",
     "1", "0", "MONOCHROME2", "1.2.840.FAKE.SOPCLASS", "1.2.3.FAKE.STUDY",
     "FAKE_GENERAL_HOSPITAL", "1999-12-31", "gs://x/src5.tar", "gs://y/dst5.tar"),
    ("1.2.3.FAKE.SERIES.6", "US", "us", "ACME_SCANNER", "U7", "ABDOMEN", "us",
     "12", "6", "RGB", "1.2.840.FAKE.SOPCLASS", "1.2.3.FAKE.STUDY",
     "FAKE_GENERAL_HOSPITAL", "1999-12-31", "gs://x/src6.tar", "gs://y/dst6.tar"),
]

_FAKE_COLUMNS = [
    "series_uid", "modality", "strata", "manufacturer", "manufacturer_model",
    "body_part", "series_description", "number_of_frames", "middle_frame_index",
    "photometric_interpretation", "sop_class_uid", "study_uid", "institution_name",
    "study_date", "source_tar_path", "dest_tar_path",
]


@pytest.fixture
def fake_manifest_path(tmp_path):
    df = pd.DataFrame(_FAKE_ROWS, columns=_FAKE_COLUMNS)
    path = tmp_path / "fake_manifest.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def view(fake_manifest_path):
    return load_manifest(fake_manifest_path)


def test_wrapped_frame_holds_only_whitelisted_columns(view):
    # The strongest structural guarantee: the underlying frame carries exactly the
    # four renamed columns and nothing else.
    assert tuple(view._frame.columns) == _EXPOSED_COLUMNS
    assert set(_EXPOSED_COLUMNS) == {"modality", "stratum", "vendor", "frame_idx"}


def test_source_column_names_are_gone(view):
    cols = set(view._frame.columns)
    for dropped in ("strata", "manufacturer", "middle_frame_index", "series_uid",
                    "study_uid", "institution_name", "study_date"):
        assert dropped not in cols


def test_n_series(view):
    assert view.n_series() == 6


def test_modality_counts(view):
    assert view.modality_counts() == {"CT": 5, "US": 1}


def test_stratum_counts(view):
    assert view.stratum_counts() == {"ct_axial": 3, "ct_scout": 2, "us": 1}


def test_vendor_counts(view):
    # Vendor taken verbatim from manufacturer, never re-derived.
    assert view.vendor_counts() == {"ACME_SCANNER": 3, "BETACORP": 3}


def test_frame_idx_summary(view):
    summary = view.frame_idx_summary()
    assert summary["min"] == 0.0
    assert summary["max"] == 14.0
    assert summary["median"] == (6.0 + 10.0) / 2  # sorted: 0,0,6,10,12,14
    assert summary["mean"] == (10 + 12 + 14 + 0 + 0 + 6) / 6


def test_n_blank_control_candidates(view):
    # D-1.2: candidate pool = stratum == ct_axial (3 here). Candidates only.
    assert view.n_blank_control_candidates() == 3
    assert BLANK_CONTROL_STRATUM == "ct_axial"


def test_missing_required_column_raises(tmp_path):
    df = pd.DataFrame(_FAKE_ROWS, columns=_FAKE_COLUMNS).drop(columns=["manufacturer"])
    path = tmp_path / "broken.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="manufacturer"):
        load_manifest(path)


def test_non_numeric_frame_idx_raises(tmp_path):
    rows = [list(r) for r in _FAKE_ROWS]
    rows[0][_FAKE_COLUMNS.index("middle_frame_index")] = "not_a_number"
    df = pd.DataFrame(rows, columns=_FAKE_COLUMNS)
    path = tmp_path / "bad_frame.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="middle_frame_index|frame_idx"):
        load_manifest(path)


def test_blank_categorical_value_raises(tmp_path):
    # A blank cell in modality/strata/manufacturer must fail loud at load time,
    # not surface later as a NaN-vs-str TypeError inside _counts().
    rows = [list(r) for r in _FAKE_ROWS]
    rows[0][_FAKE_COLUMNS.index("manufacturer")] = ""
    df = pd.DataFrame(rows, columns=_FAKE_COLUMNS)
    path = tmp_path / "blank_vendor.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="vendor"):
        load_manifest(path)


def test_non_integer_frame_idx_raises(tmp_path):
    # "10.5" is numeric but not integral — must fail loud, not truncate to 10.
    rows = [list(r) for r in _FAKE_ROWS]
    rows[0][_FAKE_COLUMNS.index("middle_frame_index")] = "10.5"
    df = pd.DataFrame(rows, columns=_FAKE_COLUMNS)
    path = tmp_path / "float_frame.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-integer"):
        load_manifest(path)


def test_no_phi_value_reachable_through_public_api(view):
    """Negative test: call every public accessor and assert no dropped PHI/quasi-
    identifier sentinel appears anywhere in the returned aggregates."""
    public_methods = [
        name for name in dir(view)
        if not name.startswith("_") and callable(getattr(view, name))
    ]
    # Sanity: we actually exercised the accessors. The two keyed accessors take an
    # argument, so they cannot be called blind here — they get the same sentinel sweep
    # in their own tests below, and this assertion is what forces a future arg-taking
    # accessor to be added there too rather than silently skipped by both.
    keyed = {"attrs_for_series", "frame_idx_for_series"}
    assert set(public_methods) == {
        "n_series", "modality_counts", "stratum_counts", "vendor_counts",
        "frame_idx_summary", "n_blank_control_candidates", *keyed,
    }
    zero_arg = [name for name in public_methods if name not in keyed]

    blob = " ".join(str(getattr(view, name)()) for name in zero_arg)
    for sentinel in _PHI_SENTINELS:
        assert sentinel not in blob, f"PHI sentinel leaked through public API: {sentinel}"

    # And prove the test is meaningful: whitelisted values DO appear.
    assert "ACME_SCANNER" in blob and "ct_axial" in blob


# --- D-10.8: the keyed accessor Phase 10e joins through -----------------------------


def test_attrs_for_series_returns_the_categorical_triple(view):
    assert view.attrs_for_series("1.2.3.FAKE.SERIES.1") == ("ACME_SCANNER", "ct_axial", "CT")
    assert view.attrs_for_series("1.2.3.FAKE.SERIES.6") == ("ACME_SCANNER", "us", "US")


def test_keyed_accessors_return_no_identifier(view):
    """They take a uid but must never hand one back, nor any other quasi-identifier —
    three categorical values and one integer, and nothing else."""
    blob = " ".join(
        f"{view.attrs_for_series(row[0])} {view.frame_idx_for_series(row[0])}"
        for row in _FAKE_ROWS
    )
    for sentinel in _PHI_SENTINELS:
        assert sentinel not in blob, f"PHI sentinel leaked through a keyed accessor: {sentinel}"
    assert "ACME_SCANNER" in blob and "ct_axial" in blob


def test_frame_idx_for_series_returns_the_manifest_value(view):
    # DIAGNOSTIC ONLY — Phase 10e writes the RENDER manifest's frame_idx into gt.csv and
    # uses this one solely to count how often the two disagree.
    assert view.frame_idx_for_series("1.2.3.FAKE.SERIES.1") == 10
    assert view.frame_idx_for_series("1.2.3.FAKE.SERIES.4") == 0


def test_frame_idx_for_series_unknown_uid_raises(view):
    with pytest.raises(KeyError):
        view.frame_idx_for_series("1.2.3.FAKE.SERIES.NOT_PRESENT")


def test_attrs_for_series_unknown_uid_raises(view):
    # A silent default would put some other series' vendor on a scored gt.csv row.
    with pytest.raises(KeyError):
        view.attrs_for_series("1.2.3.FAKE.SERIES.NOT_PRESENT")


def test_duplicate_series_uid_raises(tmp_path):
    # The 10e join is 1:1 or it is wrong.
    rows = [list(r) for r in _FAKE_ROWS]
    rows[1][_FAKE_COLUMNS.index("series_uid")] = rows[0][_FAKE_COLUMNS.index("series_uid")]
    df = pd.DataFrame(rows, columns=_FAKE_COLUMNS)
    path = tmp_path / "dup_uid.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate series_uid"):
        load_manifest(path)


def test_missing_series_uid_column_raises(tmp_path):
    df = pd.DataFrame(_FAKE_ROWS, columns=_FAKE_COLUMNS).drop(columns=["series_uid"])
    path = tmp_path / "no_uid.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="series_uid"):
        load_manifest(path)


def test_series_uid_is_not_an_exposed_column():
    # It is an index key, never data. If it ever becomes a column, ManifestView's
    # whitelist guard is the thing that would stop noticing.
    assert "series_uid" not in _EXPOSED_COLUMNS
