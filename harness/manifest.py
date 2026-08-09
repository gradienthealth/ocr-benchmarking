"""Manifest adapter (Phase 1) — the ONE audited place that touches `manifest.csv`.

`manifest.csv` is PHI-adjacent (a Limited Data Set: full dates, UIDs, institution
pseudonyms — see CLAUDE.md §3). This module is the only door through which manifest
data leaves, and it is deliberately narrow:

  * It projects the manifest down to exactly FOUR whitelisted, renamed columns at
    LOAD time (`_COLUMN_MAP`). Every other column — study_date, all UIDs,
    institution_name, patient fields — is dropped immediately and never enters a
    `ManifestView`, so it is structurally unreachable through the public API.
  * Every public accessor returns COUNTS / SUMMARIES only — an int, a rate, or a
    dict of counts. No accessor returns a row, a list of identifiers, or any raw
    quasi-identifier value.

D-10.8 (resolved 2026-08-09, Phase 10e): `series_uid` is loaded as an INDEX KEY only.
It is not in `_EXPOSED_COLUMNS`, no accessor returns it, and no accessor enumerates it
— the sole way to use it is `attrs_for_series(uid)`, which requires the caller to
already hold the uid and hands back three CATEGORICAL values. Both invariants above
therefore still hold: the retained columns are unchanged, and nothing that leaves this
module is an identifier. 10e needs this because `gt.csv` carries `vendor`/`stratum`/
`modality` per row and CLAUDE.md rule #7 forbids re-deriving any of them.

Whitelist (deny-by-default): a column Gradient adds later is silently dropped, not
leaked. `vendor`/`stratum` come straight from the manifest columns and are NEVER
re-derived from model strings (CLAUDE.md §6.7).

D-1.1 (resolved): no ordering column exists or is needed; every accessor is an
order-independent aggregate. Do not add any sort/canonical-order requirement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Source manifest column  ->  exposed (renamed) column. THE WHITELIST.
# Anything not a key here is dropped at load and never reachable downstream.
_COLUMN_MAP: dict[str, str] = {
    "modality": "modality",
    "strata": "stratum",
    "manufacturer": "vendor",
    "middle_frame_index": "frame_idx",
}

_EXPOSED_COLUMNS = tuple(_COLUMN_MAP.values())  # ("modality", "stratum", "vendor", "frame_idx")

# The join key (D-10.8). Deliberately NOT in `_COLUMN_MAP`: it becomes the DataFrame
# INDEX and is never a retained column, so `ManifestView.__init__`'s "refuses
# non-whitelisted columns" guard keeps working unmodified and no aggregate accessor can
# accidentally surface a UID.
_INDEX_COLUMN = "series_uid"

# D-1.2 (resolved 2026-06-30): there is NO control-flag column. `ct_axial` is
# Gradient's DESIGNATED no-text negative-control stratum (plain axial CT has no
# burned-in header/footer text by design), used to measure a model's hallucination
# floor. The control set = the rows with this stratum. NOTE: the canonical bucket
# manifest has 66 ct_axial rows; the stale local 1000-row BQ sample has 75. This
# module just counts whatever file it loads — feed it the canonical 966-row manifest.
BLANK_CONTROL_STRATUM = "ct_axial"


def load_manifest(path: str | Path) -> ManifestView:
    """Read the manifest CSV and return a PHI-narrowed view.

    Reads the CSV, keeps ONLY the four whitelisted source columns (renaming them to
    the harness vocabulary), and drops everything else before any data is retained.
    Raises ValueError if a whitelisted source column is missing (fail loud).
    """
    df = pd.read_csv(path, dtype=str)

    missing = [src for src in (*_COLUMN_MAP, _INDEX_COLUMN) if src not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing required column(s): {sorted(missing)}")

    # Project to the whitelist and rename. Non-whitelisted columns are discarded here
    # and never enter the ManifestView.
    projected = df[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)

    # D-10.8: series_uid rides along as the INDEX, not as a column. Validate it here,
    # loudly — the 10e join is 1:1 or it is wrong, and a duplicate uid would silently
    # hand one series' vendor/stratum to another's scored rows.
    uids = df[_INDEX_COLUMN]
    if uids.isna().any() or (uids == "").any():
        raise ValueError(f"manifest has missing/blank values in {_INDEX_COLUMN!r}")
    if uids.duplicated().any():
        raise ValueError(
            f"manifest has duplicate {_INDEX_COLUMN} values "
            f"({int(uids.duplicated().sum())}) — the series join must be 1:1"
        )
    projected.index = pd.Index(uids, name=_INDEX_COLUMN)

    # Categorical columns must have no missing/blank values. A silent NaN here would
    # otherwise crash sorted() in ManifestView._counts() (NaN vs str comparison) the
    # first time modality_counts()/stratum_counts()/vendor_counts() is called — reject
    # it loudly at load time instead.
    for col in ("modality", "stratum", "vendor"):
        if projected[col].isna().any() or (projected[col] == "").any():
            raise ValueError(f"manifest has missing/blank values in {col!r}")

    # frame_idx is the only numeric column. Coerce loudly; build_manifest defaults it
    # to 0, so a non-numeric value means a malformed manifest, not a normal blank.
    frame_idx = pd.to_numeric(projected["frame_idx"], errors="coerce")
    if frame_idx.isna().any():
        raise ValueError("manifest has non-numeric values in middle_frame_index/frame_idx")
    # A frame index is a count of frames — it must be integral. Reject non-integer
    # numerics (e.g. "10.5") loudly instead of truncating them silently.
    if (frame_idx % 1 != 0).any():
        raise ValueError("manifest has non-integer values in middle_frame_index/frame_idx")
    projected["frame_idx"] = frame_idx.astype(int)

    return ManifestView(projected)


class ManifestView:
    """A read-only view over the four whitelisted manifest columns.

    Constructed only by `load_manifest`. Holds a DataFrame containing EXACTLY
    `_EXPOSED_COLUMNS` and nothing else. Every public method returns counts or
    summaries — never a row, an identifier, or a raw value.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        # Defensive: guarantee the wrapped frame carries only whitelisted columns,
        # so even internal mistakes cannot widen the exposure surface.
        extra = set(frame.columns) - set(_EXPOSED_COLUMNS)
        if extra:
            raise ValueError(f"ManifestView refuses non-whitelisted columns: {sorted(extra)}")
        # The index is PRESERVED, not reset (D-10.8): `load_manifest` puts `series_uid`
        # there so `attrs_for_series` can do the 10e join. It is an index, never a column,
        # so the guard above still governs everything that is retained as data. A view
        # built with some other index simply has no usable `attrs_for_series`.
        # .copy() so the view owns its data: a plain column selection is a slice of the
        # caller's DataFrame, and a later write through that caller would mutate what is
        # meant to be a read-only projection.
        self._frame = frame[list(_EXPOSED_COLUMNS)].copy()

    # --- counts-only accessors ---------------------------------------------------

    def n_series(self) -> int:
        """Total number of series (rows) in the manifest."""
        return int(len(self._frame))

    def modality_counts(self) -> dict[str, int]:
        """{modality -> count}, key-sorted for determinism."""
        return self._counts("modality")

    def stratum_counts(self) -> dict[str, int]:
        """{stratum -> count}, key-sorted for determinism."""
        return self._counts("stratum")

    def vendor_counts(self) -> dict[str, int]:
        """{vendor -> count}, key-sorted. Vendor is taken verbatim from the manifest
        manufacturer column — never re-derived (CLAUDE.md §6.7)."""
        return self._counts("vendor")

    def frame_idx_summary(self) -> dict[str, float]:
        """min/max/median/mean of frame_idx — a summary, NOT the per-row indices."""
        col = self._frame["frame_idx"]
        if col.empty:
            return {"min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0}
        return {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "mean": float(col.mean()),
        }

    def n_blank_control_candidates(self) -> int:
        """Count of blank-control CANDIDATE series (stratum == ``ct_axial``).

        D-1.2: a candidate count only. Blankness is human-confirmed downstream; this
        number is the pool to hand to that step, not a count of confirmed-blank frames.
        """
        return int((self._frame["stratum"] == BLANK_CONTROL_STRATUM).sum())

    def attrs_for_series(self, series_uid: str) -> tuple[str, str, str]:
        """`(vendor, stratum, modality)` for ONE series — the only per-row accessor.

        D-10.8, added for Phase 10e: `gt.csv` carries these three per row and CLAUDE.md
        rule #7 forbids re-deriving any of them from filenames or model strings, so the
        join has to come through this module.

        It does not weaken the module's contract. The caller must ALREADY hold the uid to
        ask, so nothing is disclosed that the caller did not have; what comes back is three
        categorical values and never an identifier; and there is no accessor that
        enumerates or returns the uids themselves.

        `frame_idx` is deliberately NOT returned. The manifest's `middle_frame_index` is
        `number_of_frames // 2` computed from a DICOM tag, and it disagrees with the frame
        actually rendered and annotated wherever the tar holds fewer frames than the tag
        claims. Phase 10e takes `frame_idx` from 10b's render manifest — the recorded fact
        — so exposing it here would only invite the wrong one.

        Raises:
            KeyError: `series_uid` is not in the manifest. Loud on purpose — a default
                would put some other series' vendor on a scored row.
        """
        try:
            row = self._frame.loc[series_uid]
        except KeyError:
            raise KeyError(
                "series_uid is not in the loaded manifest — wrong manifest, or the "
                "back-map and the manifest are from different pulls"
            ) from None
        return str(row["vendor"]), str(row["stratum"]), str(row["modality"])

    def frame_idx_for_series(self, series_uid: str) -> int:
        """`middle_frame_index` for one series — DIAGNOSTIC ONLY, never a `gt.csv` value.

        Deliberately separate from `attrs_for_series` rather than a fourth element of its
        tuple. Phase 10e takes `frame_idx` from 10b's render manifest (decision 6), because
        this number is `number_of_frames // 2` derived from a DICOM tag and disagrees with
        the frame actually rendered wherever the tar holds fewer frames than the tag claims.
        The split is the point: a caller that wants a row value reaches for
        `attrs_for_series` and cannot get this by accident, and a caller that wants to
        MEASURE the disagreement has to name this method and say so.

        Raises:
            KeyError: `series_uid` is not in the manifest.
        """
        try:
            return int(self._frame.loc[series_uid, "frame_idx"])
        except KeyError:
            raise KeyError("series_uid is not in the loaded manifest") from None

    # --- internal ----------------------------------------------------------------

    def _counts(self, column: str) -> dict[str, int]:
        counts = self._frame[column].value_counts(dropna=False)
        return {str(key): int(val) for key, val in sorted(counts.items())}
