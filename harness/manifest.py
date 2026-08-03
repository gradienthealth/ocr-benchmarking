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

    missing = [src for src in _COLUMN_MAP if src not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing required column(s): {sorted(missing)}")

    # Project to the whitelist and rename. Non-whitelisted columns are discarded here
    # and never enter the ManifestView.
    projected = df[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)

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
        self._frame = frame[list(_EXPOSED_COLUMNS)].reset_index(drop=True)

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

    # --- internal ----------------------------------------------------------------

    def _counts(self, column: str) -> dict[str, int]:
        counts = self._frame[column].value_counts(dropna=False)
        return {str(key): int(val) for key, val in sorted(counts.items())}
