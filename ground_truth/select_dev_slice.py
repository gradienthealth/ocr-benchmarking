#!/usr/bin/env python3
"""Phase 13h / D-13.4: draw the TUNING dev slice, from outside the scored set. 🔴 PHI-ADJACENT.

>>> ARNAV RUNS THIS. It reads `manifest.csv` (quasi-identifiers: full dates, institution
>>> pseudonyms, UIDs — a Limited Data Set) and writes a render-input list containing series
>>> UIDs. Claude reads only the PHI-free count table it prints: images per stratum and per
>>> vendor, and how many candidates were excluded. No UID, no date, no institution.

    .venv/bin/python ground_truth/select_dev_slice.py --manifest manifest.csv --exclude ground_truth/render_backmap_v2.csv --tar-dir data/series --out ground_truth/dev_slice_inputs.csv

(One line on purpose — a backslash-continued paste mangles the flags into escaped spaces.)

WHY A SEPARATE SLICE EXISTS AT ALL (D-13.4)
---------------------------------------------------------------------------------
Phase 13 step 6 measures each engine at stock and at tuned config. The tuned config has to be
chosen somewhere, and `gt.csv` is small and frozen: sweeping thresholds against it and
reporting the best result is fitting the test set — the tuned number becomes an optimistic
bound rather than a measurement (plan.md "Trap 1").

Holding a slice OUT of `gt_v1` was considered and rejected: four of its strata carry three or
fewer text-bearing images (`us_other` 1, `us_sonosite` 2, `ct_axial` 2,
`ct_secondary_capture` 3), so a stratum-spanning holdout either deletes a stratum from the
scored set or leaves it untuned, and it shrinks the headline set by a fifth. Annotating fresh
images instead keeps all 199 scored images and costs annotation time.

`dev_v1` is NOT `gt_v1` and must never merge into it. It gets its own file, its own hashes
and its own summary (`build_gt.py --out ground_truth/dev_v1.csv`).

WHAT THE DEFAULT DRAW IS, AND WHY
---------------------------------------------------------------------------------
26 images, ~515 boxes at `gt_v1`'s measured density — about a fifth of the annotation work
already done for the scored set.

  12 ultrasound across 6 vendor strata   the dense, faint overlays where detection
                                         thresholds actually bite (27-51 boxes/image)
   8 mg_2d                               sparse, and the highest-invention stratum measured
                                         (99.6%) — a threshold that quiets it matters
   2 ct_secondary_capture                vendor overlay glyphs, single-char `L`/`R` in scope
   2 ct_scout + 2 mg_tomo (blanks)       THE INVENTION FLOOR. Not optional: tuning a
                                         threshold on a set with no blank frames optimizes
                                         recall against noise with nothing to catch it —
                                         the `ct_scout` failure (CLAUDE.md §8).

The blanks are NOT `ct_axial`. gt_v1 uses all 66 of the manifest's 66 ct_axial series, so
none are left, and drawing from it would mean tuning against the scored set. `ct_scout` and
`mg_tomo` both returned zero text-bearing images in gt_v1, which is what makes them useless
as tuning signal and exactly right as an invention floor — see `BLANK_CONTROL_STRATA`.

DETERMINISM
---------------------------------------------------------------------------------
The draw is a pure function of (manifest contents, exclusions, quotas, `--seed`): candidates
are ordered by `sha256(seed|series_uid)`, not by `random`, so it reproduces on any machine
and any Python build. Re-running with the same arguments re-draws the same slice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

# Read directly rather than through `harness.manifest.ManifestView`: that adapter
# deliberately exposes NO accessor that enumerates series UIDs (it keeps them as a
# non-returnable index precisely so no aggregate call can leak one), and a selector's whole
# job is to choose UIDs. Using it here would mean adding that accessor, which would weaken
# the adapter for every other caller. So this module opens `manifest.csv` itself, keeps only
# the five columns it needs, and never prints a UID.
MANIFEST_COLUMNS = ("series_uid", "strata", "manufacturer", "modality", "number_of_frames")

# Strata drawn as BLANK CONTROLS — the dev slice's invention floor.
#
# NOT `ct_axial`, which is Gradient's designated no-text control stratum and the source of
# gt_v1's floor. Measured 2026-08-10: the manifest holds 66 ct_axial series and gt_v1 uses
# all 66, so there are ZERO unused ones. Reusing them here would also mean tuning thresholds
# against frames that are inside the scored set — the one thing D-13.4 exists to prevent,
# and it would contaminate the headline hallucination floor specifically.
#
# `ct_scout` and `mg_tomo` are the substitutes: both returned ZERO text-bearing images in
# gt_v1 (ct_scout's absence of burned-in text is the documented 2026-08-06 finding), and
# neither is consumed by gt_v1's negative control. Two independent sources rather than one,
# so if a human finds text in one of them the dev slice still has a floor.
BLANK_CONTROL_STRATA: tuple[str, ...] = ("ct_scout", "mg_tomo")

# Stratum -> images to draw. See the module docstring for the reasoning behind each.
DEFAULT_QUOTAS: dict[str, int] = {
    "us_ge": 3,
    "us_philips": 3,
    "us_siemens": 2,
    "us_toshiba_canon": 2,
    "us_samsung": 1,
    "us_sonosite": 1,
    "mg_2d": 8,
    "ct_secondary_capture": 2,
    "ct_scout": 2,
    "mg_tomo": 2,
}


class SelectionError(RuntimeError):
    """A guard tripped. Message is PHI-free — it names strata and counts, never a UID."""


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """The five needed columns of every manifest row. Never printed, never returned upward."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SelectionError(f"{path} is missing column(s) {missing} — wrong manifest?")
        return [{c: (row.get(c) or "").strip() for c in MANIFEST_COLUMNS} for row in reader]


def read_excluded_uids(paths: Sequence[Path]) -> set[str]:
    """Every `series_uid` already spoken for — the scored set, and any earlier draw.

    Accepts any CSV carrying a `series_uid` column, which covers 10b's back-maps and the
    drawn-sample lists. Excluding the SCORED SET is the point of this tool: a dev image that
    is also a `gt_v1` image would put the tuning set back inside the test set, which is the
    single thing D-13.4 exists to prevent.
    """
    uids: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "series_uid" not in (reader.fieldnames or []):
                raise SelectionError(
                    f"{path} has no 'series_uid' column — an exclusion file that excludes "
                    "nothing would silently let scored-set images into the tuning slice"
                )
            uids.update((row.get("series_uid") or "").strip() for row in reader)
    uids.discard("")
    return uids


def _order_key(seed: str, series_uid: str) -> str:
    """Deterministic pseudo-random order. Not `random` — its stream is version-dependent."""
    return hashlib.sha256(f"{seed}|{series_uid}".encode()).hexdigest()


def draw(
    rows: list[dict[str, str]],
    excluded: set[str],
    quotas: dict[str, int],
    seed: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Pick the slice. Returns (chosen rows, per-stratum {requested, available, drawn}).

    Within a stratum, candidates are taken ROUND-ROBIN across vendors before any vendor is
    taken twice, so an 8-image `mg_2d` quota cannot land entirely on one manufacturer and
    tune the thresholds to one scanner. plan.md's requirement is that the tuning slice span
    strata *and* vendors; the quotas handle the first and this handles the second.
    """
    by_stratum: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["series_uid"] in excluded or not row["series_uid"]:
            continue
        by_stratum[row["strata"]][row["manufacturer"] or "Unknown"].append(row)

    chosen: list[dict[str, str]] = []
    report: dict[str, dict[str, int]] = {}
    for stratum, want in sorted(quotas.items()):
        vendors = by_stratum.get(stratum, {})
        available = sum(len(v) for v in vendors.values())
        # Each vendor's candidates in deterministic order, then round-robin across vendors.
        queues = {
            vendor: sorted(cands, key=lambda r: _order_key(seed, r["series_uid"]))
            for vendor, cands in sorted(vendors.items())
        }
        picked: list[dict[str, str]] = []
        while len(picked) < want and any(queues.values()):
            for vendor in sorted(queues):
                if len(picked) >= want:
                    break
                if queues[vendor]:
                    picked.append(queues[vendor].pop(0))
        chosen.extend(picked)
        report[stratum] = {"requested": want, "available": available, "drawn": len(picked)}
    return chosen, report


def build_summary(report: dict[str, dict[str, int]], chosen: list[dict[str, str]],
                  n_manifest: int, n_excluded: int, seed: str, out: Path) -> str:
    """The only output surface. PHI-free by construction: strata, vendors, counts."""
    lines: list[str] = []
    add = lines.append

    add("=== Phase 13h / D-13.4 — dev slice draw (PHI-free) ===")
    add("")
    add(f"manifest rows      {n_manifest}")
    add(f"excluded uids      {n_excluded}  (the gt_v1 scored set and any earlier draw)")
    add(f"seed               {seed}")
    add(f"inputs written     {out}   ->  feed to render.py --inputs")
    add("")
    add("This slice is the TUNING set. It is NOT gt_v1 and must never be merged into it:")
    add("build it with `build_gt.py --out ground_truth/dev_v1.csv`, which now writes its own")
    add("dev_v1_set.sha256 and dev_v1_summary.txt instead of overwriting gt_v1's.")
    add("")

    add("-- per stratum -----------------------------------------------------------")
    add(f"  {'stratum':<24} {'want':>5} {'avail':>6} {'drawn':>6}")
    short = []
    for stratum, r in sorted(report.items()):
        flag = "" if r["drawn"] >= r["requested"] else "   <-- SHORT"
        if flag:
            short.append(stratum)
        add(f"  {stratum:<24} {r['requested']:>5} {r['available']:>6} {r['drawn']:>6}{flag}")
    add(f"  {'TOTAL':<24} {sum(r['requested'] for r in report.values()):>5} "
        f"{sum(r['available'] for r in report.values()):>6} {len(chosen):>6}")
    add("")

    add("-- per vendor (drawn) ----------------------------------------------------")
    for vendor, n in sorted(Counter(r["manufacturer"] or "Unknown" for r in chosen).items()):
        add(f"  {vendor:<24} {n:>4}")
    add("")

    controls = sum(1 for r in chosen if r["strata"] in BLANK_CONTROL_STRATA)
    add(f"blank controls drawn: {controls}  (from {', '.join(BLANK_CONTROL_STRATA)})")
    if controls == 0:
        add("  WARNING no blank frames in this slice. Tuning a threshold on a set with no")
        add("  blank frames optimizes recall against noise with no invention floor to catch")
        add("  it — the ct_scout failure (CLAUDE.md §8). Add a quota from")
        add(f"  {list(BLANK_CONTROL_STRATA)} before running the sweep.")
    else:
        add("  A human must confirm these really are blank. They are the invention floor, so")
        add("  a frame with text in it silently turns the floor into a recall measurement.")
    if short:
        add("")
        add(f"NOTE short on {short} — the manifest had fewer unexcluded candidates than the")
        add("quota asked for. Recorded, not silently dropped: a stratum that is short is a")
        add("stratum the tuning cannot be checked against.")
    add("")
    add("NEXT: a human confirms text presence by eye before any of these are used as signal.")
    add("A stratum label is not evidence of burned-in text (CLAUDE.md §8, the ct_scout case).")
    return "\n".join(lines)


def write_inputs(chosen: list[dict[str, str]], out: Path, tar_dir: str) -> None:
    """render.py's `--inputs` shape: `series_uid,path,frame_idx`.

    `frame_idx` is written EMPTY on purpose. render.py reads that as "no request — use
    n // 2", so the frame comes from the tar that is actually present rather than from the
    manifest's `middle_frame_index`, which is computed from a DICOM tag and disagrees
    wherever the tar holds fewer frames than the tag claims (D-10.8). The render manifest
    then records the frame that was really rendered, which is the value gt/dev rows carry.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["series_uid", "path", "frame_idx"])
        for row in chosen:
            writer.writerow([row["series_uid"], f"{tar_dir}/{row['series_uid']}.tar", ""])


def _parse_quota(values: Sequence[str] | None) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, int] = {}
    for item in values:
        stratum, _, count = item.partition("=")
        if not stratum or not count.isdigit():
            raise SelectionError(f"--quota expects stratum=N, got {item!r}")
        quotas[stratum] = int(count)
    return quotas


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True, help="the canonical manifest.csv")
    ap.add_argument("--exclude", type=Path, action="append", required=True, metavar="CSV",
                    help="repeatable; any CSV with a series_uid column — pass gt_v1's back-map")
    ap.add_argument("--out", type=Path, default=Path("ground_truth/dev_slice_inputs.csv"))
    ap.add_argument("--tar-dir", default="data/series",
                    help="local dir holding <series_uid>.tar, used to build the path column")
    ap.add_argument("--quota", action="append", metavar="STRATUM=N",
                    help=f"repeatable; default {DEFAULT_QUOTAS}")
    ap.add_argument("--seed", default="dev_v1", help="draw seed; same seed -> same slice")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        quotas = _parse_quota(args.quota)
        rows = read_manifest_rows(args.manifest)
        excluded = read_excluded_uids(args.exclude)
        chosen, report = draw(rows, excluded, quotas, args.seed)
        if not chosen:
            raise SelectionError(
                "nothing drawn — every candidate was excluded, or no quota stratum exists "
                "in this manifest. Check --exclude and --quota."
            )
        write_inputs(chosen, args.out, args.tar_dir.rstrip("/"))
    except SelectionError as exc:
        # The message is PHI-free by construction; the traceback could carry a row.
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    print(build_summary(report, chosen, len(rows), len(excluded), args.seed, args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
