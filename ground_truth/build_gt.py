#!/usr/bin/env python3
"""Phase 10e: the 10d human review records → the frozen, hashed `gt.csv`. 🔴 PHI-TOUCHING.

>>> ARNAV RUNS THIS. Claude never reads `gt.csv`, a review record, a render, a back-map, or
>>> a row of `manifest.csv`. The PHI-free summary is the only thing that comes back.

    .venv/bin/python ground_truth/build_gt.py \
        --set      renders/v2:ground_truth/seed_v2 \
        --groups   ground_truth/seed_groups_v2.csv \
        --backmap  ground_truth/render_backmap_v2.csv \
        --manifest manifest.csv \
        --drawn    gt_sample_v2.csv

Every input is an explicit argument. There are five `render_backmap*.csv` files in this repo
and picking the wrong one mislabels every row in the artifact everything else is scored
against — so nothing here is defaulted to "the obvious one".

WHAT THIS WRITES
---------------------------------------------------------------------------------
  ground_truth/gt.csv                  PHI (token_text). GITIGNORED. The yardstick.
  ground_truth/gt.csv.sha256           PHI-FREE, COMMITTED. Content hash. Phase 13
                                       re-verifies it and hard-aborts on mismatch (D-13.1).
  ground_truth/gt_set.sha256           PHI-FREE, COMMITTED. SCOPE hash — see below.
  ground_truth/text_presence_v2.csv    PHI-FREE, COMMITTED. image_id,has_text.
  ground_truth/gt_summary.txt          PHI-FREE. The summary, also printed.

WHY TWO HASHES
---------------------------------------------------------------------------------
A confirmed-blank image contributes ZERO rows, so it is ABSENT from `gt.csv` rather than
present-and-empty. Two consequences, and each gets its own artifact:

  * `gt.csv` alone cannot distinguish "reviewed and found blank" from "never in the set".
    A build over 199 images and a build over only the 102 text-bearing ones produce a
    BYTE-IDENTICAL `gt.csv` and therefore the same content hash. `gt_set.sha256` pins the
    scope independently, so a silently shrunken scored set cannot hide behind a matching
    content hash.
  * The recall denominator is not in `gt.csv` either. `text_presence_v2.csv` is the only
    record of which images were reviewed-and-blank, and 97 of 199 depend on it. A stratum
    with no text-bearing images has NO recall denominator, so a downstream mean over it
    reads a meaningless saturated 100% — the summary names those strata explicitly
    (CLAUDE.md §8, the `ct_scout` lesson).

THE GATE
---------------------------------------------------------------------------------
Rows come from the 10d review records and NOWHERE else. There is no seed fallback, no
"seed if no review" path, no `--use-seed`, no `--force`, no `--skip-missing`. The 10c seed
is opened only by `review_gt.collect()`, and only to count how much the human changed —
never to produce a row.

Completeness is checked BIDIRECTIONALLY before anything is written: an image in the scored
set with no record fails, a `deferred` record fails, and an ORPHAN record whose `image_id`
is not in the scored set fails too. An orphan means a pilot-round file bled into the
directory or the `image_id` formula changed and produced a disjoint set; both would
silently score the wrong thing.

PHI RULES BAKED IN HERE
  - Nothing PHI to stdout, ever. No token strings, no box contents, no unhashed filenames,
    no DataFrame previews. `image_id` is 8 hex of a sha256 and is safe to name; every
    diagnostic below names images that way and never by content.
  - 10a's `validate_gt()` is what checks the written file, and its report is PHI-free by
    construction (it carries codes and column names, never values) — which is why this
    module can print it.
  - `token_text` is copied BYTE-FOR-BYTE. `normalize()` is frozen and applied at match
    time; normalizing here would double-normalize and corrupt the yardstick. It is
    deliberately not imported.
  - `vendor`/`stratum`/`modality` come from `manifest.csv` through the audited adapter and
    are never re-derived from filenames or model strings (CLAUDE.md rule #7).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

from ground_truth.gt_schema import COLUMNS, GTValidationError, validate_gt

# 10d is committed and pinned; its loaders are reused rather than re-implemented, so the
# scored set, the duplicate-image_id rules and the seed-quality counters can never drift
# between the tool that produced the records and the tool that consumes them.
from ground_truth.review_gt import (
    MANIFEST_COLUMNS,
    MANIFEST_NAME,
    SetupError,
    collect,
    load_groups,
    load_pairs,
    parse_set,
    read_json,
)
from harness.manifest import BLANK_CONTROL_STRATUM, load_manifest
from harness.matching import iou

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_DIR = REPO_ROOT / "ground_truth" / "review"
DEFAULT_ROUND2_DIR = REPO_ROOT / "ground_truth" / "review_r2"
DEFAULT_OUT = REPO_ROOT / "ground_truth" / "gt.csv"

# D-10.7: round 2 pairs a token with its re-review by overlap. `matching.py` exposes 0.5
# only as a default ARGUMENT (`match(..., iou_thr=0.5)`), not as an importable constant, so
# this is a deliberate second copy rather than a reuse — named here so a future change to
# D-4.2's threshold has one grep to find. It governs the self-agreement DIAGNOSTIC only and
# never touches scoring; the two are free to differ, but they should differ on purpose.
SELF_AGREEMENT_IOU = 0.5


class BuildError(Exception):
    """A refusal to build. Every message is PHI-free (counts and image_ids only)."""


# --- inputs ---------------------------------------------------------------------------


def read_render_manifests(specs: list[str]) -> dict[str, dict[str, int]]:
    """`image_id -> {frame_idx, w, h}` from 10b's PHI-FREE render manifest(s).

    D-10.8/decision 6: this is the authority for `frame_idx` — the frame that was actually
    rendered and annotated, and the preimage of that image's `image_id`. `manifest.csv`'s
    `middle_frame_index` is `number_of_frames // 2` from a DICOM tag and disagrees wherever
    the tar holds fewer frames than the tag claims; using it would put a frame index on the
    row that does not describe the pixels the human looked at.
    """
    out: dict[str, dict[str, int]] = {}
    for spec in specs:
        renders_dir, _ = parse_set(spec)
        path = renders_dir / MANIFEST_NAME
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows and tuple(rows[0]) != MANIFEST_COLUMNS:
            raise BuildError(
                f"{path} header is {tuple(rows[0])}, expected {MANIFEST_COLUMNS} — "
                "that file was written by something other than 10b's render.py"
            )
        for row in rows:
            # load_pairs() has already rejected a duplicate image_id whose pixels differ,
            # so a repeat here is the same image and its geometry is the same too.
            out[row["image_id"]] = {
                "frame_idx": int(row["frame_idx"]),
                "w": int(row["w"]),
                "h": int(row["h"]),
            }
    return out


def read_backmap(path: Path) -> dict[str, str]:
    """`image_id -> series_uid` from 10b's back-map. PHI-adjacent: never printed."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = {"image_id", "series_uid"} - set(rows[0] if rows else {})
    if missing:
        raise BuildError(f"{path} is missing column(s) {sorted(missing)} — wrong back-map?")
    mapping: dict[str, str] = {}
    for row in rows:
        image_id, uid = row["image_id"], row["series_uid"]
        prev = mapping.get(image_id)
        if prev is not None and prev != uid:
            # Two different series claiming one image_id makes vendor/stratum arbitrary.
            raise BuildError(f"back-map maps image_id {image_id} to two different series")
        mapping[image_id] = uid
    return mapping


def count_drawn(path: Path | None) -> int | None:
    """Row count of the drawn sub-manifest. A COUNT — no row is ever read out of it."""
    if path is None:
        return None
    # csv.reader, not a line count: a quoted embedded newline in a series_description
    # would otherwise inflate the reconciliation number by one per occurrence.
    with path.open(newline="", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def read_records(review_dir: Path) -> dict[str, dict]:
    """`image_id -> review record`, globbing `*.json` ONLY.

    Not `*`: a 10d save writes `<image_id>.json.<pid>.<tid>.part` first and a killed
    process leaves that behind. A `.part` orphan is inert, but a loose glob would parse one
    as a review record and score a half-written file.
    """
    if not review_dir.is_dir():
        raise BuildError(f"review dir {review_dir} does not exist")
    records: dict[str, dict] = {}
    for path in sorted(review_dir.glob("*.json")):
        record = read_json(path)
        # The filename is NOT trusted as the identity. A copied or renamed record — the
        # obvious way to "fill" a missing one, and a plausible accident when merging
        # batches — would otherwise sail through the completeness gate and attach one
        # image's tokens and boxes to another image's series_uid, vendor and frame_idx.
        # Nothing downstream could catch it: the boxes are in bounds, so 10a validates
        # clean and both hashes look healthy.
        inner = record.get("image_id")
        if inner != path.stem:
            raise BuildError(
                f"review record {path.name} declares image_id {inner!r}, which does not "
                f"match its filename. A record's identity is the id INSIDE it; a copied or "
                f"renamed file would silently give one image another's tokens."
            )
        records[path.stem] = record
    return records


# --- the gate -------------------------------------------------------------------------


def check_completeness(scored: set[str], records: dict[str, dict]) -> None:
    """Refuse to build unless every scored image has a decided review record.

    Bidirectional on purpose. Missing/deferred is the obvious half; the orphan half catches
    a pilot-round file bleeding into the directory, or an `image_id` formula change that
    produced a set disjoint from the one that was actually annotated. Either would build a
    `gt.csv` that looks fine and describes the wrong images.

    Raises BuildError listing every offending `image_id` — a hash, so naming them is safe.
    """
    missing = sorted(scored - set(records))
    orphan = sorted(set(records) - scored)
    deferred = sorted(i for i in scored & set(records) if records[i]["state"] == "deferred")

    problems: list[str] = []
    if missing:
        problems.append(
            f"{len(missing)} scored image(s) have NO review record: {missing}\n"
            "  Annotate them in 10d. There is no --skip-missing: an unreviewed image "
            "must be impossible to include, not merely warned about."
        )
    if deferred:
        problems.append(
            f"{len(deferred)} review record(s) are still 'deferred': {deferred}\n"
            "  A deferred image is UNDECIDED. Resolve each one in 10d before building."
        )
    if orphan:
        problems.append(
            f"{len(orphan)} review record(s) are NOT in the scored set: {orphan}\n"
            "  Either a record from another batch is in this review dir, or the "
            "image_id formula changed and the sets are disjoint. Do not build past it."
        )
    if problems:
        raise BuildError(
            "REFUSING TO BUILD — the scored set is not fully reviewed.\n\n"
            + "\n\n".join(problems)
        )


# --- rows -----------------------------------------------------------------------------


def build_rows(
    scored: list[str],
    records: dict[str, dict],
    renders: dict[str, dict[str, int]],
    backmap: dict[str, str],
    manifest,
) -> tuple[list[list[str]], dict[str, bool]]:
    """Every `gt.csv` row, plus `image_id -> has_text`.

    NOTE there is NO `stratum == "ct_axial"` special case, and that is the point. A blank
    record emits zero rows by itself, and a control frame that turns out to have text emits
    normal rows through the identical path — which is exactly how it "leaves the control
    set" (plan §2.3). A stratum filter would instead make that frame VANISH, deleting the
    one observation the negative control exists to surface.
    """
    rows: list[list[str]] = []
    has_text: dict[str, bool] = {}

    for image_id in scored:
        record = records[image_id]
        try:
            series_uid = backmap[image_id]
        except KeyError:
            raise BuildError(
                f"image_id {image_id} is not in the back-map — the back-map and the "
                "renders dir are from different batches"
            ) from None
        vendor, stratum, modality = manifest.attrs_for_series(series_uid)
        frame_idx = renders[image_id]["frame_idx"]

        tokens = record["tokens"]
        has_text[image_id] = bool(tokens)
        for token in tokens:
            x0, y0, x1, y1 = token["box"]
            rows.append([
                image_id,
                series_uid,
                modality,
                vendor,
                stratum,
                str(frame_idx),
                # RAW. No strip, no case fold, no Unicode fiddling — normalize() is frozen
                # and applied at match time; doing it here double-normalizes the yardstick.
                token["text"],
                _num(x0), _num(y0), _num(x1), _num(y1),
                token["label"],
            ])
    return rows, has_text


def _num(value: float) -> str:
    """Coordinates as canonical, round-trippable text so a rerun is byte-identical."""
    return repr(float(value))


# --- writing --------------------------------------------------------------------------


def write_csv(path: Path, header: tuple[str, ...] | list[str], rows: list[list[str]]) -> None:
    """Deterministic CSV: UTF-8, no BOM, LF, minimal quoting, fixed column order."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(header))
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_set(image_ids: list[str]) -> str:
    """D-10.2 scope hash: sha256 over the sorted image_ids, newline-joined and terminated.

    Full, untruncated, lowercase hex. The 8-char truncation elsewhere in this repo is 10b's
    `image_id` scheme and is unrelated — do not borrow it here.
    """
    payload = "".join(f"{i}\n" for i in sorted(image_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_gt(out: Path, rows: list[list[str]], image_sizes: dict[str, tuple[int, int]]) -> str:
    """Write, validate, and only then put `gt.csv` in place. Returns its sha256.

    Atomic in the same sense as 10d's record writer: the candidate is written to `.part`,
    checked against 10a's frozen spec, and `os.replace`d into position only if it is clean.
    A failed build therefore cannot leave a half-valid `gt.csv` sitting where Phase 13 would
    later hash it.
    """
    part = out.with_suffix(out.suffix + ".part")
    write_csv(part, COLUMNS, rows)
    report = validate_gt(part, image_sizes)
    if not report.ok:
        part.unlink(missing_ok=True)
        # report.summary() is PHI-free by construction (codes + column names, no values).
        raise BuildError(
            "REFUSING TO BUILD — the generated rows violate the frozen 10a spec.\n"
            "No gt.csv was written.\n\n" + report.summary()
        )
    os.replace(part, out)
    return sha256_file(out)


# --- self-agreement (D-10.7) ----------------------------------------------------------


def self_agreement(records: dict[str, dict], round2_dir: Path) -> dict | None:
    """Token-level round-1 vs round-2 agreement, or None if round 2 has not been run.

    None is a DISTINCT return, not a zero: a summary that printed `0.0%` for "not measured"
    would be indistinguishable from a catastrophic result. The caller prints a sentence, not
    a number, when this is None.

    Round 2 never feeds `gt.csv` — round 1 is the ground truth. This only measures it.
    """
    if not round2_dir.is_dir():
        return None
    round2 = {p.stem: read_json(p) for p in sorted(round2_dir.glob("*.json"))}
    # A DEFERRED round-2 record is undecided, not "the reviewer found nothing". Its token
    # list is empty by 10d's own rule, so counting it would turn every round-1 token on that
    # image into a round-1-only miss and depress the agreement figure with no sign that the
    # image was never actually ruled on. Excluded and counted separately instead.
    deferred2 = sorted(i for i, r in round2.items() if r.get("state") == "deferred")
    shared = sorted(set(records) & set(round2) - set(deferred2))
    if not shared:
        return None

    out = {"images": len(shared), "deferred2": len(deferred2), "matched": 0,
           "text_disagree": 0, "label_disagree": 0, "r1_only": 0, "r2_only": 0}
    for image_id in shared:
        r1 = records[image_id]["tokens"]
        r2 = list(round2[image_id]["tokens"])
        unmatched2 = list(range(len(r2)))
        for tok1 in r1:
            best, best_iou = None, SELF_AGREEMENT_IOU
            for j in unmatched2:
                overlap = iou(tuple(tok1["box"]), tuple(r2[j]["box"]))
                if overlap >= best_iou:
                    best, best_iou = j, overlap
            if best is None:
                out["r1_only"] += 1
                continue
            unmatched2.remove(best)
            out["matched"] += 1
            if tok1["text"] != r2[best]["text"]:
                out["text_disagree"] += 1
            if tok1["label"] != r2[best]["label"]:
                out["label_disagree"] += 1
        out["r2_only"] += len(unmatched2)
    return out


# --- summary --------------------------------------------------------------------------


def _pc(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total else "n/a"


def build_summary(ctx: dict) -> str:
    """The ONLY output surface. PHI-free by construction: counts, hashes, hashed ids."""
    out: list[str] = []
    add = out.append

    if ctx["invalidated"]:
        add("!" * 78)
        add("!! HASH CHANGED — EVERY NUMBER PREVIOUSLY REPORTED AGAINST THIS gt.csv IS VOID")
        for which, old, new in ctx["invalidated"]:
            add(f"!!   {which}: {old}")
            add(f"!!   {' ' * len(which)}  -> {new}")
        add("!! Results across two gt.csv builds are not comparable (CLAUDE.md rule #8).")
        add("!! Re-run every engine; do not mix the old numbers with the new ones.")
        add("!" * 78)
        add("")

    add("=== Phase 10e — gt.csv build summary (PHI-free) ===")
    add("")
    add(f"gt.csv sha256      {ctx['gt_hash']}")
    add(f"gt_set.sha256      {ctx['set_hash']}")
    add("                   (scope, not contents — a blank image changes this one only)")
    add("")

    add("-- scored set ------------------------------------------------------------")
    drawn = "n/a (--drawn not given)" if ctx["drawn"] is None else ctx["drawn"]
    add(f"drawn {drawn}  ->  rendered {ctx['rendered']}  ->  reviewed {ctx['reviewed']}"
        f"  (deferred {ctx['deferred']}, missing {ctx['missing']})")
    if ctx["drawn"] is not None and ctx["drawn"] > ctx["rendered"]:
        add(f"  NOTE {ctx['drawn'] - ctx['rendered']} drawn series never rendered. Recorded, "
            "not fatal — they are")
        add("       a fact about finished work, and aborting here would only invite --force.")
    elif ctx["drawn"] is not None and ctx["drawn"] < ctx["rendered"]:
        # More rendered than drawn: several --set dirs, or the wrong --drawn file. Not
        # arithmetic to subtract in the other direction and present as a shortfall.
        add(f"  NOTE {ctx['rendered'] - ctx['drawn']} MORE images rendered than the drawn file "
            "lists — several --set")
        add("       dirs, or --drawn points at a different sample. Check before trusting this.")
    add(f"frame_idx disagreements vs manifest.middle_frame_index: "
        f"{ctx['frame_disagree']} of {ctx['rendered']}")
    add("  (the render manifest wins — it records the frame actually annotated)")
    add("")

    add("-- manifest loaded -------------------------------------------------------")
    add(f"rows {ctx['manifest_rows']}   {BLANK_CONTROL_STRATUM} {ctx['manifest_controls']}")
    add("  (canonical pull is 966 / 66; anything else means a stale or partial manifest)")
    add("")

    add("-- gt.csv ----------------------------------------------------------------")
    add(f"rows {ctx['n_rows']}   images with rows {ctx['n_with_text']}"
        f"   images blank {ctx['n_blank']}  ({_pc(ctx['n_blank'], ctx['rendered'])})")
    add(f"labels   KEEP {ctx['labels'].get('KEEP', 0)}   PHI {ctx['labels'].get('PHI', 0)}")
    add(f"blank-control survivors ({BLANK_CONTROL_STRATUM} with zero rows): "
        f"{ctx['controls_blank']} of {ctx['controls_total']}")
    if ctx["controls_left"]:
        add(f"  {ctx['controls_left']} control frame(s) HAVE text and left the control set: "
            f"{ctx['controls_left_ids']}")
        add("   That is the finding the negative control exists to surface, not an error.")
    add("")

    add("per stratum:")
    add(f"  {'stratum':28} {'images':>7} {'w/text':>7} {'rows':>7} {'KEEP':>7} {'PHI':>7}")
    for stratum in sorted(ctx["by_stratum"]):
        s = ctx["by_stratum"][stratum]
        add(f"  {stratum:28} {s['images']:>7} {s['with_text']:>7} {s['rows']:>7} "
            f"{s['keep']:>7} {s['phi']:>7}")
    add("")
    add("per vendor:")
    for vendor in sorted(ctx["by_vendor"]):
        add(f"  {vendor:28} {ctx['by_vendor'][vendor]:>7} rows")
    add("")

    for stratum in ctx["zero_text_strata"]:
        add(f"WARNING  stratum {stratum!r} has NO text-bearing images. It has no recall")
        add("         denominator, so any per-image recall mean over it reads a")
        add("         meaningless saturated 100%. Report it as such or exclude it.")
    if ctx["zero_text_strata"]:
        add("")

    add("-- 10d review + seed quality --------------------------------------------")
    rounds = ", ".join(f"round {r}: {n}" for r, n in ctx["rounds"])
    add(f"annotation rounds in the review dir: {rounds}")
    if [r for r, _ in ctx["rounds"] if r != 1]:
        add("  NOTE round 1 is the ground truth. A round-2 record in the review dir means a")
        add("       re-review was saved over the original instead of into review_r2/.")
    add(ctx["review_stats"])
    add("")

    add("-- D-10.7 self-agreement -------------------------------------------------")
    agree = ctx["agreement"]
    if agree is None:
        add("self-agreement: not yet run (round 2 absent)")
        add("  ~10% of the set gets re-reviewed after a gap into ground_truth/review_r2/.")
    else:
        paired = agree["matched"]
        add(f"images re-reviewed {agree['images']}   tokens paired at IoU>={SELF_AGREEMENT_IOU} "
            f"{paired}")
        add(f"  text agreement   {paired - agree['text_disagree']}/{paired} "
            f"({_pc(paired - agree['text_disagree'], paired)})")
        add(f"  label agreement  {paired - agree['label_disagree']}/{paired} "
            f"({_pc(paired - agree['label_disagree'], paired)})")
        add(f"  round-1 only {agree['r1_only']}   round-2 only {agree['r2_only']}")
        if agree["deferred2"]:
            add(f"  {agree['deferred2']} round-2 record(s) are DEFERRED and excluded — an")
            add("   undecided re-review is not a disagreement. Resolve them to include them.")
        add("  Round 1 is the ground truth; round 2 measures it and never feeds gt.csv.")
    return "\n".join(out)


# --- main -----------------------------------------------------------------------------


def build(args: argparse.Namespace) -> str:
    groups = load_groups(args.groups)
    # 10d's own loader: it validates the renders/seed pairing and already refuses a
    # duplicate image_id whose pixels or seed provenance differ.
    images, _dup = load_pairs(args.sets, groups)
    renders = read_render_manifests(args.sets)

    scored_set = set(images)
    records = read_records(args.review_dir)
    check_completeness(scored_set, records)   # nothing is written before this returns

    scored = sorted(scored_set)
    backmap = read_backmap(args.backmap)
    manifest = load_manifest(args.manifest)

    rows, has_text = build_rows(scored, records, renders, backmap, manifest)

    # resolve() first: with a bare `--out gt.csv` the parent is "", which would scatter the
    # three sibling artifacts into the CWD while gt.csv itself lands elsewhere.
    out = args.out.resolve()
    sibling = out.parent

    # Read the previous hashes BEFORE writing, so a change can be reported as an
    # invalidation instead of quietly replacing the file everything was scored against.
    gt_hash_path = Path(str(out) + ".sha256")
    set_hash_path = sibling / "gt_set.sha256"
    prev_gt = gt_hash_path.read_text().strip() if gt_hash_path.is_file() else None
    prev_set = set_hash_path.read_text().strip() if set_hash_path.is_file() else None

    image_sizes = {i: (renders[i]["w"], renders[i]["h"]) for i in scored}
    gt_hash = write_gt(out, rows, image_sizes)
    set_hash = sha256_set(scored)
    gt_hash_path.write_text(gt_hash + "\n", encoding="utf-8")
    set_hash_path.write_text(set_hash + "\n", encoding="utf-8")

    presence_path = sibling / args.text_presence_name
    write_csv(presence_path, ("image_id", "has_text"),
              [[i, "1" if has_text[i] else "0"] for i in scored])

    invalidated = []
    if prev_gt is not None and prev_gt != gt_hash:
        invalidated.append(("gt.csv.sha256   ", prev_gt, gt_hash))
    if prev_set is not None and prev_set != set_hash:
        invalidated.append(("gt_set.sha256   ", prev_set, set_hash))

    # --- aggregate, all counted from what was actually loaded ---------------------
    by_stratum: dict[str, dict[str, int]] = {}
    by_vendor: Counter = Counter()
    labels: Counter = Counter()
    for image_id in scored:
        stratum = manifest.attrs_for_series(backmap[image_id])[1]
        s = by_stratum.setdefault(
            stratum, {"images": 0, "with_text": 0, "rows": 0, "keep": 0, "phi": 0})
        s["images"] += 1
        s["with_text"] += int(has_text[image_id])
    for row in rows:
        stratum, vendor, label = row[4], row[3], row[11]
        by_stratum[stratum]["rows"] += 1
        by_stratum[stratum]["keep" if label == "KEEP" else "phi"] += 1
        by_vendor[vendor] += 1
        labels[label] += 1

    controls = [i for i in scored
                if manifest.attrs_for_series(backmap[i])[1] == BLANK_CONTROL_STRATUM]
    controls_left = sorted(i for i in controls if has_text[i])

    summary = build_summary({
        "invalidated": invalidated,
        "gt_hash": gt_hash,
        "set_hash": set_hash,
        "drawn": count_drawn(args.drawn),
        "rendered": len(scored),
        # Counted, not asserted: the gate makes deferred/missing provably 0, and printing
        # the computed value rather than a literal is what would surface a broken gate.
        "reviewed": sum(1 for i in scored if records[i]["state"] != "deferred"),
        "deferred": sum(1 for i in scored if records[i]["state"] == "deferred"),
        "missing": len(scored_set - set(records)),
        "frame_disagree": _frame_disagreements(scored, renders, backmap, manifest),
        "manifest_rows": manifest.n_series(),
        "manifest_controls": manifest.n_blank_control_candidates(),
        "n_rows": len(rows),
        "n_with_text": sum(has_text.values()),
        "n_blank": len(scored) - sum(has_text.values()),
        "labels": labels,
        "controls_total": len(controls),
        "controls_blank": len(controls) - len(controls_left),
        "controls_left": len(controls_left),
        "controls_left_ids": controls_left,
        "by_stratum": by_stratum,
        "by_vendor": dict(by_vendor),
        "zero_text_strata": sorted(k for k, v in by_stratum.items() if v["with_text"] == 0),
        "rounds": sorted(Counter(r.get("round", 1) for r in records.values()).items()),
        "review_stats": _review_stats_block(images, args.review_dir),
        "agreement": self_agreement(records, args.round2_dir),
    })
    (sibling / "gt_summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def _frame_disagreements(scored, renders, backmap, manifest) -> int:
    """How many rendered frames differ from `manifest.middle_frame_index`.

    Reported, never reconciled. The two legitimately differ when the tar holds fewer frames
    than `NumberOfFrames` claims. Goes through the adapter's diagnostic-only accessor, not a
    second reader over `manifest.csv` — this module opens that file through exactly one door.
    """
    return sum(
        1 for i in scored
        if manifest.frame_idx_for_series(backmap[i]) != renders[i]["frame_idx"]
    )


def _review_stats_block(images: dict[str, dict], review_dir: Path) -> str:
    """10d's own per-stratum counters, reused verbatim so the two tools cannot drift.

    `collect()` resolves records through `review_gt.record_path()`, which reads the
    module-level `REVIEW_DIR` at call time — so without the rebinding below it would ignore
    `--review-dir` entirely and silently report counters for a DIFFERENT set of records than
    the ones that produced `gt.csv` (all-zero if that directory is empty, or worse, a stale
    directory's numbers presented as this build's provenance).

    Rebinding a module global is not pretty. The clean fix is a `review_dir` parameter on
    `review_gt.collect()`/`record_path()`, but 10d is committed and this phase must not
    modify it. Contained here, restored in `finally`, and covered by a test.
    """
    import ground_truth.review_gt as review_gt_module

    previous = review_gt_module.REVIEW_DIR
    review_gt_module.REVIEW_DIR = review_dir
    try:
        stats = collect(images)
    finally:
        review_gt_module.REVIEW_DIR = previous
    lines = []
    for key in ["ALL"] + sorted(k for k in stats if k != "ALL"):
        s = stats[key]
        total = s["accepted"] + s["edited"]
        # `unreviewed` is printed even though the gate makes it provably 0, because a
        # non-zero value here is the tell that these counters were resolved against a
        # DIFFERENT review directory than the one that produced the rows above.
        lines.append(
            f"  {key:28} images {s['total']:>4}  accepted {s['accepted']:>4}  "
            f"edited {s['edited']:>4}  deferred {s['deferred']:>3}  "
            f"unreviewed {s['unreviewed']:>3}")
        if total:
            lines.append(
                f"  {'':28} seed: unchanged {s['unchanged']}  text-fixed {s['text_fixed']}  "
                f"box-fixed {s['box_fixed']}  deleted {s['deleted']}  added {s['added']}")
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    """The CLI, factored out so tests exercise the same defaults the user gets."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    metavar="RENDERS:SEED", help="10b renders dir : 10c seed dir")
    ap.add_argument("--groups", type=Path, required=True, help="image_id,group CSV")
    ap.add_argument("--backmap", type=Path, required=True,
                    help="render_backmap CSV for THIS set — there are several; be explicit")
    ap.add_argument("--manifest", type=Path, required=True, help="the canonical manifest.csv")
    ap.add_argument("--drawn", type=Path, default=None,
                    help="the drawn sub-manifest, for the reconciliation line (count only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    ap.add_argument("--round2-dir", type=Path, default=DEFAULT_ROUND2_DIR)
    ap.add_argument("--text-presence-name", default="text_presence_v2.csv")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)

    try:
        print(build(args))
    except (BuildError, GTValidationError, SetupError, ValueError, KeyError) as exc:
        # These five carry counts, codes, paths and image_ids only — never a field value.
        # SetupError is 10d's (missing PNG/seed, duplicate image_id, malformed --set) and
        # ValueError is the manifest adapter's; both would otherwise escape as tracebacks.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Last resort. An unanticipated exception's MESSAGE may quote the value that broke
        # it — a malformed row can put token_text where a coordinate belongs — so the type
        # is printed and the message is deliberately dropped. Re-run under a debugger
        # locally if you need it; this process must not be the thing that leaks PHI.
        print(f"unexpected {type(exc).__name__} — message withheld because an exception "
              f"message can quote a field value. No gt.csv was written by this run.",
              file=sys.stderr)
        del exc
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
