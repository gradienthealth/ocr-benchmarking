#!/usr/bin/env python3
"""Phase 13d — the Qwen3-VL reader's HALLUCINATION FLOOR. 🔴 PHI-TOUCHING.

>>> ARNAV RUNS THIS. It opens real renders. Claude reads only the PHI-free report it writes:
>>> counts, rates, identities, hashes. No token string and no pixel is printed or stored.

    .venv/bin/python experiments/reader_negative_control.py --renders renders/v2 --text-presence ground_truth/text_presence_v2.csv --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --out experiments/qwen3vl_negative_control

(One line on purpose — a backslash-continued paste mangles the flags into escaped spaces.)

WHY A READER NEEDS CONTROL *BOXES*, NOT JUST BLANK IMAGES
---------------------------------------------------------------------------------
An end-to-end engine invents text by drawing a box where there is none, so a blank frame is
already a negative control for it. A reader is HANDED its boxes, so on blank frames it is
handed nothing, reads nothing, and scores a hallucination rate of exactly 0.0 — a number that
means "we never asked", not "it never invents". This script asks: it lays a deterministic grid
of token-shaped boxes over each confirmed-blank frame and counts how many come back non-empty.

Every non-empty read here is an invention, full stop. `harness.reading.read_image` accepts
control boxes ONLY on an image with zero GT tokens, precisely so this can never be run over a
text-bearing frame, where a *correct* read of unannotated glyphs would be miscounted as
invention (see its docstring).

plan.md requires this to run BEFORE any accuracy number for the arm is quoted. A generative
reader is the case the rule was written for: a CRNN garbles, a language model invents something
plausible, and a plausible invention is the failure that survives review.

WHAT IT NEEDS AND WHAT IT WRITES
---------------------------------------------------------------------------------
`--text-presence` is the authority for which frames are blank: 10e's PHI-free
`image_id,has_text` CSV, the human judgement recorded before any run. Nothing here re-derives
blankness from pixels or from a detector — CLAUDE.md §8's "a 0-count from an unvalidated
detector is meaningless" applies to deciding the control set too.

`--gt` is read for exactly two things, neither of which is scoring: the set of `image_id`s that
HAVE tokens (to catch a frame mislabelled blank — the guard `read_image` cannot enforce when
this script passes an empty ground truth), and the median token box height per stratum (to size
the control boxes like real ones). `token_text` is never read, no allowlist is built, and no
token string is ever held in memory.

HOW MANY BLANK FRAMES — DO NOT ASSUME 64
---------------------------------------------------------------------------------
13d's spec says "the 64 blank control images", which is the `ct_axial` count (66 minus the two
that turned out to carry text). The control set this script actually uses is every `has_text=0`
row in `--text-presence`, which for `text_presence_v2.csv` is **97** — `ct_axial` plus the
`ct_scout` and `mg_tomo` blanks that 13h drew its dev-slice floor from. Wider is better
coverage, and it makes `harness/reading.py`'s "the blank control set is all ct_axial" remark
out of date: the per-stratum breakdown in the report is the thing to quote, not one number.

Writes, all PHI-free, under `--out`:
  reading_negative_control.json   `run_reading`'s own aggregate (counts, rates, identity)
  negative_control_report.json    the same headline numbers plus the control-box spec and the
                                  reader config (with the pinned model revision, rule #9)
  negative_control_report.txt     the table, also printed

RUNTIME — CHECK THIS BEFORE STARTING
---------------------------------------------------------------------------------
Cost is per CROP: `n_blank_images x --boxes-per-image` forward passes. On the 8-core avx2 box
one fp32 crop measured ~26 s idle (`experiments/qwen3vl_cpu_timing.py`; ~46 s with another job
on the CPU), so the 97-frame blank set at 4 boxes is ~2.8 hours idle, and at 9 boxes ~6.3. The
script prints its own estimate before it starts. Start at 4 and raise it only if the floor comes
back at exactly zero and you want a tighter bound on "zero".

The floor this produces is an UPPER bound: a free-form refusal ("There is no readable text.")
is scored as a read, because recognizing arbitrary prose as abstention means guessing at intent,
and a wrong guess in that direction HIDES an invention instead of merely inflating the count
(`Qwen3VLReader._is_abstention`).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # runnable as a script, not only as a module
    sys.path.insert(0, str(REPO_ROOT))

# Reused, not re-implemented: 13h's sweep already resolves renders + back-map + manifest into
# `ImageRef`s with the authoritative stamping rules (rule #7 for vendor/stratum/modality, D-10.8
# for frame_idx). A second copy here would be a second place for those rules to drift.
from experiments.sweep_stock_vs_tuned import SweepError, load_dev_images  # noqa: E402
from harness.readers.read_qwen3vl import Qwen3VLReader  # noqa: E402
from harness.reading import BBox, run_reading  # noqa: E402

# --- the control-box grid (constants, recorded in the report) -----------------------------

BOX_HEIGHT_FRAC = 0.035
# Box height as a fraction of image height, so one spec works across render sizes. Burned-in
# overlay text is small; this lands near a real token's height on a 512-1024px frame.

BOX_MIN_HEIGHT = 16
# Floor in pixels. Below this the 48px crop normalization becomes a big upscale of almost
# nothing, which is a different question from the one being asked.

BOX_ASPECT = 6.0
# width = 6 x height, about the shape of an 8-10 character ID. The box shape matters: a VLM
# handed a square of noise is being asked a different question than one handed a text-shaped
# strip, and the strip is what the real arm hands it.

POSITIONS: tuple[tuple[float, float], ...] = (
    (0.02, 0.02),  # the four corners first — where vendor overlays and burned-in IDs live,
    (0.98, 0.02),  # so a partial run still samples the places text would actually be
    (0.02, 0.98),
    (0.98, 0.98),
    (0.50, 0.02),  # then edge midpoints
    (0.50, 0.98),
    (0.02, 0.50),
    (0.98, 0.50),
    (0.50, 0.50),  # then dead centre, where anatomy is and text almost never is
)
# Anchors as (x, y) fractions of the frame, each interpreted as the box's CENTRE and then
# clamped inside the image. Ordered, not random: `--boxes-per-image N` takes the first N, so
# two runs at the same N are the same boxes, and a bigger N is a superset of a smaller one.


def control_boxes_for(
    width: int, height: int, count: int, box_height: float | None = None
) -> list[BBox]:
    """The first `count` grid boxes for one frame, clamped inside it. Deterministic.

    `box_height` is the measured median GT token height for this frame's stratum when one
    exists (see `read_gt_geometry`); the fraction-of-frame fallback is only for a stratum with
    no annotated tokens at all. The difference is not cosmetic: at 3.5% of a large mammo frame
    the box is ~90px tall and `crop_for_reading` DOWNSCALES it to 48px, while every real token
    crop in the scored set is an upscale. A floor measured on downscaled crops is a floor for a
    different input than the accuracy it is supposed to bound.
    """
    box_h = box_height if box_height else max(BOX_MIN_HEIGHT, height * BOX_HEIGHT_FRAC)
    box_h = max(BOX_MIN_HEIGHT, min(float(box_h), float(height)))
    box_w = box_h * BOX_ASPECT
    boxes: list[BBox] = []
    for fx, fy in POSITIONS[:count]:
        cx, cy = fx * width, fy * height
        x0 = min(max(0.0, cx - box_w / 2), max(0.0, width - box_w))
        y0 = min(max(0.0, cy - box_h / 2), max(0.0, height - box_h))
        boxes.append((x0, y0, min(width, x0 + box_w), min(height, y0 + box_h)))
    return boxes


# --- the control set ----------------------------------------------------------------------


def read_blank_ids(path: Path) -> set[str]:
    """`image_id`s recorded as has_text=0 by 10e. The human judgement, not a detector's.

    PHI-free file (hashed ids + a boolean), which is why this one input may be read directly.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not {"image_id", "has_text"} <= set(reader.fieldnames):
            raise SweepError(
                f"{path} header is {reader.fieldnames}, expected image_id,has_text — that is "
                "not a text-presence file written by ground_truth/build_gt.py"
            )
        return {row["image_id"] for row in reader if row["has_text"].strip() == "0"}


def read_gt_geometry(path: Path) -> tuple[set[str], dict[str, float], float | None]:
    """From `gt.csv`: which images have tokens, and how tall a token box is per stratum.

    Returns `(image_ids, median_box_height_by_stratum, median_box_height_overall)`.

    Reads four columns — `image_id`, `stratum`, `y0`, `y1` — and deliberately never touches
    `token_text`, the PHI column. Nothing derived here is PHI: an id set and a few pixel
    medians, all of which land in the PHI-free report.

    Two jobs, both of which the reviewer of an invention floor will ask about:
      1. The CROSS-CHECK. `read_image`'s "control boxes only on a zero-GT image" guard cannot
         fire when this script passes `{}` as the ground truth, so the guard is re-created
         here against the frozen file itself. A frame mislabelled `has_text=0` — two of the 66
         `ct_axial` controls turned out to carry text — would otherwise have its real text read
         correctly and counted as an invention, with nothing anywhere to catch it.
      2. The SCALE. Control boxes sized like real token boxes, so the floor is measured on the
         same kind of crop as the accuracy it bounds.
    """
    heights: dict[str, list[float]] = {}
    image_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        needed = {"image_id", "stratum", "y0", "y1"}
        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
            raise SweepError(f"{path} is missing one of {sorted(needed)} — not a gt.csv")
        for row in reader:
            image_ids.add(row["image_id"])
            heights.setdefault(row["stratum"], []).append(float(row["y1"]) - float(row["y0"]))

    by_stratum = {s: statistics.median(v) for s, v in heights.items() if v}
    everything = [h for v in heights.values() for h in v]
    return image_ids, by_stratum, (statistics.median(everything) if everything else None)


def guard_presence_matches_gt(
    blanks: set[str], gt_ids: set[str], presence_path: Path, gt_path: Path
) -> None:
    """Refuse to run if anything called blank has tokens in `gt.csv`. No override exists.

    This re-creates, against the frozen file, the guard `read_image` cannot apply here: this
    script passes an empty ground truth, so `read_image`'s "control boxes only on a zero-GT
    image" check is structurally unable to fire. A frame wrongly marked `has_text=0` would have
    its real burned-in text read CORRECTLY and counted as an invention — a fabricated number on
    the Added axis, in the one report whose entire purpose is that number.

    Image ids are hashed and PHI-free, so naming the offenders in the error is safe and is the
    only way the operator can go fix the right rows.
    """
    contradicted = sorted(blanks & gt_ids)
    if contradicted:
        shown = contradicted[:5]
        raise SweepError(
            f"{len(contradicted)} image(s) are marked has_text=0 in {presence_path} but HAVE "
            f"tokens in {gt_path}: {shown}{' ...' if len(contradicted) > len(shown) else ''}. "
            "Reading real text on a frame this run calls blank scores it as INVENTION, which is "
            "the one number this procedure exists to produce. Fix the presence file (two of the "
            "66 ct_axial controls really did turn out to carry text) before running."
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--renders", type=Path, required=True, help="10b's render dir")
    ap.add_argument("--text-presence", type=Path, required=True, help="10e's image_id,has_text")
    ap.add_argument("--backmap", type=Path, required=True, help="10b's image_id -> series_uid")
    ap.add_argument("--manifest", type=Path, required=True, help="Gradient's manifest.csv")
    ap.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="the frozen gt.csv — read for the has_text cross-check and token box heights ONLY",
    )
    ap.add_argument("--out", type=Path, required=True, help="output dir (PHI-free artifacts)")
    ap.add_argument(
        "--boxes-per-image",
        type=int,
        default=4,
        help=f"control boxes per blank frame, 1..{len(POSITIONS)} (default 4) — see RUNTIME",
    )
    args = ap.parse_args(argv)

    if not 1 <= args.boxes_per_image <= len(POSITIONS):
        raise SweepError(f"--boxes-per-image must be 1..{len(POSITIONS)}")

    blanks = read_blank_ids(args.text_presence)
    gt_ids, height_by_stratum, height_overall = read_gt_geometry(args.gt)

    guard_presence_matches_gt(blanks, gt_ids, args.text_presence, args.gt)

    rendered = load_dev_images(args.renders, args.backmap, args.manifest)
    images = [img for img in rendered if img.id in blanks]
    if not images:
        raise SweepError(
            f"no image in {args.renders} is marked has_text=0 in {args.text_presence}. Without "
            "a blank set there is no invention floor, and an accuracy number without a floor is "
            "the ct_scout mistake (CLAUDE.md §8) — stop rather than report one."
        )
    missing = len(blanks) - len(images)

    box_heights = {
        img.id: height_by_stratum.get(img.stratum) or height_overall for img in images
    }
    boxes = {
        img.id: control_boxes_for(img.w, img.h, args.boxes_per_image, box_heights[img.id])
        for img in images
    }
    n_crops = sum(len(b) for b in boxes.values())

    # A distinct arm label, so this run's rows can never pool with an accuracy run's — the
    # control-box spec is NOT part of `arm_config_hash` (it belongs to the caller, not the
    # reader), and without this the two are identity-indistinguishable to `aggregate()`.
    reader = Qwen3VLReader(config_id=f"floor-{args.boxes_per_image}box")
    print(
        f"{len(images)} blank frames x {args.boxes_per_image} boxes = {n_crops} crops; "
        f"at ~26 s/crop (idle box) that is ~{n_crops * 26 / 3600:.1f} h",
        file=sys.stderr,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    result = run_reading(
        images,
        {},  # no ground truth: every image here is blank by construction
        reader,
        allowlist=set(),  # no KEEP tokens exist on a blank frame, so none can be redacted
        out_path=args.out / "reading_negative_control.json",
        control_boxes=boxes,
    )

    floor = result["negative_control"]
    report = {
        "arm": "reader",
        "what": "hallucination floor — every non-empty read below is INVENTED text",
        "model_name": result["model_name"],
        "version": result["version"],
        "config_id": result["config_id"],
        "config_hash": result["config_hash"],
        "reader_config": reader.config(),  # carries the pinned model revision (rule #9)
        "control_box_spec": {
            "height_source": "median GT token height for the frame's stratum, from --gt",
            "height_frac_fallback": BOX_HEIGHT_FRAC,
            "min_height_px": BOX_MIN_HEIGHT,
            "aspect": BOX_ASPECT,
            "positions": [list(p) for p in POSITIONS[: args.boxes_per_image]],
            "median_gt_box_height_by_stratum": {
                s: round(h, 2) for s, h in sorted(height_by_stratum.items())
            },
            "median_gt_box_height_overall": (
                round(height_overall, 2) if height_overall else None
            ),
        },
        "floor_is_an_upper_bound": (
            "a free-form refusal ('there is no text here') is scored as a read, not an "
            "abstention — see Qwen3VLReader._is_abstention"
        ),
        "crop_preprocessing": result["crop_preprocessing"],
        "text_presence_source": str(args.text_presence),
        "n_blank_images_scored": len(images),
        "n_blank_ids_not_rendered": missing,
        "n_control_crops": n_crops,
        "inventions": floor["floor_count"],
        "inventions_per_image": floor["hallucinations_per_image"],
        "invention_rate_per_crop": (floor["floor_count"] / n_crops) if n_crops else None,
        "per_stratum": {
            stratum: {"added_count": stats.get("added_count")}
            for stratum, stats in result["per_stratum"].items()
        },
    }

    lines = [
        "Qwen3-VL reader — hallucination floor (negative control)",
        "=" * 60,
        f"model            {report['model_name']} @ {reader.config()['revision'][:12]}",
        f"transformers     {report['version']}    config {report['config_id']} "
        f"[{report['config_hash']}]",
        f"blank frames     {report['n_blank_images_scored']}"
        + (f"  (+{missing} blank ids not in this render dir)" if missing else ""),
        f"control crops    {report['n_control_crops']} "
        f"({args.boxes_per_image} per frame, corners first)",
        "-" * 60,
        f"INVENTIONS       {report['inventions']}",
        f"per crop         {_pct(report['invention_rate_per_crop'])}",
        f"per image        {report['inventions_per_image']}",
        "-" * 60,
        "Any number above zero is text the model produced where a human confirmed there is",
        "none. Report it beside the arm's accuracy, never averaged into it.",
    ]
    text = "\n".join(lines)
    print(text)

    (args.out / "negative_control_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out / "negative_control_report.txt").write_text(text + "\n", encoding="utf-8")
    return 0


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.4f} ({rate * 100:.2f}%)"


if __name__ == "__main__":
    raise SystemExit(main())
