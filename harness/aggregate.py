"""Phase 6 aggregation — roll per-image score rows up to per-stratum + overall. 🟢 PHI-free.

`score()` (Phase 5) emits raw, unaveraged per-image ingredients; this is where they become
rates and latency percentiles. Two CLAUDE.md invariants are preserved structurally:

- **Hallucination and omission are never averaged together.** The Found axis
  (`found_count`/`omission_count`) and the Added axis (`added_count`) stay in separate keys;
  no combined "accuracy"/"error" number is ever produced.
- **False-redaction rate is the headline** — `sum(false_redaction_count) / sum(keep_total)`,
  KEEP-denominated. CER/WER are rolled up ONLY inside a nested `diagnostic` sub-dict (macro
  mean over the per-image `diagnostic` values); they share no key with the ranking group, so
  a refactor cannot accidentally promote them into the ranking signal — exactly as the scorer
  keeps them quarantined per-image. Phase 7's report reads them from `diagnostic`.

Latency is reported as mean/median/p95, split into the model's own time (`elapsed`, the
primary `run_func` call) and the verifier's time (`verifier_elapsed`) — never merged, so the
gated setup's two costs stay legible. The negative-control hallucination floor is aggregated
over just the confirmed-blank rows.

Grouping key is `row["stratum"]`, which `run_harness` stamps authoritatively from the manifest
(CLAUDE.md rule #7) — so blank controls (zero GT tokens, `None` stratum from `score()`) still
land in their real stratum. Aggregation is an order-independent reduction: shuffling `rows`
yields an identical result.

`detection_view()` (Phase 13e) is a SECOND, self-contained view of the same rows — boxes vs
GT at the matcher's IoU bar, strings ignored — so a loss can be attributed to *finding* the
text rather than *reading* it. It adds no inference and re-runs nothing; it re-reads counts
`score()` already returns. It hangs off the aggregate under `"detection"` and is reported in
its own table, never merged into the rates above.

PHI-free: consumes only the numeric/string ingredients in the score rows — never token text
or pixels.
"""

from __future__ import annotations

import statistics
from typing import Optional


def _latency_stats(values: list[float]) -> Optional[dict]:
    """mean/median/p95 over a list of latencies, or None if empty.

    p95 uses linear interpolation between the two nearest ranks (statistics.quantiles,
    exclusive method); for one value it degenerates to that value.
    """
    if not values:
        return None
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "median": v, "p95": v}
    ordered = sorted(values)
    # 100 cut points -> the 95th is the p95 boundary; interpolates between neighbours.
    p95 = statistics.quantiles(ordered, n=100, method="exclusive")[94]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": float(p95),
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    """numerator/denominator, or None when the denominator is 0 (no division by zero)."""
    return (numerator / denominator) if denominator else None


def _group_by_stratum(rows: list[dict]) -> dict[str, list[dict]]:
    """Group score rows by `row["stratum"]`, the manifest-stamped key (CLAUDE.md rule #7).

    One definition shared by every rollup in this module. If the grouping rule ever changes
    — a `None` stratum, a re-stamp — the detector view and the end-to-end tables cannot
    drift apart into grouping the same run two different ways.
    """
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    return by_stratum


def _reject_unidentified_rows(rows: list[dict]) -> None:
    """Refuse a batch that cannot be shown to come from ONE engine build + config arm.

    Every rollup entry point in this module calls this BEFORE summing anything, so there is
    no door into the module through which two engine versions, or two arms of one engine,
    can be pooled into a single number (rule #9). See `aggregate()`'s Raises section for
    what each check is for; the checks run in this order because the tuple comparison at the
    end is blind to a batch that is uniformly blank.
    """
    if not rows:
        return

    # Blank-version check FIRST (D-8.4). The mixed-pair check below cannot catch this:
    # rows that all carry version "" collapse to ONE pair, so a batch with no version
    # information at all looks maximally consistent. Counts/indices only — never row
    # contents — so this message stays PHI-free.
    blank = [i for i, r in enumerate(rows) if not str(r.get("version") or "").strip()]
    if blank:
        raise ValueError(
            f"aggregate(): {len(blank)} of {len(rows)} rows carry no engine version "
            f"(first at row index {blank[0]}). An unversioned row cannot be shown to "
            "come from the same engine build as its neighbours, and an all-blank batch "
            "would pass the (model_name, version) consistency check as one pair "
            "(rule #9)."
        )

    # Same hole, same fix, for the SECOND model in a gated run (D-9.1). Only rows that
    # actually ran a verifier are in scope — `verifier_elapsed` (not the identity
    # fields) is the signal, since a primary-only row legitimately has no verifier
    # identity at all.
    blank_verifier = [
        i
        for i, r in enumerate(rows)
        if r.get("verifier_elapsed") is not None
        and (
            not str(r.get("verifier_model_name") or "").strip()
            or not str(r.get("verifier_version") or "").strip()
        )
    ]
    if blank_verifier:
        raise ValueError(
            f"aggregate(): {len(blank_verifier)} of {len(rows)} rows ran a verifier but "
            f"carry no verifier identity (first at row index {blank_verifier[0]}). A "
            "gated row cannot be shown to come from the same verifier build as its "
            "neighbours without one, for the same reason an unversioned primary row is "
            "rejected above."
        )

    # Blank-config check, for the same reason the blank-version check runs above and
    # BEFORE the tuple comparison (D-13.5): rows that ALL carry a blank `config_hash`
    # collapse to one consistent tuple, so the comparison below would pass them as a
    # single identity — which is the collision this field exists to prevent.
    unconfigured = [
        i for i, r in enumerate(rows) if not str(r.get("config_hash") or "").strip()
    ]
    if unconfigured:
        raise ValueError(
            f"aggregate(): {len(unconfigured)} of {len(rows)} rows carry no "
            f"`config_hash` (first at row index {unconfigured[0]}). Two arms of the "
            "same engine at the same version — docTR stock vs tuned vs parseq, "
            "PP-OCRv6 stock vs tuned thresholds — are indistinguishable without it and "
            "would be averaged into one meaningless row. Produce it with "
            "Runner.config_hash() from the runner's declared config() (D-13.5)."
        )

    pairs = {
        (
            r.get("model_name"),
            r.get("version"),
            r.get("config_id"),
            r.get("config_hash"),
            r.get("verifier_model_name"),
            r.get("verifier_version"),
        )
        for r in rows
    }
    if len(pairs) > 1:
        conflicting = sorted(pairs, key=lambda p: tuple(str(x) for x in p))
        raise ValueError(
            "aggregate(): refusing to blend rows from different (model_name, version, "
            "config_id, config_hash, verifier_model_name, verifier_version) tuples: "
            f"{conflicting}"
        )


def _group_stats(rows: list[dict]) -> dict:
    """Aggregate one group of score rows (a stratum, or the whole set).

    Every count is an order-independent sum; the Found and Added axes are kept in
    disjoint keys and never combined.
    """
    n = len(rows)
    keep_total = sum(r["keep_total"] for r in rows)
    fr_count = sum(r["false_redaction_count"] for r in rows)
    kem_count = sum(r["keep_exact_match_count"] for r in rows)
    costs = [r["cost"] for r in rows]
    verifier_latencies = [
        r["verifier_elapsed"] for r in rows if r.get("verifier_elapsed") is not None
    ]

    # --- diagnostic-only rollup (CER/WER, char censoring) — NEVER a ranking signal ---
    # Kept in a nested key so it can't be mistaken for a headline metric, mirroring how
    # score() quarantines it per-image. CER/WER are macro-averaged over the images that
    # actually produced a value (jiwer returns None when an image has no matched pairs);
    # char counts are order-independent sums.
    diagnostics = [r.get("diagnostic") or {} for r in rows]
    cers = [d["cer"] for d in diagnostics if d.get("cer") is not None]
    wers = [d["wer"] for d in diagnostics if d.get("wer") is not None]
    keep_chars_total = sum(d.get("keep_chars_total", 0) for d in diagnostics)
    keep_chars_censored = sum(d.get("keep_chars_censored", 0) for d in diagnostics)

    return {
        "n_images": n,
        # --- headline: false redaction (KEEP-denominated) ---
        "false_redaction_count": fr_count,
        "keep_total": keep_total,
        "false_redaction_rate": _rate(fr_count, keep_total),
        # --- KEEP read fidelity ---
        "keep_exact_match_count": kem_count,
        "keep_exact_match_rate": _rate(kem_count, keep_total),
        # --- Found (omission) axis — never blended with Added ---
        "found_count": sum(r["found_count"] for r in rows),
        "omission_count": sum(r["omission_count"] for r in rows),
        "gt_total": sum(r["gt_total"] for r in rows),
        # --- Added (hallucination) axis — never blended with Found ---
        "added_count": sum(r["added_count"] for r in rows),
        # --- latency, split primary vs verifier ---
        "latency": _latency_stats([r["elapsed"] for r in rows]),
        "verifier_latency": _latency_stats(verifier_latencies),
        # --- cost ---
        "cost": {"sum": sum(costs), "mean": (sum(costs) / n) if n else None},
        # --- DIAGNOSTIC ONLY — never a ranking signal (nested, see comment above) ---
        "diagnostic": {
            "cer_mean": statistics.fmean(cers) if cers else None,
            "wer_mean": statistics.fmean(wers) if wers else None,
            "n_cer_images": len(cers),
            "keep_chars_total": keep_chars_total,
            "keep_chars_censored": keep_chars_censored,
            "char_censor_rate": _rate(keep_chars_censored, keep_chars_total),
        },
    }


# --- detector-only view (Phase 13e) --------------------------------------------
# A SECOND view of the SAME rows, answering a different question: did the engine *find*
# the text, ignoring entirely whether it *read* it. Nothing here consults a string — only
# the counts the IoU >= `iou_thr` matcher already produced. It is rendered as its own
# table and never merged with the end-to-end or reader numbers: three questions, three
# tables.
#
# Two denominators are kept strictly apart, because collapsing them is the exact defect
# that made the 2026-08-06 `ct_scout` sweep meaningless (CLAUDE.md §8):
#
# - RECALL is denominated in GT tokens on TEXT-BEARING images only. A blank control has
#   zero GT tokens, so averaging it in as a per-image "100% recall" invents signal out of
#   images that had nothing to find. Here it cannot happen structurally: recall is a
#   pooled sum(found)/sum(gt_total), and a group with no text-bearing images has a zero
#   denominator -> None -> `n/a` with a reason, never 100%.
# - PRECISION is denominated in predictions on those same text-bearing images. Predictions
#   on blank controls are 100% false positives by construction, so they are pulled out
#   into the hallucination FLOOR and reported on their own line — averaging them into a
#   detection precision would smear the floor across the score it exists to bound.


def _detection_group(rows: list[dict]) -> dict:
    """Detection-only counts for one group of score rows (a stratum, or the whole set).

    Only rows an IoU matcher actually produced are counted. Two kinds are EXCLUDED — and
    counted, never silently folded in, because both would read as a *better* detector than
    was measured:

    - **box-free rows.** The box-free path returns `hallucinations=[]` unconditionally
      (matching.py — with no per-word structure there is no "prediction overlapping no GT
      token" to point at), so such a row contributes a fake `added_count == 0` and reads as
      perfect detection precision.
    - **reading-arm rows** (`iou_thr is None`, Phase 13b). That arm is *handed the GT box*
      and runs no matcher at all, so every GT token is "found" by construction: including it
      would print a saturated 100% detection recall, the exact failure this table exists to
      prevent. `reading.py` says as much where it stamps the None.

    The exclusion covers excluded CONTROL rows too, so `n_control_images`/`floor_boxes` here
    can be smaller than the run-wide `negative_control` block — deliberately: such a frame
    reports no boxes because it structurally cannot, not because none were invented, so
    counting it would dilute the floor toward zero. The report names the excluded rows.

    Known property, inherited from the matcher and NOT worked around here: a prediction with
    `bbox=None` in a boxed output can never be matched (matching.py) yet is still counted a
    hallucination, so it lands in `added_count` and depresses detection precision for a box
    that never existed. Changing that is a change to the frozen scoring path; it is recorded
    here rather than silently compensated for.
    """
    box_free = [r for r in rows if r.get("box_free")]
    # `iou_thr is None` == no matcher ran (13b's reading arm). A row that never went
    # through matching.py cannot answer "did it FIND the text" — the boxes were a given.
    no_matcher = [r for r in rows if not r.get("box_free") and r.get("iou_thr") is None]
    boxed = [r for r in rows if not r.get("box_free") and r.get("iou_thr") is not None]
    # `negative_control` is score()'s "this image had zero GT tokens" flag. NOTE it means
    # exactly that and nothing more: an image absent from the loaded ground truth also
    # lands here (run_harness passes `ground_truth.get(img.id, [])`), so running an engine
    # over a wider image set than the loaded gt.csv covers would move real text-bearing
    # frames into the floor. That is a property of the run's inputs, not something this
    # view can detect — the `gt_set.sha256` in the run metadata is what pins it.
    control = [r for r in boxed if r["negative_control"]]
    scored = [r for r in boxed if not r["negative_control"]]
    # Counted so the report can say WHY a floor is missing rather than printing a 0 that
    # reads as "measured, and the engine invented nothing."
    excluded_control = [r for r in box_free + no_matcher if r["negative_control"]]

    gt_total = sum(r["gt_total"] for r in scored)
    found = sum(r["found_count"] for r in scored)
    added = sum(r["added_count"] for r in scored)
    predicted = found + added  # greedy 1:1 assignment -> every prediction is in exactly one
    floor_boxes = sum(r["added_count"] for r in control)

    # Every `n/a` states which zero it came from, and each branch claims only what is true
    # of THIS group: the empty group asserts no controls, and a group that also holds
    # box-free rows says so rather than blaming a control set for the missing denominator.
    n_excluded = len(box_free) + len(no_matcher)
    excluded = f"; {n_excluded} row(s) excluded (no IoU matcher ran)" if n_excluded else ""
    if box_free and not no_matcher:
        unmatched_why = "box-free run — no boxes, so detection is not scoreable"
    elif no_matcher and not box_free:
        unmatched_why = (
            "reading arm — the reader is handed the GT box, so there is nothing to detect"
        )
    else:
        unmatched_why = (
            f"no IoU-matched rows — {len(box_free)} box-free, {len(no_matcher)} reading-arm"
        )

    if gt_total:
        recall_na = None
    elif not rows:
        recall_na = "no images in this group"
    elif not boxed:
        recall_na = unmatched_why
    elif scored:
        recall_na = f"no ground-truth tokens on the boxed text-bearing images{excluded}"
    else:
        recall_na = (
            f"no boxed text-bearing images — all {len(control)} boxed image(s) are blank "
            f"controls{excluded}, so there is no recall denominator"
        )

    if predicted:
        precision_na = None
    elif not rows:
        precision_na = "no images in this group"
    elif not boxed:
        precision_na = unmatched_why
    elif scored:
        precision_na = f"the engine predicted nothing on the boxed text-bearing images{excluded}"
    else:
        precision_na = (
            f"no boxed text-bearing images — all {len(control)} boxed image(s) are blank "
            f"controls{excluded}, and their predictions are counted in the hallucination floor"
        )

    return {
        # Denominator bookkeeping — the split is the point, so it is reported, not implied.
        # `n_text_images`, NOT `n_images`: the sibling `_group_stats` already publishes
        # `n_images` meaning EVERY image in the group, and two different denominators under
        # one key is how a detection number gets compared against an end-to-end one.
        "n_text_images": len(scored),  # text-bearing + boxed: the recall/precision base
        "n_control_images": len(control),  # blank controls: floor only, never in the above
        "n_box_free_excluded": len(box_free),
        "n_reading_arm_excluded": len(no_matcher),
        "n_excluded_controls": len(excluded_control),  # why a floor can be missing
        # Found axis (detection recall)
        "gt_total": gt_total,
        "found_count": found,
        "omission_count": sum(r["omission_count"] for r in scored),
        "detection_recall": _rate(found, gt_total),
        "recall_na_reason": recall_na,
        # Added axis (detection precision) — separate axis, never averaged with the above
        "added_count": added,
        "predicted_count": predicted,
        "detection_precision": _rate(found, predicted),
        "precision_na_reason": precision_na,
        # hallucination floor, held out of both rates above
        "floor_boxes": floor_boxes,
        "floor_boxes_per_image": _rate(floor_boxes, len(control)),
    }


def detection_view(rows: list[dict]) -> dict:
    """The detector-only view of a run: per-stratum + overall detection counts and rates.

    Adds no inference and re-runs no engine — it is a re-reading of the counts `score()`
    already returned, so a loss can be attributed to *finding* the text rather than
    *reading* it. Order-independent (sums + sorted strata), like every other rollup here.

    `iou_thr` is the threshold every row was matched at, or None if the batch disagrees —
    which would mean two incomparable matchings were pooled (D-4.2).

    Raises:
        ValueError: the same engine/config identity failures `aggregate()` rejects (D-8.4,
            D-9.1, D-13.5, D-8.3). This is a public rollup entry point in its own right, so
            it must not become the one door into the module that lets two arms of an engine
            — or two engine versions — be pooled into a single number (rule #9).
    """
    _reject_unidentified_rows(rows)
    by_stratum = _group_by_stratum(rows)
    thresholds = {r.get("iou_thr") for r in rows}
    return {
        "iou_thr": thresholds.pop() if len(thresholds) == 1 else None,
        "overall": _detection_group(rows),
        "per_stratum": {
            stratum: _detection_group(by_stratum[stratum])
            for stratum in sorted(by_stratum, key=str)
        },
    }


def aggregate(rows: list[dict]) -> dict:
    """Roll score rows up to per-stratum + overall stats plus the negative-control floor.

    Returns:
        {
          "model_name": <carried from the rows, for Phase 7>,
          "version": <carried from the rows (D-8.3); always non-empty, see Raises>,
          "config_id": <carried from the rows (D-13.5); the human label for this arm>,
          "config_hash": <carried from the rows (D-13.5); always non-empty, see Raises>,
          "verifier_model_name": <carried from the rows; None if no row ran a verifier>,
          "verifier_version": <carried from the rows; None if no row ran a verifier>,
          "n_images": int,
          "overall": <group stats over all rows>,
          "per_stratum": {stratum: <group stats>, ...},   # sorted by stratum
          "negative_control": {floor_count, n_control_images, hallucinations_per_image},
          "detection": <detection_view(rows)>,   # detector-only second view, own table
        }

    Empty input yields zero counts and None rates throughout — never an exception.

    Raises:
        ValueError: any row carries a blank/missing `version` (D-8.4). Checked before the
            mixed-pair check, which is blind to it: an all-blank batch is one consistent
            pair. `OCROutput` also rejects a blank version at construction, so this is the
            backstop for rows assembled by hand rather than by a runner.
        ValueError: a row ran a verifier (`verifier_elapsed` is not None) but carries a
            blank/missing `verifier_model_name`/`verifier_version` (D-9.1) — same hole as
            above, extended to the second model in a gated run. `run_harness` also rejects
            this at call time; this is the backstop for hand-built rows.
        ValueError: rows come from more than one (model_name, version, verifier_model_name,
            verifier_version) tuple (D-8.3, extended by D-9.1) — an engine or verifier
            version bump invalidates prior results like a gt.csv change, so a mixed batch
            must fail loudly rather than silently average across versions. This also catches
            blending a primary-only run with a gated run: one has verifier identity `None`,
            the other doesn't, so they are never the same tuple.
    """
    _reject_unidentified_rows(rows)
    by_stratum = _group_by_stratum(rows)

    control_rows = [r for r in rows if r["negative_control"]]
    floor_count = sum(r["negative_control_floor_count"] for r in control_rows)
    n_control = len(control_rows)

    return {
        "model_name": rows[0].get("model_name") if rows else None,
        "version": rows[0].get("version") if rows else None,
        "config_id": rows[0].get("config_id") if rows else None,
        "config_hash": rows[0].get("config_hash") if rows else None,
        "verifier_model_name": rows[0].get("verifier_model_name") if rows else None,
        "verifier_version": rows[0].get("verifier_version") if rows else None,
        "n_images": len(rows),
        "overall": _group_stats(rows),
        "per_stratum": {
            stratum: _group_stats(by_stratum[stratum])
            # str() key so a None stratum (shouldn't occur post-stamp) can't break sorting
            for stratum in sorted(by_stratum, key=str)
        },
        "negative_control": {
            "floor_count": floor_count,
            "n_control_images": n_control,
            "hallucinations_per_image": _rate(floor_count, n_control),
        },
        # Detector-only view: same rows, different question (found vs read). Nested under
        # its own key and rendered as its own table — never merged into the rates above.
        "detection": detection_view(rows),
    }
