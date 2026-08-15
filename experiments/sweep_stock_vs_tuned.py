#!/usr/bin/env python3
"""Phase 13h — the stock-vs-tuned threshold sweep, on the dev slice. 🔴 PHI-TOUCHING.

>>> ARNAV RUNS THIS. It reads dev-slice renders and dev_v1.csv (both PHI). Claude reads only
>>> the PHI-free report it writes: counts, rates, latencies, config values, hashes.

    .venv/bin/python experiments/sweep_stock_vs_tuned.py --renders renders/dev_v1 --gt ground_truth/dev_v1.csv --backmap ground_truth/render_backmap_dev_v1.csv --manifest manifest.csv --out experiments/sweep_dev_v1

(One line on purpose — a backslash-continued paste mangles the flags into escaped spaces.)

D-13.4 — WHY THIS REFUSES TO TOUCH gt_v1
---------------------------------------------------------------------------------
`gt.csv` is small and frozen. Sweeping thresholds against it and reporting the best result is
fitting the test set: the tuned number stops being a measurement and becomes an optimistic
bound, and the stock-vs-tuned delta gets inflated by exactly the amount of overfitting
(plan.md, "Trap 1"). D-13.4 resolved this by annotating `dev_v1`, a slice drawn from OUTSIDE
the scored set, so the tuned config is chosen without ever seeing `gt_v1`.

Two guards enforce that, because the failure is silent and permanent:
  1. `--gt` may not BE `ground_truth/gt.csv` (path identity), and
  2. `--gt`'s sha256 may not equal the committed `gt.csv.sha256` (content identity, which
     catches a copy of the scored set under another name).
Neither has an override flag. A `--force` here would be a flag whose only use is to void
every number the project reports.

PICKING THE METRIC AFTER THE FACT — `--from-report`
---------------------------------------------------------------------------------
Which metric ranks the configs is an open question (RANK_METRICS below), and the honest
answer is "look at the numbers first". Every trial's full `overall` and `per_stratum` block
is in `sweep_report.json`, so the choice does not have to be made before a 2.5-hour sweep:

    .venv/bin/python experiments/sweep_stock_vs_tuned.py --from-report experiments/sweep_dev_v1_doctr --from-report experiments/sweep_dev_v1_paddle --rank-by keep_exact_match_rate --freeze

That re-runs SELECTION ONLY — same `select_from_trials`, no engine constructed, no render
opened, no `--gt` needed — and freezes the winners under the metric named. Reports may be
sharded per engine (one process per engine is how this is run, so a crash costs one engine
instead of all thirteen trials); shards from different dev sets, or two shards claiming the
same engine, are refused rather than merged. The D-13.4 scored-set guard is re-applied to
the dev set the report records, so the offline route is not a way around it.

WHAT IT WRITES (all PHI-FREE, all under --out)
---------------------------------------------------------------------------------
  sweep_report.txt     the table, also printed
  sweep_report.json    the same numbers, machine-readable, for the final report
  (with --freeze)      experiments/tuned_configs.json — the winning config per engine

Nothing here writes a token string, a box, a raw_response, or an image path. The aggregate
that `run_harness` returns is counts and rates only; per-image score rows are never written.

SELECTION RULE — PER-STRATUM, NEVER POOLED
---------------------------------------------------------------------------------
A config wins on the pooled ranking metric ONLY IF it also fails to regress against stock in
any single stratum by more than `--max-stratum-regression`. A gate calibrated on one stratum
can delete real text on another: `conf>=60, len>=2`, chosen on CT because it flattened
invention, deleted 29-47% of genuine burned-in text on ultrasound (CLAUDE.md §8). Invention
rates differ by two orders of magnitude across strata, so a pooled mean can improve while a
whole stratum is being destroyed. Disqualified configs are listed in the report with the
stratum that disqualified them — silently dropping them would read as "nothing else was
tried".

Sweep size is recorded per engine. The protocol does not need an upper-bound disclaimer
(that was option (c), not the one chosen), but the number of configurations tried is the
audit trail for the claim that it doesn't.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # runnable as a script, not only as a module
    sys.path.insert(0, str(REPO_ROOT))

from ground_truth.build_gt import read_backmap  # noqa: E402
from ground_truth.gt_schema import load_gt  # noqa: E402
from ground_truth.render import MANIFEST_COLUMNS, MANIFEST_NAME  # noqa: E402
from harness.contract import normalize  # noqa: E402
from harness.harness import ImageRef, run_harness  # noqa: E402
from harness.manifest import load_manifest  # noqa: E402
from harness.runners.arms import TUNED_CONFIGS  # noqa: E402

FROZEN_GT = REPO_ROOT / "ground_truth" / "gt.csv"
FROZEN_GT_HASH = REPO_ROOT / "ground_truth" / "gt.csv.sha256"

ENGINES = ("doctr", "pp-ocrv6_medium", "easyocr")

# Ranking metrics, and which direction is better. Both are exposed rather than hardcoded
# because the project's own two documents disagree about which one leads: CLAUDE.md §1 calls
# false redaction the headline metric, while the reading-quality framing ranks on KEEP exact
# match. The sweep does not resolve that — it RECORDS which metric selected the config, so a
# reader of the final report can never be unsure what "tuned" was tuned for.
RANK_METRICS = {
    "false_redaction_rate": "lower",
    "keep_exact_match_rate": "higher",
}


class SweepError(RuntimeError):
    """A guard tripped. Never caught in this module — it must reach the operator."""


# --- the D-13.4 guard -------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard_not_the_scored_set(gt_path: Path) -> None:
    """Refuse to sweep against `gt_v1`, by path AND by content. No override exists.

    The two checks are complementary: the path check catches the obvious invocation, the
    hash check catches `cp ground_truth/gt.csv /tmp/dev.csv`. A rebuilt-but-identical scored
    set trips the second; a renamed one trips it too.
    """
    if gt_path.resolve() == FROZEN_GT.resolve():
        raise SweepError(
            f"--gt is {FROZEN_GT} — the FROZEN SCORED SET. Tuning against it fits the test "
            "set and turns every tuned number into an optimistic bound (D-13.4, plan.md "
            "Trap 1). Point --gt at the dev slice (ground_truth/dev_v1.csv)."
        )
    if not FROZEN_GT_HASH.is_file():
        return  # nothing to compare against; the path check above still applies
    frozen = FROZEN_GT_HASH.read_text(encoding="utf-8").strip()
    if frozen and sha256_file(gt_path) == frozen:
        raise SweepError(
            f"--gt {gt_path} has the same sha256 as the frozen gt.csv ({frozen[:12]}...) — "
            "it IS the scored set under another name. Tuning on it voids the measurement "
            "(D-13.4). Use the dev slice."
        )


def guard_report_not_the_scored_set(gt_name: str, gt_hash: str) -> None:
    """The same D-13.4 refusal, applied to a report's recorded dev set.

    `--from-report` reaches `freeze_winners` without ever touching `--gt`, so the guard above
    never runs on that path. Without this, a sweep that WAS run against the scored set could
    be frozen from its own report — the offline route would be a hole straight through the
    one rule the module says has no override.
    """
    if gt_name == FROZEN_GT.name:
        raise SweepError(
            f"the report's dev set is named {gt_name!r} — that is the FROZEN SCORED SET. "
            "Freezing a config selected on it fits the test set (D-13.4)."
        )
    if not FROZEN_GT_HASH.is_file():
        return
    frozen = FROZEN_GT_HASH.read_text(encoding="utf-8").strip()
    if frozen and gt_hash == frozen:
        raise SweepError(
            f"the report's dev set has the same sha256 as the frozen gt.csv ({frozen[:12]}...) "
            "— it IS the scored set under another name. Its winners must not be frozen (D-13.4)."
        )


# --- inputs -----------------------------------------------------------------------------


def load_reports(paths: Sequence[Path]) -> dict:
    """Merge sharded `sweep_report.json` files into one ctx. PHI-free in, PHI-free out.

    Each path is a `--out` directory or the JSON inside it. Sharding by engine is the normal
    way this sweep is run — the engines are independent and one process per engine gives a
    write point per engine instead of one after all thirteen trials — so the offline path has
    to be able to put the shards back together.

    Three things are refused rather than merged, because each would produce a `tuned_configs.json`
    whose entries were measured on different footing while the file claims a single `dev_set`:
      * shards recording different dev sets (name, hash, or image count),
      * the same engine appearing in two shards, which would silently pick one, and
      * a dev set that is really the frozen scored set (the guard above).
    """
    ctxs: list[tuple[Path, dict]] = []
    for path in paths:
        json_path = path / "sweep_report.json" if path.is_dir() else path
        if not json_path.is_file():
            raise SweepError(
                f"{json_path} does not exist. --from-report takes a sweep --out directory or "
                "its sweep_report.json; a shard that has not finished has not written one yet."
            )
        ctxs.append((json_path, json.loads(json_path.read_text(encoding="utf-8"))))

    identity = {k: ctxs[0][1].get(k) for k in ("gt_name", "gt_hash", "n_images")}
    for json_path, ctx in ctxs[1:]:
        other = {k: ctx.get(k) for k in ("gt_name", "gt_hash", "n_images")}
        if other != identity:
            raise SweepError(
                f"{json_path} was swept against a different dev set ({other}) than "
                f"{ctxs[0][0]} ({identity}). Freezing across them would write one dev_set "
                "into tuned_configs.json while its entries came from two."
            )

    guard_report_not_the_scored_set(identity["gt_name"], identity["gt_hash"])

    engines: list[dict] = []
    seen: dict[str, Path] = {}
    for json_path, ctx in ctxs:
        for res in ctx.get("engines", []):
            name = res["engine"]
            if name in seen:
                raise SweepError(
                    f"{name} appears in both {seen[name]} and {json_path}. Which measurement "
                    "wins is not a choice this script may make silently — pass one of them."
                )
            seen[name] = json_path
            engines.append(res)
    if not engines:
        raise SweepError(f"no engine results in {[str(p) for p, _ in ctxs]}")

    merged = dict(identity)
    merged["engines"] = engines
    merged["source_reports"] = [str(p) for p, _ in ctxs]
    return merged


def read_render_manifest(renders_dir: Path) -> dict[str, dict[str, int]]:
    """`image_id -> {frame_idx, w, h}` from 10b's PHI-FREE render manifest.

    D-10.8: the render manifest is the authority for `frame_idx` — the frame actually
    rendered and annotated — never `manifest.csv`'s `middle_frame_index`.
    """
    path = renders_dir / MANIFEST_NAME
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != MANIFEST_COLUMNS:
            raise SweepError(
                f"{path} header is {reader.fieldnames}, expected {list(MANIFEST_COLUMNS)} — "
                "that file was written by something other than 10b's render.py"
            )
        return {
            row["image_id"]: {
                "frame_idx": int(row["frame_idx"]),
                "w": int(row["w"]),
                "h": int(row["h"]),
            }
            for row in reader
        }


def load_dev_images(renders_dir: Path, backmap_path: Path, manifest_path: Path) -> list[ImageRef]:
    """Every rendered dev-slice image as a pixel-free `ImageRef`.

    `stratum`/`modality`/`vendor` come from `manifest.csv` via the adapter, never re-derived
    from a filename or a model string (CLAUDE.md rule #7). `frame_idx` comes from the render
    manifest, which is that rule's one recorded exception (D-10.8).
    """
    renders = read_render_manifest(renders_dir)
    backmap = read_backmap(backmap_path)
    manifest = load_manifest(manifest_path)

    images: list[ImageRef] = []
    for image_id in sorted(renders):  # sorted for determinism, not for meaning (D-1.1)
        series_uid = backmap.get(image_id)
        if series_uid is None:
            raise SweepError(
                f"image_id {image_id} is in the render manifest but not the back-map — "
                "wrong --backmap for these renders? Its stratum and vendor would be unknown."
            )
        # (vendor, stratum, modality) — that order, not the ImageRef field order. Unpacking
        # it as (modality, stratum, vendor) reads plausibly and mislabels every row.
        vendor, stratum, modality = manifest.attrs_for_series(series_uid)
        geom = renders[image_id]
        images.append(
            ImageRef(
                id=image_id,
                path=str(renders_dir / f"{image_id}.png"),
                w=geom["w"],
                h=geom["h"],
                stratum=stratum,
                modality=modality,
                vendor=vendor,
                frame_idx=geom["frame_idx"],
            )
        )
    return images


def build_allowlist(gt: dict[str, list]) -> set[str]:
    """The redaction pipeline's known-good set: normalized KEEP token values.

    PHI, held in memory only. Never written, never printed — the report this script emits
    carries counts and rates, and this set is neither.
    """
    return {
        normalize(token.token_text)
        for tokens in gt.values()
        for token in tokens
        if token.label == "KEEP"
    }


# --- the grids --------------------------------------------------------------------------


def doctr_grid() -> list[dict[str, Any]]:
    """docTR's detector postprocess thresholds. Stock is `{}` (no override, D-9.2).

    The `db_resnet50` architecture override that plan.md also allows under "tuned" is
    deliberately NOT in this grid: swapping the architecture swaps the WEIGHTS, so folding it
    in would make the step-6 finding read "thresholds gained N points" when part of N came
    from a different model. If it is to be measured, it is its own arm.
    """
    return [
        {},
        {"bin_thresh": 0.3, "box_thresh": 0.1},
        {"bin_thresh": 0.3, "box_thresh": 0.3},
        {"bin_thresh": 0.1, "box_thresh": 0.05},
        {"bin_thresh": 0.5, "box_thresh": 0.3},
    ]


def paddle_grid() -> list[dict[str, Any]]:
    """PP-OCRv6's four detection knobs. Stock is `{}` (PaddleOCR's own defaults).

    `text_det_limit_side_len` leads the grid because step 6 flags it as mattering most here:
    burned-in tokens are small, and the default resize can shrink them below what the
    detector resolves. Raising it costs compute — the report carries latency per config so a
    tuned win that is really a compute win is visible as one.
    """
    return [
        {},
        {"text_det_limit_side_len": 1280},
        {"text_det_limit_side_len": 1920},
        {"text_det_limit_side_len": 1280, "text_det_thresh": 0.2},
        {"text_det_limit_side_len": 1280, "text_det_box_thresh": 0.3},
        {"text_det_limit_side_len": 1280, "text_det_unclip_ratio": 2.0},
    ]


def easyocr_grid() -> list[dict[str, Any]]:
    """EasyOCR's pair is FIXED, not swept — both ends are already known.

    Its stock arm is the library default (0.7/0.4) and its tuned arm is the 0.2/0.2 config
    Phase 9 shipped. The question this pair answers is not "what is the best threshold" but
    "how much of EasyOCR's over-redaction is EasyOCR and how much is our threshold choice"
    (step 6). Sweeping it would answer a question nobody asked and would spend budget on the
    arm that is explicitly the floor, not a contender.
    """
    from harness.runners.run_easyocr import (
        DETECTION_THRESHOLD,
        LIBRARY_LOW_TEXT,
        LIBRARY_TEXT_THRESHOLD,
    )

    return [
        {"text_threshold": LIBRARY_TEXT_THRESHOLD, "low_text": LIBRARY_LOW_TEXT},
        {"text_threshold": DETECTION_THRESHOLD, "low_text": DETECTION_THRESHOLD},
    ]


GRIDS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "doctr": doctr_grid,
    "pp-ocrv6_medium": paddle_grid,
    "easyocr": easyocr_grid,
}


def build_runner(engine: str, kwargs: dict[str, Any]):
    """Construct one engine at one config. Imports are local so one missing engine is not fatal."""
    if engine == "doctr":
        from harness.runners.run_doctr import DoctrRunner

        return DoctrRunner(**kwargs)
    if engine == "pp-ocrv6_medium":
        from harness.runners.run_paddle_v6 import PaddleV6Runner

        return PaddleV6Runner(**kwargs)
    from harness.runners.run_easyocr import EasyOcrRunner

    return EasyOcrRunner(**kwargs)


# --- the sweep --------------------------------------------------------------------------


def run_trial(runner, images: list[ImageRef], gt: dict, allowlist: set[str]) -> dict:
    """One config over the whole dev slice. Returns `aggregate()`'s PHI-free dict."""
    return run_harness(images, runner.run, gt, allowlist=allowlist)


def stratum_regressions(stock: dict, cand: dict, metric: str, tol: float) -> list[dict]:
    """Every stratum where `cand` is worse than `stock` by more than `tol`.

    Compared per stratum and never pooled: the pooled mean is exactly what hid a 29-47%
    deletion of genuine ultrasound text the last time a gate was accepted on one number
    (CLAUDE.md §8). A stratum missing from either side is skipped, not treated as equal —
    it has no denominator to compare.
    """
    better = RANK_METRICS[metric]
    out = []
    for stratum, cand_stats in sorted(cand.get("per_stratum", {}).items()):
        stock_stats = stock.get("per_stratum", {}).get(stratum)
        if not stock_stats:
            continue
        a, b = stock_stats.get(metric), cand_stats.get(metric)
        if a is None or b is None:
            continue
        delta = (b - a) if better == "higher" else (a - b)  # positive = candidate improved
        if delta < -tol:
            out.append({"stratum": stratum, "stock": a, "candidate": b, "delta": delta})
    return out


def sweep_engine(
    engine: str,
    images: list[ImageRef],
    gt: dict,
    allowlist: set[str],
    metric: str,
    tol: float,
    budget: int | None,
) -> dict:
    """Every config in `engine`'s grid, scored, with the winner selected and disclosed."""
    grid = GRIDS[engine]()
    if budget is not None:
        # Truncation is announced, never silent: a capped sweep that reads as a complete one
        # would make "we tried everything" false in the report.
        dropped = max(0, len(grid) - budget)
        grid = grid[:budget]
    else:
        dropped = 0

    trials: list[dict] = []
    for kwargs in grid:
        runner = build_runner(engine, kwargs)
        agg = run_trial(runner, images, gt, allowlist)
        trials.append({
            "config": dict(kwargs),
            "config_id": runner.config_id,
            "config_hash": runner.config_hash(),
            "declared_config": runner.config(),
            "version": runner.version,
            "overall": agg["overall"],
            "per_stratum": agg["per_stratum"],
            "negative_control": agg["negative_control"],
        })

    return select_from_trials(engine, trials, metric, tol, dropped)


def select_from_trials(
    engine: str,
    trials: list[dict],
    metric: str,
    tol: float,
    dropped: int = 0,
) -> dict:
    """Choose the winner among already-measured `trials`. Pure — builds no engine, reads no pixel.

    Split out of `sweep_engine` so `--from-report` can re-run selection over trials that were
    measured in an earlier run. Selection reads only `config_id`, `overall` and `per_stratum`,
    and `sweep_report.json` carries all three — so re-ranking under the other metric costs a
    file read instead of a second sweep of every engine.

    Both callers go through this one function on purpose. A second implementation for the
    offline path is exactly how the printed report and the frozen `tuned_configs.json` would
    come to disagree about which config won, with nothing to flag the divergence.
    """
    stock = next((t for t in trials if t["config_id"] == "stock"), None)
    if stock is None:
        raise SweepError(
            f"{engine}'s grid produced no stock arm — the delta this whole step reports is "
            "stock-vs-tuned, so a sweep without a stock baseline measures nothing."
        )

    better = RANK_METRICS[metric]
    eligible, disqualified = [], []
    for trial in trials:
        if trial is stock:
            continue
        regressions = stratum_regressions(stock, trial, metric, tol)
        if regressions:
            disqualified.append({"config": trial["config"], "regressions": regressions})
        else:
            eligible.append(trial)

    def key(t: dict) -> float:
        value = t["overall"].get(metric)
        if value is None:
            # No denominator -> cannot rank. Sorted last under either direction.
            return float("inf") if better == "lower" else float("-inf")
        return value

    ranked = sorted(eligible, key=key, reverse=(better == "higher"))
    winner = ranked[0] if ranked else None

    # A candidate must actually BEAT stock on the pooled metric to win. Without this, a grid
    # where every candidate is worse than stock — but by less than `tol` in each individual
    # stratum, so nothing is disqualified — crowns the least-bad one and freezes it as the
    # tuned arm. Step 6 would then report a tuning LOSS as the tuned result. "No candidate
    # beat stock" is a finding; a negative delta presented as a winner is a wrong number.
    if winner is not None:
        stock_value, win_value = stock["overall"].get(metric), winner["overall"].get(metric)
        if stock_value is None or win_value is None:
            winner = None
        elif (win_value >= stock_value) if better == "lower" else (win_value <= stock_value):
            winner = None

    return {
        "engine": engine,
        "version": trials[0]["version"],
        "sweep_size": len(trials),
        "grid_truncated_by_budget": dropped,
        "rank_metric": metric,
        "rank_direction": better,
        "max_stratum_regression": tol,
        "stock": stock,
        "trials": trials,
        "disqualified": disqualified,
        "winner": winner,
    }


# --- the report (PHI-free) --------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_report(ctx: dict) -> str:
    """The only human-facing output. Counts, rates, config values, hashes — no token text."""
    out: list[str] = []
    add = out.append

    add("=== Phase 13h — stock vs tuned, dev-slice sweep (PHI-free) ===")
    add("")
    add(f"dev set            {ctx['gt_name']}  sha256 {ctx['gt_hash']}")
    add(f"images             {ctx['n_images']}")
    add(f"ranked on          {ctx['rank_metric']} ({ctx['rank_direction']} is better)")
    add(f"stratum guard      a config is disqualified if any stratum regresses > "
        f"{ctx['max_stratum_regression']}")
    add("")
    add("D-13.4: tuned on a dev slice drawn from OUTSIDE gt_v1, so these thresholds were")
    add("chosen without seeing the scored set. The winners below are FROZEN — re-tuning")
    add("after seeing gt_v1 numbers reintroduces the bias the slice was annotated to avoid.")
    add("")

    for res in ctx["engines"]:
        add(f"-- {res['engine']} {res['version']} " + "-" * max(0, 56 - len(res['engine'])))
        add(f"configurations tried: {res['sweep_size']}"
            + (f"  (grid truncated by --budget: {res['grid_truncated_by_budget']} dropped)"
               if res["grid_truncated_by_budget"] else ""))
        add("")
        add(f"  {'config_id':<10} {'hash':<14} {'false_red':>10} {'keep_exact':>11} "
            f"{'added':>7} {'omitted':>8} {'med_ms':>8}")
        for trial in res["trials"]:
            o = trial["overall"]
            # "median", not "p50": aggregate._latency_stats emits mean/median/p95 and
            # nothing else, so a p50 lookup silently prints n/a on every row and deletes
            # the latency evidence this column exists to show (raising
            # text_det_limit_side_len costs compute, so a tuned win can be a compute win).
            latency = (o.get("latency") or {}).get("median")
            add(f"  {trial['config_id']:<10} {trial['config_hash']:<14} "
                f"{_fmt(o.get('false_redaction_rate')):>10} "
                f"{_fmt(o.get('keep_exact_match_rate')):>11} "
                f"{o.get('added_count', 0):>7} {o.get('omission_count', 0):>8} "
                f"{('n/a' if latency is None else f'{latency * 1000:.0f}'):>8}")
            if trial["config"]:
                add(f"             {json.dumps(trial['config'], sort_keys=True)}")
        add("")

        if res["disqualified"]:
            add("  DISQUALIFIED on the per-stratum guard (pooled numbers looked fine):")
            for d in res["disqualified"]:
                worst = min(d["regressions"], key=lambda r: r["delta"])
                add(f"    {json.dumps(d['config'], sort_keys=True)}")
                add(f"      worst: {worst['stratum']} {_fmt(worst['stock'])} -> "
                    f"{_fmt(worst['candidate'])} (delta {worst['delta']:+.4f})")
            add("")

        winner = res["winner"]
        if winner is None:
            add("  WINNER: none — every candidate regressed a stratum. The stock arm stands;")
            add("  do NOT freeze a tuned config for this engine on this evidence.")
        else:
            stock_value = res["stock"]["overall"].get(res["rank_metric"])
            win_value = winner["overall"].get(res["rank_metric"])
            add(f"  WINNER: {json.dumps(winner['config'], sort_keys=True)}")
            add(f"    config_hash {winner['config_hash']}")
            add(f"    {res['rank_metric']}: {_fmt(stock_value)} (stock) -> {_fmt(win_value)}")
            add("    Report this as its own finding — the delta is not folded into the")
            add("    headline number, and stock and tuned are separate rows, never averaged.")
        add("")

    add("-- per-stratum, winner vs stock ------------------------------------------")
    add("(the pooled number above can improve while a stratum is destroyed — CLAUDE.md §8)")
    for res in ctx["engines"]:
        # Printed for every engine, INCLUDING one with no winner. "Nothing beat stock" is
        # the case where the per-stratum breakdown matters most — it is the evidence for
        # that verdict — and omitting it would leave the reader with a bare refusal.
        winner = res["winner"]
        add(f"  {res['engine']}:" + ("" if winner else "   (no winner — stock arm only)"))
        add(f"    {'stratum':<24} {'stock':>10} {'tuned':>10} {'delta':>10}")
        for stratum in sorted(res["stock"]["per_stratum"]):
            a = res["stock"]["per_stratum"][stratum].get(res["rank_metric"])
            b = None if winner is None else (
                winner["per_stratum"].get(stratum, {}).get(res["rank_metric"])
            )
            if a is None or b is None:
                add(f"    {stratum:<24} {_fmt(a):>10} {_fmt(b):>10} {'n/a':>10}")
                continue
            delta = (b - a) if res["rank_direction"] == "higher" else (a - b)
            add(f"    {stratum:<24} {_fmt(a):>10} {_fmt(b):>10} {delta:>+10.4f}")
    add("")
    return "\n".join(out)


def freeze_winners(ctx: dict, path: Path) -> list[str]:
    """Write the winning config per engine into `tuned_configs.json`. Returns engines written.

    Only engines with a winner are written, and an engine that was SWEPT WITHOUT producing
    one has any previous entry DELETED — it does not merely go unwritten. Merging without
    deleting is the trap: re-sweep on a new dev set, doctr now has no winner, the run prints
    "froze nothing", and `build_arm("doctr:tuned")` still resolves to the stale config from
    the old sweep — now sitting under this file's freshly overwritten `dev_set` and
    `rank_metric`, which claim it came from the new one.

    Engines NOT in this run are left untouched: a `--engine easyocr` run must not silently
    unfreeze doctr.
    """
    doc: dict[str, Any] = {"engines": {}}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("engines", {})

    doc["d_13_4"] = (
        "Tuned on dev_v1, a slice annotated from OUTSIDE the frozen gt_v1 scored set, so "
        "these thresholds were chosen without seeing the set they are scored on. FROZEN: "
        "re-tuning after seeing gt_v1 numbers reintroduces the bias the slice exists to "
        "avoid."
    )
    doc["dev_set"] = {"csv": ctx["gt_name"], "sha256": ctx["gt_hash"], "n_images": ctx["n_images"]}
    doc["rank_metric"] = ctx["rank_metric"]

    # Where the selection happened. Popped when absent for the same reason a winner-less
    # engine's entry is deleted rather than left: a stale "reselected offline from these
    # shards" line sitting above freshly measured winners describes a run that did not happen.
    if ctx.get("source_reports"):
        doc["selected_offline_from"] = list(ctx["source_reports"])
    else:
        doc.pop("selected_offline_from", None)

    written = []
    for res in ctx["engines"]:
        if res["engine"] == "easyocr":
            # Its pair is fixed by definition, not selected: `easyocr:tuned` IS the shipped
            # 0.2/0.2 config and `arms.build_arm` constructs it directly. Writing a "frozen
            # winner" for it would imply a choice the sweep did not make.
            continue
        if res["winner"] is None:
            # Swept, nothing won -> the previous answer for this engine is now stale.
            doc["engines"].pop(res["engine"], None)
            continue
        doc["engines"][res["engine"]] = {
            "config": res["winner"]["config"],
            "config_hash": res["winner"]["config_hash"],
            "engine_version": res["version"],
            "sweep_size": res["sweep_size"],
            "rank_metric": res["rank_metric"],
            "max_stratum_regression": res["max_stratum_regression"],
        }
        written.append(res["engine"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


# --- main -------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The four PHI inputs and --out are required to SWEEP and meaningless to --from-report,
    # which reads finished reports and builds no engine. argparse cannot express "required
    # unless"; main() enforces it so the error names the missing flag instead of a stack trace.
    ap.add_argument("--renders", type=Path, default=None,
                    help="dev-slice renders dir (with render_manifest.csv)")
    ap.add_argument("--gt", type=Path, default=None,
                    help="the DEV slice csv — never ground_truth/gt.csv")
    ap.add_argument("--backmap", type=Path, default=None, help="10b's image_id -> series_uid map")
    ap.add_argument("--manifest", type=Path, default=None, help="the canonical manifest.csv")
    ap.add_argument("--out", type=Path, default=None, help="PHI-free report dir")
    ap.add_argument("--from-report", type=Path, action="append", default=None, metavar="PATH",
                    help="re-select winners from finished sweep_report.json shards instead of "
                         "sweeping; repeatable. Runs no engine and reads no pixel.")
    ap.add_argument("--engine", action="append", choices=ENGINES, default=None,
                    help="repeatable; default all three")
    ap.add_argument("--rank-by", choices=sorted(RANK_METRICS), default="false_redaction_rate")
    ap.add_argument("--max-stratum-regression", type=float, default=0.02,
                    help="disqualify a config regressing any stratum by more than this")
    ap.add_argument("--budget", type=int, default=None,
                    help="max configs per engine; equal across engines, and disclosed")
    # `.relative_to` raises when the constant has been redirected outside the repo (a test
    # pointing it at tmp_path), which would turn --help into a traceback.
    shown = (TUNED_CONFIGS.relative_to(REPO_ROOT)
             if TUNED_CONFIGS.is_relative_to(REPO_ROOT) else TUNED_CONFIGS)
    ap.add_argument("--freeze", action="store_true", help=f"write the winners into {shown}")
    return ap.parse_args(argv)


def _reselect(args: argparse.Namespace) -> dict:
    """`--from-report`: re-rank finished trials under `--rank-by`. No engine, no pixel, no GT."""
    merged = load_reports(args.from_report)
    wanted = set(args.engine or [])
    results = [
        select_from_trials(res["engine"], res["trials"], args.rank_by,
                           args.max_stratum_regression,
                           res.get("grid_truncated_by_budget", 0))
        for res in merged["engines"]
        if not wanted or res["engine"] in wanted
    ]
    if not results:
        raise SweepError(
            f"--engine {sorted(wanted)} selected none of the engines in these reports "
            f"({sorted(r['engine'] for r in merged['engines'])})."
        )
    return {
        **{k: merged[k] for k in ("gt_name", "gt_hash", "n_images")},
        "rank_metric": args.rank_by,
        "rank_direction": RANK_METRICS[args.rank_by],
        "max_stratum_regression": args.max_stratum_regression,
        "source_reports": merged["source_reports"],
        "engines": results,
    }


def _sweep(args: argparse.Namespace) -> dict:
    """The measuring path: build every engine at every config and score the dev slice."""
    missing = [f"--{n}" for n in ("renders", "gt", "backmap", "manifest") if getattr(args, n) is None]
    if missing:
        raise SweepError(f"sweeping needs {', '.join(missing)} (or pass --from-report instead)")

    # FIRST, before any engine is built or any pixel is read.
    guard_not_the_scored_set(args.gt)

    images = load_dev_images(args.renders, args.backmap, args.manifest)
    image_sizes = {img.id: (img.w, img.h) for img in images}
    gt = load_gt(args.gt, image_sizes)
    allowlist = build_allowlist(gt)

    engines: Iterable[str] = args.engine or ENGINES
    results = [
        sweep_engine(engine, images, gt, allowlist, args.rank_by,
                     args.max_stratum_regression, args.budget)
        for engine in engines
    ]

    return {
        "gt_name": args.gt.name,
        "gt_hash": sha256_file(args.gt),
        "n_images": len(images),
        "rank_metric": args.rank_by,
        "rank_direction": RANK_METRICS[args.rank_by],
        "max_stratum_regression": args.max_stratum_regression,
        "engines": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    offline = bool(args.from_report)
    ctx = _reselect(args) if offline else _sweep(args)

    report = build_report(ctx)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        # Named apart from a measured sweep's output on the offline path. A re-ranked report
        # dropped into a shard directory as `sweep_report.json` would be indistinguishable from
        # the measurement, and would then be re-readable by --from-report as if it were one.
        stem = "reselected_report" if offline else "sweep_report"
        (args.out / f"{stem}.txt").write_text(report + "\n", encoding="utf-8")
        (args.out / f"{stem}.json").write_text(
            json.dumps(ctx, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    elif not offline:
        raise SweepError("--out is required when sweeping — the measurement must be written down")
    print(report)

    if offline:
        print(f"re-ranked on {args.rank_by} from {', '.join(ctx['source_reports'])} "
              "— no engine was run, no dev-slice pixel was read.")

    if args.freeze:
        written = freeze_winners(ctx, TUNED_CONFIGS)
        print(f"froze tuned config for: {written or 'nothing — no engine had a winner'}")
    else:
        print("NOT frozen (--freeze not given). build_arm('<engine>:tuned') stays unavailable.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
